#include <algorithm>
#include <array>
#include <chrono>
#include <filesystem>
#include <iostream>
#include <string>
#include <fstream>

#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>
#include <xtensor/xadapt.hpp>
#include <xtensor/xindex_view.hpp>
#include <xtensor/xio.hpp>
#include <xtensor/xpad.hpp>
#include <xtensor/xtensor.hpp>
#include <xtensor-io/xnpz.hpp>
#include <xtensor/xhistogram.hpp>
#include <xtensor/xrandom.hpp>
#include <xtensor/xmanipulation.hpp>
#include "lrucache.hpp"

using namespace std::string_literals;
using ImageSize = std::array<size_t, 2>;

constexpr size_t EMBEDDINGS_CACHE_SIZE = 1024;
constexpr size_t IMAGE_ENCODER_INPUT_SIZE = 256;
const ov::Shape INPUT_SHAPE = {IMAGE_ENCODER_INPUT_SIZE, IMAGE_ENCODER_INPUT_SIZE, 3};

bool starts_with(const std::string& str, const std::string& prefix) {
  return str.size() >= prefix.size() && str.compare(0, prefix.size(), prefix) == 0;
}

bool ends_with(const std::string& str, const std::string& suffix) {
  return str.size() >= suffix.size() && str.compare(str.size() - suffix.size(), suffix.size(), suffix) == 0;
}

bool contains_substr(const std::string& str, const std::string& substr) {
  return str.find(substr) != std::string::npos;
}

std::array<size_t, 2> get_preprocess_shape(size_t oldh, size_t oldw, size_t encoder_input_size = IMAGE_ENCODER_INPUT_SIZE) {
  double scale = 1.0 * encoder_input_size / std::max(oldh, oldw);
  size_t newh = scale * oldh + 0.5;
  size_t neww = scale * oldw + 0.5;
  return {newh, neww};
}

void majority_voting(std::string seg_file, xt::xtensor<uint16_t, 3>& axial, xt::xtensor<uint16_t, 3>& coronal, xt::xtensor<uint16_t, 3>& sagittal) {
  xt::xtensor<uint16_t, 4> stacked_arrays = xt::stack(xt::xtuple(axial, coronal, sagittal), 3);
  // std::cout << "stacked_arrays size " << xt::adapt(stacked_arrays.shape()) << std::endl;

    // Create an array to store the result
    xt::xarray<uint16_t> result = xt::zeros<uint16_t>({axial.shape()[0], axial.shape()[1], axial.shape()[2]});

    // Apply majority voting along the last axis
    // Iterate over each element in the 3D volume
    for (size_t i = 0; i < axial.shape()[0]; ++i) {
        for (size_t j = 0; j < axial.shape()[1]; ++j) {
            for (size_t k = 0; k < axial.shape()[2]; ++k) {
                // Extract the 1D array of values along the new axis
                auto values = xt::view(stacked_arrays, i, j, k, xt::all());

                // Compute the bin counts
                auto counts = xt::bincount(values);

                // Find the value with the maximum count (the majority value)
                auto majority_value = xt::argmax(counts)();

                // Assign the majority value to the result
                result(i, j, k) = majority_value;
            }
        }
    }

  xt::dump_npz(seg_file, "segs", result, true);
}

xt::xtensor<float, 1> get_bbox(xt::xtensor<float, 2>& mask) {
  auto indices = xt::where(mask > 0);
  auto y_indices = indices[0], x_indices = indices[1];
  auto x_min = *std::min_element(x_indices.begin(), x_indices.end());
  auto x_max = *std::max_element(x_indices.begin(), x_indices.end());
  auto y_min = *std::min_element(y_indices.begin(), y_indices.end());
  auto y_max = *std::max_element(y_indices.begin(), y_indices.end());
  return {(float)x_min, (float)y_min, (float)x_max, (float)y_max};
}

template <class T>
T cast_npy_file(xt::detail::npy_file& npy_file) {
  auto m_typestring = npy_file.m_typestring;
  if (m_typestring == "|u1") {
    return npy_file.cast<uint8_t>();
  } else if (m_typestring == "<u2") {
    return npy_file.cast<uint16_t>();
  } else if (m_typestring == "<u4") {
    return npy_file.cast<uint32_t>();
  } else if (m_typestring == "<u8") {
    return npy_file.cast<uint64_t>();
  } else if (m_typestring == "|i1") {
    return npy_file.cast<int8_t>();
  } else if (m_typestring == "<i2") {
    return npy_file.cast<int16_t>();
  } else if (m_typestring == "<i4") {
    return npy_file.cast<int32_t>();
  } else if (m_typestring == "<i8") {
    return npy_file.cast<int64_t>();
  } else if (m_typestring == "<f4") {
    return npy_file.cast<float>();
  } else if (m_typestring == "<f8") {
    return npy_file.cast<double>();
  }
  XTENSOR_THROW(std::runtime_error, "Cast error: unknown format "s + m_typestring);
}

struct Encoder {
  ov::CompiledModel model;
  ov::InferRequest infer_request;
  ImageSize original_size, new_size;

  Encoder(ov::Core& core, const std::string& model_path) {
    model = core.compile_model(model_path, "CPU");
    infer_request = model.create_infer_request();
  }

  void set_sizes(const ImageSize& original_size, const ImageSize& new_size) {
    this->original_size = original_size;
    this->new_size = new_size;
  }

  ov::Tensor encode_image(const ov::Tensor& input_tensor) {
    infer_request.set_input_tensor(input_tensor);
    infer_request.infer();
    return infer_request.get_output_tensor();
  }

xt::xtensor<uint8_t, 2> get_img_2d(xt::xtensor<uint8_t, 3>& img_3D, int i, std::string orientation) {
    xt::xtensor<uint8_t, 2> slice;

    if (orientation == "axial") {
        slice = xt::view(img_3D, i, xt::all(), xt::all());
    } else if (orientation == "coronal") {
        slice = xt::view(img_3D, xt::all(), i, xt::all());
    } else if (orientation == "sagittal") {
        slice = xt::view(img_3D, xt::all(), xt::all(), i);
    } else {
        throw std::invalid_argument("Invalid orientation");
    }
    return slice;
}

  xt::xtensor<float, 3> preprocess_2D(xt::xtensor<uint8_t, 3>& original_img, size_t encoder_input_size = IMAGE_ENCODER_INPUT_SIZE) {
    assert(original_img.shape()[2] == 3);
    cv::Mat mat1(cv::Size(original_size[1], original_size[0]), CV_8UC3, original_img.data()), mat2;
    cv::resize(mat1, mat2, cv::Size(new_size[1], new_size[0]), cv::INTER_LINEAR);

    xt::xtensor<float, 3> img = xt::adapt((uint8_t*)mat2.data, mat2.total() * mat2.channels(), xt::no_ownership(), std::vector<int>{mat2.rows, mat2.cols, mat2.channels()});
    img = (img - xt::amin(img)()) / std::clamp(xt::amax(img)() - xt::amin(img)(), 1e-8f, 1e18f);
    return xt::pad(img, {{0, encoder_input_size - new_size[0]}, {0, encoder_input_size - new_size[1]}, {0, 0}});
  }

  xt::xtensor<float, 4> preprocess_efficient_sam_2D(xt::xtensor<uint8_t, 3>& original_img, size_t encoder_input_size = IMAGE_ENCODER_INPUT_SIZE) {
    assert(original_img.shape()[2] == 3);
    // std::cout << "preprocess efficient sam" << std::endl;
    cv::Mat mat1(cv::Size(original_size[1], original_size[0]), CV_8UC3, original_img.data());
    // cv::resize(mat1, mat2, cv::Size(new_size[1], new_size[0]), cv::INTER_LINEAR);

    xt::xtensor<float, 3> img = xt::adapt((uint8_t*)mat1.data, mat1.total() * mat1.channels(), xt::no_ownership(), std::vector<int>{mat1.rows, mat1.cols, mat1.channels()});
    // Permute the dimensions from (H, W, C) to (C, H, W)
    xt::xtensor<float, 3> img_tensor_permuted = xt::transpose(img, {2, 0, 1});
    // Add a new dimension at the first position
    xt::xtensor<float, 4> img_tensor = xt::expand_dims(img_tensor_permuted, 0);
    // std::cout << "preprocess efficient sam 2d " <<  xt::adapt(img_tensor.shape()) << std::endl;
    img_tensor = (img_tensor) / std::clamp(xt::amax(img_tensor)(), 1e-8f, 1e18f);
    return img_tensor;
  }
  

  xt::xtensor<float, 4> preprocess_efficient_sam_3D(xt::xtensor<uint8_t, 3>& original_img, int z, std::string view, size_t encoder_input_size = IMAGE_ENCODER_INPUT_SIZE) {
    xt::xtensor<uint8_t, 2> data = get_img_2d(original_img, z, view);
    // xt::xtensor<uint8_t, 3> img_3c = xt::repeat(data, 3, 2);
    // auto expanded_input = xt::expand_dims(data, 2);  // Shape becomes {shape[0], shape[1], 1}
    // auto img_3c = xt::repeat(expanded_input, 3, 2);  // Repeat along the new axis

    // auto output_reshaped = xt::reshape_view(img_3c, {original_size[0], original_size[1], 3});
    // std::cout << "preprocess efficient sam" <<  xt::adapt(data.shape()) << std::endl;
    cv::Mat mat1(cv::Size(original_size[1], original_size[0]), CV_8UC1, data.data());
    // cv::resize(mat1, mat2, cv::Size(new_size[1], new_size[0]), cv::INTER_LINEAR);

    xt::xtensor<float, 3> img = xt::adapt((uint8_t*)mat1.data, mat1.total() * mat1.channels(), xt::no_ownership(), std::vector<int>{mat1.rows, mat1.cols, mat1.channels()});
    // Permute the dimensions from (H, W, C) to (C, H, W)
    xt::xtensor<float, 3> img_tensor_permuted = xt::transpose(img, {2, 0, 1});
    // Add a new dimension at the first position
    xt::xtensor<float, 4> img_tensor = xt::expand_dims(img_tensor_permuted, 0);
    // std::cout << "preprocess efficient sam 3d " <<  xt::adapt(img.shape()) << std::endl;

    img_tensor = (img_tensor) / std::clamp(xt::amax(img_tensor)(), 1e-8f, 1e18f);
    return xt::repeat(img_tensor, 3, 1);
  }


  xt::xtensor<float, 3> preprocess_3D(xt::xtensor<uint8_t, 3>& original_img, int z, std::string view, size_t encoder_input_size = IMAGE_ENCODER_INPUT_SIZE) {
    auto data = get_img_2d(original_img, z, view);
    // std::cout << "preprocess_3D " << data << std::endl;
    cv::Mat mat1(cv::Size(original_size[1], original_size[0]), CV_8UC1, data.data()), mat2;
    // std::cout << "convert to mat" << std::endl;
    cv::resize(mat1, mat2, cv::Size(new_size[1], new_size[0]), cv::INTER_LINEAR);
    // std::cout << "resize" << std::endl;
    xt::xtensor<float, 3> img = xt::adapt((uint8_t*)mat2.data, mat2.total(), xt::no_ownership(), std::vector<int>{mat2.rows, mat2.cols, 1});
    img = (img - xt::amin(img)()) / std::clamp(xt::amax(img)() - xt::amin(img)(), 1e-8f, 1e18f);
    // std::cout << "clamp " << xt::adapt(img.shape()) << std::endl;

    return xt::repeat(xt::pad(img, {{0, encoder_input_size - new_size[0]}, {0, encoder_input_size - new_size[1]}, {0, 0}}), 3, 2);
  }
};

struct Decoder {
  ov::CompiledModel model;
  ov::InferRequest infer_request;
  ImageSize original_size, new_size;

  Decoder(ov::Core& core, const std::string& model_path) {
    model = core.compile_model(model_path, "CPU");
    infer_request = model.create_infer_request();
  }

  void set_sizes(const ImageSize& original_size, const ImageSize& new_size) {
    this->original_size = original_size;
    this->new_size = new_size;
  }

  void set_embedding_tensor(const ov::Tensor& embedding_tensor) {
    infer_request.set_input_tensor(0, embedding_tensor);
  }

  xt::xtensor<float, 2> decode_mask(const ov::Tensor& box_tensor, size_t encoder_input_size = IMAGE_ENCODER_INPUT_SIZE, const ov::Tensor& label_tensor = ov::Tensor(), const ov::Tensor& original_size_tensor = ov::Tensor(), bool extra_params = false) {
    infer_request.set_input_tensor(1, box_tensor);
    // std::cout << "label_tensor " << label_tensor.get_shape() << std::endl;
    if (extra_params) {
      infer_request.set_input_tensor(2, label_tensor);
      infer_request.set_input_tensor(3, original_size_tensor);
    }
    infer_request.infer();

    xt::xtensor<float, 2> mask = xt::adapt(infer_request.get_output_tensor().data<float>(), encoder_input_size * encoder_input_size, xt::no_ownership(), std::vector<int>{(int)encoder_input_size, (int)encoder_input_size});
    mask = xt::view(mask, xt::range(_, new_size[0]), xt::range(_, new_size[1]));

    cv::Mat mat1(cv::Size(new_size[1], new_size[0]), CV_32FC1, mask.data()), mat2;
    cv::resize(mat1, mat2, cv::Size(original_size[1], original_size[0]), cv::INTER_LINEAR);
    return xt::adapt((float*)mat2.data, mat2.total(), xt::no_ownership(), std::vector<int>{mat2.rows, mat2.cols});
  }
};

void infer_2d(std::string img_file, std::string seg_file, Encoder& encoder, Decoder& decoder, size_t encoder_input_size = IMAGE_ENCODER_INPUT_SIZE, ov::Shape input_shape = INPUT_SHAPE) {
  auto npz_data = xt::load_npz(img_file);
  auto original_img = cast_npy_file<xt::xtensor<uint8_t, 3>>(npz_data["imgs"]);
  auto boxes = cast_npy_file<xt::xtensor<float, 2>>(npz_data["boxes"]);
  assert(original_img.shape()[2] == 3);
  assert(boxes.shape()[1] == 4);

  ImageSize original_size = {original_img.shape()[0], original_img.shape()[1]};
  ImageSize new_size = get_preprocess_shape(original_size[0], original_size[1], encoder_input_size);
  boxes /= (std::max(original_size[0], original_size[1]));
  encoder.set_sizes(original_size, new_size);
  decoder.set_sizes(original_size, new_size);
  auto img = encoder.preprocess_2D(original_img, encoder_input_size);
  ov::Tensor input_tensor(ov::element::f32, input_shape, img.data());

  // auto encoder_start = std::chrono::high_resolution_clock::now();
  ov::Tensor embedding_tensor = encoder.encode_image(input_tensor);
  // auto encoder_finish = std::chrono::high_resolution_clock::now();
  // std::cout << "Encoded image in " << std::chrono::duration_cast<std::chrono::milliseconds>(encoder_finish - encoder_start).count() << "ms\n";

  xt::xtensor<uint16_t, 2> segs = xt::zeros<uint16_t>({original_size[0], original_size[1]});

  decoder.set_embedding_tensor(embedding_tensor);
  for (int i = 0; i < boxes.shape()[0]; ++i) {
    ov::Tensor box_tensor(ov::element::f32, {4}, boxes.data() + i * 4);
    // auto decoder_start = std::chrono::high_resolution_clock::now();
    auto mask = decoder.decode_mask(box_tensor, encoder_input_size);
    // auto decoder_finish = std::chrono::high_resolution_clock::now();
    // std::cout << "Decoded box " << (i + 1) << " in " << std::chrono::duration_cast<std::chrono::milliseconds>(decoder_finish - decoder_start).count() << "ms\n";
    xt::filtration(segs, mask > 0) = i + 1;
  }

  xt::dump_npz(seg_file, "segs", segs, true);
}

void infer_efficientsam_2d(std::string img_file, std::string seg_file, Encoder& encoder, Decoder& decoder, size_t encoder_input_size = IMAGE_ENCODER_INPUT_SIZE) {
  auto npz_data = xt::load_npz(img_file);
  auto original_img = cast_npy_file<xt::xtensor<uint8_t, 3>>(npz_data["imgs"]);
  auto boxes = cast_npy_file<xt::xtensor<float, 2>>(npz_data["boxes"]);
  assert(original_img.shape()[2] == 3);
  assert(boxes.shape()[1] == 4);
  auto select_point = [&](xt::xtensor<float, 2>& box) {
    uint16_t x_min = box(0,0), y_min = box(0,1);
    uint16_t x_max = box(1,0), y_max = box(1,1);
    auto x = xt::random::randint<uint16_t>({1}, x_min, x_max);
    auto y = xt::random::randint<uint16_t>({1}, y_min, y_max);
    return xt::xtensor<float, 2> {{(float)x(0), (float)y(0)}};
  };

  ImageSize original_size = {(int64_t)original_img.shape()[0], (int64_t)original_img.shape()[1]};
  ImageSize new_size = {IMAGE_ENCODER_INPUT_SIZE, IMAGE_ENCODER_INPUT_SIZE};
  ov::Shape input_shape = {1, 3, original_img.shape()[0], original_img.shape()[1]};
  encoder.set_sizes(original_size, new_size);
  decoder.set_sizes(original_size, new_size);

  auto img = encoder.preprocess_efficient_sam_2D(original_img, encoder_input_size);
  ov::Tensor input_tensor(ov::element::f32, input_shape, img.data());
  ov::Tensor embedding_tensor = encoder.encode_image(input_tensor);
  xt::xtensor<uint16_t, 2> segs = xt::zeros<uint16_t>({original_size[0], original_size[1]});

  decoder.set_embedding_tensor(embedding_tensor);
  for (int i = 0; i < boxes.shape()[0]; ++i) {
    xt::xtensor<float, 2> box_tensor = {{(float)boxes(i, 0), (float)boxes(i, 1)}, {(float)boxes(i, 2), (float)boxes(i, 3)}};
    xt::xtensor<float, 2> point_tensor = select_point(box_tensor);
    xt::xtensor<float, 2> coord = xt::concatenate(xt::xtuple(point_tensor, box_tensor));
    auto coord_extended = xt::expand_dims(coord, 0);
    ov::Tensor coord_tensor(ov::element::f32, {1,3,2}, coord_extended.data());
    std::vector<float> labels_array = {1, 2, 3};
    auto labels = xt::adapt(labels_array, std::vector<int>{1, 3});
    ov::Tensor label_tensor(ov::element::f32, {1, 3}, labels.data());
    auto original_size_tensor = ov::Tensor(ov::element::i64, {2}, original_size.data());
    auto mask = decoder.decode_mask(coord_tensor, IMAGE_ENCODER_INPUT_SIZE, label_tensor, original_size_tensor, true);
    xt::filtration(segs, mask > 0) = i + 1;
  }
  xt::dump_npz(seg_file, "segs", segs, true);
}


xt::xtensor<uint16_t, 3> infer_efficientsam_3d(std::string img_file, std::string seg_file, Encoder& encoder, Decoder& decoder, std::string view, size_t encoder_input_size = IMAGE_ENCODER_INPUT_SIZE) {
  auto npz_data = xt::load_npz(img_file);
  auto original_img = cast_npy_file<xt::xtensor<uint8_t, 3>>(npz_data["imgs"]);
  auto boxes = cast_npy_file<xt::xtensor<float, 2>>(npz_data["boxes"]);
  assert(boxes.shape()[1] == 6);
  ImageSize original_size, new_size;

  auto select_point = [&](xt::xtensor<float, 2>& box) {
    uint16_t x_min = box(0,0), y_min = box(0,1);
    uint16_t x_max = box(1,0), y_max = box(1,1);
    auto x = xt::random::randint<uint16_t>({1}, x_min, x_max);
    auto y = xt::random::randint<uint16_t>({1}, y_min, y_max);
    return xt::xtensor<float, 2> {{(float)x(0), (float)y(0)}};
  };

  if (view == "axial") {
    original_size = {original_img.shape()[1], original_img.shape()[2]};
  }
  else if (view == "coronal") {
    original_size = {original_img.shape()[0], original_img.shape()[2]};
  }
  else if (view == "sagittal") {
    original_size = {original_img.shape()[0], original_img.shape()[1]};
  }
  else {
    throw std::runtime_error("Unknown view: " + view);
  }

  // ImageSize original_size = {(int64_t)original_img.shape()[0], (int64_t)original_img.shape()[1]};
  new_size = {IMAGE_ENCODER_INPUT_SIZE, IMAGE_ENCODER_INPUT_SIZE};
  ov::Shape input_shape = {1, 3, original_img.shape()[0], original_img.shape()[1]};
  encoder.set_sizes(original_size, new_size);
  decoder.set_sizes(original_size, new_size);
  // std::cout << "original size " << original_size[0] << " " << original_size[1] << std::endl;
  // std::cout << "new size " << new_size[0] << " " << new_size[1] << std::endl;

  cache::lru_cache<int, ov::Tensor> cached_embeddings(EMBEDDINGS_CACHE_SIZE);
  auto get_embedding = [&](int z) {
    if (cached_embeddings.exists(z)) {
      return cached_embeddings.get(z);
    }
    auto img = encoder.preprocess_efficient_sam_3D(original_img, z, view, encoder_input_size);
    // std::cout << "get_embedding " << view << " " << xt::adapt(img.shape()) << std::endl;
    ov::Tensor input_tensor(ov::element::f32, input_shape, img.data());
    ov::Tensor embedding_tensor = encoder.encode_image(input_tensor);
    cached_embeddings.put(z, embedding_tensor);
    return embedding_tensor;
  };
  auto process_slice = [&](int z, xt::xtensor<float, 2>& box_tensor) {
    ov::Tensor embedding_tensor = get_embedding(z);
    // std::cout << "box_tensor " << xt::adapt(box_tensor.shape()) << std::endl;
    // ov::Tensor box_tensor(ov::element::f32, {4}, box.data());
    xt::xtensor<float, 2> point_tensor = select_point(box_tensor);
    xt::xtensor<float, 2> coord = xt::concatenate(xt::xtuple(point_tensor, box_tensor));
    auto coord_extended = xt::expand_dims(coord, 0);
    ov::Tensor coord_tensor(ov::element::f32, {1,3,2}, coord_extended.data());

    std::vector<float> labels_array = {1, 2, 3};
    auto labels = xt::adapt(labels_array, std::vector<int>{1, 3});
    ov::Tensor label_tensor(ov::element::f32, {1, 3}, labels.data());
    auto original_size_tensor = ov::Tensor(ov::element::i64, {2}, original_size.data());
    
    decoder.set_embedding_tensor(embedding_tensor);
    return decoder.decode_mask(coord_tensor, IMAGE_ENCODER_INPUT_SIZE, label_tensor, original_size_tensor, true);
  };

  auto select_default_box = [&](xt::xtensor<float, 1>& box, std::string view) {
    // std::cout << "Selecting default box " << " " << box << " " << view << std::endl;
    uint16_t x_min = box(0), y_min = box(1), z_min = box(2);
    uint16_t x_max = box(3), y_max = box(4), z_max = box(5);
    struct result {xt::xtensor<float, 2> box_default; uint16_t z_middle; uint16_t z_max; uint16_t z_min;};

    // std::cout << "get original_img.shape() " << box << " " << xt::adapt(original_img.shape()) << std::endl;

    if (view == "axial") {
      z_min = std::max(z_min, uint16_t(0));
      z_max = std::min(z_max, uint16_t(original_img.shape()[0]));
      uint16_t z_middle = (z_min + z_max) / 2;

      xt::xtensor<float, 2> box_default = {{(float)x_min, (float)y_min}, {(float)x_max, (float)y_max}};
      // std::cout << "Selecting axial box " << " " << xt::adapt(box_default.shape()) << " " <<  z_middle << " " << z_max << " " << z_min << std::endl;
      
      return result { box_default, z_middle, z_max, z_min };
    }
    else if (view == "coronal") {
      y_min = std::max(y_min, uint16_t(0));
      y_max = std::min(y_max, uint16_t(original_img.shape()[1]));
      uint16_t y_middle = (y_min + y_max) / 2;
      // std::cout << "get y min max " <<  y_middle << " " << y_max << " " << y_min << std::endl;

      xt::xtensor<float, 2> box_default = {{(float)x_min, (float)z_min}, {(float)x_max, (float)z_max}};
      // std::cout << "Selecting coronal box " << " " << box_default << " " <<  y_middle << " " << y_max << " " << y_min << std::endl;
      
      return result { box_default, y_middle, y_max, y_min };
    }
    else if (view == "sagittal") {
      x_min = std::max(x_min, uint16_t(0));
      x_max = std::min(x_max, uint16_t(original_img.shape()[2]));
      uint16_t x_middle = (x_min + x_max) / 2;

      xt::xtensor<float, 2> box_default = {{(float)y_min, (float)z_min}, {(float)y_max, (float)z_max}};
      // std::cout << "Selecting sag box " << " " << box_default << " " <<  x_middle << " " << x_max << " " << x_min << std::endl;

      return result { box_default, x_middle, x_max, x_min };
    }
    else {
      throw std::runtime_error("Unknown view: " + view);
    }
  };

  xt::xtensor<uint16_t, 3> segs = xt::zeros_like(original_img);

  for (int i = 0; i < boxes.shape()[0]; ++i) {
    xt::xtensor<float, 1> given_box = {(float)boxes(i, 0), (float)boxes(i, 1), (float)boxes(i, 2), (float)boxes(i, 3), (float)boxes(i, 4), (float)boxes(i, 5)};
    // xt::xtensor<float, 1> box_default, uint16_t i_middle, uint16_t i_max, uint16_t i_min = select_default_box(boxes(i), view);
    auto [box_default, i_middle, i_max, i_min] = select_default_box(given_box, view);
    // std::cout << "Selecting default box " << " " << xt::adapt(box_default.shape()) << " " << i_middle << " " << i_max << " " << i_min << std::endl;
    xt::xtensor<float, 2> box_middle;
    {
      auto mask_middle = process_slice(i_middle, box_default);
      // std::cout << "mask_middle " << mask_middle << std::endl;
      if (view == "axial") {
        xt::filtration(xt::view(segs, i_middle, xt::all(), xt::all()), mask_middle > 0) = i + 1;
      } else if (view == "coronal") {
        xt::filtration(xt::view(segs, xt::all(), i_middle, xt::all()), mask_middle > 0) = i + 1;
      } else if (view == "sagittal") {
        xt::filtration(xt::view(segs, xt::all(), xt::all(), i_middle), mask_middle > 0) = i + 1;
      }
      // xt::filtration(xt::view(segs, i_middle, xt::all(), xt::all()), mask_middle > 0) = i + 1;
      if (xt::amax(mask_middle)() > 0) {
        box_middle = get_bbox(mask_middle);
      } else {
        box_middle = box_default;
      }
    }
    // std::cout << "infer box_middle " << box_middle << std::endl;
    // infer z_middle+1 to z_max-1
    auto last_box = box_middle;
    for (int z = i_middle + 1; z < i_max; ++z) {
      auto mask = process_slice(z, last_box);
      if (view == "axial") {
        xt::filtration(xt::view(segs, z, xt::all(), xt::all()), mask > 0) = i + 1;
      }
      else if (view == "coronal") {
        xt::filtration(xt::view(segs, xt::all(), z, xt::all()), mask > 0) = i + 1;
      }
      else if (view == "sagittal") {
        xt::filtration(xt::view(segs, xt::all(), xt::all(), z), mask > 0) = i + 1;
      }
      if (xt::amax(mask)() > 0) {
        last_box = get_bbox(mask);
      } else {
        last_box = box_default;
      }
    }
    std::cout << "infer z_middle+1 to z_max-1 " << last_box << std::endl;
    // infer z_min to z_middle-1
    last_box = box_middle;
    for (int z = i_middle - 1; z >= i_min; --z) {
      auto mask = process_slice(z, last_box);
      if (view == "axial") {
        xt::filtration(xt::view(segs, z, xt::all(), xt::all()), mask > 0) = i + 1;
      }
      else if (view == "coronal") {
        xt::filtration(xt::view(segs, xt::all(), z, xt::all()), mask > 0) = i + 1;
      }
      else if (view == "sagittal") {
        xt::filtration(xt::view(segs, xt::all(), xt::all(), z), mask > 0) = i + 1;
      }
      // xt::filtration(xt::view(segs, z, xt::all(), xt::all()), mask > 0) = i + 1;
      if (xt::amax(mask)() > 0) {
        last_box = get_bbox(mask);
      } else {
        last_box = box_default;
      }
    }
    return segs;
  }
}
xt::xtensor<uint16_t, 3> infer_3d(std::string img_file, std::string seg_file, Encoder& encoder, Decoder& decoder, std::string view, size_t encoder_input_size = IMAGE_ENCODER_INPUT_SIZE, ov::Shape input_shape = INPUT_SHAPE) {
  auto npz_data = xt::load_npz(img_file);
  auto original_img = cast_npy_file<xt::xtensor<uint8_t, 3>>(npz_data["imgs"]);
  auto boxes = cast_npy_file<xt::xtensor<uint16_t, 2>>(npz_data["boxes"]);
  assert(boxes.shape()[1] == 6);
  ImageSize original_size, new_size;
  // std::cout << "Image size " << xt::adapt(original_img.shape()) << std::endl;

  if (view == "axial") {
    original_size = {original_img.shape()[1], original_img.shape()[2]};
  }
  else if (view == "coronal") {
    original_size = {original_img.shape()[0], original_img.shape()[2]};
  }
  else if (view == "sagittal") {
    original_size = {original_img.shape()[0], original_img.shape()[1]};
  }
  else {
    throw std::runtime_error("Unknown view: " + view);
  }
  new_size = get_preprocess_shape(original_size[0], original_size[1], encoder_input_size);

  encoder.set_sizes(original_size, new_size);
  decoder.set_sizes(original_size, new_size);

  cache::lru_cache<int, ov::Tensor> cached_embeddings(EMBEDDINGS_CACHE_SIZE);
  auto get_embedding = [&](int z) {
    if (cached_embeddings.exists(z)) {
      return cached_embeddings.get(z);
    }
    auto img = encoder.preprocess_3D(original_img, z, view, encoder_input_size);
    // std::cout << "get_embedding " << view << " " << xt::adapt(img.shape()) << std::endl;
    ov::Tensor input_tensor(ov::element::f32, input_shape, img.data());
    ov::Tensor embedding_tensor = encoder.encode_image(input_tensor);
    cached_embeddings.put(z, embedding_tensor);
    return embedding_tensor;
  };
  auto process_slice = [&](int z, xt::xtensor<float, 1>& box, size_t encoder_input_size) {
    ov::Tensor embedding_tensor = get_embedding(z);
    // std::cout << "tensor " << embedding_tensor.get_shape() << std::endl;
    ov::Tensor box_tensor(ov::element::f32, {4}, box.data());
    decoder.set_embedding_tensor(embedding_tensor);
    return decoder.decode_mask(box_tensor, encoder_input_size);
  };


  auto select_default_box = [&](xt::xtensor<float, 1>& box, std::string view) {
    // std::cout << "Selecting default box " << " " << box << " " << view << std::endl;
    uint16_t x_min = box(0), y_min = box(1), z_min = box(2);
    uint16_t x_max = box(3), y_max = box(4), z_max = box(5);
    struct result {xt::xtensor<float, 1> box_default; uint16_t z_middle; uint16_t z_max; uint16_t z_min;};

    // std::cout << "get original_img.shape() " << box << " " << xt::adapt(original_img.shape()) << std::endl;

    if (view == "axial") {
      z_min = std::max(z_min, uint16_t(0));
      z_max = std::min(z_max, uint16_t(original_img.shape()[0]));
      uint16_t z_middle = (z_min + z_max) / 2;

      xt::xtensor<float, 1> box_default = {(float)x_min, (float)y_min, (float)x_max, (float)y_max};
      // std::cout << "Selecting axial box " << " " << box_default << " " <<  z_middle << " " << z_max << " " << z_min << std::endl;
      
      return result { box_default, z_middle, z_max, z_min };
    }
    else if (view == "coronal") {
      y_min = std::max(y_min, uint16_t(0));
      y_max = std::min(y_max, uint16_t(original_img.shape()[1]));
      uint16_t y_middle = (y_min + y_max) / 2;
      // std::cout << "get y min max " <<  y_middle << " " << y_max << " " << y_min << std::endl;

      xt::xtensor<float, 1> box_default = {(float)x_min, (float)z_min, (float)x_max, (float)z_max};
      // std::cout << "Selecting coronal box " << " " << box_default << " " <<  y_middle << " " << y_max << " " << y_min << std::endl;
      
      return result { box_default, y_middle, y_max, y_min };
    }
    else if (view == "sagittal") {
      x_min = std::max(x_min, uint16_t(0));
      x_max = std::min(x_max, uint16_t(original_img.shape()[2]));
      uint16_t x_middle = (x_min + x_max) / 2;

      xt::xtensor<float, 1> box_default = {(float)y_min, (float)z_min, (float)y_max, (float)z_max};
      // std::cout << "Selecting sag box " << " " << box_default << " " <<  x_middle << " " << x_max << " " << x_min << std::endl;

      return result { box_default, x_middle, x_max, x_min };
    }
    else {
      throw std::runtime_error("Unknown view: " + view);
    }
  };

  xt::xtensor<uint16_t, 3> segs = xt::zeros_like(original_img);
  for (int i = 0; i < boxes.shape()[0]; ++i) {
    // uint16_t x_min = boxes(i, 0), y_min = boxes(i, 1), z_min = boxes(i, 2);
    // uint16_t x_max = boxes(i, 3), y_max = boxes(i, 4), z_max = boxes(i, 5);
    // z_min = std::max(z_min, uint16_t(0));
    // z_max = std::min(z_max, uint16_t(original_img.shape()[0]));
    // uint16_t z_middle = (z_min + z_max) / 2;

    xt::xtensor<float, 1> given_box = {(float)boxes(i, 0), (float)boxes(i, 1), (float)boxes(i, 2), (float)boxes(i, 3), (float)boxes(i, 4), (float)boxes(i, 5)};
    // xt::xtensor<float, 1> box_default, uint16_t i_middle, uint16_t i_max, uint16_t i_min = select_default_box(boxes(i), view);
    auto [box_default, i_middle, i_max, i_min] = select_default_box(given_box, view);
    box_default /= std::max(original_size[0], original_size[1]);

    // std::cout << "Processing box " << i << " " << box_default << " " << i_middle << " " << i_max << " " << i_min << std::endl;
    // infer z_middle
    xt::xtensor<float, 1> box_middle;
    {
      auto mask_middle = process_slice(i_middle, box_default, encoder_input_size);
      // std::cout << "mask_middle " << mask_middle << std::endl;
      if (view == "axial") {
        xt::filtration(xt::view(segs, i_middle, xt::all(), xt::all()), mask_middle > 0) = i + 1;
      } else if (view == "coronal") {
        xt::filtration(xt::view(segs, xt::all(), i_middle, xt::all()), mask_middle > 0) = i + 1;
      } else if (view == "sagittal") {
        xt::filtration(xt::view(segs, xt::all(), xt::all(), i_middle), mask_middle > 0) = i + 1;
      }
      // xt::filtration(xt::view(segs, i_middle, xt::all(), xt::all()), mask_middle > 0) = i + 1;
      if (xt::amax(mask_middle)() > 0) {
        box_middle = get_bbox(mask_middle) / std::max(original_size[0], original_size[1]);
      } else {
        box_middle = box_default;
      }
    }
    // std::cout << "infer box_middle " << box_middle << std::endl;
    // infer z_middle+1 to z_max-1
    auto last_box = box_middle;
    for (int z = i_middle + 1; z < i_max; ++z) {
      auto mask = process_slice(z, last_box, encoder_input_size);
      if (view == "axial") {
        xt::filtration(xt::view(segs, z, xt::all(), xt::all()), mask > 0) = i + 1;
      }
      else if (view == "coronal") {
        xt::filtration(xt::view(segs, xt::all(), z, xt::all()), mask > 0) = i + 1;
      }
      else if (view == "sagittal") {
        xt::filtration(xt::view(segs, xt::all(), xt::all(), z), mask > 0) = i + 1;
      }
      if (xt::amax(mask)() > 0) {
        last_box = get_bbox(mask) / std::max(original_size[0], original_size[1]);
      } else {
        last_box = box_default;
      }
    }
    // std::cout << "infer z_middle+1 to z_max-1 " << last_box << std::endl;
    // infer z_min to z_middle-1
    last_box = box_middle;
    for (int z = i_middle - 1; z >= i_min; --z) {
      auto mask = process_slice(z, last_box, encoder_input_size);
      if (view == "axial") {
        xt::filtration(xt::view(segs, z, xt::all(), xt::all()), mask > 0) = i + 1;
      }
      else if (view == "coronal") {
        xt::filtration(xt::view(segs, xt::all(), z, xt::all()), mask > 0) = i + 1;
      }
      else if (view == "sagittal") {
        xt::filtration(xt::view(segs, xt::all(), xt::all(), z), mask > 0) = i + 1;
      }
      // xt::filtration(xt::view(segs, z, xt::all(), xt::all()), mask > 0) = i + 1;
      if (xt::amax(mask)() > 0) {
        last_box = get_bbox(mask) / std::max(original_size[0], original_size[1]);
      } else {
        last_box = box_default;
      }
    }
  }
  // std::cout << "Saving segs " << seg_file << std::endl;
  // xt::dump_npz(seg_file, "segs", segs, true);
  return segs;
}


int main(int argc, char** argv) {
  if (argc != 8) {
    std::cerr << "Usage: " << argv[0] << " <encoder.xml> <decoder.xml> <model cache folder> <imgs folder> <segs folder>\n";
    return 1;
  }

  ov::Core core;
  core.set_property("CPU", ov::hint::inference_precision(ov::element::f32));
  core.set_property("CPU", ov::hint::execution_mode(ov::hint::ExecutionMode::ACCURACY));
  core.set_property("CPU", ov::hint::performance_mode(ov::hint::PerformanceMode::LATENCY));
  core.set_property("CPU", ov::hint::num_requests(1));
  core.set_property(ov::cache_dir(argv[5]));
  Encoder encoder(core, argv[1]);
  Decoder decoder(core, argv[2]);
  Encoder encoder_efficientvit(core, argv[3]);
  Decoder decoder_efficientvit(core, argv[4]);

  std::filesystem::path imgs_folder(argv[6]);
  if (!std::filesystem::is_directory(imgs_folder)) {
    throw std::runtime_error(imgs_folder.string() + " is not a folder");
  }

  std::filesystem::path segs_folder(argv[7]);
  if (!std::filesystem::exists(segs_folder) && !std::filesystem::create_directory(segs_folder)) {
    throw std::runtime_error("Failed to create " + segs_folder.string());
  }
  if (!std::filesystem::is_directory(segs_folder)) {
    throw std::runtime_error(segs_folder.string() + " is not a folder");
  }

  std::string filename = (segs_folder / "runtime.txt").string();
  std::ofstream file;

  // Check if the file exists
  std::ifstream infile(filename);
  if (infile.good()) {
      // File exists, open in append mode
      file.open(filename, std::ios::app);
  } else {
      // File does not exist, create and open in write mode
      file.open(filename, std::ios::out);
  }

  for (const auto& entry : std::filesystem::directory_iterator(imgs_folder)) {
    if (!entry.is_regular_file()) {
      continue;
    }

    auto base_name = entry.path().filename().string();
    if (ends_with(base_name, ".npz")) {
      auto img_file = entry.path().string();
      auto seg_file = (segs_folder / entry.path().filename()).string();

      // std::cout << "Processing " << base_name << std::endl;
      auto infer_start = std::chrono::high_resolution_clock::now();
      if (starts_with(base_name, "2D") && (contains_substr(base_name, "Microscope") || contains_substr(base_name, "X-Ray"))) {
        constexpr size_t IMAGE_ENCODER_INPUT_SIZE_EFFICIENTVIT = 512;
        const ov::Shape INPUT_SHAPE_EFFICIENTVIT = {IMAGE_ENCODER_INPUT_SIZE_EFFICIENTVIT, IMAGE_ENCODER_INPUT_SIZE_EFFICIENTVIT, 3};
        infer_2d(img_file, seg_file, encoder_efficientvit, decoder_efficientvit, IMAGE_ENCODER_INPUT_SIZE_EFFICIENTVIT, INPUT_SHAPE_EFFICIENTVIT);
      } else if (starts_with(base_name, "3D") && contains_substr(base_name, "PET")) {
        xt::xtensor<uint16_t, 3> axial = infer_3d(img_file, seg_file, encoder, decoder, "axial");
        xt::xtensor<uint16_t, 3> coronal = infer_3d(img_file, seg_file, encoder, decoder, "coronal");
        xt::xtensor<uint16_t, 3> sagittal = infer_3d(img_file, seg_file, encoder, decoder, "sagittal");
        majority_voting(seg_file, axial, coronal, sagittal);
      }
      else if (starts_with(base_name, "3D")) {
        xt::xtensor<uint16_t, 3> axial = infer_3d(img_file, seg_file, encoder, decoder, "axial");
        xt::dump_npz(seg_file, "segs", axial, true);
      }
      else {
        infer_2d(img_file, seg_file, encoder, decoder);
      }
      auto infer_finish = std::chrono::high_resolution_clock::now();
      std::cout << "Inferred " << base_name << " in " << std::chrono::duration_cast<std::chrono::milliseconds>(infer_finish - infer_start).count() << "ms\n";
      
      // Append text to the file or write to the new file
      if (file.is_open()) {
          file << base_name << ", " << std::chrono::duration_cast<std::chrono::milliseconds>(infer_finish - infer_start).count() << std::endl;
      } else {
          std::cerr << "Unable to open the file." << std::endl;
      }
    }
  }

  return 0;
}