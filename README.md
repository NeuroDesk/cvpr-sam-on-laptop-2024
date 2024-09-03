# [Modality-Specific Strategies for Medical Image Segmentation using Lightweight SAM Architectures](https://openreview.net/forum?id=bEQ2KJ9Cgw)  
 
<img src="assets\figure1-cvpr24_challenge_methods_post.png" alt="method" width="700"/>  

This repository is our approach to the challenge [CVPR 2024: SEGMENT ANYTHING IN MEDICAL IMAGES ON LAPTOP](https://www.codabench.org/competitions/1847/#/pages-tab). This challenge focuses on efficiently segmenting multi-modality medical image using lightweight bounding box-based segmentation model on CPU.


## Evaluation:

Segmentation accuracy metrics:

- Dice Similarity Coefficient (DSC)
- Normalized Surface Distance (NSD)

Segmentation efficiency metrics:

- Running time (s)

## Method:

Pretrained models and publicly available datasets (registered before April 15, 2024)

- Pretrain Models other than SAM: https://paperswithcode.com/sota/interactive-segmentation-on-grabcut?p=reviving-iterative-training-with-mask

Hardware specs:

- CPU: Intel® Xeon(R) W-2133 @ 3.60GHz RAM: 8G
- Docker version 20.10.13

## Challeng rule agreement consent

https://drive.google.com/file/d/1XnrKAntAwZo3neEEMNBrKU0h4udoN64h/view

### Installation

The codebase is tested with: `Ubuntu 20.04` | Python `3.10` | `CUDA 11.8` | `Pytorch 2.1.2`

1. Create a virtual environment `conda create -n medsam python=3.10 -y` and activate it `conda activate medsam`
2. Install [Pytorch 2.0](https://pytorch.org/get-started/locally/)
3. `git clone https://github.com/NeuroDesk/cvpr-sam-on-laptop-2024.git`
4. Enter the MedSAM folder `cd cvpr-sam-on-laptop-2024` and run `pip install -e .`

### Quick tutorial on making submissions to CVPR 2024 MedSAM on Laptop Challenge

#### Sanity test

- Download the LiteMedSAM, finetuned LiteMedSAM, EfficientSAM onnx models [here](https://files.au-1.osf.io/v1/resources/u8tny/providers/osfstorage/6618c57de65c6053727d9cbf/?zip=) and put it in `work_dir/onnx_models`. After unzipping, the directory structure should be following. The LiteMedSAM_finetuned_PET_Mircoscope folder should be renamed as LiteMedSAM_finetuned and the LiteMedSAM folder should be renamed as LiteMedSAM_preprocess. The model files within them should be renamed as shown below.

```
work_dir/
├── onnx_models
│   ├── EfficientSAM
│   │   ├── efficient_sam_vitt_decoder.quant.onnx
│   │   └── efficient_sam_vitt_encoder.quant.onnx
│   ├── LiteMedSAM_finetuned
│   │   ├── litemedsam_decoder.onnx
│   │   └── litemedsam_encoder.onnx
│   └── LiteMedSAM_preprocess
│       ├── litemedsam_decoder.onnx
│       └── litemedsam_encoder.onnx
```

- Download the demo data [here](https://drive.google.com/drive/folders/1t3Rs9QbfGSEv2fIFlk8vi7jc0SclD1cq?usp=sharing). The directory structure should be following.

```
test_demo/
├── gts
│   ├── *.npz
│   └── *.npz
├── imgs
│   ├── *.npz
│   └── *.npz
└── segs
    ├── *.npz
    └── *.npz
```
- Run the following command to compile the C++ code and run the inference on the data.

```
cd cpp
cmake -S . -B build -D CMAKE_BUILD_TYPE=Release
cmake --build build --verbose -j$(nproc)
./main litemedsam-encoder.xml litemedsam-decoder.xml efficientvit-encoder.xml efficientvit-decoder.xml /workspace/outputs/ /workspace/inputs/ /workspace/outputs/
```

#### Build Docker

```bash
docker build -f Dockerfile.cpp -t hawken50.fat .
slim build --target hawken50.fat --tag hawken50 --http-probe=false --include-workdir --mount $PWD/test_demo/test_input/:/workspace/inputs/ --mount $PWD/test_demo/segs/:/workspace/outputs/ --exec "sh predict.sh"
docker save hawken50 | gzip -c > hawken50.tar.gz
```

> Note: don't forget the `.` in the end

Run the docker on the testing demo images

```bash
docker container run -m 8G --name hawken50 --rm -v $PWD/test_demo/imgs/:/workspace/inputs/ -v $PWD/test_demo/hawken50/:/workspace/outputs/ hawken50:latest /bin/bash -c "sh predict.sh"
```

> Note: please run `chmod -R 777 ./*` if you run into `Permission denied` error.

Save docker

```bash
docker save hawken50 | gzip -c > hawken50.tar.gz
```

#### Compute Metrics

```bash
python evaluation/compute_metrics.py -s test_demo/hawken50 -g test_demo/gts -csv_dir ./metrics.csv
```

#### Export ONNX model
Download the LiteMedSAM, finetuned LiteMedSAM, EfficientSAM checkpoints [here](https://files.au-1.osf.io/v1/resources/u8tny/providers/osfstorage/6649998e915ae40b30e8993a/?zip=) and put it in `work_dir/checkpoints`. 
Change `--checkpoint` and `--model-type` argument to export from different checkpoints where `vit_h`, `vit_l`, `vit_b`, `vit_t` for LiteMedSAM and `vitt`, `vits` for EfficientSAM.

```bash
python onnx_decoder_exporter.py --checkpoint work_dir/checkpoints/lite_medsam.pth --output work_dir/onnx_models/lite_medsam_encoder.onnx --model-type vit_t --return-single-mask
```

#### Export OpenVINO model

To export OpenVINO model, you need to install OpenVINO toolkit. Follow the instructions [here](https://docs.openvino.ai/2022.3/openvino_docs_install_guides_overview.html) to install OpenVINO toolkit.

Once you have the exported ONNX model from the previous step, you can convert it to OpenVINO model using the following command.

```bash
ovc lite_medsam_encoder.onnx
```

 





### Acknowledgements

We thank the authors of [LiteMedSAM](https://github.com/bowang-lab/MedSAM/tree/LiteMedSAM) and [EfficientSAM](https://github.com/yformer/EfficientSAM) for making their source code publicly available.
