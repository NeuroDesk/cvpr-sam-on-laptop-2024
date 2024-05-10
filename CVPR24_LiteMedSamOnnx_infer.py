from os import makedirs
from os.path import join, basename
from glob import glob
from tqdm import tqdm
from time import time
import numpy as np
import torch
import torch.nn.functional as F

import cv2
import argparse
from collections import OrderedDict
import pandas as pd
from datetime import datetime
import onnxruntime
#%% set seeds
torch.set_float32_matmul_precision('high')
torch.manual_seed(2024)
np.random.seed(2024)

parser = argparse.ArgumentParser()

parser.add_argument(
    '-i',
    '--input_dir',
    type=str,
    default='test_demo/imgs/',
    # required=True,
    help='root directory of the data',
)
parser.add_argument(
    '-o',
    '--output_dir',
    type=str,
    default='test_demo/segs/',
    help='directory to save the prediction',
)
parser.add_argument(
    '-model_path',
    type=str,
    default="work_dir",
    help='path to the checkpoint of MedSAM-Lite',
)
parser.add_argument(
    '-device',
    type=str,
    default="cpu",
    help='device to run the inference',
)
parser.add_argument(
    '-num_workers',
    type=int,
    default=4,
    help='number of workers for inference with multiprocessing',
)


args = parser.parse_args()

data_root = args.input_dir
pred_save_dir = args.output_dir
num_workers = args.num_workers

makedirs(pred_save_dir, exist_ok=True)
device = torch.device(args.device)

def resize_longest_side(image, target_length):
    """
    Resize image to target_length while keeping the aspect ratio
    Expects a numpy array with shape HxWxC in uint8 format.
    """
    oldh, oldw = image.shape[0], image.shape[1]
    scale = target_length * 1.0 / max(oldh, oldw)
    newh, neww = oldh * scale, oldw * scale
    neww, newh = int(neww + 0.5), int(newh + 0.5)
    target_size = (neww, newh)

    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

def pad_image(image, target_size):
    """
    Pad image to target_size
    Expects a numpy array with shape HxWxC in uint8 format.
    """
    # Pad
    h, w = image.shape[0], image.shape[1]
    padh = target_size - h
    padw = target_size - w
    if len(image.shape) == 3: ## Pad image
        image_padded = np.pad(image, ((0, padh), (0, padw), (0, 0)))
    else: ## Pad gt mask
        image_padded = np.pad(image, ((0, padh), (0, padw)))

    return image_padded

def postprocess_masks(masks, new_size, original_size):
    """
    Do cropping and resizing

    Parameters
    ----------
    masks : torch.Tensor
        masks predicted by the model
    new_size : tuple
        the shape of the image after resizing to the longest side of 256
    original_size : tuple
        the original shape of the image

    Returns
    -------
    torch.Tensor
        the upsampled mask to the original size
    """
    # Crop
    masks = masks[..., :new_size[0], :new_size[1]]
    # Resize
    masks = F.interpolate(
        masks,
        size=(original_size[0], original_size[1]),
        mode="bilinear",
        align_corners=False,
    )

    return masks


def get_bbox_from_mask(mask_256, bbox_shift=7):
    """
    Get the bounding box coordinates from the mask (256x256)

    Parameters
    ----------
    mask_256 : numpy.ndarray
        the mask of the resized image

    bbox_shift : int
        Add perturbation to the bounding box coordinates
    
    Returns
    -------
    numpy.ndarray
        bounding box coordinates in the resized image
    """
    y_indices, x_indices = np.where(mask_256 > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates and test the robustness
    # this can be removed if you do not want to test the robustness
    H, W = mask_256.shape
    x_min = max(0, x_min - bbox_shift)
    x_max = min(W, x_max + bbox_shift)
    y_min = max(0, y_min - bbox_shift)
    y_max = min(H, y_max + bbox_shift)

    bboxes256 = np.array([x_min, y_min, x_max, y_max])

    return bboxes256

def get_point_from_box(box):
    x_min, y_min, x_max, y_max = box
    x_indices = np.array([x_min, x_max])
    y_indices = np.array([y_min, y_max])
    # Select random points
    x_point = np.random.randint(np.min(x_indices), np.max(x_indices), 1)[0]
    y_point = np.random.randint(np.min(y_indices), np.max(y_indices), 1)[0]

    return [x_point, y_point]


def resize_box_to_image_size(box, original_size, image_size):
    """
    the input bounding box is obtained from the original image
    here, we rescale it to the coordinates of the resized image

    Parameters
    ----------
    box : numpy.ndarray
        bounding box coordinates in the original image
    original_size : tuple
        the original size of the image

    Returns
    -------
    numpy.ndarray
        bounding box coordinates in the resized image
    """
    new_box = np.zeros_like(box)
    ratio = image_size / max(original_size)
    for i in range(len(box)):
        new_box[i] = int(box[i] * ratio)

    return new_box

def medsam_preprocess(img_2d, image_size):
    if len(img_2d.shape) == 2:
        img_3c = np.repeat(img_2d[:, :, None], 3, axis=-1)
    else:
        img_3c = img_2d
    img_1024 = resize_longest_side(img_3c, image_size)

    img_1024_norm = (img_1024 - img_1024.min()) / np.clip(
        img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
    )
    img_1024_padded = pad_image(img_1024_norm, image_size)
    img_1024_tensor = torch.tensor(img_1024_padded).float().permute(2, 0, 1).unsqueeze(0)
    return img_1024_tensor

def efficientsam_preprocess(img_2d):
    if len(img_2d.shape) == 2:
        img_3c = np.repeat(img_2d[:, :, None], 3, axis=-1)
    else:
        img_3c = img_2d
    img_norm = img_3c / img_3c.max()
    img_tensor = torch.tensor(img_norm).float().permute(2, 0, 1).unsqueeze(0)
    return img_tensor

@torch.no_grad()
def onnx_decoder_inference(decoder_session, image_embedding_slice, input_points, input_box, original_size):
    if len(input_points)>0:
        input_label = np.array([1])
    if input_box is None:
        onnx_coord = np.concatenate([input_points[0], np.array([[0.0, 0.0]])], axis=0)[None, :, :]
        onnx_label = np.concatenate([input_label, np.array([-1])], axis=0)[None, :].astype(np.float32)

    else:
        onnx_box_coords = input_box.reshape(2, 2)
        onnx_box_labels = np.array([2,3])
        if len(input_points) == 0:
            onnx_coord = onnx_box_coords[None, :, :].astype(np.float32)
            onnx_label = onnx_box_labels[None, :].astype(np.float32)
        else:
            onnx_coord = np.concatenate([input_points, onnx_box_coords], axis=0)[None, :, :].astype(np.float32)
            onnx_label = np.concatenate([input_label, onnx_box_labels], axis=0)[None, :].astype(np.float32)

    # print("image_embedding_slice type", image_embedding_slice.type())
    decoder_inputs = {
        "image_embeddings": np.array(image_embedding_slice),
        "batched_point_coords": onnx_coord,
        "batched_point_labels": onnx_label,
        "orig_im_size": np.array(original_size, dtype=np.int64)
    }
    masks, scores, low_res_logits = decoder_session.run(None, decoder_inputs)
    return masks[0,0,:,:], scores

def majority_voting(npz_name, axial, coronal, sagittal):
    # Stack the arrays along a new axis to create a 4D array
    stacked_arrays = np.stack((axial, coronal, sagittal), axis=-1)

    # Use np.apply_along_axis to apply the majority voting function along the last axis
    result = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=-1, arr=stacked_arrays)
    result = result.astype(np.uint8)
    np.savez_compressed(
        join(pred_save_dir, npz_name),
        segs=result,
    )  


def MedSAM_infer_npz_2D(img_npz_file, encoder_session, decoder_session, image_size):
    # gc.collect()

    npz_name = basename(img_npz_file)
    npz_data = np.load(img_npz_file, 'r', allow_pickle=True) # (H, W, 3)
    img_3c = npz_data['imgs'] # (H, W, 3)
    assert np.max(img_3c)<256, f'input data should be in range [0, 255], but got {np.unique(img_3c)}'
    H, W = img_3c.shape[:2]
    boxes = npz_data['boxes']
    segs = np.zeros(img_3c.shape[:2], dtype=np.uint8)

    ## preprocessing
    if "Microscope" in npz_name:
        img_tensor = efficientsam_preprocess(img_3c)
    else:
        img_tensor  = medsam_preprocess(img_3c, image_size)

    with torch.no_grad():
        image_embedding = encoder_session.run(None, {'input_image': img_tensor.cpu().numpy()})[0]
        # image_embedding = medsam_lite_model.image_encoder(img_256_tensor)

    for idx, box in enumerate(boxes, start=1):
        if "Microscope" in npz_name:
            box_resized = box[None, ...]
            point = [get_point_from_box(box)]
        else:
            box_resized = resize_box_to_image_size(box, (H, W), image_size)
            box_resized = box_resized[None, ...] # (1, 4)
            point = []
        medsam_mask, iou_pred = onnx_decoder_inference(decoder_session, image_embedding, point, box_resized, [H, W])
        segs[medsam_mask>0] = idx
   
    np.savez_compressed(
        join(pred_save_dir, npz_name),
        segs=segs,
    )

def select_middle_slice(box3D, view):
    x_min, y_min, z_min, x_max, y_max, z_max = box3D
    if view == 'axial':
        assert z_min < z_max, f"z_min should be smaller than z_max, but got {z_min=} and {z_max=}"
        mid_slice_bbox_2d = np.array([x_min, y_min, x_max, y_max])
        z_middle = int((z_max - z_min)/2 + z_min)
        return mid_slice_bbox_2d, z_middle, z_min, z_max
    if view == 'coronal':
        assert y_min < y_max, f"y_min should be smaller than y_max, but got {y_min=} and {y_max=}"
        mid_slice_bbox_2d = np.array([x_min, z_min, x_max, z_max])
        y_middle = int((y_max - y_min)/2 + y_min)
        return mid_slice_bbox_2d, y_middle, y_min, y_max
    if view == 'sagittal':
        assert x_min < x_max, f"x_min should be smaller than x_max, but got {x_min=} and {x_max=}"
        mid_slice_bbox_2d = np.array([y_min, z_min, y_max, z_max])
        x_middle = int((x_max - x_min)/2 + x_min)
        return mid_slice_bbox_2d, x_middle, x_min, x_max

def get_img_2d(img_3D, i, orientation):
    if orientation == 'axial':
        return img_3D[i, :, :]
    elif orientation == 'coronal':
        return img_3D[:, i, :]
    elif orientation == 'sagittal':
        return img_3D[:, :, i]

def get_pre_seg(segs_3d_temp, i, orientation):
    if orientation == 'axial':
        return segs_3d_temp[i, :, :]
    elif orientation == 'coronal':
        return segs_3d_temp[:, i, :]
    elif orientation == 'sagittal':
        return segs_3d_temp[:, :, i]

def update_segs_3d_temp(segs_3d_temp, img_2d_seg, i, idx, orientation):
    if orientation == 'axial':
        segs_3d_temp[i,:,:][img_2d_seg>0] = idx
    elif orientation == 'coronal':
        segs_3d_temp[:,i, :][img_2d_seg>0] = idx
    elif orientation == 'sagittal':
        segs_3d_temp[:, :, i][img_2d_seg>0] = idx
    return segs_3d_temp

def MedSAM_infer_npz_3D(img_npz_file, encoder_session, decoder_session, image_size, view):
    npz_name = basename(img_npz_file)
    npz_data = np.load(img_npz_file, 'r', allow_pickle=True)
    img_3D = npz_data['imgs'] # (D, H, W)
    segs = np.zeros_like(img_3D, dtype=np.uint8) 
    boxes_3D = npz_data['boxes'] # [[x_min, y_min, z_min, x_max, y_max, z_max]]

    for idx, box3D in enumerate(boxes_3D, start=1):
        segs_3d_temp = np.zeros_like(img_3D, dtype=np.uint8) 
        mid_slice_bbox_2d, i_middle, i_min, i_max = select_middle_slice(box3D, view)
        # infer from middle slice to the z_max
        # print(npz_name, 'infer from middle slice to the z_max')
        
        for i in range(i_middle, i_max):
            img_2d = get_img_2d(img_3D, i, view)
            H, W = img_2d.shape
            img_tensor = medsam_preprocess(img_2d, image_size)

            # get the image embedding
            with torch.no_grad():
                image_embedding = encoder_session.run(None, {'input_image': img_tensor.cpu().numpy()})[0]

            if i == i_middle:
                box_resized = resize_box_to_image_size(mid_slice_bbox_2d, (H, W), image_size)
                pre_seg_resized = []
            else:
                pre_seg = get_pre_seg(segs_3d_temp, i-1, view)

                if np.max(pre_seg) > 0 and ('MR' in npz_name or 'CT' in npz_name):
                    pre_seg_resized = resize_longest_side(pre_seg, image_size)
                    pre_seg_resized = pad_image(pre_seg_resized, image_size)
                    # print("pre_seg", pre_seg.shape, pre_seg_resized.shape)

                    box_resized = get_bbox_from_mask(pre_seg_resized)
                else:
                    box_resized = resize_box_to_image_size(mid_slice_bbox_2d, (H, W), image_size)
            # img_2d_seg, iou_pred = medsam_inference(medsam_lite_model, image_embedding, box_256, [new_H, new_W], [H, W])
            # print("[new_H, new_W], [H, W]", [new_H, new_W], [H, W])
            img_2d_seg, iou_pred = onnx_decoder_inference(decoder_session, image_embedding, [], box_resized, [H, W])
            # print(img_2d_seg.shape)

            segs_3d_temp = update_segs_3d_temp(segs_3d_temp, img_2d_seg, i, idx, view)
        
        # infer from middle slice to the z_max
        # print(npz_name, 'infer from middle slice to the z_min')
        for i in range(i_middle-1, i_min, -1):
            img_2d = get_img_2d(img_3D, i, view)
            H, W = img_2d.shape
            img_tensor = medsam_preprocess(img_2d, image_size)

            # get the image embedding
            with torch.no_grad():
                image_embedding = encoder_session.run(None, {'input_image': img_tensor.cpu().numpy()})[0]

            pre_seg = get_pre_seg(segs_3d_temp,i+1, view)

            if np.max(pre_seg) > 0 and ('MR' in npz_name or 'CT' in npz_name):
                pre_seg_resized = resize_longest_side(pre_seg, image_size)
                pre_seg_resized = pad_image(pre_seg_resized, image_size)
                box_resized = get_bbox_from_mask(pre_seg_resized)
            else:
                box_resized = resize_box_to_image_size(mid_slice_bbox_2d, (H, W), image_size)

            # img_2d_seg, iou_pred = medsam_inference(medsam_lite_model, image_embedding, box_resized, [new_H, new_W], [H, W])
            img_2d_seg, iou_pred = onnx_decoder_inference(decoder_session, image_embedding, [], box_resized, [H, W])

            segs_3d_temp = update_segs_3d_temp(segs_3d_temp, img_2d_seg, i, idx, view)

        segs[segs_3d_temp>0] = idx
    if "PET" in npz_name:
        return segs
    else:
        np.savez_compressed(
            join(pred_save_dir, npz_name),
            segs=segs,
        )

if __name__ == '__main__':
    img_npz_files = sorted(glob(join(data_root, '*.npz'), recursive=True))
    efficiency = OrderedDict()
    efficiency['case'] = []
    efficiency['time'] = []

    with torch.no_grad():
        litemedsam_encoder_onnx_path = glob(join(args.model_path, 'LiteMedSAM_preprocess', '*encoder.onnx'))[0]
        litemedsam_decoder_onnx_path = glob(join(args.model_path, 'LiteMedSAM_preprocess', '*decoder.onnx'))[0]
        efficientsam_encoder_onnx_path = glob(join(args.model_path, 'EfficientSAM', '*encoder.quant.onnx'))[0]
        efficientsam_decoder_onnx_path = glob(join(args.model_path, 'EfficientSAM', '*decoder.quant.onnx'))[0]
        pet_finetune_encoder_onnx_path = glob(join(args.model_path, 'LiteMedSAM_finetuned', '*encoder.onnx'))[0]
        pet_finetune_decoder_onnx_path = glob(join(args.model_path, 'LiteMedSAM_finetuned', '*decoder.onnx'))[0]

        # print("litemedsam_encoder_onnx_path", litemedsam_encoder_onnx_path, "litemedsam_decoder_onnx_path", litemedsam_decoder_onnx_path)
        options = onnxruntime.SessionOptions()
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.intra_op_num_threads = num_workers
        options.inter_op_num_threads = 2
        litemedsam_encoder_session = onnxruntime.InferenceSession(litemedsam_encoder_onnx_path, sess_options=options, providers=['CPUExecutionProvider'])
        litemedsam_decoder_session = onnxruntime.InferenceSession(litemedsam_decoder_onnx_path, sess_options=options, providers=['CPUExecutionProvider'])
        efficientsam_encoder_session = onnxruntime.InferenceSession(efficientsam_encoder_onnx_path, sess_options=options, providers=['CPUExecutionProvider'])
        efficientsam_decoder_session = onnxruntime.InferenceSession(efficientsam_decoder_onnx_path, sess_options=options, providers=['CPUExecutionProvider'])
        pet_finetune_encoder_session = onnxruntime.InferenceSession(pet_finetune_encoder_onnx_path, sess_options=options, providers=['CPUExecutionProvider'])
        pet_finetune_decoder_session = onnxruntime.InferenceSession(pet_finetune_decoder_onnx_path, sess_options=options, providers=['CPUExecutionProvider'])

    image_size = 256

    for img_npz_file in tqdm(img_npz_files):
        start_time = time()
        if basename(img_npz_file).startswith('2D') and "Microscope" in basename(img_npz_file):
            MedSAM_infer_npz_2D(img_npz_file, efficientsam_encoder_session, efficientsam_decoder_session, image_size)
        elif basename(img_npz_file).startswith('3D') and "PET" in basename(img_npz_file):
            axial = MedSAM_infer_npz_3D(img_npz_file, pet_finetune_encoder_session, pet_finetune_decoder_session, image_size, 'axial')
            coronal = MedSAM_infer_npz_3D(img_npz_file, pet_finetune_encoder_session, pet_finetune_decoder_session, image_size, 'coronal')
            sagittal = MedSAM_infer_npz_3D(img_npz_file, pet_finetune_encoder_session, pet_finetune_decoder_session, image_size, 'sagittal')
            majority_voting(basename(img_npz_file), axial, coronal, sagittal)
        elif basename(img_npz_file).startswith('3D'):
            # print("filename", basename(img_npz_file))
            axial = MedSAM_infer_npz_3D(img_npz_file, litemedsam_encoder_session, litemedsam_decoder_session, image_size, 'axial')
        else:
            MedSAM_infer_npz_2D(img_npz_file, litemedsam_encoder_session, litemedsam_decoder_session, image_size)

        end_time = time()
        efficiency['case'].append(basename(img_npz_file))
        efficiency['time'].append(end_time - start_time)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(current_time, 'file name:', basename(img_npz_file), 'time cost:', np.round(end_time - start_time, 4))
    efficiency_df = pd.DataFrame(efficiency)
    efficiency_df.to_csv(join(pred_save_dir, 'efficiency.csv'), index=False)
