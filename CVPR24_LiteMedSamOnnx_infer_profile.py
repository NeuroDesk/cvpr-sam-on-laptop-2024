"""Runs inference on a single test image file (for profiling).

This file is meant to be run using the line profiler library.
It cannot be executed through a regular python interpreter.

Instructions:
https://github.com/pyutils/line_profiler
"""


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
parser.add_argument(
    '--save_overlay',
    default=False,
    action='store_true',
    help='whether to save the overlay image'
)
parser.add_argument(
    '-f',
    '--file',
    type=str,
    help='The specific npz file to be inferred on.\
        If not provided, the first file in the input directory is used.',
)

args = parser.parse_args()

data_root = args.input_dir
pred_save_dir = args.output_dir
num_workers = args.num_workers

makedirs(pred_save_dir, exist_ok=True)
device = torch.device(args.device)
image_size = 256

# use the specific npz file given or select the first one in the input directory
img_npz_file = args.file if args.file else sorted(glob(join(data_root, '*.npz'), 
                                                       recursive=True))[0]
    

def resize_longest_side(image, target_length=256):
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

def pad_image(image, target_size=256):
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


def get_bbox256(mask_256, bbox_shift=3):
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

def resize_box_to_256(box, original_size):
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
    ratio = 256 / max(original_size)
    for i in range(len(box)):
        new_box[i] = int(box[i] * ratio)

    return new_box

@torch.no_grad()
def onnx_decoder_inference(decoder_session, image_embedding_slice, input_points, input_box, new_size, original_size):
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

    # onnx_coord = transform.apply_coords(onnx_coord, [stacked_img.shape[0], stacked_img.shape[1]]).astype(np.float32)
    onnx_mask_input = np.zeros((1, 1, 256, 256), dtype=np.float32)
    onnx_has_mask_input = np.zeros(1, dtype=np.float32)
    # print("image_embedding_slice type", image_embedding_slice.type())
    decoder_inputs = {
        "image_embeddings": np.array(image_embedding_slice),
        "point_coords": onnx_coord,
        "point_labels": onnx_label,
        "mask_input": onnx_mask_input,
        "has_mask_input": onnx_has_mask_input,
        "orig_im_size": np.array(original_size, dtype=np.float32)
    }
    masks, scores, low_res_logits = decoder_session.run(None, decoder_inputs)
    # if ori_mask:
    low_res_logits = torch.tensor(low_res_logits)
    low_res_pred = postprocess_masks(low_res_logits, new_size, original_size)
    low_res_pred = torch.sigmoid(low_res_pred)  
    low_res_pred = low_res_pred.squeeze().cpu().numpy()  
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)
    # else:
        # mask = (masks > 0).astype(np.uint8)
    return medsam_seg, scores

@profile
def MedSAM_infer_npz_2D(img_npz_file, encoder_session, decoder_session):
    # gc.collect()

    npz_name = basename(img_npz_file)
    npz_data = np.load(img_npz_file, 'r', allow_pickle=True) # (H, W, 3)
    img_3c = npz_data['imgs'] # (H, W, 3)
    assert np.max(img_3c)<256, f'input data should be in range [0, 255], but got {np.unique(img_3c)}'
    H, W = img_3c.shape[:2]
    boxes = npz_data['boxes']
    segs = np.zeros(img_3c.shape[:2], dtype=np.uint8)

    ## preprocessing
    img_256 = resize_longest_side(img_3c, 256)
    newh, neww = img_256.shape[:2]
    img_256_norm = (img_256 - img_256.min()) / np.clip(
        img_256.max() - img_256.min(), a_min=1e-8, a_max=None
    )
    img_256_padded = pad_image(img_256_norm, 256)
    img_256_tensor = torch.tensor(img_256_padded).float().permute(2, 0, 1).unsqueeze(0).to(device)
    with torch.no_grad():
        image_embedding = encoder_session.run(None, {'input_image': img_256_tensor.cpu().numpy()})[0]
        # image_embedding = medsam_lite_model.image_encoder(img_256_tensor)

    for idx, box in enumerate(boxes, start=1):
        box256 = resize_box_to_256(box, original_size=(H, W))
        box256 = box256[None, ...] # (1, 4)
        # medsam_mask, iou_pred = medsam_inference(medsam_lite_model, image_embedding, box256, (newh, neww), (H, W))
        medsam_mask, iou_pred = onnx_decoder_inference(decoder_session, image_embedding, [], box256, (newh, neww), [H, W])
        # print("medsam_mask", medsam_mask.shape, medsam_mask.dtype, segs.shape)
        segs[medsam_mask>0] = idx
   
    np.savez_compressed(
        join(pred_save_dir, npz_name),
        segs=segs,
    )

@profile
def MedSAM_infer_npz_3D(img_npz_file, encoder_session, decoder_session):
    # gc.collect()

    npz_name = basename(img_npz_file)
    npz_data = np.load(img_npz_file, 'r', allow_pickle=True)
    img_3D = npz_data['imgs'] # (D, H, W)
    segs = np.zeros_like(img_3D, dtype=np.uint8) 
    boxes_3D = npz_data['boxes'] # [[x_min, y_min, z_min, x_max, y_max, z_max]]

    for idx, box3D in enumerate(boxes_3D, start=1):
        segs_3d_temp = np.zeros_like(img_3D, dtype=np.uint8) 
        x_min, y_min, z_min, x_max, y_max, z_max = box3D
        assert z_min < z_max, f"z_min should be smaller than z_max, but got {z_min=} and {z_max=}"
        mid_slice_bbox_2d = np.array([x_min, y_min, x_max, y_max])
        z_middle = int((z_max - z_min)/2 + z_min)

        # infer from middle slice to the z_max
        # print(npz_name, 'infer from middle slice to the z_max')
        for z in range(z_middle, z_max):
            img_2d = img_3D[z, :, :]
            if len(img_2d.shape) == 2:
                img_3c = np.repeat(img_2d[:, :, None], 3, axis=-1)
            else:
                img_3c = img_2d
            H, W, _ = img_3c.shape

            img_256 = resize_longest_side(img_3c, 256)
            new_H, new_W = img_256.shape[:2]

            img_256 = (img_256 - img_256.min()) / np.clip(
                img_256.max() - img_256.min(), a_min=1e-8, a_max=None
            )  # normalize to [0, 1], (H, W, 3)
            ## Pad image to 256x256
            img_256 = pad_image(img_256)
            
            # convert the shape to (3, H, W)
            img_256_tensor = torch.tensor(img_256).float().permute(2, 0, 1).unsqueeze(0).to(device)
            # get the image embedding
            with torch.no_grad():
                image_embedding = encoder_session.run(None, {'input_image': img_256_tensor.cpu().numpy()})[0]
                # image_embedding = medsam_lite_model.image_encoder(img_256_tensor) # (1, 256, 64, 64)
            if z == z_middle:
                box_256 = resize_box_to_256(mid_slice_bbox_2d, original_size=(H, W))
            else:
                pre_seg = segs[z-1, :, :]
                if np.max(pre_seg) > 0:
                    pre_seg256 = resize_longest_side(pre_seg)
                    pre_seg256 = pad_image(pre_seg256)
                    box_256 = get_bbox256(pre_seg256)
                else:
                    box_256 = resize_box_to_256(mid_slice_bbox_2d, original_size=(H, W))
            # img_2d_seg, iou_pred = medsam_inference(medsam_lite_model, image_embedding, box_256, [new_H, new_W], [H, W])
            img_2d_seg, iou_pred = onnx_decoder_inference(decoder_session, image_embedding, [], box_256, [new_H, new_W], [H, W])
            
            segs_3d_temp[z, img_2d_seg>0] = idx
        
        # infer from middle slice to the z_max
        # print(npz_name, 'infer from middle slice to the z_min')
        for z in range(z_middle-1, z_min, -1):
            img_2d = img_3D[z, :, :]
            if len(img_2d.shape) == 2:
                img_3c = np.repeat(img_2d[:, :, None], 3, axis=-1)
            else:
                img_3c = img_2d
            H, W, _ = img_3c.shape

            img_256 = resize_longest_side(img_3c)
            new_H, new_W = img_256.shape[:2]

            img_256 = (img_256 - img_256.min()) / np.clip(
                img_256.max() - img_256.min(), a_min=1e-8, a_max=None
            )  # normalize to [0, 1], (H, W, 3)
            ## Pad image to 256x256
            img_256 = pad_image(img_256)

            img_256_tensor = torch.tensor(img_256).float().permute(2, 0, 1).unsqueeze(0).to(device)
            # get the image embedding
            with torch.no_grad():
                image_embedding = encoder_session.run(None, {'input_image': img_256_tensor.cpu().numpy()})[0]

            pre_seg = segs[z+1, :, :]
            if np.max(pre_seg) > 0:
                pre_seg256 = resize_longest_side(pre_seg)
                pre_seg256 = pad_image(pre_seg256)
                box_256 = get_bbox256(pre_seg256)
            else:
                scale_256 = 256 / max(H, W)
                box_256 = mid_slice_bbox_2d * scale_256
            # img_2d_seg, iou_pred = medsam_inference(medsam_lite_model, image_embedding, box_256, [new_H, new_W], [H, W])
            img_2d_seg, iou_pred = onnx_decoder_inference(decoder_session, image_embedding, [], box_256, [new_H, new_W], [H, W])
            
            segs_3d_temp[z, img_2d_seg>0] = idx
        segs[segs_3d_temp>0] = idx
    np.savez_compressed(
        join(pred_save_dir, npz_name),
        segs=segs,
    )            

@profile
def main():
    efficiency = OrderedDict()
    efficiency['case'] = []
    efficiency['time'] = []

    with torch.no_grad():
        encoder_onnx_path = glob(join(args.model_path, '*encoder*.onnx'))[0]
        decoder_onnx_path = glob(join(args.model_path, '*decoder*.onnx'))[0]
        options = onnxruntime.SessionOptions()
        options.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_DISABLE_ALL
        options.intra_op_num_threads = num_workers
        options.enable_mem_pattern = False
        options.enable_cpu_mem_arena = False
        options.enable_mem_reuse = False
        encoder_session = onnxruntime.InferenceSession(encoder_onnx_path, sess_options=options, providers=['CPUExecutionProvider'])
        decoder_session = onnxruntime.InferenceSession(decoder_onnx_path, sess_options=options, providers=['CPUExecutionProvider'])

    start_time = time()
    print('Processing:', img_npz_file)
    if basename(img_npz_file).startswith('3D'):
        MedSAM_infer_npz_3D(img_npz_file, encoder_session, decoder_session)
    else:
        MedSAM_infer_npz_2D(img_npz_file, encoder_session, decoder_session)
    end_time = time()
    efficiency['case'].append(basename(img_npz_file))
    efficiency['time'].append(end_time - start_time)
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(current_time, 'file name:', basename(img_npz_file), 'time cost:', np.round(end_time - start_time, 4))
    
    efficiency_df = pd.DataFrame(efficiency)
    efficiency_df.to_csv(join(pred_save_dir, 'efficiency.csv'), index=False)


if __name__ == '__main__':
    main()