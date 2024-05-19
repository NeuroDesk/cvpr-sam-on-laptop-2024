import matplotlib.pyplot as plt
import numpy as np
import torch
from torchvision.transforms import ToTensor
from collections import OrderedDict
import os
from os.path import join, isfile, basename
from time import time
from datetime import datetime
from glob import glob
import pandas as pd
from tqdm import tqdm
import argparse
import gc
from src.visual_util import show_mask, show_box
from src.infer_util import efficientsam_infer, get_bbox, medsam_inference, medsam_preprocess, resize_box_to_target
from src.efficient_sam.build_efficient_sam import build_efficient_sam_vitt, build_efficient_sam_vits
from src.segment_anything import sam_model_registry
from src.litemedsam.build_sam import build_sam_vit_t

def infer_npz_2D(model, model_name, img_npz_file, pred_save_dir, save_overlay=False, png_save_dir=None):
    npz_name = basename(img_npz_file)
    npz_data = np.load(img_npz_file, 'r', allow_pickle=True) # (H, W, 3)
    img_3c = npz_data['imgs'] # (H, W, 3)
    # print(f'input data shape: {img_3c.shape}')
    assert np.max(img_3c)<256, f'input data should be in range [0, 255], but got {np.unique(img_3c)}'
    H, W = img_3c.shape[:2]
    boxes = npz_data['boxes']
    segs = np.zeros(img_3c.shape[:2], dtype=np.uint8)

    if model_name == 'efficientsam' or "Microscope" in npz_name or "X-Ray" in npz_name:
        img_tensor = ToTensor()(img_3c)
        img_tensor = img_tensor[None, ...]
        ## preprocessing
        img_1024 = model.preprocess(img_tensor)
        # print(img_1024.shape, model.image_encoder.img_size)
    elif model_name == 'medsam':
        img_1024, newh, neww = medsam_preprocess(img_3c, 1024)
    elif model_name == 'litemedsam':
        img_1024, newh, neww = medsam_preprocess(img_3c, 256)
    print(npz_name,f'img3c {img_3c.shape}, img1024{img_1024.shape}, model{model.image_encoder.img_size}')
    with torch.no_grad():
        image_embedding = model.image_encoder(img_1024)

    for idx, box in enumerate(boxes, start=1):
        if model_name == 'efficientsam' or "Microscope" in npz_name or "X-Ray" in npz_name:
            mask = efficientsam_infer(image_embedding, box, model, H,W)
        elif model_name == 'medsam':
            box1024 = resize_box_to_target(box, original_size=(H, W), target_size=1024)
            box1024 = box1024[None, ...] # (1, 4)
            mask, iou = medsam_inference(model, image_embedding, box1024, (newh, neww), (H, W))
        elif model_name == 'litemedsam':
            box256 = resize_box_to_target(box, original_size=(H, W), target_size=256)
            box256 = box256[None, ...]
            mask, iou = medsam_inference(model, image_embedding, box256, (newh, neww), (H, W))

        segs[mask>0] = idx
    if pred_save_dir is not None:
        np.savez_compressed(
            join(pred_save_dir, npz_name),
            segs=segs,
        )
    if save_overlay:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(img_3c)
        ax[1].imshow(img_3c)
        ax[0].set_title("Image")
        ax[1].set_title("EfficientSAM Segmentation")
        ax[0].axis('off')
        ax[1].axis('off')

        for i, box in enumerate(boxes):
            color = np.random.rand(3)
            box_viz = box
            show_box(box_viz, ax[1], edgecolor=color)
            show_mask((segs == i+1).astype(np.uint8), ax[1], mask_color=color)

        plt.tight_layout()
        plt.savefig(join(png_save_dir, npz_name.split(".")[0] + '.png'), dpi=300)
        plt.close()
    return img_3c, segs, img_3c.shape

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

def get_img_2d(img_3D, i, view):
    if view == 'axial':
        return img_3D[i, :, :]
    elif view == 'coronal':
        return img_3D[:, i, :]
    elif view == 'sagittal':
        return img_3D[:, :, i]

def get_pre_seg(segs_3d_temp, i, view):
    if view == 'axial':
        return segs_3d_temp[i, :, :]
    elif view == 'coronal':
        return segs_3d_temp[:, i, :]
    elif view == 'sagittal':
        return segs_3d_temp[:, :, i]

def update_segs_3d_temp(segs_3d_temp, img_2d_seg, i, idx, view):
    if view == 'axial':
        segs_3d_temp[i,:,:][img_2d_seg>0] = idx
    elif view == 'coronal':
        segs_3d_temp[:,i, :][img_2d_seg>0] = idx
    elif view == 'sagittal':
        segs_3d_temp[:, :, i][img_2d_seg>0] = idx
    return segs_3d_temp

def majority_voting(npz_name, axial, coronal, sagittal,pred_save_dir):
    # Stack the arrays along a new axis to create a 4D array
    stacked_arrays = np.stack((axial, coronal, sagittal), axis=-1)

    # Use np.apply_along_axis to apply the majority voting function along the last axis
    result = np.apply_along_axis(lambda x: np.bincount(x).argmax(), axis=-1, arr=stacked_arrays)
    result = result.astype(np.uint8)
    np.savez_compressed(
        join(pred_save_dir, npz_name),
        segs=result,
    )  


def infer_npz_3D(view, model, model_name, img_npz_file, pred_save_dir, save_overlay=False, png_save_dir=None):
    npz_name = basename(img_npz_file)
    npz_data = np.load(img_npz_file, 'r', allow_pickle=True) # (H, W, 3)
    img_3D = npz_data['imgs'] # (D, H, W)
    # print(f'input data shape: {img_3D.shape}')
    spacing = npz_data['spacing'] # not used in this demo because it treats each slice independently
    segs = np.zeros_like(img_3D, dtype=np.uint8) 
    boxes_3D = npz_data['boxes'] # [[x_min, y_min, z_min, x_max, y_max, z_max]]

    for idx, box3D in enumerate(boxes_3D, start=1):
        segs_3d_temp = np.zeros_like(img_3D, dtype=np.uint8) 
        mid_slice_bbox_2d, i_middle, i_min, i_max = select_middle_slice(box3D, view)

        # infer from middle slice to the i_max
        if view == 'axial':
            img_shape = img_3D.shape[0]
        elif view == 'coronal':
            img_shape = img_3D.shape[1]
        else:
            img_shape = img_3D.shape[2]

        i_max = min(i_max+1, img_shape)
        for i in range(i_middle, i_max):
            img_2d = get_img_2d(img_3D, i, view)
            if len(img_2d.shape) == 2:
                img_3c = np.repeat(img_2d[:, :, None], 3, axis=-1)
            else:
                img_3c = img_2d
            H, W, _ = img_3c.shape

            if model_name == 'efficientsam':
                img_tensor = ToTensor()(img_3c)
                img_tensor = img_tensor[None, ...]
                img_1024 = model.preprocess(img_tensor)
            elif model_name == 'medsam':
                img_1024, newh, neww = medsam_preprocess(img_3c, 1024)
            elif model_name == 'litemedsam':
                img_1024, newh, neww = medsam_preprocess(img_3c, 256)

            # get the image embedding
            with torch.no_grad():
                image_embedding = model.image_encoder(img_1024) # (1, 256, 64, 64)
            
            if i != i_middle:
                pre_seg = get_pre_seg(segs_3d_temp, i-1, view)
                if np.max(pre_seg) > 0 and ('MR' in npz_name or 'CT' in npz_name):
                    box = get_bbox(pre_seg, bbox_shift=7)
                else:
                    box = mid_slice_bbox_2d
            else:
                box = mid_slice_bbox_2d
            if model_name == 'efficientsam':
                mask = efficientsam_infer(image_embedding, box, model, H,W)
            elif model_name == 'medsam':
                box1024 = resize_box_to_target(box, original_size=(H, W), target_size=1024)
                box1024 = box1024[None, ...]
                mask, iou_pred = medsam_inference(model, image_embedding, box1024, [newh, neww], [H, W])
            elif model_name == 'litemedsam':
                box256 = resize_box_to_target(box, original_size=(H, W), target_size=256)
                box256 = box256[None, ...]
                mask, iou_pred = medsam_inference(model, image_embedding, box256, [newh, neww], [H, W])

            segs_3d_temp = update_segs_3d_temp(segs_3d_temp, mask, i, idx, view)

        # infer from middle slice to the z_min
        for i in range(i_middle-1, i_min-1, -1):
            img_2d = get_img_2d(img_3D, i, view)
            if len(img_2d.shape) == 2:
                img_3c = np.repeat(img_2d[:, :, None], 3, axis=-1)
            else:
                img_3c = img_2d
            H, W, _ = img_3c.shape

            if model_name == 'efficientsam':
                img_tensor = ToTensor()(img_3c)
                img_tensor = img_tensor[None, ...]
                img_1024 = model.preprocess(img_tensor)
            elif model_name == 'medsam':
                img_1024, newh, neww = medsam_preprocess(img_3c, 1024)
            elif model_name == 'litemedsam':
                img_1024, newh, neww = medsam_preprocess(img_3c, 256)

            # get the image embedding
            with torch.no_grad():
                image_embedding = model.image_encoder(img_1024) # (1, 256, 64, 64)

            pre_seg = segs_3d_temp[i+1, :, :]
            if np.max(pre_seg) > 0 and ('MR' in npz_name or 'CT' in npz_name):
                box = get_bbox(pre_seg, bbox_shift=7)
            else:
                box = mid_slice_bbox_2d
            if model_name == 'efficientsam':
                mask = efficientsam_infer(image_embedding, box, model, H,W)
            elif model_name == 'medsam':
                box1024 = resize_box_to_target(box, original_size=(H, W), target_size=1024)
                box1024 = box1024[None, ...]
                mask, iou_pred = medsam_inference(model, image_embedding, box1024, [newh, neww], [H, W])
            elif model_name == 'litemedsam':
                box256 = resize_box_to_target(box, original_size=(H, W), target_size=256)
                box256 = box256[None, ...]
                mask, iou_pred = medsam_inference(model, image_embedding, box256, [newh, neww], [H, W])

            segs_3d_temp = update_segs_3d_temp(segs_3d_temp, mask, i, idx, view)

        segs[segs_3d_temp>0] = idx
    if pred_save_dir is not None:
        np.savez_compressed(
            join(pred_save_dir, npz_name),
            segs=segs,
        )
    # visualize image, mask and bounding box
    if save_overlay:
        idx = int(segs.shape[0] / 2)
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(img_3D[idx], cmap='gray')
        ax[1].imshow(img_3D[idx], cmap='gray')
        ax[0].set_title("Image")
        ax[1].set_title("MedSAM Segmentation")
        ax[0].axis('off')
        ax[1].axis('off')

        for i, box3D in enumerate(boxes_3D, start=1):
            if np.sum(segs[idx]==i) > 0:
                color = np.random.rand(3)
                x_min, y_min, z_min, x_max, y_max, z_max = box3D
                box_viz = np.array([x_min, y_min, x_max, y_max])
                show_box(box_viz, ax[1], edgecolor=color)
                show_mask(segs[idx]==i, ax[1], mask_color=color)

        plt.tight_layout()
        plt.savefig(join(png_save_dir, npz_name.split(".")[0] + '.png'), dpi=300)
        plt.close()
    return img_3D, segs, img_3D.shape

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='SAM inference')
    parser.add_argument('--data_root', type=str, default='data', help='root directory of the data')
    parser.add_argument('--pred_save_dir', type=str, default='segs', help='directory to save the prediction')
    parser.add_argument('--save_overlay', type=bool, default=True, help='whether to save the overlay image')
    parser.add_argument('--png_save_dir', type=str, default='overlay', help='directory to save the overlay image')
    parser.add_argument('--device', type=str, default="cpu", help='device to run the inference')
    parser.add_argument('--model_name', type=str, choices=['efficientsam', 'medsam','litemedsam'], help='model name to use for inference')
    parser.add_argument('--checkpoint_path', type=str, help='checkpoint file to load the model')
    parser.add_argument('--efficientsam_path', type=str, help='checkpoint file to load the efficientsam model')
    parser.add_argument('--finetuned_pet_path', type=str, help='checkpoint file to load the fine-tuned model for PET')
    args = parser.parse_args()
    
    os.makedirs(args.pred_save_dir, exist_ok=True)
    if args.save_overlay:
        assert args.png_save_dir is not None, "Please specify the directory to save the overlay image"
        os.makedirs(args.png_save_dir, exist_ok=True)

    torch.set_float32_matmul_precision('high')
    torch.manual_seed(2024)
    torch.cuda.manual_seed(2024)
    np.random.seed(2024)

    img_npz_files = sorted(glob(join(args.data_root, '*.npz'), recursive=True))
    efficiency = OrderedDict()
    efficiency['case'] = []
    efficiency['image size'] = []
    efficiency['time'] = []

    if args.model_name == 'efficientsam':
        model = build_efficient_sam_vitt(args.checkpoint_path)
        if 'vits' in args.checkpoint_path:
            model = build_efficient_sam_vits(args.checkpoint_path)
    elif args.model_name == 'medsam':
        model = sam_model_registry["vit_b"](checkpoint=args.checkpoint_path)
    elif args.model_name == 'litemedsam':
        model = build_sam_vit_t(args.checkpoint_path)       


    model.to(args.device)
    model.eval()

    for img_npz_file in tqdm(img_npz_files):
        if not os.path.exists(os.path.join(args.pred_save_dir, basename(img_npz_file))):
            gc.collect()

            if basename(img_npz_file).startswith('3D') and "PET" in basename(img_npz_file):
                finetuned_model = build_sam_vit_t(args.finetuned_pet_path)
                start_time = time()
                imgs, segs_axial, image_size = infer_npz_3D('axial', finetuned_model, args.model_name, img_npz_file, None)
                _, segs_coronal, _ = infer_npz_3D('coronal', finetuned_model, args.model_name, img_npz_file, None)
                _, segs_sagittal, _ = infer_npz_3D('sagittal', finetuned_model, args.model_name, img_npz_file, None)
                majority_voting(basename(img_npz_file), segs_axial, segs_coronal, segs_sagittal, args.pred_save_dir)
            elif basename(img_npz_file).startswith('3D'):
                start_time = time()
                imgs, segs, image_size = infer_npz_3D('axial', model, args.model_name, img_npz_file, args.pred_save_dir)
            elif basename(img_npz_file).startswith('2D') and "Microscope" in basename(img_npz_file):
                efficientmodel = build_efficient_sam_vitt(args.efficientsam_path)
                start_time = time()
                imgs, segs, image_size = infer_npz_2D(efficientmodel, 'efficientsam', img_npz_file, args.pred_save_dir)
            elif basename(img_npz_file).startswith('2D') and "X-Ray" in basename(img_npz_file):
                efficientmodel = build_efficient_sam_vitt(args.efficientsam_path)
                start_time = time()
                imgs, segs, image_size = infer_npz_2D(efficientmodel, 'efficientsam', img_npz_file, args.pred_save_dir)
            else:
                start_time = time()
                imgs, segs, image_size = infer_npz_2D(model, args.model_name, img_npz_file, args.pred_save_dir)
            end_time = time()
            efficiency['case'].append(basename(img_npz_file))
            efficiency['image size'].append(image_size)
            efficiency['time'].append(end_time - start_time)
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(current_time, 'file name:', basename(img_npz_file), 'image size', image_size, 'time cost:', np.round(end_time - start_time, 4))
        efficiency_df = pd.DataFrame(efficiency)
        efficiency_df.to_csv(join(args.pred_save_dir, 'efficiency.csv'), index=False)
