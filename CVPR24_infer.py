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

def infer_npz_2D(model, model_name, img_npz_file, pred_save_dir, save_overlay, png_save_dir):
    npz_name = basename(img_npz_file)
    npz_data = np.load(img_npz_file, 'r', allow_pickle=True) # (H, W, 3)
    img_3c = npz_data['imgs'] # (H, W, 3)
    # print(f'input data shape: {img_3c.shape}')
    assert np.max(img_3c)<256, f'input data should be in range [0, 255], but got {np.unique(img_3c)}'
    H, W = img_3c.shape[:2]
    boxes = npz_data['boxes']
    segs = np.zeros(img_3c.shape[:2], dtype=np.uint8)

    if model_name == 'efficientsam':
        img_tensor = ToTensor()(img_3c)
        img_tensor = img_tensor[None, ...]
        ## preprocessing
        img_1024 = model.preprocess(img_tensor)
    elif model_name == 'medsam':
        img_1024, newh, neww = medsam_preprocess(img_3c, 1024)
    elif model_name == 'litemedsam':
        img_1024, newh, neww = medsam_preprocess(img_3c, 256)

    with torch.no_grad():
        image_embedding = model.image_encoder(img_1024)

    for idx, box in enumerate(boxes, start=1):
        if model_name == 'efficientsam':
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

def infer_npz_3D(model, model_name, img_npz_file, pred_save_dir, save_overlay, png_save_dir):
    npz_name = basename(img_npz_file)
    npz_data = np.load(img_npz_file, 'r', allow_pickle=True) # (H, W, 3)
    img_3D = npz_data['imgs'] # (D, H, W)
    # print(f'input data shape: {img_3D.shape}')
    spacing = npz_data['spacing'] # not used in this demo because it treats each slice independently
    segs = np.zeros_like(img_3D, dtype=np.uint8) 
    boxes_3D = npz_data['boxes'] # [[x_min, y_min, z_min, x_max, y_max, z_max]]

    for idx, box3D in enumerate(boxes_3D, start=1):
        segs_3d_temp = np.zeros_like(img_3D, dtype=np.uint8) 
        x_min, y_min, z_min, x_max, y_max, z_max = box3D
        assert z_min < z_max, f"z_min should be smaller than z_max, but got {z_min=} and {z_max=}"
        mid_slice_bbox_2d = np.array([x_min, y_min, x_max, y_max])
        z_middle = int((z_max - z_min)/2 + z_min)

        # infer from middle slice to the z_max
        for z in range(z_middle, z_max):
            img_2d = img_3D[z, :, :]
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
            
            if z != z_middle:
                pre_seg = segs_3d_temp[z-1, :, :]
                if np.max(pre_seg) > 0:
                    box = get_bbox(pre_seg)
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

            segs_3d_temp[z, mask>0] = idx
        
        # infer from middle slice to the z_min
        for z in range(z_middle-1, z_min, -1):
            img_2d = img_3D[z, :, :]
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

            pre_seg = segs_3d_temp[z+1, :, :]
            if np.max(pre_seg) > 0:
                box = get_bbox(pre_seg)
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

            segs_3d_temp[z, mask>0] = idx
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
        start_time = time()
        gc.collect()

        if basename(img_npz_file).startswith('3D'):
            imgs, segs, image_size = infer_npz_3D(model, args.model_name, img_npz_file, args.pred_save_dir, args.save_overlay, args.png_save_dir)
        else:
             imgs, segs, image_size = infer_npz_2D(model, args.model_name, img_npz_file, args.pred_save_dir, args.save_overlay, args.png_save_dir)
        end_time = time()
        efficiency['case'].append(basename(img_npz_file))
        efficiency['image size'].append(image_size)
        efficiency['time'].append(end_time - start_time)
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(current_time, 'file name:', basename(img_npz_file), 'image size', image_size, 'time cost:', np.round(end_time - start_time, 4))
    efficiency_df = pd.DataFrame(efficiency)
    efficiency_df.to_csv(join(args.pred_save_dir, 'efficiency.csv'), index=False)
