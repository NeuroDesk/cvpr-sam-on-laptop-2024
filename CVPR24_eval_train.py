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
import torch.nn.functional as F

def infer_npz_2D(zoom, zoom_enlarge, model, model_name, img_npz_file, pred_save_dir, save_overlay, png_save_dir):
    npz_name = basename(img_npz_file)
    npz_data = np.load(img_npz_file, 'r', allow_pickle=True) # (H, W, 3)
    img_3c = npz_data['imgs'] # (H, W, 3)
    # print(f'input data shape: {img_3c.shape}')
    assert np.max(img_3c)<256, f'input data should be in range [0, 255], but got {np.unique(img_3c)}'
    H, W = img_3c.shape[:2]
    gts = npz_data['gts']
    gts_labels = [l for l in np.unique(gts) if l != 0]
    # print(f'gts_labels: {gts_labels}')
    boxes = []
    segs = np.zeros(img_3c.shape[:2], dtype=np.uint8)

    if model_name == 'efficientsam'or "Microscopy" in npz_name or "X-Ray" in npz_name:
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

    for idx, label in enumerate(gts_labels):
        mask_label = np.uint8(gts == label)
        box = get_bbox(mask_label, bbox_shift=7)
        boxes.append(box)
        print(f'{npz_name} box: {box} label: {label}')
        bbox_area = (box[2] - box[0]) * (box[3] - box[1])
        if zoom and bbox_area <= 1024:
            # Crop
            original_size = (H, W)
            # print('original size:', original_size)
            # crop_xmin, crop_ymin, crop_xmax, crop_ymax = box[0], box[1], box[2], box[3]
            crop_xmin = max(0, box[0]- zoom_enlarge)
            crop_ymin =  max(0, box[1]- zoom_enlarge)
            crop_xmax = min(box[2] + zoom_enlarge, mask_label.shape[1])
            crop_ymax = min(box[3]+ zoom_enlarge, mask_label.shape[0])

            print('crop:', crop_xmin, crop_ymin, crop_xmax, crop_ymax, 'image shape:', img_3c.shape)
            img_3c_crop = img_3c[ crop_ymin:crop_ymax, crop_xmin:crop_xmax]
            print('after crop img_3c shape:', img_3c_crop.shape)
            mask_label = mask_label[crop_ymin:crop_ymax, crop_xmin:crop_xmax]
            print('mask_label:', np.unique(mask_label), 'mask_label shape:', mask_label.shape)
            box = get_bbox(mask_label, bbox_shift=7)
            box_area = (box[2] - box[0]) * (box[3] - box[1])
            print('new box:', box, 'box area:', box_area)
            # zoom_H, zoom_W = box[3] - box[1], box[2] - box[0]
            zoom_H, zoom_W = crop_ymax - crop_ymin, crop_xmax - crop_xmin
            # H, W = img_3c.shape[:2]
            if model_name == 'efficientsam'or "Microscopy" in npz_name or "X-Ray" in npz_name:
                img_tensor = ToTensor()(img_3c_crop)
                img_tensor = img_tensor[None, ...]
                ## preprocessing
                img_1024 = model.preprocess(img_tensor)
            elif model_name == 'medsam':
                img_1024, newh, neww = medsam_preprocess(img_3c_crop, 1024)
            elif model_name == 'litemedsam':
                img_1024, newh, neww = medsam_preprocess(img_3c_crop, 256)

            with torch.no_grad():
                image_embedding = model.image_encoder(img_1024)



        if model_name == 'efficientsam' or "Microscopy" in npz_name or "X-Ray" in npz_name:
            if zoom and bbox_area <= 1024:
                mask = efficientsam_infer(image_embedding, box, model, zoom_H, zoom_W)
            else:
                mask = efficientsam_infer(image_embedding, box, model, H, W)
        elif model_name == 'medsam':
            if zoom and bbox_area <= 1024:
                box1024 = resize_box_to_target(box, original_size=(zoom_H, zoom_W), target_size=1024)
                box1024 = box1024[None, ...] # (1, 4)
                mask, iou = medsam_inference(model, image_embedding, box1024, (newh, neww), (zoom_H, zoom_W))
            else:
                box1024 = resize_box_to_target(box, original_size=(H, W), target_size=1024)
                box1024 = box1024[None, ...]
                mask, iou = medsam_inference(model, image_embedding, box, (newh, neww), (H, W))
        elif model_name == 'litemedsam':
            if zoom and bbox_area <= 1024:
                box256 = resize_box_to_target(box, original_size=(zoom_H, zoom_W), target_size=256)
                box256 = box256[None, ...]
                mask, iou = medsam_inference(model, image_embedding, box256, (newh, neww), (zoom_H, zoom_W))
            else:
                box256 = resize_box_to_target(box, original_size=(H, W), target_size=256)
                box256 = box256[None, ...]
                mask, iou = medsam_inference(model, image_embedding, box256, (newh, neww), (H, W))

        if zoom and bbox_area <= 1024:
            print('mask shape:', mask.shape, np.unique(mask))
            mask_temp = np.zeros(original_size, dtype=np.uint8)
            mask_temp[crop_ymin:crop_ymax, crop_xmin:crop_xmax] += mask 
            segs[mask_temp>0] = label
        else:
            segs[mask>0] = label

    if pred_save_dir is not None:
        # print(f'save to {join(pred_save_dir, npz_name)}', np.unique(segs))

        np.savez_compressed(
            join(pred_save_dir, npz_name),
            segs=segs,
        )
    if save_overlay:
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(img_3c)
        ax[0].imshow(gts,alpha=0.5)
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

def update_segs_3d_temp(segs_3d_temp, img_2d_seg, i, label, view):
    if view == 'axial':
        segs_3d_temp[i,:,:][img_2d_seg>0] = label
    elif view == 'coronal':
        segs_3d_temp[:,i, :][img_2d_seg>0] = label
    elif view == 'sagittal':
        segs_3d_temp[:, :, i][img_2d_seg>0] = label
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

def infer_npz_3D(view, model, model_name, img_npz_file, pred_save_dir, save_overlay, png_save_dir):
    npz_name = basename(img_npz_file)
    npz_data = np.load(img_npz_file, 'r', allow_pickle=True) # (H, W, 3)
    img_3D = npz_data['imgs'] # (D, H, W)
    # print(f'input data shape: {img_3D.shape}')
    spacing = npz_data['spacing'] # not used in this demo because it treats each slice independently
    segs = np.zeros_like(img_3D, dtype=np.uint8) 
    gts = npz_data['gts'] # [[x_min, y_min, z_min, x_max, y_max, z_max]]
    gts_label = [l for l in np.unique(gts) if l != 0]
    # print(f'gts_label: {gts_label}')
    boxes_3D = []

    for idx, label in enumerate(gts_label):
        mask_label = np.uint8(gts == label)
        box3D = np.where(mask_label>0)
        z_indice, y_indice, x_indice = box3D

        segs_3d_temp = np.zeros_like(img_3D, dtype=np.uint8) 
        x_min, y_min, z_min, x_max, y_max, z_max = np.min(x_indice), np.min(y_indice), np.min(z_indice), np.max(x_indice), np.max(y_indice), np.max(z_indice)
        # print(f'{npz_name} box3D: {x_min, y_min, z_min, x_max, y_max, z_max}')
        boxes_3D.append(np.array([x_min, y_min, z_min, x_max, y_max, z_max]))
        assert z_min < z_max, f"z_min should be smaller than z_max, but got {z_min=} and {z_max=}"
        
        mid_slice_bbox_2d, i_middle, i_min, i_max = select_middle_slice(np.array([x_min, y_min, z_min, x_max, y_max, z_max]), view)

        # infer from middle slice to the i_max
        if view == 'axial':
            img_shape = img_3D.shape[0]
        elif view == 'coronal':
            img_shape = img_3D.shape[1]
        else:
            img_shape = img_3D.shape[2]

        i_max = min(i_max+1, img_shape)
        # print(f'Infer from {view} {i_middle} to {i_max}')
        # print(img_3D.shape)
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
                if np.max(pre_seg) > 0:
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

            segs_3d_temp = update_segs_3d_temp(segs_3d_temp, mask, i, label, view)
        
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

            pre_seg = get_pre_seg(segs_3d_temp, i+1, view)
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

            segs_3d_temp = update_segs_3d_temp(segs_3d_temp, mask, i, label, view)

        segs[segs_3d_temp>0] = label
    if pred_save_dir is not None:
        # print(f'save to {join(pred_save_dir, npz_name)}', np.unique(segs))
        np.savez_compressed(
            join(pred_save_dir, npz_name),
            segs=segs,
        )
    # visualize image, mask and bounding box
    if save_overlay:
        idx = int(segs.shape[0] / 2)
        fig, ax = plt.subplots(1, 2, figsize=(10, 5))
        ax[0].imshow(img_3D[idx], cmap='gray')
        ax[0].imshow(gts[idx],alpha=0.5)
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
    parser.add_argument('--save_overlay', type=bool, default=False, help='whether to save the overlay image')
    parser.add_argument('--png_save_dir', type=str, default='overlay', help='directory to save the overlay image')
    parser.add_argument('--device', type=str, default="cpu", help='device to run the inference')
    parser.add_argument('--model_name', type=str, choices=['efficientsam', 'medsam','litemedsam'], help='model name to use for inference')
    parser.add_argument('--checkpoint_path', type=str, help='checkpoint file to load the model')
    parser.add_argument('--efficientsam_path', type=str, help='checkpoint file to load the efficientsam model')
    parser.add_argument('--finetuned_pet_path', type=str, help='checkpoint file to load the fine-tuned model for PET')
    parser.add_argument('--zoom', type=bool, default=False, help='whether to zoom the image')
    parser.add_argument('--zoom_enlarge', type=int, default=0, help='enlarge factor for zooming the image')
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
    # invalid_files = [f'{args.data_root}/XRay/Chest-Xray-Masks-and-Labels/XRay_Chest-Xray-Masks-and-Labels_MCUCXR_0301_1.npz',f'{args.data_root}/XRay/Chest-Xray-Masks-and-Labels/XRay_Chest-Xray-Masks-and-Labels_MCUCXR_0309_1.npz']
    # img_npz_files = [npz_path for npz_path in img_npz_files if npz_path not in invalid_files]
    # # img_npz_files = [npz_path for npz_path in img_npz_files if 'CVPR24-PostChallenge-PET' not in npz_path]
    # img_npz_files = [npz_path for npz_path in img_npz_files if 'PET' not in npz_path]
    # img_npz_files = [npz_path for npz_path in img_npz_files if 'XRay' in npz_path or 'OCT' in npz_path or 'US' in npz_path]
    # img_npz_files = np.random.choice(img_npz_files, 3000)
    efficiency = OrderedDict()
    efficiency['case'] = []
    efficiency['modality'] = []
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
        print('Processing:', img_npz_file)
        dimension = basename(img_npz_file).split('_')[0]
        modality = basename(img_npz_file).split('_')[1]
        # modality = basename(os.path.dirname(os.path.dirname(img_npz_file)))
        # if modality in ['CT', 'MR', 'PET']:
        #     dimension = '3D'
        # else:
        #     dimension = '2D'
        if not os.path.exists(os.path.join(args.pred_save_dir, basename(img_npz_file))):
            gc.collect()

            if dimension == '3D' and "PET" in basename(img_npz_file):
                finetuned_model = build_sam_vit_t(args.finetuned_pet_path)
                start_time = time()
                imgs, segs_axial, image_size = infer_npz_3D('axial', finetuned_model, args.model_name, img_npz_file, None,
                                                            args.save_overlay, args.png_save_dir)
                _, segs_coronal, _ = infer_npz_3D('coronal', finetuned_model, args.model_name, img_npz_file, None,
                                                    args.save_overlay, args.png_save_dir)
                _, segs_sagittal, _ = infer_npz_3D('sagittal', finetuned_model, args.model_name, img_npz_file, None,
                                                    args.save_overlay, args.png_save_dir)
                majority_voting(basename(img_npz_file), segs_axial, segs_coronal, segs_sagittal, args.pred_save_dir)
            elif dimension == '3D':
                start_time = time()
                imgs, segs, image_size = infer_npz_3D('axial', model, args.model_name, img_npz_file, args.pred_save_dir,
                                                        args.save_overlay, args.png_save_dir)
            elif dimension == '2D' and "Microscopy" in basename(img_npz_file):
                efficientmodel = build_efficient_sam_vitt(args.efficientsam_path)
                start_time = time()
                imgs, segs, image_size = infer_npz_2D(args.zoom, args.zoom_enlarge, efficientmodel, 'efficientsam', img_npz_file, args.pred_save_dir,
                                                        args.save_overlay, args.png_save_dir)
            elif dimension == '2D' and "X-Ray" in basename(img_npz_file):
                efficientmodel = build_efficient_sam_vitt(args.efficientsam_path)
                start_time = time()
                imgs, segs, image_size = infer_npz_2D(args.zoom, args.zoom_enlarge, efficientmodel, 'efficientsam', img_npz_file, args.pred_save_dir,
                                                        args.save_overlay, args.png_save_dir)
            else:
                start_time = time()
                imgs, segs, image_size = infer_npz_2D(args.zoom, args.zoom_enlarge, model, args.model_name, img_npz_file, args.pred_save_dir,
                                                        args.save_overlay, args.png_save_dir)
            # print('segs', np.unique(segs))
            end_time = time()
            efficiency['case'].append(basename(img_npz_file))
            efficiency['image size'].append(image_size)
            efficiency['modality'].append(modality)
            efficiency['time'].append(end_time - start_time)
            current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            print(current_time, 'file name:', basename(img_npz_file), 'image size', image_size, 'time cost:', np.round(end_time - start_time, 4))
    efficiency_df = pd.DataFrame(efficiency)
    efficiency_df.to_csv(join(args.pred_save_dir, 'efficiency.csv'), index=False)
