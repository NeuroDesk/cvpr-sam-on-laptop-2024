import torch
import numpy as np
import cv2

def resize_longest_side(image, target_length=1024):
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

def pad_image(image, target_size=1024):
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

def get_bbox(mask, bbox_shift=3):
    """
    Get the bounding box coordinates from the mask (256x256)

    Parameters
    ----------
    mask_1024 : numpy.ndarray
        the mask of the resized image

    bbox_shift : int
        Add perturbation to the bounding box coordinates
    
    Returns
    -------
    numpy.ndarray
        bounding box coordinates in the resized image
    """
    y_indices, x_indices = np.where(mask > 0)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates and test the robustness
    # this can be removed if you do not want to test the robustness
    H, W = mask.shape
    x_min = max(0, x_min - bbox_shift)
    x_max = min(W, x_max + bbox_shift)
    y_min = max(0, y_min - bbox_shift)
    y_max = min(H, y_max + bbox_shift)

    bboxes = np.array([x_min, y_min, x_max, y_max])

    return bboxes

def resize_box_to_1024(box, original_size):
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
    ratio = 1024 / max(original_size)
    for i in range(len(box)):
        new_box[i] = int(box[i] * ratio)

    return new_box

def medsam_preprocess(img_3c, image_size):
    img_1024 = resize_longest_side(img_3c, image_size)
    newh, neww = img_1024.shape[:2]
    img_1024_norm = (img_1024 - img_1024.min()) / np.clip(
        img_1024.max() - img_1024.min(), a_min=1e-8, a_max=None
    )
    img_1024_padded = pad_image(img_1024_norm, image_size)
    img_1024_tensor = torch.tensor(img_1024_padded).float().permute(2, 0, 1).unsqueeze(0)
    return img_1024_tensor, newh, neww

@torch.no_grad()
def medsam_inference(medsam_model, img_embed, box_1024, new_size, original_size):
    """
    Perform inference using the MedSAM model.

    Args:
        medsam_model (MedSAMModel): The MedSAM model.
        img_embed (torch.Tensor): The image embeddings.
        box_1024 (numpy.ndarray): The bounding box coordinates.
        new_size (tuple): The new size of the image.
        original_size (tuple): The original size of the image.
    Returns:
        tuple: A tuple containing the segmented image and the intersection over union (IoU) score.
    """
    box_torch = torch.as_tensor(box_1024[None, None, ...], dtype=torch.float, device=img_embed.device)
    
    sparse_embeddings, dense_embeddings = medsam_model.prompt_encoder(
        points = None,
        boxes = box_torch,
        masks = None,
    )
    low_res_logits, iou = medsam_model.mask_decoder(
        image_embeddings=img_embed, # (B, 256, 64, 64)
        image_pe=medsam_model.prompt_encoder.get_dense_pe(), # (1, 256, 64, 64)
        sparse_prompt_embeddings=sparse_embeddings, # (B, 2, 256)
        dense_prompt_embeddings=dense_embeddings, # (B, 256, 64, 64)
        multimask_output=False
    )


    low_res_pred = medsam_model.postprocess_masks(low_res_logits, new_size, original_size)

    low_res_pred = torch.sigmoid(low_res_pred)  
    # print(low_res_logits.shape, new_size, original_size, low_res_pred.shape)
    low_res_pred = low_res_pred.squeeze().cpu().numpy()  
    medsam_seg = (low_res_pred > 0.5).astype(np.uint8)

    return medsam_seg, iou

@torch.no_grad()
def efficientsam_infer(embddings, box, model, H, W):
    input_label = np.array([2,3])
    
    pts_sampled = torch.reshape(torch.tensor(box), [1, 1, -1, 2])
    pts_labels = torch.reshape(torch.tensor(input_label), [1, 1, -1])
    predicted_logits, predicted_iou = model.predict_masks(embddings, pts_sampled, pts_labels, True, H,W, H, W)
    sorted_ids = torch.argsort(predicted_iou, dim=-1, descending=True)
    predicted_iou = torch.take_along_dim(predicted_iou, sorted_ids, dim=2)
    predicted_logits = torch.take_along_dim(
        predicted_logits, sorted_ids[..., None, None], dim=2
    )

    return torch.ge(predicted_logits[0, 0, 0, :, :], 0).cpu().detach().numpy()

