# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
from torch.nn import functional as F
import torchvision.transforms as transforms
from torchvision.transforms.functional import resize
from typing import Tuple, List

from typing import Tuple
from src.litemedsam.modeling import Sam
from src.litemedsam.utils.amg import calculate_stability_score


class SamResize:
    def __init__(self, size: int) -> None:
        self.size = size

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        h, w, _ = image.shape
        long_side = max(h, w)
        if long_side != self.size:
            return self.apply_image(image)
        else:
            return image

    def apply_image(self, image: torch.Tensor) -> torch.Tensor:
        """
        Expects a torch tensor with shape HxWxC in float format.
        """
        h, w, _ = image.shape
        long_side = max(h, w)
        if long_side != self.size:
            target_size = self.get_preprocess_shape(image.shape[0], image.shape[1], self.size)
            x = resize(image.permute(2, 0, 1), target_size)
            return x
        else:
            return image.permute(2, 0, 1)

    @staticmethod
    def get_preprocess_shape(oldh: int, oldw: int, long_side_length: int) -> Tuple[int, int]:
        """
        Compute the output size given input size and target long side length.
        """
        scale = long_side_length * 1.0 / max(oldh, oldw)
        newh, neww = oldh * scale, oldw * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return (newh, neww)

    def __repr__(self) -> str:
        return f"{type(self).__name__}(size={self.size})"


class EncoderModel(nn.Module):
    """
    This model should not be called directly, but is used in ONNX export.
    It combines the image encoder of Sam, with some functions modified to enable model tracing. 
    Also supports extra options controlling what information. 
    See the ONNX export script for details.
    """

    def __init__(
        self,
        model: Sam,
        use_preprocess: bool,
        pixel_mean: List[float] = [123.675 / 255, 116.28 / 255, 103.53 / 255],
        pixel_std: List[float] = [58.395 / 255, 57.12 / 255, 57.375 / 255],
    ):
        super().__init__()

        self.pixel_mean = torch.tensor(pixel_mean, dtype=torch.float)
        self.pixel_std = torch.tensor(pixel_std, dtype=torch.float)

        self.model = model
        self.image_size = model.image_encoder.img_size
        self.image_encoder = self.model.image_encoder
        self.use_preprocess = use_preprocess
        self.resize_transform = SamResize(size=1024)

    @torch.no_grad()
    def forward(self, input_image):
        if self.use_preprocess:
            input_image = self.preprocess(input_image)
        image_embeddings = self.image_encoder(input_image)
        return image_embeddings

    def preprocess(self, x: torch.Tensor) -> torch.Tensor:
        
        # Resize & Permute to (C,H,W)
        x = self.resize_transform(x)

        # Normalize
        x = x.float() / 255
        x = transforms.Normalize(mean=self.pixel_mean, std=self.pixel_std)(x)

        # Pad
        h, w = x.shape[-2:]
        th, tw = self.image_size[1], self.image_size[1]
        assert th >= h and tw >= w
        padh = th - h
        padw = tw - w
        x = F.pad(x, (0, padw, 0, padh), value=0)

        # Expand
        x = torch.unsqueeze(x, 0)

        return x

class DecoderModel(nn.Module):
    """
    This model should not be called directly, but is used in ONNX export.
    It combines the prompt encoder, mask decoder, and mask postprocessing of Sam,
    with some functions modified to enable model tracing. Also supports extra
    options controlling what information. See the ONNX export script for details.
    """

    def __init__(
        self,
        model: Sam,
        return_single_mask: bool,
        use_stability_score: bool = False,
        return_extra_metrics: bool = False,
    ) -> None:
        super().__init__()
        self.mask_decoder = model.mask_decoder
        self.model = model
        self.img_size = model.image_encoder.img_size
        self.return_single_mask = return_single_mask
        self.use_stability_score = use_stability_score
        self.stability_score_offset = 1.0
        self.return_extra_metrics = return_extra_metrics

    @staticmethod
    def resize_longest_image_size(
        input_image_size: torch.Tensor, longest_side: int
    ) -> torch.Tensor:
        input_image_size = input_image_size.to(torch.float32)
        scale = longest_side / torch.max(input_image_size)
        transformed_size = scale * input_image_size
        transformed_size = torch.floor(transformed_size + 0.5).to(torch.int64)
        return transformed_size

    def _embed_points(
        self, point_coords: torch.Tensor, point_labels: torch.Tensor
    ) -> torch.Tensor:
        point_coords = point_coords + 0.5
        point_coords = point_coords / self.img_size
        point_embedding = self.model.prompt_encoder.pe_layer._pe_encoding(point_coords)
        point_labels = point_labels.unsqueeze(-1).expand_as(point_embedding)

        point_embedding = point_embedding * (point_labels != -1)
        point_embedding = (
            point_embedding
            + self.model.prompt_encoder.not_a_point_embed.weight * (point_labels == -1)
        )

        for i in range(self.model.prompt_encoder.num_point_embeddings):
            point_embedding = (
                point_embedding
                + self.model.prompt_encoder.point_embeddings[i].weight
                * (point_labels == i)
            )

        return point_embedding

    def _embed_masks(
        self, input_mask: torch.Tensor, has_mask_input: torch.Tensor
    ) -> torch.Tensor:
        mask_embedding = has_mask_input * self.model.prompt_encoder.mask_downscaling(
            input_mask
        )
        mask_embedding = mask_embedding + (
            1 - has_mask_input
        ) * self.model.prompt_encoder.no_mask_embed.weight.reshape(1, -1, 1, 1)
        return mask_embedding

    def mask_postprocessing(
        self, masks: torch.Tensor, orig_im_size: torch.Tensor
    ) -> torch.Tensor:
        masks = F.interpolate(
            masks,
            size=(self.img_size, self.img_size),
            mode="bilinear",
            align_corners=False,
        )

        prepadded_size = self.resize_longest_image_size(orig_im_size, self.img_size).to(
            torch.int64
        )
        masks = masks[..., : prepadded_size[0], : prepadded_size[1]]  # type: ignore

        orig_im_size = orig_im_size.to(torch.int64)
        h, w = orig_im_size[0], orig_im_size[1]
        masks = F.interpolate(masks, size=(h, w), mode="bilinear", align_corners=False)
        return masks

    def select_masks(
        self, masks: torch.Tensor, iou_preds: torch.Tensor, num_points: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Determine if we should return the multiclick mask or not from the number of points.
        # The reweighting is used to avoid control flow.
        score_reweight = torch.tensor(
            [[1000] + [0] * (self.model.mask_decoder.num_mask_tokens - 1)]
        ).to(iou_preds.device)
        score = iou_preds + (num_points - 2.5) * score_reweight
        best_idx = torch.argmax(score, dim=1)
        masks = masks[torch.arange(masks.shape[0]), 0, :, :].unsqueeze(1)
        iou_preds = iou_preds[torch.arange(masks.shape[0]), best_idx].unsqueeze(1)

        return masks, iou_preds

    @torch.no_grad()
    def forward(
        self,
        image_embeddings: torch.Tensor,
        batched_point_coords: torch.Tensor,
        batched_point_labels: torch.Tensor,
        orig_im_size: torch.Tensor,
    ):
        mask_input = torch.zeros((1, 1, 256, 256), dtype=torch.float32)
        has_mask_input = torch.zeros(1, dtype=torch.float32)
        sparse_embedding = self._embed_points(batched_point_coords, batched_point_labels)
        dense_embedding = self._embed_masks(mask_input, has_mask_input)

        masks, scores = self.model.mask_decoder.predict_masks(
            image_embeddings=image_embeddings,
            image_pe=self.model.prompt_encoder.get_dense_pe(),
            sparse_prompt_embeddings=sparse_embedding,
            dense_prompt_embeddings=dense_embedding,
        )
        print("masks", masks.shape, "scores", scores.shape)
        if self.use_stability_score:
            scores = calculate_stability_score(
                masks, self.model.mask_threshold, self.stability_score_offset
            )

        if self.return_single_mask:
            masks, scores = self.select_masks(masks, scores, batched_point_coords.shape[1])

        upscaled_masks = self.mask_postprocessing(masks, orig_im_size)

        if self.return_extra_metrics:
            stability_scores = calculate_stability_score(
                upscaled_masks, self.model.mask_threshold, self.stability_score_offset
            )
            areas = (upscaled_masks > self.model.mask_threshold).sum(-1).sum(-1)
            return upscaled_masks, scores, stability_scores, areas, masks

        return upscaled_masks, scores, masks
