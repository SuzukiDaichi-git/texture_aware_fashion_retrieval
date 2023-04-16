# Copyright (c) Facebook, Inc. and its affiliates.

from typing import Dict

import torch
from mmf.models.composition import NormalizationLayer
from mmf.models.fashionvil.base import FashionViLBaseModel
from mmf.models.fashionvil.combiner import Combiner
from torch import Tensor


class FashionViLForContrastive(FashionViLBaseModel):
    def __init__(self, config):
        super().__init__(config)
        self.norm_layer = NormalizationLayer()
        feature_dim = 768
        projection_dim = 2560
        hidden_dim = 5120
        self.img_combiner = Combiner(feature_dim, projection_dim, hidden_dim)
        self.txt_combiner = Combiner(feature_dim, projection_dim, hidden_dim)

    def flatten_for_bert(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        to_be_flattened = ["input_ids_g", "segment_ids_g", "input_ids_l", "segment_ids_l"]
        to_be_flattened_dim = ["image_g", "image_l"]
        flattened = self.flatten(sample_list, to_be_flattened, to_be_flattened_dim)
        return flattened

    def add_post_flatten_params(
        self, sample_list: Dict[str, Tensor]
    ) -> Dict[str, Tensor]:
        b, l, _ = sample_list["image_g"].shape
        device = sample_list["image_g"].device
        sample_list["visual_embeddings_type"] = torch.zeros(
            (b, l), device=device
        ).long()
        return sample_list

    def _forward(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
        visual_embeddings_g, _, _ = self.bert.get_image_embedding(
            sample_list["image_g"],
            sample_list["visual_embeddings_type"],
        )
        visual_embeddings_l, _, _ = self.bert.get_image_embedding(
            sample_list["image_l"],
            sample_list["visual_embeddings_type"],
        )
        visual_embeddings = self.img_combiner.combine_features(visual_embeddings_g, visual_embeddings_l)
        visual_embeddings = visual_embeddings_g
        # visual_embeddings, _, _ = self.bert.get_image_embedding(
        #     sample_list["image"],
        #     sample_list["visual_embeddings_type"],
        # )
        visual_embeddings = visual_embeddings.mean(dim=1)
        visual_embeddings = self.norm_layer(visual_embeddings)

        text_embeddings_g, _, _ = self.bert.get_text_embedding(
            sample_list["input_ids_g"],
            sample_list["segment_ids_g"],
            sample_list["input_mask_g"],
        )
        masks_g = sample_list["input_mask_g"]
        text_embeddings_g = text_embeddings_g * masks_g.unsqueeze(2)
        text_embeddings_g = torch.sum(text_embeddings_g, dim=1) / (
            torch.sum(masks_g, dim=1, keepdim=True)
        )
        text_embeddings_l, _, _ = self.bert.get_text_embedding(
            sample_list["input_ids_l"],
            sample_list["segment_ids_l"],
            sample_list["input_mask_l"],
        )
        masks_l = sample_list["input_mask_l"]
        text_embeddings_l = text_embeddings_l * masks_l.unsqueeze(2)
        text_embeddings_l = torch.sum(text_embeddings_l, dim=1) / (
            torch.sum(masks_l, dim=1, keepdim=True)
        )
        text_embeddings = self.txt_combiner.combine_features(text_embeddings_g, text_embeddings_l)
        # text_embeddings = text_embeddings_g
        # text_embeddings = text_embeddings[:, 0]
        
        text_embeddings = self.norm_layer(text_embeddings)

        output_dict = {
            "scores": visual_embeddings,
            "targets": text_embeddings,
        }
        return output_dict




# class FashionViLForContrastive(FashionViLBaseModel):
#     def __init__(self, config):
#         super().__init__(config)
#         self.norm_layer = NormalizationLayer()

#     def flatten_for_bert(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
#         to_be_flattened = ["input_ids", "segment_ids"]
#         to_be_flattened_dim = ["image"]
#         flattened = self.flatten(sample_list, to_be_flattened, to_be_flattened_dim)
#         return flattened

#     def add_post_flatten_params(
#         self, sample_list: Dict[str, Tensor]
#     ) -> Dict[str, Tensor]:
#         b, l, _ = sample_list["image"].shape
#         device = sample_list["image"].device
#         sample_list["visual_embeddings_type"] = torch.zeros(
#             (b, l), device=device
#         ).long()
#         return sample_list

#     def _forward(self, sample_list: Dict[str, Tensor]) -> Dict[str, Tensor]:
#         visual_embeddings, _, _ = self.bert.get_image_embedding(
#             sample_list["image"],
#             sample_list["visual_embeddings_type"],
#         )
#         visual_embeddings = visual_embeddings.mean(dim=1)
#         visual_embeddings = self.norm_layer(visual_embeddings)

#         text_embeddings, _, _ = self.bert.get_text_embedding(
#             sample_list["input_ids"],
#             sample_list["segment_ids"],
#             sample_list["input_mask"],
#         )
#         # text_embeddings = text_embeddings[:, 0]
#         masks = sample_list["input_mask"]
#         text_embeddings = text_embeddings * masks.unsqueeze(2)
#         text_embeddings = torch.sum(text_embeddings, dim=1) / (
#             torch.sum(masks, dim=1, keepdim=True)
#         )
#         text_embeddings = self.norm_layer(text_embeddings)

#         output_dict = {
#             "scores": visual_embeddings,
#             "targets": text_embeddings,
#         }
#         return output_dict
