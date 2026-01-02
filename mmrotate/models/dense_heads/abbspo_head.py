# Copyright (c) OpenMMLab. All rights reserved.
import copy
import math
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch import Tensor

from mmdet.models.utils import filter_scores_and_topk, multi_apply
from mmdet.structures import SampleList
from mmdet.structures.bbox import cat_boxes, get_box_tensor
from mmdet.utils import (ConfigType, InstanceList, OptConfigType,
                         OptInstanceList, reduce_mean)
from mmengine import ConfigDict
from mmengine.structures import InstanceData

from mmrotate.models.dense_heads.rotated_fcos_head import RotatedFCOSHead
from mmrotate.registry import MODELS
from mmrotate.structures import RotatedBoxes

from mmdet.models.utils import unpack_gt_instances

INF = 1e8


@MODELS.register_module()
class ABBSPOHead(RotatedFCOSHead):
    """Anchor-free head used in `ABBSPO`.

    This head is based on H2RBox-v2 and adds:
        - ABBS Module (Adaptive Bounding Box Scaling Module): Optimal scale search for bounding box width/height
        - SPA loss (Symmetric Prior Angle loss): Predictive angle refinement via symmetry-based image similarity
        - Proposal sampling strategy based on quality assessment

    Args:
        num_classes (int): Number of categories excluding the background.
        in_channels (int): Number of channels in the input feature map.
        angle_version (str): Angle representation. Defaults to 'le90'.
        use_hbbox_loss (bool): If true, use horizontal bbox loss and
            loss_angle should not be None. Defaults to False.
        scale_angle (bool): If true, add scale to angle pred branch.
            Defaults to True.
        angle_coder (ConfigType): Config of angle coder.
        h_bbox_coder (ConfigType): Config of horizontal bbox coder.
        bbox_coder (ConfigType): Config of bbox coder.
        loss_cls (ConfigType): Config of classification loss.
        loss_bbox (ConfigType): Config of localization loss.
        loss_centerness (ConfigType): Config of centerness loss.
        loss_angle (OptConfigType): Config of angle loss.
        loss_symmetry_ss (ConfigType): Config of symmetry consistency loss.
        loss_spa (ConfigType): Config of SPA loss.
        rotation_agnostic_classes (list): IDs of rotation-agnostic categories.
        agnostic_resize_classes (list): IDs of categories with additional
            resize during inference.
        use_circumiou_loss (bool): Whether to use circum-IOU loss.
        use_standalone_angle (bool): If True, angle is learned by SS only.
        use_reweighted_loss_bbox (bool): Whether to reweight bbox loss
            with symmetry loss.
        scale (float): Base scaling factor (not directly used; kept for
            compatibility).
        scale_set_w (list[tuple[float, float]]): Range of width scale factors.
        scale_set_h (list[tuple[float, float]]): Range of height scale factors.
        top_ratio (float): Ratio for top-k selection in proposal sampling.
        angle_weight_spa (float): Weight of SPA loss.
        angle_weight_ss (float): Weight of symmetry loss.
        regularization_alpha (float): Weight of regularization term using
            GT-based loss.
    """

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 angle_version: str = 'le90',
                 use_hbbox_loss: bool = False,
                 scale_angle: bool = False,
                 angle_coder: ConfigType = dict(type='PseudoAngleCoder'),
                 h_bbox_coder: ConfigType = dict(
                     type='mmdet.DistancePointBBoxCoder'),
                 bbox_coder: ConfigType = dict(type='DistanceAnglePointCoder'),
                 loss_cls: ConfigType = dict(
                     type='mmdet.FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox: ConfigType = dict(
                     type='RotatedIoULoss', loss_weight=1.0),
                 loss_centerness: ConfigType = dict(
                     type='mmdet.CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_angle: OptConfigType = None,
                 loss_symmetry_ss: ConfigType = dict(
                     type='H2RBoxV2ConsistencyIoULoss'),
                 loss_spa: ConfigType = dict(type='SPALoss'),
                 rotation_agnostic_classes: list = None,
                 agnostic_resize_classes: list = None,
                 use_circumiou_loss: bool = True,
                 use_standalone_angle: bool = True,
                 use_reweighted_loss_bbox: bool = False,
                 scale: float = 1.1,
                 scale_set_w: List[Tuple[float, float]] = [
                     (1.0, 1.0), (1.0, 1.1), (1.1, 1.2),
                     (1.2, 1.3), (1.3, 1.4), (1.4, 1.5)],
                 scale_set_h: List[Tuple[float, float]] = [
                     (1.0, 1.0), (1.0, 1.1), (1.1, 1.2),
                     (1.2, 1.3), (1.3, 1.4), (1.4, 1.5)],
                 top_ratio: float = 0.1,
                 angle_weight_spa: float = 0.05,
                 angle_weight_ss: float = 0.6,
                 regularization_alpha: float = 0.01,
                 **kwargs) -> None:
        super().__init__(
            num_classes=num_classes,
            in_channels=in_channels,
            angle_version=angle_version,
            use_hbbox_loss=use_hbbox_loss,
            scale_angle=scale_angle,
            angle_coder=angle_coder,
            h_bbox_coder=h_bbox_coder,
            bbox_coder=bbox_coder,
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_centerness=loss_centerness,
            loss_angle=loss_angle,
            **kwargs)

        self.loss_symmetry_ss = MODELS.build(loss_symmetry_ss)
        self.loss_spa = MODELS.build(loss_spa)

        self.rotation_agnostic_classes = rotation_agnostic_classes
        self.agnostic_resize_classes = agnostic_resize_classes

        self.use_circumiou_loss = use_circumiou_loss
        self.use_standalone_angle = use_standalone_angle
        self.use_reweighted_loss_bbox = use_reweighted_loss_bbox

        self.scale = scale
        self.scale_set_w = scale_set_w
        self.scale_set_h = scale_set_h
        self.top_ratio = top_ratio
        self.angle_weight_spa = angle_weight_spa
        self.angle_weight_ss = angle_weight_ss
        self.regularization_alpha = regularization_alpha


    def obb2xyxy(self, rbboxes: Tensor) -> Tensor:
        """Convert rotated boxes (x, y, w, h, a) to horizontal boxes
        (x1, y1, x2, y2)."""
        w = rbboxes[:, 2::5]
        h = rbboxes[:, 3::5]
        a = rbboxes[:, 4::5].detach()
        cosa = torch.cos(a).abs()
        sina = torch.sin(a).abs()
        hbbox_w = cosa * w + sina * h
        hbbox_h = sina * w + cosa * h
        dx = rbboxes[..., 0]
        dy = rbboxes[..., 1]
        dw = hbbox_w.reshape(-1)
        dh = hbbox_h.reshape(-1)
        x1 = dx - dw / 2
        y1 = dy - dh / 2
        x2 = dx + dw / 2
        y2 = dy + dh / 2
        return torch.stack((x1, y1, x2, y2), -1)

    def nested_projection(self, pred: Tensor,
                          target: Tensor) -> Tuple[Tensor, Tensor]:
        """Project oriented boxes into their enclosing horizontal boxes.

        Args:
            pred (Tensor): Predicted OBBs (N, 5).
            target (Tensor): Target OBBs (N, 5).

        Returns:
            Tuple[Tensor, Tensor]: Projected HBBs (pred, target) with shape
                (N, 4) each.
        """
        target_xy1 = target[..., 0:2] - target[..., 2:4] / 2
        target_xy2 = target[..., 0:2] + target[..., 2:4] / 2
        target_projected = torch.cat((target_xy1, target_xy2), -1)

        pred_xy = pred[..., 0:2]
        pred_wh = pred[..., 2:4]
        da = pred[..., 4] - target[..., 4]
        cosa = torch.cos(da).abs()
        sina = torch.sin(da).abs()
        pred_wh = torch.matmul(
            torch.stack((cosa, sina, sina, cosa), -1).view(
                *cosa.shape, 2, 2),
            pred_wh[..., None])[..., 0]
        pred_xy1 = pred_xy - pred_wh / 2
        pred_xy2 = pred_xy + pred_wh / 2
        pred_projected = torch.cat((pred_xy1, pred_xy2), -1)
        return pred_projected, target_projected

    def _get_rotation_agnostic_mask(self, cls: Tensor) -> Tensor:
        """Return boolean mask for rotation-agnostic classes."""
        _rot_agnostic_mask = torch.zeros_like(cls, dtype=torch.bool)
        if self.rotation_agnostic_classes is None:
            return _rot_agnostic_mask
        for c in self.rotation_agnostic_classes:
            _rot_agnostic_mask = torch.logical_or(_rot_agnostic_mask, cls == c)
        return _rot_agnostic_mask

    def loss(self,
             x: Tuple[Tensor],
             batch_inputs: List[Tensor],
             batch_data_samples: SampleList,
             rot: float) -> Dict[str, Tensor]:
        """Override FCOS head loss to accept extra inputs.

        Args:
            x (Tuple[Tensor]): Features from backbone/neck.
            batch_inputs (list[Tensor]): Input images (N, C, H, W) from
                detector.
            batch_data_samples (SampleList): Data samples with GTs.
            rot (float): Rotation angle used to generate rotated views.

        Returns:
            dict: A dictionary of loss components.
        """
        outs = self(x)
        (batch_gt_instances, batch_gt_instances_ignore,
         batch_img_metas) = unpack_gt_instances(batch_data_samples)
        loss_inputs = outs + (
            batch_inputs,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore,
            rot,
        )
        losses = self.loss_by_feat(*loss_inputs)
        return losses

    def loss_by_feat(
        self,
        cls_scores: List[Tensor],
        bbox_preds: List[Tensor],
        angle_preds: List[Tensor],
        centernesses: List[Tensor],
        batch_inputs: List[Tensor],
        batch_gt_instances: InstanceList,
        batch_img_metas: List[dict],
        batch_gt_instances_ignore: OptInstanceList = None,
        rot: float = 0.0
    ) -> Dict[str, Tensor]:
        """Calculate loss based on head outputs.

        This is an extended version of H2RBox-v2 loss that:
            - splits views (ori/rot/flp),
            - performs scale search on GT boxes,
            - introduces SPA loss and proposal sampling.

        Args:
            cls_scores (list[Tensor]): Classification scores for each level.
            bbox_preds (list[Tensor]): Box deltas for each level.
            angle_preds (list[Tensor]): Angle logits for each level.
            centernesses (list[Tensor]): Centerness for each level.
            batch_inputs (list[Tensor]): Original input images (N, C, H, W).
            batch_gt_instances (list[InstanceData]): GT for each image.
            batch_img_metas (list[dict]): Image metadata.
            batch_gt_instances_ignore (list[InstanceData], optional):
                GT to ignore. Defaults to None.
            rot (float): Rotation angle used for one of the views.

        Returns:
            dict[str, Tensor]: Loss dictionary.
        """
        assert len(cls_scores) == len(bbox_preds) == len(angle_preds) == len(
            centernesses)

        num_views = 3

        # split bbox / angle predictions into views
        bbox_preds_split = [
            torch.chunk(bbox_pred, num_views, dim=0)
            for bbox_pred in bbox_preds
        ]
        angle_preds_split = [
            torch.chunk(angle_pred, num_views, dim=0)
            for angle_pred in angle_preds
        ]
        bbox_preds_ori = [split[0] for split in bbox_preds_split]
        angle_preds_ori = [split[0] for split in angle_preds_split]

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        num_proposals_each_level = [
            featmap.size(-1) * featmap.size(-2) for featmap in cls_scores
        ]

        all_level_points = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds[0].dtype,
            device=bbox_preds[0].device)
        all_level_points_ori = self.prior_generator.grid_priors(
            featmap_sizes,
            dtype=bbox_preds_ori[0].dtype,
            device=bbox_preds_ori[0].device)

        labels, bbox_targets, angle_targets, bid_targets = self.get_targets(
            all_level_points, batch_gt_instances)

        num_imgs = cls_scores[0].size(0)

        # flatten outputs
        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_angle_preds = [
            angle_pred.permute(0, 2, 3, 1).reshape(
                -1, self.angle_coder.encode_size)
            for angle_pred in angle_preds
        ]
        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(-1)
            for centerness in centernesses
        ]

        flatten_cls_scores = torch.cat(flatten_cls_scores)
        flatten_bbox_preds = torch.cat(flatten_bbox_preds)
        flatten_angle_preds = torch.cat(flatten_angle_preds)
        flatten_centerness = torch.cat(flatten_centerness)

        flatten_labels = torch.cat(labels)
        flatten_bbox_targets = torch.cat(bbox_targets)
        flatten_angle_targets = torch.cat(angle_targets)
        flatten_bid_targets = torch.cat(bid_targets)

        flatten_points = torch.cat(
            [points.repeat(num_imgs, 1) for points in all_level_points])

        bg_class_ind = self.num_classes
        pos_inds = ((flatten_labels >= 0) &
                    (flatten_labels < bg_class_ind)).nonzero().reshape(-1)
        num_pos = torch.tensor(
            len(pos_inds), dtype=torch.float, device=bbox_preds[0].device)
        num_pos = max(reduce_mean(num_pos), 1.0)

        loss_cls = self.loss_cls(
            flatten_cls_scores, flatten_labels, avg_factor=num_pos)

        pos_bbox_preds = flatten_bbox_preds[pos_inds]
        pos_angle_preds = flatten_angle_preds[pos_inds]
        pos_centerness = flatten_centerness[pos_inds]
        pos_bbox_targets = flatten_bbox_targets[pos_inds]
        pos_angle_targets = flatten_angle_targets[pos_inds]
        pos_bid_targets = flatten_bid_targets[pos_inds]
        pos_centerness_targets = self.centerness_target(pos_bbox_targets)

        centerness_denorm = max(
            reduce_mean(pos_centerness_targets.sum().detach()), 1e-6)

        if len(pos_inds) > 0:
            pos_points = flatten_points[pos_inds]
            pos_labels = flatten_labels[pos_inds]
            pos_cls_scores = flatten_cls_scores[pos_inds]

            pos_decoded_angle_preds = self.angle_coder.decode(
                pos_angle_preds, keepdim=True)

            if self.use_standalone_angle:
                pos_decoded_angle_preds = pos_decoded_angle_preds.detach()

            if self.rotation_agnostic_classes:
                pos_agnostic_mask = self._get_rotation_agnostic_mask(
                    pos_labels)
                pos_decoded_angle_preds[pos_agnostic_mask] = 0
                target_mask = torch.abs(
                    pos_angle_targets[pos_agnostic_mask]) < torch.pi / 4
                pos_angle_targets[pos_agnostic_mask] = torch.where(
                    target_mask, 0, -torch.pi / 2)

            pos_bbox_preds = torch.cat(
                [pos_bbox_preds, pos_decoded_angle_preds], dim=-1)
            pos_bbox_targets = torch.cat(
                [pos_bbox_targets, pos_angle_targets], dim=-1)

            pos_decoded_bbox_preds = self.bbox_coder.decode(
                pos_points, pos_bbox_preds)
            pos_decoded_bbox_targets = self.bbox_coder.decode(
                pos_points, pos_bbox_targets)

            # scale search
            S_w = self.scale_set_w
            S_h = self.scale_set_h

            scale_factors_w = [
                torch.empty(1).uniform_(s_low, s_high).item()
                for (s_low, s_high) in S_w
            ]
            scale_factors_h = [
                torch.empty(1).uniform_(s_low, s_high).item()
                for (s_low, s_high) in S_h
            ]
            scale_combinations = [
                (sw, sh) for sw in scale_factors_w for sh in scale_factors_h
            ]

            rotated_mask = (pos_bid_targets % 1 == 0.4)

            theta = pos_decoded_bbox_preds[:, 4].detach()
            theta[rotated_mask] = theta[rotated_mask] - rot

            theta_mod = torch.remainder(theta, math.pi / 2)
            pi_over_four = math.pi / 4
            condition_increase = theta_mod <= pi_over_four

            scaled_bbox_target = []
            for sw, sh in scale_combinations:
                scale_increase_w = 1 + (sw - 1) * (theta_mod / pi_over_four)
                scale_decrease_w = (
                    sw - (sw - 1) * ((theta_mod - pi_over_four) /
                                     pi_over_four))
                scale_factor_w = torch.where(condition_increase,
                                             scale_increase_w,
                                             scale_decrease_w)

                scale_increase_h = 1 + (sh - 1) * (theta_mod / pi_over_four)
                scale_decrease_h = (
                    sh - (sh - 1) * ((theta_mod - pi_over_four) /
                                     pi_over_four))
                scale_factor_h = torch.where(condition_increase,
                                             scale_increase_h,
                                             scale_decrease_h)

                scaled = pos_decoded_bbox_targets.clone()
                scaled[:, 2] *= scale_factor_w
                scaled[:, 3] *= scale_factor_h
                scaled_bbox_target.append(scaled)

            scaled_bbox_target_tensor = torch.stack(scaled_bbox_target, dim=0)
            scaled_gt = scaled_bbox_target_tensor

            # NOTE: current implementation effectively uses only min loss
            # computed from last loop; kept as-is to preserve behavior.
            loss_bbox = pos_decoded_bbox_targets.new_tensor(
                0.0, device=pos_decoded_bbox_targets.device)

            loss = self.loss_bbox(
                *self.nested_projection(pos_decoded_bbox_preds, scaled_gt),
                weight=pos_centerness_targets,
                avg_factor=centerness_denorm)
            min_loss, _ = loss.min(dim=0)

            loss_bbox_gt = self.loss_bbox(
                *self.nested_projection(pos_decoded_bbox_preds,
                                        pos_decoded_bbox_targets),
                weight=pos_centerness_targets,
                avg_factor=centerness_denorm)

            loss_bbox_gt = loss_bbox_gt.mean()
            min_loss = min_loss.mean()
            loss_bbox_proposals = (
                min_loss + self.regularization_alpha * loss_bbox_gt)
            loss_bbox += loss_bbox_proposals.sum()

            loss_centerness = self.loss_centerness(
                pos_centerness,
                pos_centerness_targets,
                avg_factor=num_pos)

            # Self-supervision of angle (same as H2RBox-v2 style)
            bid, idx = torch.unique(pos_bid_targets, return_inverse=True)
            compacted_bid_targets = torch.empty_like(bid).index_reduce_(
                0, idx, pos_bid_targets, 'mean', include_self=False)

            _, bidx, bcnt = torch.unique(
                compacted_bid_targets.long(),
                return_inverse=True,
                return_counts=True)
            bmsk = bcnt[bidx] == 3

            compacted_angle_targets = torch.empty_like(bid).index_reduce_(
                0, idx, pos_angle_targets[:, 0], 'mean',
                include_self=False)[bmsk].view(-1, 3)
            compacted_angle_preds = torch.empty(
                *bid.shape, pos_angle_preds.shape[-1],
                device=bid.device).index_reduce_(
                    0, idx, pos_angle_preds, 'mean',
                    include_self=False)[bmsk].view(
                        -1, 3, pos_angle_preds.shape[-1])

            compacted_angle_preds = self.angle_coder.decode(
                compacted_angle_preds, keepdim=False)
            compacted_agnostic_mask = None
            if self.rotation_agnostic_classes:
                compacted_labels = torch.empty(
                    bid.shape, dtype=pos_labels.dtype,
                    device=bid.device).index_reduce_(
                        0, idx, pos_labels, 'mean',
                        include_self=False)[bmsk].view(-1, 3)[:, 0]
                compacted_agnostic_mask = self._get_rotation_agnostic_mask(
                    compacted_labels)

            loss_symmetry_ss = self.loss_symmetry_ss(
                compacted_angle_preds[:, 0], compacted_angle_preds[:, 1],
                compacted_angle_preds[:, 2], compacted_angle_targets[:, 0],
                compacted_angle_targets[:, 1], compacted_agnostic_mask)

            # SPA Loss + proposal sampling
            num_levels = len(cls_scores)
            num_imgs_ori = batch_inputs.shape[0]  # should be views-merged batch, ori count

            inds_level_interval = np.cumsum(num_proposals_each_level)
            num_inds = num_imgs_ori * inds_level_interval[-1]

            # reshape only "ori" part
            flatten_cls_scores_ori = flatten_cls_scores[:num_inds].reshape(
                num_imgs_ori, -1, self.num_classes)
            flatten_bbox_preds_ori = flatten_bbox_preds[:num_inds].reshape(
                num_imgs_ori, -1, 4)
            flatten_angle_preds_ori = flatten_angle_preds[:num_inds].reshape(
                num_imgs_ori, -1, self.angle_coder.encode_size)
            flatten_bbox_targets_ori = flatten_bbox_targets[:num_inds].reshape(
                num_imgs_ori, -1, 4)
            flatten_angle_targets_ori = flatten_angle_targets[:num_inds].reshape(
                num_imgs_ori, -1, 1)
            flatten_labels_ori = flatten_labels[:num_inds].reshape(
                num_imgs_ori, -1)
            flatten_points_ori = flatten_points[:num_inds].reshape(
                num_imgs_ori, -1, 2)
            flatten_bid_targets_ori = flatten_bid_targets[:num_inds].reshape(
                num_imgs_ori, -1)

            # positive indices per image
            pos_inds_ori = []
            pos_cnts_ori = []
            for i in range(num_imgs_ori):
                pos_inds_1 = ((flatten_labels_ori[i] >= 0) &
                              (flatten_labels_ori[i] < bg_class_ind)).nonzero().reshape(-1)
                pos_inds_ori.append(pos_inds_1)
                pos_cnts_ori.append(pos_inds_1.numel())

            num_pos_ori_total = torch.tensor(
                float(sum(pos_cnts_ori)),
                dtype=torch.float,
                device=bbox_preds[0].device)
            num_pos_ori_global = max(reduce_mean(num_pos_ori_total), 1.0)

            # build stride-per-prediction vector
            points_per_level_ori = [points.size(0) for points in all_level_points_ori]
            strides_per_level_ori = self.strides

            strides_per_prediction_ori = []
            for stride, num_points in zip(strides_per_level_ori, points_per_level_ori):
                strides_per_prediction_ori.extend([stride] * num_points)
            strides_per_prediction_ori = torch.tensor(
                strides_per_prediction_ori,
                dtype=torch.float,
                device=flatten_bbox_preds_ori.device)

            pos_decoded_bbox_preds_ori_list = []
            pos_labels_ori_list = []

            for i in range(num_imgs_ori):
                pos_inds_1 = pos_inds_ori[i]

                # keep list alignment for SPALoss
                if pos_inds_1.numel() == 0:
                    pos_decoded_bbox_preds_ori_list.append(None)
                    pos_labels_ori_list.append(None)
                    continue

                pos_cls_scores_ori = flatten_cls_scores_ori[i][pos_inds_1]
                pos_bbox_preds_ori = flatten_bbox_preds_ori[i][pos_inds_1]
                pos_labels_ori = flatten_labels_ori[i][pos_inds_1]
                pos_angle_preds_ori = flatten_angle_preds_ori[i][pos_inds_1]
                pos_points_ori = flatten_points_ori[i][pos_inds_1]

                pos_decoded_angle_preds_ori = self.angle_coder.decode(
                    pos_angle_preds_ori, keepdim=True)
                pos_bbox_preds_ori = torch.cat(
                    [pos_bbox_preds_ori, pos_decoded_angle_preds_ori], dim=-1)
                pos_decoded_bbox_preds_ori = self.bbox_coder.decode(
                    pos_points_ori, pos_bbox_preds_ori)

                pos_bbox_targets_ori = flatten_bbox_targets_ori[i][pos_inds_1]
                pos_angle_targets_ori = flatten_angle_targets_ori[i][pos_inds_1]
                pos_centerness_targets_ori = self.centerness_target(pos_bbox_targets_ori)

                pos_bbox_targets_ori = torch.cat(
                    [pos_bbox_targets_ori, pos_angle_targets_ori], dim=-1)
                pos_decoded_bbox_targets_ori = self.bbox_coder.decode(
                    pos_points_ori, pos_bbox_targets_ori)

                pos_bid_targets_ori = flatten_bid_targets_ori[i][pos_inds_1]

                with torch.no_grad():
                    points_quality_assessment, _, _ = self.points_quality_assessment(
                        pos_cls_scores_ori,
                        pos_decoded_bbox_preds_ori,
                        pos_labels_ori,
                        pos_decoded_bbox_targets_ori,
                        pos_centerness_targets_ori,
                        centerness_denorm,
                        num_pos_ori_global,    
                        pos_inds_ori[i])

                    labels_t, _ = self.point_samples_selection(
                        points_quality_assessment,
                        flatten_labels_ori[i],
                        pos_inds_ori[i],
                        pos_bid_targets_ori,
                        pos_centerness_targets_ori,
                        num_proposals_each_level=num_proposals_each_level,
                        num_level=num_levels)

                pos_inds_new = ((labels_t >= 0) &
                                (labels_t < bg_class_ind)).nonzero().reshape(-1)

                # if nothing selected, keep None to preserve alignment
                if pos_inds_new.numel() == 0:
                    pos_decoded_bbox_preds_ori_list.append(None)
                    pos_labels_ori_list.append(None)
                    continue

                pos_bbox_preds_ori = flatten_bbox_preds_ori[i][pos_inds_new]
                pos_labels_ori = flatten_labels_ori[i][pos_inds_new]
                pos_angle_preds_ori = flatten_angle_preds_ori[i][pos_inds_new]
                pos_points_ori = flatten_points_ori[i][pos_inds_new]

                pos_decoded_angle_preds_ori = self.angle_coder.decode(
                    pos_angle_preds_ori, keepdim=True)
                pos_bbox_preds_ori = torch.cat(
                    [pos_bbox_preds_ori, pos_decoded_angle_preds_ori], dim=-1)
                pos_decoded_bbox_preds_ori = self.bbox_coder.decode(
                    pos_points_ori, pos_bbox_preds_ori)

                stride_per_pos_ori = strides_per_prediction_ori[pos_inds_new]
                pos_decoded_bbox_preds_ori[:, 2] *= stride_per_pos_ori
                pos_decoded_bbox_preds_ori[:, 3] *= stride_per_pos_ori

                pos_decoded_bbox_preds_ori_list.append(pos_decoded_bbox_preds_ori)
                pos_labels_ori_list.append(pos_labels_ori)

            loss_spa = self.loss_spa(
                batch_inputs,
                pos_decoded_bbox_preds_ori_list,
                pos_labels_ori_list)

            loss_angle = (self.angle_weight_spa * loss_spa +
                          self.angle_weight_ss * loss_symmetry_ss)

            if self.use_reweighted_loss_bbox:
                loss_bbox = math.exp(-loss_symmetry_ss.item()) * loss_bbox
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_centerness = pos_centerness.sum()
            loss_angle = pos_angle_preds.sum()

        return dict(
            loss_cls=loss_cls,
            loss_bbox=loss_bbox,
            loss_centerness=loss_centerness,
            loss_angle=loss_angle)

    def get_targets(
        self,
        points: List[Tensor],
        batch_gt_instances: InstanceList
    ) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        """Compute regression, classification and angle targets for points.

        Args:
            points (list[Tensor]): Points of each FPN level,
                each has shape (num_points, 2).
            batch_gt_instances (list[InstanceData]): GT instances.

        Returns:
            tuple:
                - concat_lvl_labels (list[Tensor])
                - concat_lvl_bbox_targets (list[Tensor])
                - concat_lvl_angle_targets (list[Tensor])
                - concat_lvl_id_targets (list[Tensor])
        """
        assert len(points) == len(self.regress_ranges)
        num_levels = len(points)

        expanded_regress_ranges = [
            points[i].new_tensor(self.regress_ranges[i])[None].expand_as(
                points[i]) for i in range(num_levels)
        ]
        concat_regress_ranges = torch.cat(expanded_regress_ranges, dim=0)
        concat_points = torch.cat(points, dim=0)

        num_points = [center.size(0) for center in points]

        labels_list, bbox_targets_list, angle_targets_list, id_targets_list = \
            multi_apply(
                self._get_targets_single,
                batch_gt_instances,
                points=concat_points,
                regress_ranges=concat_regress_ranges,
                num_points_per_lvl=num_points)

        labels_list = [labels.split(num_points, 0) for labels in labels_list]
        bbox_targets_list = [
            bbox_targets.split(num_points, 0)
            for bbox_targets in bbox_targets_list
        ]
        angle_targets_list = [
            angle_targets.split(num_points, 0)
            for angle_targets in angle_targets_list
        ]
        id_targets_list = [
            id_targets.split(num_points, 0) for id_targets in id_targets_list
        ]

        concat_lvl_labels = []
        concat_lvl_bbox_targets = []
        concat_lvl_angle_targets = []
        concat_lvl_id_targets = []
        for i in range(num_levels):
            concat_lvl_labels.append(
                torch.cat([labels[i] for labels in labels_list]))
            bbox_targets = torch.cat(
                [bbox_targets[i] for bbox_targets in bbox_targets_list])
            angle_targets = torch.cat(
                [angle_targets[i] for angle_targets in angle_targets_list])
            id_targets = torch.cat(
                [id_targets[i] for id_targets in id_targets_list])
            if self.norm_on_bbox:
                bbox_targets = bbox_targets / self.strides[i]
            concat_lvl_bbox_targets.append(bbox_targets)
            concat_lvl_angle_targets.append(angle_targets)
            concat_lvl_id_targets.append(id_targets)

        return (concat_lvl_labels, concat_lvl_bbox_targets,
                concat_lvl_angle_targets, concat_lvl_id_targets)

    def _get_targets_single(
        self,
        gt_instances: InstanceData,
        points: Tensor,
        regress_ranges: Tensor,
        num_points_per_lvl: List[int]
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Compute regression and classification targets for a single image."""
        num_points = points.size(0)
        num_gts = len(gt_instances)
        gt_bboxes = gt_instances.bboxes
        gt_labels = gt_instances.labels
        gt_bid = gt_instances.bid

        if num_gts == 0:
            return (gt_labels.new_full((num_points,), self.num_classes),
                    gt_bboxes.new_zeros((num_points, 4)),
                    gt_bboxes.new_zeros((num_points, 1)),
                    gt_bboxes.new_zeros((num_points,)))

        areas = gt_bboxes.areas
        gt_bboxes = gt_bboxes.regularize_boxes(self.angle_version)

        areas = areas[None].repeat(num_points, 1)
        regress_ranges = regress_ranges[:, None, :].expand(
            num_points, num_gts, 2)
        points = points[:, None, :].expand(num_points, num_gts, 2)
        gt_bboxes = gt_bboxes[None].expand(num_points, num_gts, 5)

        gt_ctr, gt_wh, gt_angle = torch.split(gt_bboxes, [2, 2, 1], dim=2)

        cos_angle, sin_angle = torch.cos(gt_angle), torch.sin(gt_angle)
        rot_matrix = torch.cat(
            [cos_angle, sin_angle, -sin_angle, cos_angle],
            dim=-1).reshape(num_points, num_gts, 2, 2)
        offset = points - gt_ctr
        offset = torch.matmul(rot_matrix, offset[..., None]).squeeze(-1)

        w, h = gt_wh[..., 0], gt_wh[..., 1]
        offset_x, offset_y = offset[..., 0], offset[..., 1]
        left = w / 2 + offset_x
        right = w / 2 - offset_x
        top = h / 2 + offset_y
        bottom = h / 2 - offset_y
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        inside_gt_bbox_mask = bbox_targets.min(-1)[0] > 0
        if self.center_sampling:
            radius = self.center_sample_radius
            stride = offset.new_zeros(offset.shape)

            lvl_begin = 0
            for lvl_idx, num_points_lvl in enumerate(num_points_per_lvl):
                lvl_end = lvl_begin + num_points_lvl
                stride[lvl_begin:lvl_end] = self.strides[lvl_idx] * radius
                lvl_begin = lvl_end

            inside_center_bbox_mask = (abs(offset) < stride).all(dim=-1)
            inside_gt_bbox_mask = torch.logical_and(inside_center_bbox_mask,
                                                    inside_gt_bbox_mask)

        max_regress_distance = bbox_targets.max(-1)[0]
        inside_regress_range = (
            (max_regress_distance >= regress_ranges[..., 0]) &
            (max_regress_distance <= regress_ranges[..., 1]))

        areas[inside_gt_bbox_mask == 0] = INF
        areas[inside_regress_range == 0] = INF
        min_area, min_area_inds = areas.min(dim=1)

        labels = gt_labels[min_area_inds]
        labels[min_area == INF] = self.num_classes
        bbox_targets = bbox_targets[range(num_points), min_area_inds]
        angle_targets = gt_angle[range(num_points), min_area_inds]
        bid_targets = gt_bid[min_area_inds]

        return labels, bbox_targets, angle_targets, bid_targets

    def _predict_by_feat_single(
        self,
        cls_score_list: List[Tensor],
        bbox_pred_list: List[Tensor],
        angle_pred_list: List[Tensor],
        score_factor_list: List[Tensor],
        mlvl_priors: List[Tensor],
        img_meta: dict,
        cfg: ConfigDict,
        rescale: bool = False,
        with_nms: bool = True
    ) -> InstanceData:
        """Transform features of a single image into bbox results."""
        if score_factor_list[0] is None:
            with_score_factors = False
        else:
            with_score_factors = True

        cfg = self.test_cfg if cfg is None else cfg
        cfg = copy.deepcopy(cfg)
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1)

        mlvl_bbox_preds = []
        mlvl_valid_priors = []
        mlvl_scores = []
        mlvl_labels = []
        if with_score_factors:
            mlvl_score_factors = []
        else:
            mlvl_score_factors = None

        for (cls_score, bbox_pred, angle_pred, score_factor,
             priors) in zip(cls_score_list, bbox_pred_list, angle_pred_list,
                            score_factor_list, mlvl_priors):

            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            angle_pred = angle_pred.permute(1, 2, 0).reshape(
                -1, self.angle_coder.encode_size)
            if with_score_factors:
                score_factor = score_factor.permute(1, 2,
                                                    0).reshape(-1).sigmoid()
            cls_score = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)[:, :-1]

            score_thr = cfg.get('score_thr', 0)
            results = filter_scores_and_topk(
                scores,
                score_thr,
                nms_pre,
                dict(
                    bbox_pred=bbox_pred,
                    angle_pred=angle_pred,
                    priors=priors))
            scores, labels, keep_idxs, filtered_results = results

            bbox_pred = filtered_results['bbox_pred']
            angle_pred = filtered_results['angle_pred']
            priors = filtered_results['priors']

            decoded_angle = self.angle_coder.decode(angle_pred, keepdim=True)
            bbox_pred = torch.cat([bbox_pred, decoded_angle], dim=-1)

            if with_score_factors:
                score_factor = score_factor[keep_idxs]

            mlvl_bbox_preds.append(bbox_pred)
            mlvl_valid_priors.append(priors)
            mlvl_scores.append(scores)
            mlvl_labels.append(labels)

            if with_score_factors:
                mlvl_score_factors.append(score_factor)

        bbox_pred = torch.cat(mlvl_bbox_preds)
        priors = cat_boxes(mlvl_valid_priors)
        bboxes = self.bbox_coder.decode(priors, bbox_pred, max_shape=img_shape)

        results = InstanceData()
        results.bboxes = RotatedBoxes(bboxes)
        results.scores = torch.cat(mlvl_scores)
        results.labels = torch.cat(mlvl_labels)
        if with_score_factors:
            results.score_factors = torch.cat(mlvl_score_factors)

        if self.rotation_agnostic_classes:
            bboxes = get_box_tensor(results.bboxes)
            for cls_id in self.rotation_agnostic_classes:
                bboxes[results.labels == cls_id, -1] = 0
            if self.agnostic_resize_classes:
                for cls_id in self.agnostic_resize_classes:
                    bboxes[results.labels == cls_id, 2:4] *= 0.85
            results.bboxes = RotatedBoxes(bboxes)

        results = self._bbox_post_process(
            results=results,
            cfg=cfg,
            rescale=rescale,
            with_nms=with_nms,
            img_meta=img_meta)

        return results

    def points_quality_assessment(
        self,
        cls_score: Tensor,
        pos_decoded_bbox_preds: Tensor,
        label: Tensor,
        pos_decoded_bbox_targets: Tensor,
        cls_centerness: Tensor,
        centerness_denorm: Tensor,
        num_pos: Tensor,
        pos_inds: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """Assess quality of points for proposal sampling.

        Args:
            cls_score (Tensor): Classification scores of positive points.
            pos_decoded_bbox_preds (Tensor): Decoded predicted boxes.
            label (Tensor): Class labels of positive points.
            pos_decoded_bbox_targets (Tensor): Decoded GT boxes.
            cls_centerness (Tensor): Centerness targets.
            centerness_denorm (Tensor): Normalization factor for centerness.
            num_pos (Tensor): Number of positive samples.
            pos_inds (Tensor): Indices of positive samples.

        Returns:
            tuple[Tensor, Tensor, Tensor]:
                - quality_assess (Tensor): quality scores (used for sampling)
                - qua_cls (Tensor): classification quality term
                - qua_loc (Tensor): localization quality term
        """
        xy_preds = pos_decoded_bbox_preds[:, :2]
        xy_targets = pos_decoded_bbox_targets[:, :2]
        xy_diff = ((xy_preds[:, 0] - xy_targets[:, 0])**2 +
                   (xy_preds[:, 1] - xy_targets[:, 1])**2)

        qua_cls = self.loss_cls(
            cls_score,
            label,
            avg_factor=num_pos,
            reduction_override='none')
        qua_cls = qua_cls.sum(-1)

        pred_proj, target_proj = self.nested_projection(
            pos_decoded_bbox_preds, pos_decoded_bbox_targets)
        qua_loc = self.loss_bbox(
            pred_proj,
            target_proj,
            weight=cls_centerness,
            avg_factor=centerness_denorm,
            reduction_override='none')

        quality = qua_cls + qua_loc
        return quality, qua_cls, qua_loc

    def point_samples_selection(
        self,
        quality_assess: Tensor,
        label: Tensor,
        pos_inds: Tensor,
        pos_bid_targets: Tensor,
        cls_centerness: Tensor,
        num_proposals_each_level: List[int],
        num_level: int
    ) -> Tuple[Tensor, int]:
        """Select proposal samples based on quality assessment.

        Args:
            quality_assess (Tensor): Quality scores for positive samples.
            label (Tensor): Classification labels for all samples.
            pos_inds (Tensor): Indices of positive samples.
            pos_bid_targets (Tensor): BID targets of positive samples.
            cls_centerness (Tensor): Centerness targets for positive samples.
            num_proposals_each_level (list[int]): Number of proposals per level.
            num_level (int): Number of FPN levels.

        Returns:
            tuple:
                - label (Tensor): Updated labels (low-quality samples set to BG).
                - num_pos (int): Number of positive samples after selection.
        """
        if len(pos_inds) == 0:
            return label, 0

        pos_bid_targets = pos_bid_targets.long()
        num_gt = pos_bid_targets.max()

        num_proposals_each_level_ = num_proposals_each_level.copy()
        num_proposals_each_level_.insert(0, 0)
        inds_level_interval = np.cumsum(num_proposals_each_level_)
        pos_level_mask = []
        for i in range(num_level):
            mask = ((pos_inds >= inds_level_interval[i]) &
                    (pos_inds < inds_level_interval[i + 1]))
            pos_level_mask.append(mask)

        pos_inds_after_select = []
        ignore_inds_after_select = []

        for gt_ind in range(num_gt):
            pos_inds_select = []
            pos_loss_select = []
            gt_mask = pos_bid_targets == (gt_ind + 1)
            for level in range(num_level):
                level_mask = pos_level_mask[level]
                level_gt_mask = level_mask & gt_mask
                if level_gt_mask.sum() == 0:
                    continue
                value, topk_inds = quality_assess[level_gt_mask].topk(
                    min(level_gt_mask.sum(), 6),
                    largest=False)
                valid_mask = ~torch.isinf(value)
                value = value[valid_mask]
                topk_inds = topk_inds[valid_mask]

                pos_inds_select.append(pos_inds[level_gt_mask][topk_inds])
                pos_loss_select.append(value)

            if len(pos_inds_select) == 0:
                continue

            pos_inds_select = torch.cat(pos_inds_select)
            pos_loss_select = torch.cat(pos_loss_select)

            if len(pos_inds_select) < 2:
                pos_inds_after_select.append(pos_inds_select)
                ignore_inds_after_select.append(
                    pos_inds_select.new_tensor([]))
            else:
                pos_loss_select, sort_inds = pos_loss_select.sort()
                pos_inds_select = pos_inds_select[sort_inds]
                topk = math.ceil(pos_loss_select.shape[0] * self.top_ratio)
                pos_inds_select_topk = pos_inds_select[:topk]
                pos_inds_after_select.append(pos_inds_select_topk)
                ignore_inds_after_select.append(
                    pos_inds_select_topk.new_tensor([]))

        pos_inds_after_select = torch.cat(pos_inds_after_select)
        ignore_inds_after_select = torch.cat(ignore_inds_after_select)

        reassign_mask = (pos_inds.unsqueeze(1) != pos_inds_after_select).all(1)
        reassign_ids = pos_inds[reassign_mask]

        label_t = label.clone()
        label_t[reassign_ids] = self.num_classes
        num_pos = len(pos_inds_after_select)

        return label_t, num_pos
