"""Training actor: forward pass + Eq. 9 loss composition (§3.5, §4.1).

§3.5 Eq. 9:  L = L_focal + alpha * L1 + beta * L_GIoU
§4.1: "set the loss weights to alpha = 5, and beta = 2" — focal coefficient is 1
(Eq. 9, user-confirmed).

L1/GIoU are computed on the DECODED box (peak score + gathered size/offset),
the OSTrack convention [UNSPECIFIED] branch mapping in the paper ("The size
and offset regression branches are supervised using L1 loss and GIoU loss").
Focal loss on the gaussian center label (OSTrack/CenterNet convention
[UNSPECIFIED]).

Structure follows the stock OSTrack actor (data keys, num_queries handling,
NaN guard, status keys); the loss internals are a verbatim port of
dinotrack_v1/train/actor.py — CenterNet gaussian labels with the radius-0
positive-center fix (lib/utils/dino/heatmap.py), the v1 FocalLoss, and
L1/GIoU on decoded xyxy boxes against clamped GT.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

from . import BaseActor
from lib.utils.dino.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy, giou_loss
from lib.utils.dino.focal_loss import FocalLoss
from lib.utils.dino.heatmap import generate_heatmap


class DINOTrackActor(BaseActor):
    """ Actor for training DINOTrack models """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.cfg = cfg
        # §3.5 Hx/P = 256/16 = 16
        self.feat_sz = cfg.DATA.SEARCH.SIZE // cfg.MODEL.BACKBONE.STRIDE
        self.focal = FocalLoss(alpha=cfg.TRAIN.FOCAL_ALPHA, beta=cfg.TRAIN.FOCAL_BETA)

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template_images',
                   'search_images', 'search_anno'.
            template_images: (N_t, batch, 3, 128, 128)
            search_images: (N_s, batch, 3, 256, 256)
        returns:
            loss    - the training loss
            status  - dict containing detailed losses
        """
        out_dict = self.forward_pass(data)
        loss, status = self.compute_losses(out_dict, data)
        return loss, status

    def forward_pass(self, data):
        # currently only support 1 template and 1 search region
        assert len(data['template_images']) == 1
        assert len(data['search_images']) == 1

        template_img = data['template_images'][0].view(
            -1, *data['template_images'].shape[2:])       # (batch, 3, 128, 128)
        search_img = data['search_images'][0].view(
            -1, *data['search_images'].shape[2:])         # (batch, 3, 256, 256)

        out_dict = self.net(template=template_img,
                            search=search_img,
                            ce_template_mask=None,
                            ce_keep_rate=None,
                            return_last_attn=False)
        return out_dict

    def compute_losses(self, pred_dict, gt_dict):
        gt_xywh = gt_dict['search_anno'][-1]              # (B,4) normalized xywh in crop

        # --- Eq. 9 first term: focal loss on the gaussian center label
        # (lib/utils/dino/heatmap.py: CornerNet radius incl. the radius-0 fix)
        gt_heat = generate_heatmap(gt_xywh.detach().cpu(), self.feat_sz).to(gt_xywh.device)
        gt_heat = gt_heat.unsqueeze(1)                    # (B,1,16,16)
        loss_focal = self.focal(pred_dict['score_map'], gt_heat)

        # --- Eq. 9 last terms: L1 + GIoU on the decoded box
        pred_boxes = pred_dict['pred_boxes']              # (B,1,4) cxcywh normalized
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        pred_xyxy = box_cxcywh_to_xyxy(pred_boxes.squeeze(1))        # (B,4)
        gt_xyxy = box_xywh_to_xyxy(gt_xywh).clamp(0.0, 1.0)          # (B,4)

        loss_giou, iou = giou_loss(pred_xyxy, gt_xyxy)
        loss_l1 = F.l1_loss(pred_xyxy, gt_xyxy)

        loss = (self.loss_weight['giou'] * loss_giou
                + self.loss_weight['l1'] * loss_l1
                + self.loss_weight['focal'] * loss_focal)

        mean_iou = iou.detach().mean()
        status = {"Loss/total": loss.item(),
                  "Loss/giou": loss_giou.item(),
                  "Loss/l1": loss_l1.item(),
                  "Loss/location": loss_focal.item(),
                  "IoU": mean_iou.item()}
        return loss, status
