from . import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
from lib.utils.merge import merge_template_search
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate


class TBSITrackActor(BaseActor):
    """ Actor for training TBSI_Track models """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict = self.forward_pass(data)

        # compute losses
        loss, status = self.compute_losses(out_dict, data['visible'])

        return loss, status

    def forward_pass(self, data):
        # currently only support 1 template and 1 search region
        # assert len(data['visible']['template_images']) == 1
        # assert len(data['visible']['search_images']) == 1

        template_img_v = data['visible']['template_images'][0].view(-1, *data['visible']['template_images'].shape[2:])  # (batch, 3, 128, 128)
        template_img_i = data['infrared']['template_images'][0].view(-1, *data['infrared']['template_images'].shape[2:])  # (batch, 3, 128, 128)   

        template_img_online_v = data['visible']['template_images'][1].view(-1, *data['visible']['template_images'].shape[2:])  # (batch, 3, 128, 128)
        template_img_online_i = data['infrared']['template_images'][1].view(-1, *data['infrared']['template_images'].shape[2:])  # (batch, 3, 128, 128)      
        
        search_img_v = data['visible']['search_images'][0].view(-1, *data['visible']['search_images'].shape[2:])  # (batch, 3, 320, 320)
        search_img_i = data['infrared']['search_images'][0].view(-1, *data['infrared']['search_images'].shape[2:])  # (batch, 3, 320, 320)

        box_mask_z = None
        ce_keep_rate = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            box_mask_z = generate_mask_cond(self.cfg, template_img_v.shape[0], template_img_v.device,
                                            data['visible']['template_anno'][0])

            ce_start_epoch = self.cfg.TRAIN.CE_START_EPOCH
            ce_warm_epoch = self.cfg.TRAIN.CE_WARM_EPOCH
            ce_keep_rate = adjust_keep_rate(data['epoch'], warmup_epochs=ce_start_epoch,
                                                total_epochs=ce_start_epoch + ce_warm_epoch,
                                                ITERS_PER_EPOCH=1,
                                                base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0])
        search_bboxes = box_xywh_to_xyxy(data['visible']['search_anno'][0].clone())
        out_dict = self.net(template=[template_img_v, template_img_i],
                            template_online=[template_img_online_v, template_img_online_i],
                            search=[search_img_v, search_img_i],
                            ce_template_mask=box_mask_z,
                            ce_keep_rate=ce_keep_rate,
                            return_last_attn=False, gt_bboxes=search_bboxes)

        return out_dict

    def compute_losses(self, pred_dict, gt_dict, return_status=True):
        # gt gaussian map
        gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        
        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)

        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                           max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute giou and iou
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        # compute location loss
        if 'score_map' in pred_dict:
            location_loss = self.objective['focal'](pred_dict['score_map'], gt_gaussian_maps)
        else:
            location_loss = torch.tensor(0.0, device=l1_loss.device)

        if 'cls_logits' in pred_dict:
            gt_label = gt_dict['label'].reshape(-1, 1)
            cls_loss = self.objective['score'](pred_dict['cls_logits'], gt_label)
        else:
             cls_loss = torch.tensor(0.0, device=l1_loss.device)
        # weighted sum
        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss + self.loss_weight['focal'] * location_loss + self.loss_weight['score'] * cls_loss
        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "Loss/location": location_loss.item(),
                      "Loss/cls":cls_loss.item(),
                      "IoU": mean_iou.item()}
            return loss, status
        else:
            return loss
