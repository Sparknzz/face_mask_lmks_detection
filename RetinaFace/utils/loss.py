import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from utils.box_utils import match, log_sum_exp

class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).

        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.

        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)

    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, prior_for_matching, bkg_label, neg_mining, neg_pos, neg_overlap, encode_target):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = [0.1, 0.2]

    def forward(self, predictions, priors, targets):
        """Multibox Loss
        Args:
            predictions (tuple): A tuple containing loc preds, conf preds,
            and prior boxes from SSD net.
                conf shape: torch.size(batch_size, num_priors, num_classes) # n, 3(anchor), 3
                loc shape: torch.size(batch_size,num_priors,4) # n, 3, 4 as we only count each anchor for only one cls not cal for every cls, for rcnn it does for every cls
                priors shape: torch.size(num_priors, 4)

            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size, num_objs, 5] (last idx is the label).
        """
        loc_data, conf_data, landm_data = predictions
        
        priors = priors
        num = loc_data.size(0)
        num_priors = (priors.size(0))

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        landm_t = torch.Tensor(num, num_priors, 10)
        conf_t = torch.LongTensor(num, num_priors)

        for idx in range(num):
            truths = targets[idx][:, :4].data
            labels = targets[idx][:, -1].data # label only has 1 and -1 and 2, 1 and 2 used for cls
            landms = targets[idx][:, 4:14].data
            defaults = priors.data
            match(self.threshold, truths, defaults, self.variance, labels, landms, loc_t, conf_t, landm_t, idx)

        if 1: # use gpu
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()
            landm_t = landm_t.cuda()

        # cos label file including -1 landmarks so ignore this data for landmark regression
        zeros = torch.tensor(0).cuda()
        pos = conf_t > zeros
        num_pos_landm = pos.long().sum(1, keepdim=True)
        N1 = max(num_pos_landm.data.sum().float(), 1)

        # 1. get index for face class
        face_tensor = torch.tensor(1).cuda()
        # Shape: [batch,num_priors,10]
        face_pos = conf_t == face_tensor
        face_pos_idx = face_pos.unsqueeze(face_pos.dim()).expand_as(landm_data) # 32, 16800, 10
        face_landm_p = landm_data[face_pos_idx].view(-1, 10)
        face_landm_t = landm_t[face_pos_idx].view(-1, 10)

        # 2. get index for mask class, set all these target landmarks to 0.
        # conf_t is target and anchor, if anchor matched target bbox cls 1, then the conf_t is 1, if anchor matched cls2, conf_t is 2
        # after match conf_t, we can get all cls2's anchors.
        mask_tensor = torch.tensor(2).cuda()
        mask_pos = conf_t == mask_tensor
        mask_pos_idx = mask_pos.unsqueeze(mask_pos.dim()).expand_as(landm_data)
        mask_landm_p = landm_data[mask_pos_idx].view(-1, 10)
        mask_landm_t = landm_t[mask_pos_idx].view(-1, 10)

        mask_landm_p[:, 4:] = 0
        mask_landm_t[:, 4:] = 0

        landm_p = torch.cat([face_landm_p, mask_landm_p], 0)
        landm_t = torch.cat([face_landm_t, mask_landm_t], 0)

        loss_landm = F.smooth_l1_loss(landm_p, landm_t, reduction='sum')
        ##############################################################################################################
        zeros = torch.tensor(0).cuda()
        pos = conf_t != zeros

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, reduction='sum')

        ##############################################################################################################
        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        # Note here, as label file including -1 label, so we have to make them to 0~2
        no_landmark_pos = conf_t < zeros
        conf_t[no_landmark_pos] = 1
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        # Hard Negative Mining
        loss_c[pos.view(-1, 1)] = 0 # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1,self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, reduction='sum')
        ##############################################################################################################

        # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        N = max(num_pos.data.sum().float(), 1)
        loss_l /= N
        loss_c /= N
        loss_landm /= N1

        return loss_l, loss_c, loss_landm
