import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable



def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]

def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                     boxes[:, :2] + boxes[:, 2:]/2), 1)  # xmax, ymax


def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, 4].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        landms: (tensor) Ground truth landms, Shape [num_obj, 10].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        landm_t: (tensor) Tensor to be filled w/ endcoded landm targets.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location 2)confidence 3)landm preds.
    """
    # jaccard index
    overlaps = jaccard(
        truths,
        point_form(priors)
    )
    # (Bipartite Matching)
    # [1,num_objects] best prior for each ground truth
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)

    # ignore hard gt
    valid_gt_idx = best_prior_overlap[:, 0] >= 0.2

    best_prior_idx_filter = best_prior_idx[valid_gt_idx, :]
    
    if best_prior_idx_filter.shape[0] <= 0:
        loc_t[idx] = 0
        conf_t[idx] = 0
        return

    # [1, num_priors] best ground truth for each prior, each gt match an anchor, we called best_prior_idx, each anchor match a gt, we call best_truth_idx
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_idx_filter.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    best_truth_overlap.index_fill_(0, best_prior_idx_filter, 2)  # ensure best prior

    for j in range(best_prior_idx.size(0)):     # 判别此anchor是预测哪一个boxes
        best_truth_idx[best_prior_idx[j]] = j
    
    matches = truths[best_truth_idx]            # Shape: [num_priors,4] 此处为每一个anchor对应的bbox取出来
    conf = labels[best_truth_idx]               # Shape: [num_priors]      此处为每一个anchor对应的label取出来
    conf[best_truth_overlap < threshold] = 0    # label as background   overlap<0.35的全部作为负样本
    loc = encode(matches, priors, variances)

    loc_t[idx] = loc    # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior


def encode(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 4].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
    """

    # dist b/t match center and prior's center
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]
    # encode variance
    g_cxcy /= (variances[0] * priors[:, 2:])
    # match wh / prior wh
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    # return target for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]


def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers # 50000, 2
    explaination: 
        in my opinion, because each anchor has multiple cls confidence, so if the anchor is negative, 
        then it should be sum all cls confidence and still a minimum value. so this function is to measure the predict logits and max logits.

        on the other hand, the across all examples in a batch, every image has the different loss, we have to average the loss. 
        give them a highest score, change whole loss distribution. we can get a batch loss.
    """
    x_max = x.data.max()
    avg_batch_loss = torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max
    return avg_batch_loss

def log_sigmoid(x):
    return torch.clamp(x, max=0)-torch.log(torch.exp(-torch.abs(x))+1)


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

    def __init__(self, num_classes, overlap_thresh, prior_for_matching, bkg_label, neg_mining, neg_pos, neg_overlap):
        super(MultiBoxLoss, self).__init__()
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
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
                conf shape: torch.size(batch_size, num_priors, num_classes) # n, anchor, 2
                loc shape: torch.size(batch_size,num_priors,4) # n, H*W*3, 4 as we only count each anchor for only one cls not cal for every cls, for rcnn it does for every cls
                priors shape: torch.size(num_priors, 4)

            ground_truth (tensor): Ground truth boxes and labels for a batch,
                shape: [batch_size, num_objs, 5] (last idx is the label).
        """
        # loc_data, conf_data, landm_data = predictions

        loc_data, conf_data = predictions
        
        priors = priors
        num = loc_data.size(0)
        num_priors = (priors.size(0))

        # match priors (default boxes) and ground truth boxes
        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)

        for idx in range(num): # for every image in a batch match them
            truths = targets[idx][:, :4].data
            labels = targets[idx][:, -1].data # labels is 1 or 2. face or face_mask
            defaults = priors.data
            match(self.threshold, truths, defaults, self.variance, labels, loc_t, conf_t, idx)

        if 1: # use gpu
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()

        ###########################################################################################################
        # 1. localization regression loss 
        zeros = torch.tensor(0).cuda()
        pos = conf_t != zeros # conf_t means cls 1 or 2's anchor, these anchor will contributes to localization loss

        # Localization Loss (Smooth L1)
        # Shape: [batch,num_priors,4]
        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4) # loc_data is batch, num_anchor, 4
        loc_t = loc_t[pos_idx].view(-1, 4) # no need to care which anchor belongs to which cls, only care the matched gt and matched anchor to regression
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

        ##############################################################################################################
        # Compute max conf across batch for hard negative mining
        batch_conf = conf_data.view(-1, self.num_classes)
        # note here is the softmax loss. for each x x represents each anchor, but why the sel.numclass need to be added?????
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1)) # total_num_anchor, 1  conf_t.view(-1, 1) each anchor to a gt num 1/2
        # loss_c shape batch*num_anchor, 1, it represents the anchors loss. loss is a constant value, but gradients is not same

        # Hard Negative Mining
        loss_c[pos.view(-1, 1)] = 0 # filter out pos boxes for now 
        loss_c = loss_c.view(num, -1)

        _, loss_idx = loss_c.sort(1, descending=True) # each image, get the error anchor
        _, idx_rank = loss_idx.sort(1)

        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1) # make them 1 : 1

        neg = idx_rank < num_neg.expand_as(idx_rank)
        # Confidence Loss Including Positive and Negative Examples
        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)

        conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos + neg).gt(0)]

        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)
        ##############################################################################################################
        N = num_pos.data.sum()
        loss_l /= N
        loss_c /= N

        return loss_l, loss_c

# ######################################### sigmoid loss mode ###########################################
# # different with softmax is softmax are 0 , 1, 2.  sigmoid should be one hot encoding 0, 1 or 1, 0.
# # anyway the same thing is they all contributes to the loss.

# # Compute max conf across batch for hard negative mining
# batch_conf = conf_data.view(-1, self.num_classes)

# # how to determine the loss
# # as matched_target_idx including 0, 1, 2 cls, but when we use sigmoid, we only have 2 neural, so the index would be out of bounds
# # we only care about the zeros
# loss_c = -F.logsigmoid(batch_conf) # cos we only care the negative loss, so don't care about which anchor is 1 or 2, just count the anchor loss

# loss_c[pos.view(-1, 1)] = 0 # filter out pos boxes for now 
# loss_c = loss_c.view(num, -1)

# _, loss_idx = loss_c.sort(1, descending=True)
# _, idx_rank = loss_idx.sort(1)
# num_pos = pos.long().sum(1, keepdim=True)
# num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1) # make them 1 : 1

# neg = idx_rank < num_neg.expand_as(idx_rank)

# # Confidence Loss Including Positive and Negative Examples
# pos_idx = pos.unsqueeze(2).expand_as(conf_data)
# neg_idx = neg.unsqueeze(2).expand_as(conf_data) # batch, num_anchor, 2

# conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, 1)
# targets_weighted = conf_t[(pos+neg).gt(0)].view(-1, 1)

# loss_c =  (conf_p, targets_weighted)  # batch, num_anchor, 2   # batch, num_anchor, 1