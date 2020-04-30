import torch
import torch.nn as nn
import torch.nn.functional as F


def smooth_l1_loss(y_true, y_pred):
    '''Implements Smooth-L1 loss.
    Args
    y_true: N, 4
    y_pred: N, 2, 4

    ---
        y_true and y_pred are typically: [N, 4], but could be any shape.
    return: loss [N, 4]
    '''

    diff = torch.abs(y_true - y_pred)
    less_than_one = torch.le(diff, 1)

    loss = (less_than_one * 0.5 * diff ** 2) + (1 - less_than_one) * (diff - 0.5)
    return loss

class LossFn:
    def __init__(self, cls_factor=1, box_factor=1, landmark_factor=1):
        # loss function
        self.cls_factor = cls_factor
        self.box_factor = box_factor
        self.land_factor = landmark_factor
        self.loss_cls = nn.BCELoss()
        self.loss_box = nn.MSELoss()
        self.loss_landmark = nn.MSELoss()

    def cls_loss(self, gt_label, pred_label):
        pred_label = torch.squeeze(pred_label) # batch, 2
        gt_label = torch.squeeze(gt_label) # batch, 2

        # get the mask element which >= 0, only 0 and 1 and 2 can effect the detection loss
        # mask = torch.ge(gt_label, 0)
        # valid_gt_label = torch.masked_select(gt_label,mask)
        # valid_pred_label = torch.masked_select(pred_label,mask)

        return self.loss_cls(pred_label, gt_label)*self.cls_factor


    def box_loss(self, gt_label, gt_offset, pred_offset):

        # pred_offset is n, 8 , h ,w  coresponding to 2 classes

        pred_offset = torch.squeeze(pred_offset)
        pred_offset = torch.reshape(pred_offset, (-1, 2, 4))
        gt_offset = torch.squeeze(gt_offset)

        # gt_label n, 2, if correspding to cls 1 then first 4 elements. mask for last 4 elements.
        gt_label = torch.squeeze(gt_label) # n, 2 => n, 2, 4

        # print(gt_label)
        unmask = torch.ge(torch.sum(gt_label, dim=1), 0)
        # mask = torch.eq(torch.sum(gt_label, dim=1), 0)

        # convert mask to dim index
        chose_index = torch.nonzero(unmask.data)
        chose_index = torch.squeeze(chose_index)
        
        # only valid element can effect the loss
        valid_gt_offset = gt_offset[chose_index, ...]
        valid_pred_offset = pred_offset[chose_index, ...]

        # now still need to clssify which pred_offset to reg with valid_gt valid_pred_offset[i]=>which valid_gt
        face_gt_idx = torch.eq(gt_label[chose_index, 0], 1) 
        face_gt_idx = torch.nonzero(face_gt_idx.data)
        face_gt_idx = torch.squeeze(face_gt_idx)

        mask_gt_idx = torch.eq(gt_label[chose_index, 1], 1)
        mask_gt_idx = torch.nonzero(mask_gt_idx.data)
        mask_gt_idx = torch.squeeze(mask_gt_idx)

        face_pred_offset = valid_pred_offset[face_gt_idx, 0, :]
        mask_pred_offset = valid_pred_offset[mask_gt_idx, 1, :]

        valid_gt_offset = torch.cat([valid_gt_offset[face_gt_idx], valid_gt_offset[mask_gt_idx]])
        valid_pred_offset = torch.cat([face_pred_offset, mask_pred_offset])

        return self.loss_box(valid_pred_offset, valid_gt_offset)*self.box_factor


    def landmark_loss(self,gt_label,gt_landmark,pred_landmark):
        pred_landmark = torch.squeeze(pred_landmark)
        gt_landmark = torch.squeeze(gt_landmark)
        gt_label = torch.squeeze(gt_label)
        mask = torch.eq(gt_label,-2)

        chose_index = torch.nonzero(mask.data)
        chose_index = torch.squeeze(chose_index)

        valid_gt_landmark = gt_landmark[chose_index, :]
        valid_pred_landmark = pred_landmark[chose_index, :]
        return self.loss_landmark(valid_pred_landmark,valid_gt_landmark)*self.land_factor

