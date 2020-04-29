import os
import torch
from model.pnet import PNet
from dataloader.image_reader import TrainImageReader
import image_tools
from torch.autograd.variable import Variable
from dataloader.imdb import ImageDB
from model.loss import LossFn
import datetime
import numpy as np


def metric_hit(logit, truth, threshold=0.5):
    batch_size, num_class, H, W = logit.shape

    with torch.no_grad():
        logit = logit.view(batch_size, num_class, -1)
        truth = truth.view(batch_size, num_class, -1) # batch, 2, 1

        probability = torch.sigmoid(logit)
        p = (probability>threshold).float()
        t = (truth>0.5).float()

        tp = ((p + t) == 2).float()  # True positives
        tn = ((p + t) == 0).float()  # True negatives

        tp = tp.sum(dim=[0,2])
        tn = tn.sum(dim=[0,2]) # equal to tn.sum(dim=0).squeeze() we get the face tn and mask tn seperately, eg 197 0 means 0 mask tn detect
        num_pos = t.sum(dim=[0,2])
        num_neg = batch_size*H*W - num_pos

        tp = tp.data.cpu().numpy() # face tp detect,  mask tp detect
        tn = tn.data.cpu().numpy().sum() 
        num_pos = num_pos.data.cpu().numpy()
        num_neg = num_neg.data.cpu().numpy().sum() # why sum()?? no need to sum, 

        tp = np.nan_to_num(tp/(num_pos+1e-12),0)
        tn = np.nan_to_num(tn/(num_neg+1e-12),0)

        tp = list(tp)
        num_pos = list(num_pos)

    return tn, tp, num_neg, num_pos


def train_pnet(annotation_file, model_store_path, end_epoch=16, batch_size=512, frequent=50, base_lr=0.01, use_cuda=True):

    imagedb = ImageDB(annotation_file)
    gt_imdb = imagedb.load_imdb() # imdb is a list of dict
    gt_imdb = imagedb.append_flipped_images(gt_imdb)

    if not os.path.exists(model_store_path):
        os.makedirs(model_store_path)

    net = PNet()
    net.train()

    if use_cuda:
        net.cuda()

    optimizer = torch.optim.Adam(net.parameters(), lr=base_lr)

    train_data = TrainImageReader(gt_imdb, 12, batch_size, shuffle=True)

    lossfn = LossFn()
    
    for cur_epoch in range(1, end_epoch+1):
        train_data.reset()
        face_accuracy_list=[]
        mask_accuracy_list=[]
        cls_loss_list=[]
        bbox_loss_list=[]

        for batch_idx, (image, (gt_label, gt_bbox, gt_landmark)) in enumerate(train_data):

            im_tensor = [image_tools.convert_image_to_tensor(image[i, ...]) for i in range(image.shape[0]) ]
            im_tensor = torch.stack(im_tensor)
            im_tensor = Variable(im_tensor.float())
  
            gt_label = Variable(torch.from_numpy(gt_label).float())
            gt_bbox = Variable(torch.from_numpy(gt_bbox).float())

            if use_cuda:
                im_tensor = im_tensor.cuda()
                gt_label = gt_label.cuda()
                gt_bbox = gt_bbox.cuda()

            cls_pred, box_offset_pred = net(im_tensor)
            cls_loss = lossfn.cls_loss(gt_label, cls_pred)
            box_offset_loss = lossfn.box_loss(gt_label, gt_bbox, box_offset_pred)

            all_loss = cls_loss * 1.0 + box_offset_loss * 0.5

            if batch_idx%frequent==0:
                tn, tp, num_neg, num_pos = metric_hit(cls_pred, gt_label)
                c_loss = cls_loss.data
                b_loss = box_offset_loss.data
                a_loss = all_loss.data

                print("%s : Epoch: %d, Step: %d, face_hit: %s, mask_hit: %s det loss: %s, bbox loss: %s, all_loss: %s, lr:%s " \
                    %(datetime.datetime.now(),cur_epoch,batch_idx, tp[0], tp[1], c_loss, b_loss, a_loss, base_lr))
                face_accuracy_list.append(tp[0])
                mask_accuracy_list.append(tp[1])
                cls_loss_list.append(cls_loss)
                bbox_loss_list.append(box_offset_loss)

            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()


        face_avg = np.mean(face_accuracy_list)
        mask_avg = np.mean(mask_accuracy_list)
        cls_loss_avg = torch.mean(torch.cat(cls_loss_list))
        bbox_loss_avg = torch.mean(torch.cat(bbox_loss_list))

        show7 = cls_loss_avg.data.tolist()[0]
        show8 = bbox_loss_avg.data.tolist()[0]

        print("Epoch: %d, face_avg_hit: %s, mask_avg_hit: %s, cls loss: %s, bbox loss: %s" % (cur_epoch, face_avg, mask_avg, show7, show8))
        torch.save(net.state_dict(), os.path.join(model_store_path,"pnet_epoch_%d.pt" % cur_epoch))
        torch.save(net, os.path.join(model_store_path, "pnet_epoch_model_%d.pkl" % cur_epoch))

if __name__ == "__main__":
    train_pnet(annotation_file='/root/face_mask_lmks_detection/annos/pnet_train_imglist.txt', model_store_path='/root/face_mask_lmks_detection/weights')    