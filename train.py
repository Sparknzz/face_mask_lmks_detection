import os
import torch
from model.pnet import PNet
from dataloader.image_reader import TrainImageReader
import image_tools
from torch.autograd.variable import Variable

def train_pnet(model_store_path, end_epoch,imdb, batch_size, frequent=50, base_lr=0.01, use_cuda=True):

    if not os.path.exists(model_store_path):
        os.makedirs(model_store_path)

    net = PNet(is_train=True, use_cuda=use_cuda)
    net.train()

    if use_cuda:
        net.cuda()

    optimizer = torch.optim.Adam(net.parameters(), lr=base_lr)

    train_data = TrainImageReader(imdb, 12, batch_size, shuffle=True)

    for cur_epoch in range(1,end_epoch+1):
        train_data.reset()
        accuracy_list=[]
        cls_loss_list=[]
        bbox_loss_list=[]
        # landmark_loss_list=[]

        for batch_idx, (image, (gt_label, gt_bbox, gt_landmark)) in enumerate(train_data):

            im_tensor = [image_tools.convert_image_to_tensor(image[i,:,:,:]) for i in range(image.shape[0]) ]
            im_tensor = torch.stack(im_tensor)

            im_tensor = Variable(im_tensor)
            gt_label = Variable(torch.from_numpy(gt_label).float())

            gt_bbox = Variable(torch.from_numpy(gt_bbox).float())
            # gt_landmark = Variable(torch.from_numpy(gt_landmark).float())

            if use_cuda:
                im_tensor = im_tensor.cuda()
                gt_label = gt_label.cuda()
                gt_bbox = gt_bbox.cuda()
                # gt_landmark = gt_landmark.cuda()

            cls_pred, box_offset_pred = net(im_tensor)
            # all_loss, cls_loss, offset_loss = lossfn.loss(gt_label=label_y,gt_offset=bbox_y, pred_label=cls_pred, pred_offset=box_offset_pred)

            cls_loss = lossfn.cls_loss(gt_label,cls_pred)
            box_offset_loss = lossfn.box_loss(gt_label,gt_bbox,box_offset_pred)
            # landmark_loss = lossfn.landmark_loss(gt_label,gt_landmark,landmark_offset_pred)

            all_loss = cls_loss * 1.0 + box_offset_loss * 0.5

            if batch_idx%frequent==0:
                accuracy = compute_accuracy(cls_pred,gt_label)

                show1 = accuracy.data.tolist()[0]
                show2 = cls_loss.data.tolist()[0]
                show3 = box_offset_loss.data.tolist()[0]
                show5 = all_loss.data.tolist()[0]

                print("%s : Epoch: %d, Step: %d, accuracy: %s, det loss: %s, bbox loss: %s, all_loss: %s, lr:%s "%(datetime.datetime.now(),cur_epoch,batch_idx, show1,show2,show3,show5,base_lr))
                accuracy_list.append(accuracy)
                cls_loss_list.append(cls_loss)
                bbox_loss_list.append(box_offset_loss)

            optimizer.zero_grad()
            all_loss.backward()
            optimizer.step()


        accuracy_avg = torch.mean(torch.cat(accuracy_list))
        cls_loss_avg = torch.mean(torch.cat(cls_loss_list))
        bbox_loss_avg = torch.mean(torch.cat(bbox_loss_list))
        # landmark_loss_avg = torch.mean(torch.cat(landmark_loss_list))

        show6 = accuracy_avg.data.tolist()[0]
        show7 = cls_loss_avg.data.tolist()[0]
        show8 = bbox_loss_avg.data.tolist()[0]

        print("Epoch: %d, accuracy: %s, cls loss: %s, bbox loss: %s" % (cur_epoch, show6, show7, show8))
        torch.save(net.state_dict(), os.path.join(model_store_path,"pnet_epoch_%d.pt" % cur_epoch))
        torch.save(net, os.path.join(model_store_path,"pnet_epoch_model_%d.pkl" % cur_epoch))

if __name__ == "__main__":
    train_pnet()    