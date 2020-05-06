# eda for the wider face dataset
# 'wider_face_train_bbx_gt' is orignial label file.
x1, y1, w, h, blur(0,1,2), express, illumination, occlusion(0,1,2), invalid, pose
# total : 12880 images, 159424 bboxes. as there are some labels need to be ignored. so use the clean wider_face_train.txt

# wider_face_train.txt 12880 images but bboxes is a bit less

# label.txt is the landmarks for wider face dataset. thanks for the insightface team to publish useful infos.
reference https://github.insightface.com//
# 172304 lines. total 159424 bboxes. same as orignal label. but all added landmarks.

# when training pnet don't need to care too much about landmarks, so generate 12*12 size faces first.  the script has been provided in prepare_data folder.
# gen_pnet_data.py (including mafa dataset assemble)

# mafa is the mask detection dataset. published by xxx. 
# total 25876 images. total 29452 labeled bboxes only 14110 mask labels. compare to wider face, it is obviously less faces each images. 
# without seeing bbox, we can guess the mafa dataset faces is bigger.

# put those two dataset all together. running merge dataset



# then got 188876 (29452 + 159424) labeled bbox. but some of them have to be cleaned. 
# 173534 (159424 + 14110)

# in order to boost training speed and the purpose of our case is to detect bigger face, so for those faces smaller than 40. 
# ignored them cos hard to label the correct landmarks. 
# so i only keep 39915 bboxes (22644 images, 12487 images for mafa dataset) for training purpose. see eda_all.ipynb 13547 mask faces and 26368 normal faces.
# when we training them, may be use balanced sampler.

# after runing script gen_pnet_data.py, 276800 (13660 mask positive generated, 140361 face positive generated). 
# the negative face we only use widerface because mafa dataset including lots of the non mask faces.


# 14w 13w 63w 1:1:6

# 95, 160,  91,  91 face bbox x, y, w, h
# (113, 177), (158, 172) two eys
# 7,  26,  82,  89, face occluder bbox xywh
# 1, occ_type 1(simple) 2(complex) 3(human body)
# 3, icc_degree
# 1,  1, gender and race   
# 3, orientation 1-left 2-left frontal 3-front 4=right frontal 5=right  
# -1,  -1,  -1,  -1 xywh bbox of glasses