# the script is used for generate mask and non mask labels and images.
# as there are lots of small faces in widerface, so i cleaned all small faces. only keep the faces greater than 80 to match mafa dataset.

def clean(anno_file):

    with open(anno_file, 'r') as f:
        annotations = f.readlines()


    for annotation in annotations:
        annotation = annotation.strip().split(' ')

        bbox = list(map(float, annotation[1:]))
        boxes = np.array(bbox, dtype=np.int32).reshape(-1, 4)
        img = cv2.imread(im_path)
        idx += 1
        if idx % 100 == 0:
            print(idx, "images done")

        height, width, channel = img.shape

        neg_num = 0


        for box in boxes:
            # box (x_left, y_top, x_right, y_bottom)
            x1, y1, x2, y2 = box
            w = x2 - x1 + 1
            h = y2 - y1 + 1

            # ignore small faces
            # in case the ground truth boxes of small faces are not accurate
            if max(w, h) < 40 or x1 < 0 or y1 < 0:
                continue


if __name__=='__main__':
    clean()