import os
import numpy as np

class ImageDB(object):
    def __init__(self, image_annotation_file, prefix_path='', mode='train'):
        self.prefix_path = prefix_path
        self.image_annotation_file = image_annotation_file
        self.classes = ['__background__', 'face', 'mask']
        self.num_classes = 3
        self.image_set_index = self.load_image_set_index()
        self.num_images = len(self.image_set_index)
        self.mode = mode


    def load_image_set_index(self):
        """Get image index

        Parameters:
        ----------
        Returns:
        -------
        image_set_index: str
            relative path of image
        """
        assert os.path.exists(self.image_annotation_file), 'Path does not exist: {}'.format(self.image_annotation_file)
       
        with open(self.image_annotation_file, 'r') as f:
            image_set_index = [x.strip().split(' ')[0] for x in f.readlines()]
        
        return image_set_index


    def load_imdb(self):
        """Get and save ground truth image database

        Parameters:
        ----------
        Returns:
        -------
        gt_imdb: dict
            image database with annotations
        """
        #cache_file = os.path.join(self.cache_path, self.name + '_gt_roidb.pkl')
        #if os.path.exists(cache_file):
        #    with open(cache_file, 'rb') as f:
        #        imdb = cPickle.load(f)
        #    print '{} gt imdb loaded from {}'.format(self.name, cache_file)
        #    return imdb
        gt_imdb = self.load_annotations()
        #with open(cache_file, 'wb') as f:
        #    cPickle.dump(gt_imdb, f, cPickle.HIGHEST_PROTOCOL)
        return gt_imdb


    def real_image_path(self, index):
        """Given image index, return full path

        Parameters:
        ----------
        index: str
            relative path of image
        Returns:
        -------
        image_file: str
            full path of image
        """

        index = index.replace("\\", "/")

        if not os.path.exists(index):
            image_file = os.path.join(self.prefix_path, index)
        else:
            image_file=index
        if not image_file.endswith('.jpg'):
            image_file = image_file + '.jpg'

            
        assert os.path.exists(image_file), 'Path does not exist: {}'.format(image_file)
        return image_file


    def load_annotations(self):
        """Load annotations

        Parameters:
        ----------
        Returns:
        -------
        imdb: dict
            image database with annotations
        """

        assert os.path.exists(self.image_annotation_file), 'annotations not found at {}'.format(self.image_annotation_file)
        with open(self.image_annotation_file, 'r') as f:
            annotations = f.readlines()

        imdb = []

        for i in range(self.num_images):
            annotation = annotations[i].strip().split(' ')
            im_path = annotation[0]
            # im_path = self.real_image_path(index)

            imdb_ = dict()
            imdb_['image'] = im_path

            if self.mode == 'test':
               # gt_boxes = map(float, annotation[1:])
               # boxes = np.array(bbox, dtype=np.float32).reshape(-1, 4)
               # imdb_['gt_boxes'] = boxes
                pass
            else:
                label = annotation[1]
                imdb_['label'] = int(label)
                imdb_['flipped'] = False
                imdb_['bbox_target'] = np.zeros((4,))
                imdb_['landmark_target'] = np.zeros((10,))
                if len(annotation[2:])==4:
                    bbox_target = annotation[2:6]
                    imdb_['bbox_target'] = np.array(bbox_target).astype(float)
                if len(annotation[2:])==14:
                    bbox_target = annotation[2:6]
                    imdb_['bbox_target'] = np.array(bbox_target).astype(float)
                    landmark = annotation[6:]
                    imdb_['landmark_target'] = np.array(landmark).astype(float)

            imdb.append(imdb_)
        return imdb

    def append_flipped_images(self, imdb):
        """append flipped images to imdb

        Parameters:
        ----------
        imdb: imdb
            image database
        Returns:
        -------
        imdb: dict
            image database with flipped image annotations added
        """
        print('append flipped images to imdb', len(imdb))
        for i in range(len(imdb)):
            imdb_ = imdb[i]
            m_bbox = imdb_['bbox_target'].copy()

            m_bbox[0], m_bbox[2] = -m_bbox[2], -m_bbox[0]

            entry = {'image': imdb_['image'],
                     'label': imdb_['label'],
                     'bbox_target': m_bbox,
                     'landmark_target': np.zeros((10,)),
                     'flipped': True}

            imdb.append(entry)

        self.image_set_index *= 2
        return imdb
