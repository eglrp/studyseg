import os
import json
import torch
import lib.utils.data as torchdata
import cv2
from torchvision import transforms
from scipy.misc import imread, imresize
import numpy as np
from PIL import Image

# Round x to the nearest multiple of p and x' >= x
def round2nearest_multiple(x, p):
    return ((x - 1) // p + 1) * p

class TrainDataset(torchdata.Dataset):
    def __init__(self , opt, max_sample=-1, batch_per_gpu=1):
        self.root_dataset = opt.root_dataset
        self.imgSize = opt.imgSize
        self.imgMaxSize = opt.imgMaxSize
        self.random_flip = opt.random_flip
        # max down sampling rate of network to avoid rounding during conv or pooling
        self.padding_constant = opt.padding_constant
        # down sampling rate of segm labe
        self.segm_downsampling_rate = opt.segm_downsampling_rate
        self.batch_per_gpu = batch_per_gpu

        # classify images into two classes: 1. h > w and 2. h <= w
        self.batch_record_list = [[], []]

        # override dataset length when trainig with batch_per_gpu > 1
        self.cur_idx = 0

        # mean and std
        self.img_transform = transforms.Compose([
            transforms.Normalize(mean=[102.9801, 115.9465, 122.7717], std=[1., 1., 1.])
            ])

        self.list_sample = [x for x in open(opt.root_dataset + 'VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt', 'r').read().splitlines()]

        self.if_shuffled = False
        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]
        #self.list_sample2 = [x.read().splitlines() for x in open(opt.root_dataset + 'benchmark_RELEASE/dataset/train.txt', 'r')]
        self.num_sample = len(self.list_sample) #+len(self.list_sample2)
        assert self.num_sample > 0
        print('# samples: {}'.format(self.num_sample))

    def _get_sub_batch(self):
        while True:
            # get a sample record
            this_sample = self.list_sample[self.cur_idx]
            if this_sample['height'] > this_sample['width']:
                self.batch_record_list[0].append(this_sample) # h > w, go to 1st class
            else:
                self.batch_record_list[1].append(this_sample) # h <= w, go to 2nd class

            # update current sample pointer
            self.cur_idx += 1
            if self.cur_idx >= self.num_sample:
                self.cur_idx = 0
                np.random.shuffle(self.list_sample)

            if len(self.batch_record_list[0]) == self.batch_per_gpu:
                batch_records = self.batch_record_list[0]
                self.batch_record_list[0] = []
                break
            elif len(self.batch_record_list[1]) == self.batch_per_gpu:
                batch_records = self.batch_record_list[1]
                self.batch_record_list[1] = []
                break
        return batch_records

    def __getitem__(self, index):
        # NOTE: random shuffle for the first time. shuffle in __init__ is useless
        #if not self.if_shuffled:
        #    np.random.shuffle(self.list_sample)
        #    self.if_shuffled = True

        # get sub-batch candidates
        #batch_records = self._get_sub_batch()

        # resize all images' short edges to the chosen size
        if isinstance(self.imgSize, list):
            this_short_size = np.random.choice(self.imgSize)
        else:
            this_short_size = self.imgSize

        # calculate the BATCH's height and width
        # since we concat more than one samples, the batch's h and w shall be larger than EACH sample
        '''
        batch_resized_size = np.zeros((self.batch_per_gpu, 2), np.int32)
        #print(batch_records)
        for i in range(self.batch_per_gpu):
            img_height, img_width = batch_records[i]['height'], batch_records[i]['width']
            this_scale = min(float(this_short_size) / min(img_height, img_width),# \
                    float(self.imgMaxSize) / max(img_height, img_width))
            img_resized_height, img_resized_width = img_height * this_scale, img_width * this_scale
            batch_resized_size[i, :] = img_resized_height, img_resized_width
        #print(batch_resized_size) 
        batch_resized_height = np.max(batch_resized_size[:, 0])
        batch_resized_width = np.max(batch_resized_size[:, 1])

        # Here we must pad both input image and segmentation map to size h' and w' so that p | h' and p | w'
        batch_resized_height = int(round2nearest_multiple(batch_resized_height, self.padding_constant))
        batch_resized_width = int(round2nearest_multiple(batch_resized_width, self.padding_constant))
        '''
        assert self.padding_constant >= self.segm_downsampling_rate,\
                'padding constant must be equal or large than segm downsamping rate'
        batch_images = torch.zeros(self.batch_per_gpu, 3, this_short_size, this_short_size)
        batch_segms = torch.zeros(self.batch_per_gpu, this_short_size // self.segm_downsampling_rate, \
                                this_short_size // self.segm_downsampling_rate).long()
        
        for i in range(self.batch_per_gpu):
            #this_record = batch_records[i]

            # load image and label
            image_path = os.path.join(self.root_dataset, 'VOCdevkit/VOC2012/JPEGImages' ,self.list_sample[self.cur_idx]+'.jpg' )#this_record['fpath_img'])
            segm_path = os.path.join(self.root_dataset, 'VOCdevkit/VOC2012/SegmentationClass' , self.list_sample[self.cur_idx]+'.png')#this_record['fpath_segm'])
            self.cur_idx +=1
            if self.cur_idx >= self.num_sample:
                self.cur_idx =0
                break
            img = imread(image_path, mode='RGB')
            segm = np.array(Image.open(segm_path),dtype = np.int32)

            assert(img.ndim == 3)
            assert(segm.ndim == 2)
            assert(img.shape[0] == segm.shape[0])
            assert(img.shape[1] == segm.shape[1])

            if self.random_flip == True:
                random_flip = np.random.choice([0, 1])
                if random_flip == 1:
                    img = cv2.flip(img, 1)
                    segm = cv2.flip(segm, 1)

            # note that each sample within a mini batch has different scale param
            img = imresize(img, (this_short_size, this_short_size), interp='bilinear')
            segm = imresize(segm, (this_short_size, this_short_size), interp='nearest')
            #print(img)
            #print(np.unique(segm))
            # to avoid seg label misalignment
            #segm_rounded_height = round2nearest_multiple(segm.shape[0], self.segm_downsampling_rate)
            #segm_rounded_width = round2nearest_multiple(segm.shape[1], self.segm_downsampling_rate)
            #segm_rounded = np.zeros((segm_rounded_height, segm_rounded_width), dtype='uint8')
            segm_rounded = segm
            #print(segm.shape)
            segm = imresize(segm_rounded, (segm_rounded.shape[0] // self.segm_downsampling_rate, \
                                           segm_rounded.shape[1] // self.segm_downsampling_rate), \
                            interp='nearest')
            #print(segm.shape)
             # image to float
            img = img.astype(np.float32)[:, :, ::-1] # RGB to BGR!!!
            img = img.transpose((2, 0, 1))
            img = self.img_transform(torch.from_numpy(img.copy()))

            batch_images[i] = img
            batch_segms[i] = torch.from_numpy(segm.astype(np.int)).long()
        #from IPython import embed;embed();exit();
        batch_segms[batch_segms==255] =-1
        batch_segms = batch_segms #- 1 # label from -1 to 149
        output = dict()
        output['img_data'] = batch_images#.cuda()
        output['seg_label'] = batch_segms#.cuda()
        #print(output)
        return output

    def __len__(self):
        return int(1e6) # It's a fake length due to the trick that every loader maintains its own list
        #return self.num_sampleclass


class ValDataset(torchdata.Dataset):
    def __init__(self, opt, max_sample=-1, start_idx=-1, end_idx=-1):
        self.root_dataset = opt.root_dataset
        self.imgSize = opt.imgSize
        self.imgMaxSize = opt.imgMaxSize
        # max down sampling rate of network to avoid rounding during conv or pooling
        self.padding_constant = opt.padding_constant

        # mean and std
        self.img_transform = transforms.Compose([
            transforms.Normalize(mean=[102.9801, 115.9465, 122.7717], std=[1., 1., 1.])
            ])

        self.list_sample = [x for x in open(opt.root_dataset + 'VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt', 'r').read().splitlines()] 

        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]

        if start_idx >= 0 and end_idx >= 0: # divide file list
            self.list_sample = self.list_sample[start_idx:end_idx]

        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0
        print('# samples: {}'.format(self.num_sample))

    def __getitem__(self, index):
        this_record = self.list_sample[index]
        # load image and label
        image_path = os.path.join(self.root_dataset, 'VOCdevkit/VOC2012/JPEGImages' ,self.list_sample[index]+'.jpg' )#this_record['fpath_img'])
        segm_path = os.path.join(self.root_dataset, 'VOCdevkit/VOC2012/SegmentationClass' , self.list_sample[index]+'.png')#img = imread(image_path, mode='RGB')
        img = imread(image_path, mode='RGB')
        segm = np.array(Image.open(segm_path),dtype = np.int32)
        img = img[:, :, ::-1] # BGR to RGB!!!
        segm = imread(segm_path)

        ori_height, ori_width, _ = img.shape

        img_resized_list = []
        for this_short_size in self.imgSize:
            # calculate target height and width
            '''
            scale = min(this_short_size / float(min(ori_height, ori_width)),
                    self.imgMaxSize / float(max(ori_height, ori_width)))
            target_height, target_width = int(ori_height * scale), int(ori_width * scale)
            '''
            target_height,target_width = this_short_size ,this_short_size
            # to avoid rounding in network
            target_height = round2nearest_multiple(target_height, self.padding_constant)
            target_width = round2nearest_multiple(target_width, self.padding_constant)

            # resize
            img_resized = cv2.resize(img.copy(), (target_width, target_height))

            # image to float
            img_resized = img_resized.astype(np.float32)
            img_resized = img_resized.transpose((2, 0, 1))
            img_resized = self.img_transform(torch.from_numpy(img_resized))

            img_resized = torch.unsqueeze(img_resized, 0)
            img_resized_list.append(img_resized)

        segm = torch.from_numpy(segm.astype(np.int)).long()

        batch_segms = torch.unsqueeze(segm, 0)

        batch_segms = batch_segms - 1 # label from -1 to 149
        output = dict()
        output['img_ori'] = img.copy()
        output['img_data'] = [x.contiguous() for x in img_resized_list]
        output['seg_label'] = batch_segms.contiguous()
        #output['info'] = this_record['fpath_img']
        return output

    def __len__(self):
        return self.num_sample


class TestDataset(torchdata.Dataset):
    def __init__(self, odgt, opt, max_sample=-1):
        self.imgSize = opt.imgSize
        self.imgMaxSize = opt.imgMaxSize
        # max down sampling rate of network to avoid rounding during conv or pooling
        self.padding_constant = opt.padding_constant
        # down sampling rate of segm labe
        self.segm_downsampling_rate = opt.segm_downsampling_rate

        # mean and std
        self.img_transform = transforms.Compose([
            transforms.Normalize(mean=[102.9801, 115.9465, 122.7717], std=[1., 1., 1.])
            ])

        if isinstance(odgt, list):
            self.list_sample = odgt
        elif isinstance(odgt, str):
            self.list_sample = [json.loads(x.rstrip()) for x in open(odgt, 'r')]

        if max_sample > 0:
            self.list_sample = self.list_sample[0:max_sample]
        self.num_sample = len(self.list_sample)
        assert self.num_sample > 0
        print('# samples: {}'.format(self.num_sample))

    def __getitem__(self, index):
        this_record = self.list_sample[index]
        # load image and label
        image_path = this_record['fpath_img']
        img = imread(image_path, mode='RGB')
        img = img[:, :, ::-1] # BGR to RGB!!!

        ori_height, ori_width, _ = img.shape

        img_resized_list = []
        for this_short_size in self.imgSize:
            # calculate target height and width
            scale = min(this_short_size / float(min(ori_height, ori_width)),
                    self.imgMaxSize / float(max(ori_height, ori_width)))
            target_height, target_width = int(ori_height * scale), int(ori_width * scale)

            # to avoid rounding in network
            target_height = round2nearest_multiple(target_height, self.padding_constant)
            target_width = round2nearest_multiple(target_width, self.padding_constant)

            # resize
            img_resized = cv2.resize(img.copy(), (target_width, target_height))

            # image to float
            img_resized = img_resized.astype(np.float32)
            img_resized = img_resized.transpose((2, 0, 1))
            img_resized = self.img_transform(torch.from_numpy(img_resized))

            img_resized = torch.unsqueeze(img_resized, 0)
            img_resized_list.append(img_resized)

        # segm = torch.from_numpy(segm.astype(np.int)).long()

        # batch_segms = torch.unsqueeze(segm, 0)

        # batch_segms = batch_segms - 1 # label from -1 to 149
        output = dict()
        output['img_ori'] = img.copy()
        output['img_data'] = [x.contiguous() for x in img_resized_list]
        # output['seg_label'] = batch_segms.contiguous()
        output['info'] = this_record['fpath_img']
        return output

    def __len__(self):
        return self.num_sample
