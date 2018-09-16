import socket
import timeit
from datetime import datetime
import os
import glob
from collections import OrderedDict
import numpy as np

# PyTorch includes
import torch
from torch.autograd import Variable
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import make_grid


# Tensorboard include
from tensorboardX import SummaryWriter

# Custom includes
from dataloaders import pascal, sbd, combine_dbs
from dataloaders import utils
from networks import deeplab_xception
# from dataloaders import custom_transforms as tr
from dataloaders import custom_transforms as tr

import cv2

gpu_id = 0
print('Using GPU: {} '.format(gpu_id))
# Setting parameters

testBatch = 2  # Testing batch size

model_path = 'run/run_10/models/deeplabv3+_epoch-49.pth'
print("Initializing weights from: {}...".format(model_path))
save_dir = 'doc'

net = deeplab_xception.DeepLabv3_plus(nInputChannels=3, n_classes=21, pretrained=True)
modelName = 'deeplabv3+'
criterion = utils.cross_entropy2d
net.load_state_dict(
    torch.load(model_path,
               map_location=lambda storage, loc: storage))

if gpu_id >= 0:
    torch.cuda.set_device(device=gpu_id)
    net.cuda()

composed_transforms_ts = transforms.Compose([
    tr.FixedResize(size=(512, 512)),
    tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    tr.ToTensor()])

voc_val = pascal.VOCSegmentation(split='val', transform=composed_transforms_ts)

global_mean = (0.485, 0.456, 0.406)
global_std=(0.229, 0.224, 0.225)

testloader = DataLoader(voc_val, batch_size=testBatch, shuffle=False, num_workers=0)


num_img_ts = len(testloader)
testing_loss = 0.0
testing_acc = 0.0
testing_miou = 0.0
aveGrad = 0
print("Testing Network")

'''
            if ii % (num_img_tr / 20) == 0:
                grid_image = make_grid(inputs[:3].clone().cpu().data, 3, normalize=True)
                writer.add_image('Image', grid_image, global_step)
                grid_image = make_grid(utils.decode_seg_map_sequence(torch.max(outputs[:3], 1)[1].detach().cpu().numpy()), 3, normalize=False,
                                       range=(0, 255))
                writer.add_image('Predicted label', grid_image, global_step)
                grid_image = make_grid(utils.decode_seg_map_sequence(torch.squeeze(labels[:3], 1).detach().cpu().numpy()), 3, normalize=False, range=(0, 255))
                writer.add_image('Groundtruth label', grid_image, global_step)
'''
net.eval()
images_num = 0
for ii, sample_batched in enumerate(testloader):
	inputs, labels = sample_batched['image'], sample_batched['label']

	# Forward pass of the mini-batch
	inputs, labels = Variable(inputs, requires_grad=True), Variable(labels)
	if gpu_id >= 0:
	    inputs, labels = inputs.cuda(), labels.cuda()

	with torch.no_grad():
	    outputs = net.forward(inputs)

	predictions = torch.max(outputs, 1)[1]

	loss = criterion(outputs, labels, size_average=False, batch_average=True)
	testing_loss += loss.item()
        #from IPython import embed;embed();exit();
        for num in range( testBatch):
            img = utils.decode_seg_map_sequence(predictions.cpu().numpy())[num]
            print('save images num {} '.format(images_num))
            cv2.imwrite(os.path.join(save_dir, 'img_{:04d}_pre.jpg'.format(images_num)),img.numpy().transpose(1,2,0)*256)
            img_ori = inputs[num].detach().cpu().numpy().transpose(1,2,0)*global_std+global_mean
            cv2.imwrite(os.path.join(save_dir, 'img_{:04d}_ori.jpg'.format(images_num)),img_ori*256)
            img_gt = utils.decode_seg_map_sequence(labels.detach().cpu().numpy().reshape(testBatch,512,512))[num]#*global_std+global_mean
            cv2.imwrite(os.path.join(save_dir, 'img_{:04d}_gt.jpg'.format(images_num)),img_gt.numpy().transpose(1,2,0)*256)
            images_num+=1
	total_acc,total_miou = utils.get_iou(predictions.cpu().numpy(), labels.cpu().numpy())
        
	testing_acc += total_acc
	testing_miou += total_miou
	# Print stuff
        if images_num >= 100:
            break
	if ii % num_img_ts == num_img_ts - 1:

	    testing_miou = testing_miou / num_img_ts#(ii * testBatch + inputs.data.shape[0])
            testing_loss = testing_loss / num_img_ts
	    testing_acc  = testing_acc  / num_img_ts

	    writer.add_scalar('data/testing_loss', testing_loss, epoch)
	    writer.add_scalar('data/testing_miou', testing_miou, epoch)
	    writer.add_scalar('data/testing_acc ', testing_acc  ,epoch)
	    print('Validation:')
	    print('[ numImages: %5d] ' % ( ii * testBatch + inputs.data.shape[0]) + ' Loss: %f ' % testing_loss +' Acc: %f' %testing_acc+' MIoU: %f\n ' % testing_miou)
	    testing_loss = 0
	    testing_acc = 0
	    testing_miou = 0


