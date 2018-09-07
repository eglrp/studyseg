import cv2
import numpy as np
import os
from IPython import embed
def mkdir(path):
 
	folder = os.path.exists(path)
 
	if not folder:                   
		os.makedirs(path)            
		print "---  new folder...  ---"
		print "---  OK  ---"
 
	else:
		print "---  There is this folder!  ---"

dataset_root1 = '/home/cuishuhao/data/datasets/VOC/VOCdevkit/VOC2012'

filelist1 = os.path.join(dataset_root1 , 'ImageSets/Segmentation/trainval.txt')

path_new = os.path.join(dataset_root1 , 'SegmentationEdge')

mkdir (path_new)

with open(filelist1,'r') as f:
    lines = f.read().splitlines()

def get_score( img, x ,y):
    num =[]
    present = img[x,y]
    u,v =img.shape
    i =0
    #print(u,v)
    for i in range(x):
        if present != img[x-i-1,y]:
            break
    num.append(i*256/u)
    print(i)
    for i in range(img.shape[0]-x):
         if present != img[x+i ,y]:
            break
    num.append(i*256/u)
    print(i)
    for i in range(y):
         if present != img[x,y-i-1]:
            break
    num.append(i*256/v)
    print(i)
    for i in range(img.shape[1]-y):
         #embed()
         if present != img[x ,y+i]:
            break
    num.append(i*256/v)
    print(i)
    num = np.array(num)
    #num = num.astype(np.int32)
    return num


for line in lines:
    origin = os.path.join(dataset_root1 ,'SegmentationClass', line + '.png')
    assert os.path.isfile(origin)
    img = cv2.imread(origin, cv2.IMREAD_GRAYSCALE)
    print img.shape
    new_img = np.zeros(( img.shape[0], img.shape[1], 4))
    for i in range(img.shape[0]):
        for j in range(img.shape[1]):
            new_img[i,j] = get_score(img,i,j)
    new_img =new_img.astype(np.int8)
    np.save(os.path.join(path_new,line+'.npy'),new_img) 
    #break
