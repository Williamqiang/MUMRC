import random
import os
import torch
import json
import ast
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer,CLIPProcessor
from torchvision import transforms
import logging

DATA_PATH = {
    'train_auximgs': '../data/txt/mre_train_dict.pth',     # {data_id : object_crop_img_path}
    'dev_auximgs': '../data/txt/mre_dev_dict.pth',
    'test_auximgs': '../data/txt/mre_test_dict.pth',
    'train_img2crop': '../data/img_detect/train/train_img2crop.pth',
    'dev_img2crop': '../data/img_detect/val/val_img2crop.pth',
    'test_img2crop': '../data/img_detect/test/test_img2crop.pth'
}

IMG_PATH = {
    'train': '../data/img_org/train/',
    'dev': '../data/img_org/val/',
    'test': '../data/img_org/test'
}

AUX_PATH = {
    'train': '../data/img_vg/train/crops',
    'dev': '../data/img_vg/val/crops',
    'test': '../data/img_vg/test/crops'
}

rcnn_img_path = '../data'

pixel_values=64
aux_size=128
rcnn_size=64

# print("loading clip_processor.")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
clip_processor.feature_extractor.size, clip_processor.feature_extractor.crop_size = pixel_values, pixel_values

# print("loading aux_processor.")
aux_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
aux_processor.feature_extractor.size, aux_processor.feature_extractor.crop_size = aux_size, aux_size

# print("loading rcnn_processor.")
rcnn_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
rcnn_processor.feature_extractor.size, rcnn_processor.feature_extractor.crop_size = rcnn_size, rcnn_size

Aux_dict={'train':torch.load(DATA_PATH["train_auximgs"]),'test':torch.load(DATA_PATH["test_auximgs"]),'dev':torch.load(DATA_PATH["dev_auximgs"])}
Rcnn_dict={'train':torch.load(DATA_PATH['train_img2crop']),'test':torch.load(DATA_PATH['test_img2crop']),'dev':torch.load(DATA_PATH['dev_img2crop'])}


def loading(imgid,aux_id,mode):
    Aux = Aux_dict[mode]
    Rcnn = Rcnn_dict[mode]


    IMG_path=IMG_PATH[mode]
    AUX_img_path=AUX_PATH[mode]

    try:
        img_path = os.path.join(IMG_path, imgid)
        image = Image.open(img_path).convert('RGB')
        image = clip_processor(images=image, return_tensors='pt')['pixel_values'].squeeze()
    except:
        img_path = os.path.join(IMG_path, 'inf.png')
        image = Image.open(img_path).convert('RGB')
        image = clip_processor(images=image, return_tensors='pt')['pixel_values'].squeeze()


    aux_imgs = []
    aux_img_paths = []
    imgid = imgid.split(".")[0]
    if aux_id in Aux:
        aux_img_paths  = Aux[aux_id]
        aux_img_paths = [os.path.join(AUX_img_path, path) for path in aux_img_paths]
        
    # select 3 img
    for i in range(min(3, len(aux_img_paths))):
        aux_img = Image.open(aux_img_paths[i]).convert('RGB')
        aux_img = aux_processor(images=aux_img, return_tensors='pt')['pixel_values'].squeeze()
        aux_imgs.append(aux_img)

    # padding
    for i in range(3-len(aux_imgs)):
        aux_imgs.append(torch.zeros((3, aux_size, aux_size))) 

    aux_imgs = torch.stack(aux_imgs, dim=0)  #[3,3,128,128]
    assert len(aux_imgs) == 3

    # if rcnn_img_path is not None:
    rcnn_imgs = []
    rcnn_img_paths = []
    if imgid in Rcnn:
        rcnn_img_paths = Rcnn[imgid]
        rcnn_img_paths = [os.path.join(rcnn_img_path, path) for path in rcnn_img_paths]
    
    # select 3 img
    for i in range(min(3, len(rcnn_img_paths))):
        rcnn_img = Image.open(rcnn_img_paths[i]).convert('RGB')
        rcnn_img = rcnn_processor(images=rcnn_img, return_tensors='pt')['pixel_values'].squeeze()
        rcnn_imgs.append(rcnn_img)
    
    # padding
    for i in range(3-len(rcnn_imgs)):
        rcnn_imgs.append(torch.zeros((3, rcnn_size, rcnn_size))) 

    rcnn_imgs = torch.stack(rcnn_imgs, dim=0)  #[3,3,64,64]
    assert len(rcnn_imgs) == 3

    return image,aux_imgs,rcnn_imgs  #[3,224,224] #[3,3,128,128] [3,3,64,64]
