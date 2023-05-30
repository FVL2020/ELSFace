import os,time,sys
from glob import glob, iglob
import numpy as np
import torch.utils.data as data
from PIL import Image
from torchvision import transforms
import torch
import cv2
from torch.utils.data import DataLoader
import PIL

class faces_down_int(data.Dataset):
    def __init__(self, datasets):
        assert datasets, print('no datasets specified')
        self.img_list = []
        dataset = datasets
        if dataset == 'widerfacetest':
            #img_path = "./testset"
            img_path = "./testset_ffhq"
            #img_path = "/data/dataset/celeba/test"
            list_name = (glob(os.path.join(img_path, "*.jpg")))
            list_name.sort()
            for filename in list_name:#jpg
                self.img_list.append(filename)

    def __len__(self):
        return len(self.img_list)

    def __getitem__(self, index):
        data = {}
        inp = cv2.imread(self.img_list[index], cv2.IMREAD_COLOR).astype(np.float32) / 255.
        inp32 = cv2.resize(inp,(int(32),int(32)),interpolation=cv2.INTER_CUBIC)
        #inp128 = cv2.resize(inp,(int(128),int(128)),interpolation=cv2.INTER_CUBIC)
        #inp32 = cv2.resize(inp32,(int(128),int(128)),interpolation=cv2.INTER_NEAREST)                    
        #data['edge'] = transforms.ToTensor()(canny(inp32_np, sigma=0.5))
        data['img_hr'] = torch.from_numpy(inp).float()
        data['img_lr'] = torch.from_numpy(inp32).float()
        data['imgpath'] = self.img_list[index]
        return data



def get_int_loader(dataname,bs =1):
    #dataset = faces_super(dataname, transform)
    dataset = faces_down_int(dataname)
    data_loader = DataLoader(dataset=dataset,
                             batch_size=bs,
                             shuffle=False, num_workers=2, pin_memory=True)
    return data_loader





















