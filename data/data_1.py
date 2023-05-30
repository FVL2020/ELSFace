import os, sys
import numpy as np
import cv2

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import torch.nn.functional as nnF
import random
import torch
from skimage.feature import canny

High_Data = ["/data/dataset/celeba/train/"]

'''
img_list = []
img_labels = []
attr_label = {}

fp = open(attr_txt, 'r')
for line in fp.readlines():
    if len(line.split()) != 41:
        continue
    img_name = line.split()[0]
    img_label_single = []
    for value in line.split()[1:]:
        if value == '-1':
            img_label_single.append(0)
        if value == '1':
            img_label_single.append(1)
    attr_label['{}'.format(img_name)] = img_label_single    
''' 
def single2tensor3(img):
    return torch.from_numpy(np.ascontiguousarray(img)).permute(2, 0, 1).float()

class faces_data_int(Dataset):
    def __init__(self, data_hr):
        self.hr_imgs = [os.path.join(d, i) for d in data_hr for i in os.listdir(d) if
                        os.path.isfile(os.path.join(d, i))]
        
        self.preproc = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

    def __len__(self):
        return len(self.hr_imgs)

    def __getitem__(self, index):
        data = {}
        hr = cv2.imread(self.hr_imgs[index],cv2.IMREAD_UNCHANGED)
        img_name = self.hr_imgs[index].split('/')[-1]
        #hr_sharp = cv2.imread('/data1/celeba/train_sharp/'+img_name,cv2.IMREAD_UNCHANGED)
        hr = cv2.resize(hr, (128,128), interpolation=cv2.INTER_CUBIC)
        lr = cv2.resize(hr, (32, 32), interpolation=cv2.INTER_CUBIC)
        lr_16 = cv2.resize(hr, (16,16), interpolation=cv2.INTER_CUBIC)
        lr_16 = cv2.resize(lr_16, (128,128), interpolation=cv2.INTER_NEAREST)
        #hr_sharp = cv2.resize(hr_sharp, (128,128), interpolation=cv2.INTER_CUBIC)
        hr = np.float32(hr/255.)
        lr = np.float32(lr/255.)
        lr_16 = np.float32(lr_16/255.)
        #hr_sharp = np.float32(hr_sharp/255.)
        data["lr"] = single2tensor3(lr)
        data["lr_16"] = single2tensor3(lr_16)
        data["hr"] = single2tensor3(hr)
        #data["hr_sharp"] = single2tensor3(hr_sharp)
        return data

    def get_noise(self, n):
        return torch.randn(n, 1, 64, dtype=torch.float32)

        

if __name__ == "__main__":
    data = faces_data_0(High_Data, Low_Data)
    #exit()
    loader = DataLoader(dataset=data, batch_size=16, shuffle=True)
    for i, batch in enumerate(loader):
        print("batch: ", i)
        lrs = batch["lr"].numpy()
        hrs = batch["hr"].numpy()
        downs = batch["hr_down"].numpy()

        for b in range(batch["z"].size(0)):
            lr = lrs[b]
            hr = hrs[b]
            down = downs[b]
            lr = lr.transpose(1, 2, 0)
            hr = hr.transpose(1, 2, 0)
            down = down.transpose(1, 2, 0)
            lr = (lr - lr.min()) / (lr.max() - lr.min())
            hr = (hr - hr.min()) / (hr.max() - hr.min())
            down = (down - down.min()) / (down.max() - down.min())
            cv2.imshow("lr-{}".format(b), lr)
            cv2.imshow("hr-{}".format(b), hr)
            cv2.imshow("down-{}".format(b), down)
            cv2.waitKey()
            cv2.destroyAllWindows()

    print("finished.")