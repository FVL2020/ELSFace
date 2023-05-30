import os
os.sys.path.append(os.getcwd())
from ntpath import basename
import torch
import numpy as np
from os.path import join
import cv2 as cv
import fid_score
from skimage.feature import canny
from glob import glob
from skimage.metrics import structural_similarity
from skimage.metrics import peak_signal_noise_ratio
import torch.backends.cudnn as cudnn
import torch.nn as nn
import math
from easydict import EasyDict as edict
import lpips
from torch.autograd import Variable
import torchvision.utils as vutils
from skimage.feature import canny
import cv2


def to_var(data):
    real_cpu = data
    batchsize = real_cpu.size(0)
    input = Variable(real_cpu.cuda())
    return input, batchsize

def compare_mae(img_true, img_test):
    img_true = img_true.astype(np.float32)
    img_test = img_test.astype(np.float32)
    # return np.sum(np.abs(img_true - img_test)) / np.sum(img_true + img_test)
    # return np.mean(np.abs((np.mean(img_true,2) - np.mean(img_test,2))/255))
    return np.mean(np.abs(img_true - img_test))
    
def main():
    PSNR = []
    SSIM = []
    mae = []
    LPIPS = []
    
    lpips_net = lpips.LPIPS(net='alex')
    lpips_net = lpips_net.cuda()
    input_file = sorted(glob('./sr_results'+'/*.jpg'))
    gt_file = './testset_celeba/'
    for fn in sorted(input_file):
        basename0 = basename(str(fn))
        #print(basename0)
        img_gt_uint = cv.imread(str(fn))
        img_pred_uint = cv.imread(gt_file + basename0.split('.')[0]+'.jpg')
        PSNR.append(peak_signal_noise_ratio(img_gt_uint, img_pred_uint, data_range=255))
        SSIM.append(structural_similarity(img_gt_uint, img_pred_uint, data_range=255, multichannel=True, win_size=51))
        mae.append(compare_mae(img_gt_uint, img_pred_uint))
    
    input_file = sorted(glob('./sr_results'+'/*.jpg'))
    #input_file = sorted(glob('/home/qihaoran/home/qihaoran/Face-and-Image-super-resolution-master/test_res_celeba_new'+'/*.jpg'))
    gt_file = './testset_celeba/'
    for fn in sorted(input_file):
        basename0 = basename(str(fn))
        #print(basename0)
        img_gt_uint = cv.imread(str(fn))
        img_pred_uint = cv.imread(gt_file + basename0.split('.')[0]+'.jpg')
        gt_l = lpips_net(torch.tensor(img_gt_uint.transpose(2,0,1)).cuda(),torch.tensor(img_pred_uint.transpose(2,0,1)).cuda())
        LPIPS.append(gt_l.item())
        
    Fid = fid_score.calculate_fid_given_paths(['./sr_results', './testset_celeba'], 64, True, 2048)

    print("PSNR of the test set is {}".format(np.array(PSNR).mean()))
    print("SSIM of the test set is {}".format(np.array(SSIM).mean()))
    print("MAE of the test set is {}".format(np.array(mae).mean()))
    print("LPIPS of the test set is {}".format(np.array(LPIPS).mean()))
    print("FID of the test set is {}".format(Fid))

if __name__ == '__main__':
    main()