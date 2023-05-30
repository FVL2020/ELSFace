import argparse
import os
from data.dataset import get_loader, get_int_loader, get_real_loader
import random
import time
import warnings
from model_full import High2Low_simple
from datetime import datetime
import torch.nn.functional as F
import cv2
from collections import OrderedDict
import math
import numpy as np
import random
from data.data_1 import faces_data, High_Data, Low_Data
import torchvision.models as models
import torch
from torch.cuda.amp.autocast_mode import autocast
import model
from option import args as args_ipt
from seg_model import BiSeNet
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.cuda.amp as amp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from fvcore.nn import FlopCountAnalysis, parameter_count_table
#import torchvision.models as models
from torch.utils.data import DataLoader
#from ipt_semantic import ipt_base
from datetime import datetime


parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('-d','--data', metavar='DIR', default='./data',
                    help='path to dataset')
parser.add_argument('--eval-data', metavar='DIR', default='./data',
                    help='path to eval dataset')
parser.add_argument('-s','--save-path', metavar='DIR', default='./ckpt',
                    help='path to save checkpoints')
parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N',
                    help='mini-batch size (default: 256), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=5e-5, type=float,
                    metavar='LR', help='initial learning rate', dest='lr')
parser.add_argument('--lr-policy', default='naive',
                    help='lr policy')
parser.add_argument('--warmup-epochs', default=0, type=int, metavar='N',
                    help='number of warmup epochs')
parser.add_argument('--warmup-lr-multiplier', default=0.1, type=float, metavar='W',
                    help='warmup lr multiplier')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1-4)',
                    dest='weight_decay')
parser.add_argument('--power', default=1.0, type=float,
                    metavar='P', help='power for poly learning-rate decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='./ckpt/train/checkpoint.pth', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--reset-epoch', action='store_true', 
                    help='whether to reset epoch')
parser.add_argument('--eval', action='store_true', 
                    help='only do evaluation')         
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--task', default='', type=str, metavar='string',
                    help='specific a task'
                    '["denoise30", "denoise50", "SRx2", "SRx3", "SRx4", "dehaze"] (default: none)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--fp16',action='store_true', default=False, help="\
                    use fp16 instead of fp32.")


best_acc1 = 0
# set task sets

def norm(input):
    x = (input-input.min())/(input.max()-input.min())
    output = (x*255).astype(np.uint8)
    return output
    
def main():
    args = parser.parse_args()

    now = datetime.now()
    timestr = now.strftime("%m-%d-%H_%M_%S")
    args.save_path = os.path.join(args.save_path, f"{args.task}" if args.task else "train")
    #args.save_path = os.path.join(args.save_path, timestr)
    save_path = args.save_path
    
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    print("=> creating model '{}'".format("ipt_base"))
    model_ipt = model.Model(args_ipt, ckp = None)
    model_ipt_t = High2Low_simple()
    optimizer = torch.optim.Adam(model_ipt.parameters(), args.lr,
                                betas=(0.9, 0.999),
                                weight_decay=args.weight_decay)

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            num_param=0
            for param in model_ipt.parameters():
                num_param += param.numel()
            print(num_param)
            checkpoint = torch.load('./ckpt_seg_ipt_mul_nr_sharp_rec_4_1_rec_3_5_pre/train/checkpoint.pth')
            #checkpoint_t = torch.load('./ckpt_seg_ipt_mul_nr_sharp_rec_4_1_rec_3_5_gan_0.005/train/checkpoint_t.pth')
            #if not args.reset_epoch:
            #    args.start_epoch = checkpoint['epoch']
            #args.start_epoch = 10
            model_ipt.load_state_dict(checkpoint['state_dict'])
            #model_ipt_t.load_state_dict(checkpoint_t['state_dict'])
            #optimizer.load_state_dict(checkpoint['optimizer'])
            #print("=> loaded checkpoint '{}' (epoch {})"
            #      .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True

    if args.distributed:
        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model_ipt.cuda(args.gpu)
            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            model_ipt = torch.nn.parallel.DistributedDataParallel(model_ipt, device_ids=[args.gpu])
        else:
            model_ipt.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model_ipt = torch.nn.parallel.DistributedDataParallel(model_ipt)
    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model_ipt = model_ipt.cuda(args.gpu)
    else:
        # DataParallel will divide and allocate batch_size to all available GPUs
        model_ipt = torch.nn.DataParallel(model_ipt).cuda()
        model_ipt_t = torch.nn.DataParallel(model_ipt_t)
    input_size = 48

    # Data loading code
    
    data = faces_data(High_Data, Low_Data)
    train_loader =  DataLoader(dataset=data, batch_size=4, shuffle=True)
    
    data_name = 'widerfacetest'
    test_loader = get_int_loader(data_name, 1)

    if args.distributed:
        raise RuntimeError("distributed not implemented")
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    scaler = amp.GradScaler() if args.fp16 else None
    print(args)
    test(test_loader, model_ipt, model_ipt_t, args)
         

task_map = {"denoise30": 0, "denoise50": 1, "SRx2": 2, "SRx3": 3, "SRx4": 4, "dehaze": 5}

def train(train_loader, model, criterion, optimizer, epoch, args, scaler=None):
    # train for one epoch
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    psnr_out = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    if args.lr_policy == 'naive':
        local_lr = adjust_learning_rate_naive(optimizer, epoch, args)
    elif args.lr_policy == 'step':
        local_lr = adjust_learning_rate(optimizer, epoch, args)
    elif args.lr_policy == 'epoch_poly':
        local_lr = adjust_learning_rate_epoch_poly(optimizer, epoch, args)
        
    
    for i, batch in enumerate(train_loader):
        
        target = batch["hr_bicubic"]
        # set random task
        input = batch["lr_bicubic"]
        global_iter = epoch * args.epoch_size + i
        
        if args.lr_policy == 'iter_poly':
            local_lr = adjust_learning_rate_poly(optimizer, global_iter, args)
        elif args.lr_policy == 'cosine':
            local_lr = adjust_learning_rate_cosine(optimizer, global_iter, args)
        
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            input = input.cuda(args.gpu, non_blocking=True)
            target = target.cuda(args.gpu, non_blocking=True)

        target = target.cuda()
        if scaler is None:
            # compute output
            output = model(input)
            #print(output.device, target.device)
            loss = criterion(output, target)
        else:
            with autocast():
                # compute output
                output = model(input)
                #print(output.device, target.device)
                loss = criterion(output, target)

        # measure accuracy and record loss
        output = (output * 0.5 + 0.5) * 255.
        target = (target * 0.5 + 0.5) * 255.
        psnr = PSNR()(output, target)
        losses.update(loss.item(), input.size(0))
        psnr_out.update(psnr.item(), input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()

        if scaler is None:
            # compute gradient and do SGD step
            loss.backward()
            optimizer.step()
        else:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 100 == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'PSNR {psnr.val:.3f} ({psnr.avg:.3f})\t'
                  'LR: {lr: .6f}'.format(
                   epoch, i, args.epoch_size, batch_time=batch_time,
                   data_time=data_time, loss=losses, psnr=psnr_out, lr=local_lr))


def validate(val_loader, model, criterion, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    psnr_out = AverageMeter()
    psnr_in = AverageMeter()

    # switch to evaluate mode
    model.eval()
    P = PSNR()
    with torch.no_grad():
        end = time.time()
        for i, (target, input_group) in enumerate(val_loader):
            task_id = task_map[args.task]
            input = input_group[task_id]
            model.module.set_task(task_id)
            if args.gpu is not None:
                input = input.cuda(args.gpu, non_blocking=True)
                target = target.cuda(args.gpu, non_blocking=True)
            target = target.cuda()
            # compute output
            output = model(input)
            loss = criterion(output, target)

            # measure accuracy and record loss
            output = (output * 0.5 + 0.5) * 255.
            target = (target * 0.5 + 0.5) * 255.
            psnr1 = P(output, target)
            # psnr2 = P(input.cuda(), target)
            losses.update(loss.item(), input.size(0))
            psnr_out.update(psnr1.item(), input.size(0))
            # psnr_in.update(psnr2.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'PSNR_Out {psnr1.val:.3f} ({psnr1.avg:.3f})\t'
                      'PSNR_In {psnr2.val:.3f} ({psnr2.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time, loss=losses, psnr1=psnr_out, psnr2=psnr_in
                    ))

        print(' * PSNR_Out {psnr1.val:.3f} ({psnr1.avg:.3f})\t'
                 'PSNR_In {psnr2.val:.3f} ({psnr2.avg:.3f})'.format(psnr1=psnr_out, psnr2=psnr_in))
    return psnr_out.avg


def test(test_loader, model_ipt, model_ipt_t, args):
    batch_time = AverageMeter()
    losses = AverageMeter()
    psnr_out = AverageMeter()
    psnr_in = AverageMeter()

    seg_net = BiSeNet(n_classes=19)
    seg_net = seg_net.cuda()
    seg_net.load_state_dict(torch.load("./res/cp/79999_iter.pth"))
    seg_net.eval()
    
    vggnet = models.vgg16_bn(pretrained=True).features[:15]
    vggnet = vggnet.cuda()
    
    # switch to evaluate mode
    model_ipt.eval()
    P = PSNR()
    with torch.no_grad():
        end = time.time()
        for i, batch in enumerate(test_loader):
            input = batch["img_lr"]
            #input_2 = batch["img_lr_2"]
            target = batch["img_hr"]
            input = input.permute(0,3,1,2)
            #input_2 = input_2.permute(0,3,1,2)
            target = target.permute(0,3,1,2)
            input = input.cuda()
            #input_2 = input_2.cuda()
            target = target.cuda()
            style = vggnet(target)

            seg_input = F.interpolate(input, (512,512), mode='nearest')
            seg_input = seg_net(seg_input)[0]
            #print(x_seg.size())
            seg_input_0 = (F.interpolate(seg_input, (64,64), mode='nearest'))[:,:18,:,:]
            seg_input_1 = (F.interpolate(seg_input, (128,128), mode='nearest'))[:,:18,:,:]
            
            # compute output
            time1 = time.time()
            #input_t = model_ipt_t(input)
            output = model_ipt(input,0)
            time2 = time.time()
            #print(time2-time1)
            #exit()
            '''
            output_save = output.cpu().detach().numpy().transpose(0, 2, 3, 1).squeeze(0)
            output_save = (output_save - output_save.min()) / (output_save.max() - output_save.min())
            output_save = (output_save * 255).astype(np.uint8)
            '''
            #output = (output * 0.5 + 0.5) * 255.
            #target = (target * 0.5 + 0.5) * 255.
            output = (output) * 255.
            target = (target) * 255.
            output_save = output.cpu().detach().numpy().transpose(0, 2, 3, 1).squeeze(0)
            num_param = 0
            

            psnr1 = P(output, target)
            psnr_out.update(psnr1.item(), input.size(0))
            img_name = batch['imgpath'][0].split('/')[-1]
            path = './sr_results/'+img_name.split('.')[0]+'.jpg'
            cv2.imwrite(path, output_save)
            # measure elapsed time
            batch_time.update(time2-time1)
            end = time.time()

            if i % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'PSNR_Out {psnr1.val:.3f} ({psnr1.avg:.3f})\t'.format(
                       i, len(test_loader), batch_time=batch_time, psnr1=psnr_out
                    ))
        #flops = FlopCountAnalysis(model_ipt, (input,0))
        print('number of parameter is {}'.format(num_param))
        #print("FLOPs: ", flops.total())
        print(' * PSNR_Out {psnr1.val:.3f} ({psnr1.avg:.3f})\t'
                 'PSNR_In {psnr2.val:.3f} ({psnr2.avg:.3f})'.format(psnr1=psnr_out, psnr2=psnr_in))

    return psnr_out.avg
    
def save_checkpoint(state, path='./', filename='checkpoint'):
    saved_path = os.path.join(path, filename+'.pth')
    torch.save(state, saved_path)
    '''
    if is_best:
        state_dict = state['state_dict']
        new_state_dict = OrderedDict()
        best_path = os.path.join(path, 'model_best.pth')
        for key in state_dict.keys():
            if 'module.' in key:
                new_state_dict[key.replace('module.', '')] = state_dict[key].cpu()
            else:
                new_state_dict[key] = state_dict[key].cpu()
        torch.save(new_state_dict, best_path)
    '''

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def adjust_learning_rate_naive(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr if epoch < 200 else 2/5 * args.lr
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr
    
def adjust_learning_rate_epoch_poly(optimizer, epoch, args):
    """Sets epoch poly learning rate"""
    lr = args.lr * ((1 - epoch * 1.0 / args.epochs) ** args.power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def adjust_learning_rate_poly(optimizer, global_iter, args):
    """Sets iter poly learning rate"""
    lr = args.lr * ((1 - global_iter * 1.0 / (args.epochs * args.epoch_size)) ** args.power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

def adjust_learning_rate_cosine(optimizer, global_iter, args):
    warmup_lr = args.lr * args.warmup_lr_multiplier
    max_iter = args.epochs * args.epoch_size
    warmup_iter = args.warmup_epochs * args.epoch_size
    if global_iter < warmup_iter:
        slope = (args.lr - warmup_lr) / warmup_iter
        lr = slope * global_iter + warmup_lr
    else:
        lr = 0.5 * args.lr * (1 + math.cos(math.pi * (global_iter - warmup_iter) / (max_iter - warmup_iter)))
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr

class PSNR:
    """Peak Signal to Noise Ratio
    img1 and img2 have range [0, 255]"""

    def __init__(self):
        self.name = "PSNR"

    @staticmethod
    def __call__(img1, img2):
        mse = torch.mean((img1 - img2) ** 2)
        return 20 * torch.log10(255.0 / torch.sqrt(mse))
'''
class SSIM:
    """Structure Similarity
    img1, img2: [0, 255]"""

    def __init__(self):
        self.name = "SSIM"

    @staticmethod
    def __call__(img1, img2):
        if not img1.shape == img2.shape:
            raise ValueError("Input images must have the same dimensions.")
        if img1.ndim == 2:  # Grey or Y-channel image
            return self._ssim(img1, img2)
        elif img1.ndim == 3:
            if img1.shape[2] == 3:
                ssims = []
                for i in range(3):
                    ssims.append(ssim(img1, img2))
                return np.array(ssims).mean()
            elif img1.shape[2] == 1:
                return self._ssim(np.squeeze(img1), np.squeeze(img2))
        else:
            raise ValueError("Wrong input image dimensions.")

    @staticmethod
    def _ssim(img1, img2):
        C1 = (0.01 * 255) ** 2
        C2 = (0.03 * 255) ** 2

        img1 = img1.astype(np.float64)
        img2 = img2.astype(np.float64)
        kernel = cv2.getGaussianKernel(11, 1.5)
        window = np.outer(kernel, kernel.transpose())

        mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
        mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
        mu1_sq = mu1 ** 2
        mu2_sq = mu2 ** 2
        mu1_mu2 = mu1 * mu2
        sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
        sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
        sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2

        ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / (
            (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)
        )
        return ssim_map.mean()
'''
if __name__ == '__main__':
    main()
