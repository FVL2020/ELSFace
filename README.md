# ELSFace (An Efficient Latent Style Guided Transformer-CNN Framework for Face Super-Resolution)
By Haoran Qi, Yuwei Qiu, Xing Luo, and Zhi Jin

In the Face Super-Resolution (FSR) task, it is important to precisely recover facial textures while maintaining facial contours for realistic high resolution faces. Although several CNN-based FSR methods have achieved great performance, they fail in restoring the facial contours due to the limitation of local convolutions. In contrast, Transformer-based methods which use self-attention as the basic component, are expert in modeling long-range dependencies between image patches. However, learning long-range dependencies often deteriorates facial textures due to the lack of locality. Therefore, a question is naturally raised: how to effectively combine the superiority of CNN and Transformer for better reconstructing faces? To address this issue, we propose an Efficient Latent Style guided Transformer-CNN framework for FSR called ELSFace, which can sufficiently integrate the advantages of CNN and Transformer. The framework consists of a Feature Preparation Stage and a Feature Carving Stage. Basic facial contours and textures are generated in the Feature Preparation Stage, and separately guided by latent styles, so that facial details are better repesented in reconstruction. CNN and Transformer streams in the Feature Carving Stage are used to individually restore facial textures and facial contours, respectively in a parallel recursive way. Considering the negligence of high-frequency features when learning the long-range dependencies, we design the High-Frequency Enhancement Block (HFEB) in the Transformer stream. The Sharp Loss is also proposed for better perceptual quality in optimization. Extensive experimental results demonstrate that our ELSFace can achieve the best results among all metrics compared to the state-of-the-art CNN and Transformer-based methods on commonly used datasets and real- world tasks. Meanwhile, our ELSFace method has the least model parameters and running time.


## Dependencies 
* torch==1.7.1
* torchvision==0.8.2
* numpy
* opencv-python

If there are any propmts about other necessary dependencies for training and inference, please prepare the environment with the guidance.

## Test
To test the ELSFace model, we firstly make sure the location of the pretrained model (checkpoints):
```
./ckpt/train/checkpoint.pth
```
This location should be same as the configuration ``--resume`` in the ``main_test.py``.

Meanwhile, make sure that the the configuration ``--model`` in line 47 of ``option.py`` is same as the inference model, which is also the filename of ``*.py`` in the location of ``./model/elsface``(for example). 
```
parser.add_argument('--model', default='elsface', help='model name')
```
According to this, the pretrained model can be inferenced and test with the defined test dataset.

The test dataset can be set in ``./data/dataset.py``, where ``img_path`` in line 19 can be replaced with other test dataset.

When finishing the aforementioned guidances, run:
```
python main_test.py
```
or directly set checkpoints like:
```
python main_test.py -- resume ./location/of/checkpoints
```
For the location of saving reconstructed faces, the ``main_test.py`` define that the save_dir is ``./sr_results/``, which can be changed with the parameter ``path`` in line 448 of ``main_test.py``.


## Train
This project is suitable for any other kinds of training dataset. The only thing you need to do is just change the parameter ``High_Data`` in ``./data/data_1.py``, so that your dataset can be implemented in training on ELSFace.

If your want to change the configurations of ELSFace for improvement, you can change the network parameters in the ``option.py``.

As for details of parameters in training including the learning rate, the training epoch, the ckpt_path, and other hyparameters, they can be set by the configurations in the ``main_train.py``, like:
```
parser.add_argument('--epochs', default=40, type=int, metavar='N',
                    help='number of total epochs to run')  #line 46
```
```
parser.add_argument('-s','--save-path', metavar='DIR', default='./ckpt',
                    help='path to save checkpoints')  #line 42
```
When finishing the aforementioned steps, you can train ELSFace with:
```
python main_train.py
```
Or directly define parameters like:
```
python main_train.py --save-path ./path/to/save/ckpts
```
## Evaluate
After inference, if you want to find the performance of reconstructed faces, you can set the all the locations of ``input_files`` and ``gt_files`` corresponding with the reconstructed faces and ground truth faces like:
```
input_file = sorted(glob('./sr_results'+'/*.jpg'))
gt_file = './testset_celeba/'
```
And then run:
```
python evaluate.py
```
Therefore, the PSNR, SSIM, MAE, LPIPS, and FID are all measured and printed in the shell.

## Dataset
The training dataset of ELSFace is the training dataset that officiallt divided, which can be referred with:

http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

The test dataset utilized in the paper are integrated with the BaiduNetdisk as follows:

https://pan.baidu.com/s/188FERZKG8Lap_avXQsepyA, passwd: 7xjx


## Appreciation
The codes refer to IPT. Thanks for the authors of it！

## License
This repository is released under the MIT License as found in the LICENSE file. Code in this repo is for non-commercial use only.