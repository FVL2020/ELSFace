import numpy as np
import torch
import torch.nn as nn
from model import common
#from mmcv.runner import load_checkpoint

from mmedit.models.backbones.sr_backbones.rrdb_net import RRDB
from mmedit.models.builder import build_component
from mmedit.models.common import PixelShufflePack, make_layer
from mmedit.models.registry import BACKBONES
from mmedit.utils import get_root_logger


#@BACKBONES.register_module()
    
class GLEANStyleGANv4(nn.Module):

    def __init__(self,
                 in_size,
                 out_size,
                 img_channels=3,
                 rrdb_channels=60,
                 num_rrdbs=1,
                 style_channels=60,
                 num_mlps=8,
                 channel_multiplier=2,
                 blur_kernel=[1, 3, 3, 1],
                 lr_mlp=0.01,
                 default_style_mode='mix',
                 eval_style_mode='single',
                 mix_prob=0.9,
                 pretrained=None,
                 bgr2rgb=False):

        super().__init__()
        conv=common.default_conv
        act = nn.ReLU(True)
        ckpt_http = './stylegan2-ffhq-config-f-official_20210327_171224-bce9310c.pth'
        pretrained=dict(ckpt_path=ckpt_http,prefix='generator_ema')
        # input size must be strictly smaller than output size
        if in_size >= out_size:
            raise ValueError('in_size must be smaller than out_size, but got '
                             f'{in_size} and {out_size}.')
        
        # latent bank (StyleGANv2), with weights being fixed

        self.in_size = in_size
        self.style_channels = style_channels
        #channels = self.generator.channels
        channels = {4: 60, 8: 60, 16: 60, 32: 60, 64: 60, 128: 60, 256: 60, 512: 60, 1024: 60}
        # encoder
        num_styles = int(np.log2(out_size)-5) * 2
        #num_styles = int(np.log2(out_size)-5) * 2+2
        print(num_styles)        
        encoder_res = [2**i for i in range(int(np.log2(in_size)), 1, -1)]
        self.encoder = nn.ModuleList()
        self.encoder.append(
            nn.Sequential(
                nn.Conv2d(
                    img_channels, channels[in_size], 3, 1, 1, bias=True),
                nn.LeakyReLU(negative_slope=0.2, inplace=True),
                nn.Conv2d(
                    channels[in_size], channels[in_size], 3, 1, 1, bias=True),
                nn.LeakyReLU(negative_slope=0.2, inplace=True)))
        for res in encoder_res:
            in_channels = channels[res]
            if res > 4:
                out_channels = channels[res // 2]
                block = nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 3, 2, 1, bias=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True))
            else:
                block = nn.Sequential(
                    nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=True),
                    nn.LeakyReLU(negative_slope=0.2, inplace=True),
                    nn.Flatten(),
                    nn.Linear(16 * in_channels, num_styles * style_channels))
            self.encoder.append(block)

    def forward(self, lq):
        """Forward function.

        Args:
            lq (Tensor): Input LR image with shape (n, c, h, w).

        Returns:
            Tensor: Output HR image.
        """

        h, w = lq.shape[2:]
        if h != self.in_size or w != self.in_size:
            raise AssertionError(
                f'Spatial resolution must equal in_size ({self.in_size}).'
                f' Got ({h}, {w}).')

        # encoder
        feat = lq
        encoder_features = []
        for block in self.encoder:
            feat = block(feat)
            encoder_features.append(feat)
        encoder_features = encoder_features[::-1]

        latent = encoder_features[0].view(lq.size(0), -1, self.style_channels)
        encoder_features = encoder_features[1:]

        return latent
    '''
    def init_weights(self, pretrained=None, strict=True):
        """Init weights for models.

        Args:
            pretrained (str, optional): Path for pretrained weights. If given
                None, pretrained weights will not be loaded. Defaults to None.
            strict (boo, optional): Whether strictly load the pretrained model.
                Defaults to True.
        """
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, strict=strict, logger=logger)
        elif pretrained is not None:
            raise TypeError(f'"pretrained" must be a str or None. '
                            f'But received {type(pretrained)}.')
    '''