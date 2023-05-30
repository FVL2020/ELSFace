# 2021.05.07-Changed for IPT
#            Huawei Technologies Co., Ltd. <foss@huawei.com>

# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
from model import network_swinir
from model import common, common_style_encoder
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import math
import torch
import torch.nn.functional as F
import torch.nn as nn
from torch import nn, Tensor
import os
import numpy as np
import cv2
#from einops import rearrange
import copy

def make_model(args, parent=False):
    return ipt(args)

class ipt(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(ipt, self).__init__()

        
        self.scale_idx = 0
        
        self.args = args
        
        n_feats = args.n_feats
        kernel_size = 3 
        
        
        act = nn.ReLU(True)
        
        self.sub_mean = common.MeanShift(args.rgb_range)
        self.add_mean = common.MeanShift(args.rgb_range, sign=1)
        self.head_x2 = nn.ModuleList([
            nn.Sequential(
                conv(args.n_colors, n_feats*2, kernel_size),
                common.ResBlock(conv, n_feats*2 , 3, act=act),
                common.ResBlock(conv, n_feats*2 , 3, act=act),
                common.ResBlock(conv, n_feats*2 , 3, act=act),
            ) for _ in args.scale
        ])
          
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.upsample_1 = nn.Upsample(scale_factor=2, mode='bilinear')
        self.encoder = common_style_encoder.GLEANStyleGANv4(in_size=32 ,out_size=64)
        self.cnn_x2_1 = nn.ModuleList([
            nn.Sequential(
                conv(n_feats , n_feats , kernel_size),
                common.ResBlock(conv, n_feats , 5, act=act),
                common.ResBlock(conv, n_feats , 5, act=act),
                )
        ])
        '''
        self.cnn_x2_in = conv(args.n_colors, n_feats, kernel_size)
        
        self.cnn_x2 = nn.ModuleList([
            nn.Sequential(
                common.ResBlock(conv, n_feats, 3, act=act),
                common.ResBlock(conv, n_feats, 3, act=act),
                )
        ])
        
        self.cnn_x2_1 = nn.ModuleList([
            nn.Sequential(
                conv(n_feats , n_feats , kernel_size),
                common.ResBlock(conv, n_feats , 5, act=act),
                common.ResBlock(conv, n_feats , 5, act=act),
                )
        ])
        '''
        self.t_tail_x2 = nn.ModuleList([
            nn.Sequential(Head_x2(args),
                          Head_x2(args),
                          Head_x2(args),
                          Head_x2(args),
                          )
        ])
        '''
        #self.t_tail_x2_1 = Head_x2(args)
        self.t_tail_x4 = nn.ModuleList([
            nn.Sequential(Head_x4(args),
                          Head_x4(args),
                          
                          )
        ])
        #self.t_tail_x4_1 = Head_x4(args)
        '''
        self.tail_x2 = nn.ModuleList([
            nn.Sequential(
                common.Upsampler(conv, s, n_feats , act=False),
                conv((n_feats ), args.n_colors, kernel_size)
            ) for s in args.scale
        ])
        '''
        self.tail_x4 = nn.ModuleList([
            nn.Sequential(
                conv((n_feats )*2, args.n_colors, kernel_size)
            ) for s in args.scale
        ])
        '''


        """
        QYW1
        """

        RSTB = network_swinir.RSTB

        #self.layers = nn.ModuleList()
        self.layers_1 = nn.ModuleList()
        
        self.num_layers = len(args.depths)
        self.ape=args.ape

        self.num_features = args.embed_dim

        #self.norm = nn.LayerNorm(self.num_features*2)
        self.norm_1 = nn.LayerNorm(self.num_features )
        
        # split image into non-overlapping patches
        '''
        self.patch_embed=network_swinir.PatchEmbed(
            img_size=args.patch_size, patch_size=1, in_chans=n_feats*2, embed_dim=args.embed_dim*2,
            norm_layer=nn.LayerNorm)
        '''
        self.patch_embed_1=network_swinir.PatchEmbed(
            img_size=args.patch_size*2, patch_size=1, in_chans=n_feats , embed_dim=args.embed_dim ,
            norm_layer=nn.LayerNorm)
        
        #num_patches = self.patch_embed.num_patches
        num_patches_1 = self.patch_embed_1.num_patches
       
        #patches_resolution = self.patch_embed.patches_resolution
        patches_resolution_1 = self.patch_embed_1.patches_resolution
        
        #self.patches_resolution = patches_resolution
        

        # merge non-overlapping patches into image
        '''
        self.patch_unembed = network_swinir.PatchUnEmbed(
            img_size=args.patch_size, patch_size=1, in_chans=args.embed_dim*2, embed_dim=args.embed_dim*2,
            norm_layer=nn.LayerNorm)
        '''
        self.patch_unembed_1 = network_swinir.PatchUnEmbed(
            img_size=args.patch_size*2, patch_size=1, in_chans=args.embed_dim, embed_dim=args.embed_dim ,
            norm_layer=nn.LayerNorm)

        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, args.embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        self.pos_drop = nn.Dropout(p=args.dropout_rate)

        # stochastic depth
        dpr = [x.item() for x in torch.linspace(0, args.drop_path_rate, sum(args.depths))]

        for i_layer in range(self.num_layers):
            '''
            layer = RSTB(dim=args.embed_dim*2,
                         input_resolution=(patches_resolution[0],
                                           patches_resolution[1]),
                         depth=args.depths[i_layer],
                         num_heads=args.num_heads,
                         window_size=args.window_size,
                         mlp_ratio=args.mlp_ratio,
                         qkv_bias=args.qkv_bias, qk_scale=args.qk_scale,
                         drop=args.dropout_rate, attn_drop=args.attn_drop_rate,
                         drop_path=dpr[sum(args.depths[:i_layer]):sum(args.depths[:i_layer + 1])],  # no impact on SR results
                         norm_layer=nn.LayerNorm,
                         downsample=None,
                         use_checkpoint=False,
                         img_size=args.patch_size,
                         patch_size=1,
                         resi_connection=args.resi_connection
                         )
            '''             
            layer_1 = RSTB(dim=args.embed_dim ,
                         input_resolution=(patches_resolution_1[0],
                                           patches_resolution_1[1]),
                         depth=args.depths[i_layer],
                         num_heads=args.num_heads,
                         window_size=args.window_size,
                         mlp_ratio=args.mlp_ratio,
                         qkv_bias=args.qkv_bias, qk_scale=args.qk_scale,
                         drop=args.dropout_rate, attn_drop=args.attn_drop_rate,
                         drop_path=dpr[sum(args.depths[:i_layer]):sum(args.depths[:i_layer + 1])],  # no impact on SR results
                         norm_layer=nn.LayerNorm,
                         downsample=None,
                         use_checkpoint=False,
                         img_size=args.patch_size,
                         patch_size=1,
                         resi_connection=args.resi_connection
                         )
                         
            #self.layers.append(layer)
            self.layers_1.append(layer_1)
            
        # build the last conv layer in deep feature extraction
        if args.resi_connection == '1conv':
            #self.conv_after_body = nn.Conv2d(args.embed_dim*2, n_feats*2, 3, 1, 1)
            self.conv_after_body_1 = nn.Conv2d(args.embed_dim , n_feats , 3, 1, 1)
            
        elif args.resi_connection == '3conv':
            # to save parameters and memory
            self.conv_after_body = nn.Sequential(nn.Conv2d(args.embed_dim, args.embed_dim // 4, 3, 1, 1),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(args.embed_dim // 4, args.embed_dim // 4, 1, 1, 0),
                                                 nn.LeakyReLU(negative_slope=0.2, inplace=True),
                                                 nn.Conv2d(args.embed_dim // 4, n_feats , 3, 1, 1))
    '''
    def forward_features(self, x):
        x_size = (x.shape[2], x.shape[3])
        #print(x.shape)
        x = self.patch_embed(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers:
            x = layer(x, x_size)
            x = layer(x, x_size)
            x = layer(x, x_size)                        
            
        
        x = self.norm(x)  # B L C
        x = self.patch_unembed(x, x_size)

        return x
    '''
    def forward_features_1(self, x):
        x_size = (x.shape[2], x.shape[3])
        #print(x.shape)
        x = self.patch_embed_1(x)
        if self.ape:
            x = x + self.absolute_pos_embed
        x = self.pos_drop(x)

        for layer in self.layers_1:
            x = layer(x, x_size)
            x = layer(x, x_size)
            x = layer(x, x_size)
            

        x = self.norm_1(x)  # B L C
        x = self.patch_unembed_1(x, x_size)

        return x
        
    """
    QYW1

    """

    def forward(self, x):
        input = x
        
        res_x2 = self.head_x2[0](input) 
        #res_x2 = self.conv_after_body(self.forward_features(res_x2))
        style = self.encoder(input)
        style1 = style[:,0,:].view(x.size()[0], -1, 1, 1)
        style2 = style[:,1,:].view(x.size()[0], -1, 1, 1)
        res_x2 = self.upsample_1(res_x2)    
        x_c = torch.split(res_x2, 60, dim=1)[0]
        res_x2_1 = torch.split(res_x2, 60, dim=1)[1] 
        
        x_c = x_c*style1
        '''
        hf_feature = (x_c).cpu().detach().numpy().transpose(0,2,3,1).squeeze(0)        
        hf_feature_path = './' + '15574.jpg'.split('.')[0]+'t'
        if not os.path.exists(hf_feature_path):
            os.makedirs(hf_feature_path)
        for k in range(hf_feature.shape[2]):
            feature_img = cv2.applyColorMap(np.asarray((1.0/(1+np.exp(-1*hf_feature[:,:,k]*255))*255), dtype=np.uint8), cv2.COLORMAP_JET)                                
            cv2.imwrite((hf_feature_path + '/' + str(k) + '.jpg'), feature_img)
        '''
        for _ in range(2):
            x_c = self.cnn_x2_1[0](x_c)
        
        res_x2_1 = res_x2_1*style2
        '''
        hf_feature = (res_x2_1*255).cpu().detach().numpy().transpose(0,2,3,1).squeeze(0)        
        hf_feature_path =  './' + '15574.jpg'.split('.')[0]+'c'
        if not os.path.exists(hf_feature_path):
            os.makedirs(hf_feature_path)
        for k in range(hf_feature.shape[2]):
            feature_img = cv2.applyColorMap(np.asarray((1.0/(1+np.exp(-1*hf_feature[:,:,k]*255))*255), dtype=np.uint8), cv2.COLORMAP_JET)                                
            cv2.imwrite((hf_feature_path + '/' + str(k) + '.jpg'), feature_img)
        exit()
        '''
        res_x2_1 = self.conv_after_body_1(self.forward_features_1(res_x2_1))
        '''
        l_f = (res_x2_1*255).cpu().detach().numpy().transpose(0,2,3,1).squeeze(0)
        lf = np.zeros([l_f.shape[0],l_f.shape[1]])
        for k in range(l_f.shape[2]):
            lf += (l_f[:,:,k])
        l_f = lf/(l_f.shape[2])
        l_f = cv2.dft(np.float32(l_f),flags=cv2.DFT_COMPLEX_OUTPUT)
        l_f = np.fft.fftshift(l_f)
        l_f = 20*np.log(cv2.magnitude(l_f[:,:,0],l_f[:,:,1]))
        #l_f = cv2.applyColorMap(np.asarray((l_f), dtype=np.uint8), cv2.COLORMAP_WINTER)
        #cv2.imwrite('./2097l.jpg', l_f)
        #plt.show()
        '''
        
        res_x2_1 = self.t_tail_x2[0](res_x2_1)
        
        '''
        h_f = (res_x2_1*255).cpu().detach().numpy().transpose(0,2,3,1).squeeze(0)
        hf = np.zeros([h_f.shape[0],h_f.shape[1]])
        for k in range(h_f.shape[2]):
            hf += (h_f[:,:,k])
        h_f = (hf)/(h_f.shape[2]) 
        h_f = cv2.dft(np.float32(h_f),flags=cv2.DFT_COMPLEX_OUTPUT)
        h_f = np.fft.fftshift(h_f)
        h_f = 20*np.log(cv2.magnitude(h_f[:,:,0],h_f[:,:,1]))
        #h_f = cv2.applyColorMap(np.asarray((h_f), dtype=np.uint8), cv2.COLORMAP_WINTER)
        #cv2.imwrite('./2097h.jpg', h_f)
        #exit()
        '''
        res_x2_1 = res_x2_1 + x_c
        #res_x2 = self.upsample_1(res_x2)
        output = self.tail_x2[self.scale_idx](res_x2_1)
        
        return output
        '''
        input = x
        #x: x->conv(x)  x_2->conv(2x)
        x = self.head_x2[0](input)
        x_2 = self.head_x4[0](input)
        
        
        # x.shape=[B,n_feats ,32,32]
        x = torch.cat((x, x_seg_1), dim=1)
        x_2 = torch.cat((x_2, x_seg_2), dim=1)

        res_x2 = self.conv_after_body(self.forward_features(x)) + x
        res_x4 = self.conv_after_body(self.forward_features(x_2)) + x_2

        #res_x2 = self.body_x2(x,self.scale_idx)
        #res_x2 += x
        
        #res_x4 = self.body_x4(x_2, self.scale_idx)
        #res_x4 += x_2

        res_x2 = self.tail_x2[self.scale_idx](res_x2)
        
        output = torch.cat((res_x2, res_x4), dim=1)
        output = self.tail_x4[self.scale_idx](output)
        #x = self.add_mean(x)
        
        return output
        '''
    def set_scale(self, scale_idx):
        self.scale_idx = scale_idx
        
class VisionTransformer_simple(nn.Module):
    def __init__(
        self,
        img_dim,
        patch_dim,
        num_channels,
        embedding_dim,
        num_heads,
        num_layers,
        hidden_dim,
        num_queries,
        positional_encoding_type="learned",
        dropout_rate=0,
        no_norm=False,
        mlp=False,
        pos_every=False,
        no_pos = False
    ):
        super(VisionTransformer_simple, self).__init__()

        assert embedding_dim % num_heads == 0
        assert img_dim % patch_dim == 0
        self.no_norm = no_norm
        self.mlp = mlp
        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.patch_dim = patch_dim
        self.num_channels = num_channels
        
        self.img_dim = img_dim
        self.pos_every = pos_every
        self.num_patches = int((img_dim // patch_dim) ** 2)
        self.seq_length = self.num_patches
        self.flatten_dim = patch_dim * patch_dim * num_channels
        
        self.out_dim = patch_dim * patch_dim * num_channels
        
        self.no_pos = no_pos
        
        if self.mlp==False:
            self.linear_encoding = nn.Linear(self.flatten_dim, embedding_dim)
            self.mlp_head = nn.Sequential(
                nn.Linear(embedding_dim, hidden_dim),
                nn.Dropout(dropout_rate),
                nn.ReLU(),
                nn.Linear(hidden_dim, self.out_dim),
                nn.Dropout(dropout_rate)
            )
        
            self.query_embed = nn.Embedding(num_queries, embedding_dim * self.seq_length)

        encoder_layer = TransformerEncoderLayer(embedding_dim, num_heads, hidden_dim, dropout_rate, self.no_norm)
        self.encoder = TransformerEncoder(encoder_layer, num_layers)
        
        #decoder_layer = TransformerDecoderLayer(embedding_dim, num_heads, hidden_dim, dropout_rate, self.no_norm)
        #self.decoder = TransformerDecoder(decoder_layer, num_layers)
        
        if not self.no_pos:
            self.position_encoding = LearnedPositionalEncoding(
                    self.seq_length, self.embedding_dim, self.seq_length
                )
            
        self.dropout_layer1 = nn.Dropout(dropout_rate)
        
        if no_norm:
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, std = 1/m.weight.size(1))

    def forward(self, x, query_idx, con=False):
        #x.shape=[B,n_feats ,32,32]->x.shape=[B,(n_feats )*patch_dim*patch_dim,(32//patch_dim)*(32//patch_dim)] whether padding?
        #x.shape=[(32//patch_dim)*(32//patch_dim),B,(n_feats )*patch_dim*patch_dim]
        print(x.shape)
        x = torch.nn.functional.unfold(x,self.patch_dim,stride=self.patch_dim).transpose(1,2).transpose(0,1).contiguous()
        print(x.shape)
        if self.mlp==False:
            # linear layer +res
            #  input:[(32//patch_dim)*(32//patch_dim), B, (n_feats )*patch_dim*patch_dim],output:[(32//patch_dim)*(32//patch_dim), B, (n_feats )*patch_dim*patch_dim]
            x = self.dropout_layer1(self.linear_encoding(x)) + x

            query_embed = self.query_embed.weight[query_idx].view(-1,1,self.embedding_dim).repeat(1,x.size(1), 1)
        else:
            query_embed = None

        
        if not self.no_pos:
            pos = self.position_encoding(x).transpose(0,1)
        
        if self.pos_every:
            x = self.encoder(x, pos=pos)
            #x = self.decoder(x, x, pos=pos, query_pos=query_embed)
        elif self.no_pos:
            x = self.encoder(x)
            #x = self.decoder(x, x, query_pos=query_embed)
        # this will run
        else:
            x = self.encoder(x+pos)
            #x = self.decoder(x, x, query_pos=query_embed)
        
        
        if self.mlp==False:
            x = self.mlp_head(x) + x
        
        x = x.transpose(0,1).contiguous().view(x.size(1), -1, self.flatten_dim)
        
        if con:
            con_x = x
            x = torch.nn.functional.fold(x.transpose(1,2).contiguous(),int(self.img_dim),self.patch_dim,stride=self.patch_dim)
            return x, con_x
        
        x = torch.nn.functional.fold(x.transpose(1,2).contiguous(),int(self.img_dim),self.patch_dim,stride=self.patch_dim)
        
        return x

class Head_x2(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(Head_x2, self).__init__()
        n_feats = args.n_feats
        act = nn.ReLU(True)
        
        #self.general = conv(n_feats//2, n_feats//2, kernel_size=3) 
        self.hf_block = common.HFBlock(conv, n_feats, 5, act=act)
        
        self.main = nn.ModuleList([
            nn.Sequential(
                common.ResBlock(conv, n_feats, 5, act=act),
                common.ResBlock(conv, n_feats, 5, act=act),
                common.ResBlock(conv, n_feats, 5, act=act),
                common.ResBlock(conv, n_feats, 5, act=act),
                common.ResBlock(conv, n_feats, 5, act=act)
            ) for _ in args.scale
        ]) 
        
    def forward(self, input):
        #x = self.general(input)
        x_hf = self.hf_block(input)
        output = self.main[0](x_hf)
        output += x_hf
        
        return output

class Head_x2_pro(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(Head_x2_pro, self).__init__()
        n_feats = args.n_feats
        act = nn.ReLU(True)
        
        self.general = conv(args.n_colors, n_feats//2, kernel_size=3) 
        self.hf_block = common.HFBlock(conv, n_feats//2, 5, act=act)
        
        self.main = nn.ModuleList([
            nn.Sequential(
                common.ResBlock(conv, n_feats//2, 5, act=act),
                common.ResBlock(conv, n_feats//2, 5, act=act),
                common.ResBlock(conv, n_feats//2, 5, act=act),
                common.ResBlock(conv, n_feats//2, 5, act=act),
                common.ResBlock(conv, n_feats//2, 5, act=act)
            ) for _ in range(2)
        ]) 
        
    def forward(self, input):
        x = self.general(input)
        x = self.main[0](x)
        x_hf = self.hf_block(input)
        x_hf = self.main[1](x_hf)
        x = torch.cat((x, x_hf), dim=1)
        output = x
        
        return output
        
class Head_x4(nn.Module):
    def __init__(self, args, conv=common.default_conv):
        super(Head_x4, self).__init__()
        n_feats = args.n_feats
        act = nn.ReLU(True)
        
        #self.general = conv(n_feats//2, n_feats//2, kernel_size=3) 
        self.hf_block = common.HFBlock(conv, n_feats, 5, act=act)
        self.main = nn.ModuleList([
            nn.Sequential(
                common.ResBlock(conv, n_feats, 5, act=act),
                common.ResBlock(conv, n_feats, 5, act=act),
                common.ResBlock(conv, n_feats, 5, act=act),
                common.ResBlock(conv, n_feats, 5, act=act),
                common.ResBlock(conv, n_feats, 5, act=act)
            ) for _ in args.scale
        ]) 
        
    def forward(self, input):
        #x = self.general(input)
        x_hf = self.hf_block(input)
        #x = torch.cat((x, x_hf), dim=1)
        output = self.main[0](x_hf)
        output += x_hf
        
        return output
        
class LearnedPositionalEncoding(nn.Module):
    def __init__(self, max_position_embeddings, embedding_dim, seq_length):
        super(LearnedPositionalEncoding, self).__init__()
        self.pe = nn.Embedding(max_position_embeddings, embedding_dim)
        self.seq_length = seq_length

        self.register_buffer(
            "position_ids", torch.arange(self.seq_length).expand((1, -1))
        )

    def forward(self, x, position_ids=None):
        if position_ids is None:
            position_ids = self.position_ids[:, : self.seq_length]

        position_embeddings = self.pe(position_ids)
        return position_embeddings
    
class TransformerEncoder(nn.Module):

    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, src, pos = None):
        output = src

        for layer in self.layers:
            output = layer(output, pos=pos)

        return output
    
class TransformerEncoderLayer(nn.Module):
    #d_model is embedding_dim = (n_feats )*patch_dim*patch_dim
    #dim_feedforward is hiden_dim
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, no_norm = False,
                 activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=False)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        self.norm1 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)
        
        nn.init.kaiming_uniform_(self.self_attn.in_proj_weight, a=math.sqrt(5))

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos
    
    def forward(self, src, pos = None):

        src2 = self.norm1(src)

        q = k = self.with_pos_embed(src2, pos)

        src2 = self.self_attn(q, k, src2)
        src = src + self.dropout1(src2[0])
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src2))))
        src = src + self.dropout2(src2)
        return src

    
class TransformerDecoder(nn.Module):

    def __init__(self, decoder_layer, num_layers):
        super().__init__()
        self.layers = _get_clones(decoder_layer, num_layers)
        self.num_layers = num_layers

    def forward(self, tgt, memory, pos = None, query_pos = None):
        output = tgt
        
        for layer in self.layers:
            output = layer(output, memory, pos=pos, query_pos=query_pos)

        return output

    
class TransformerDecoderLayer(nn.Module):

    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1, no_norm = False,
                 activation="relu"):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=False)
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, bias=False)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)

        self.norm1 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.norm2 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.norm3 = nn.LayerNorm(d_model) if not no_norm else nn.Identity()
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)

        self.activation = _get_activation_fn(activation)

    def with_pos_embed(self, tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward(self, tgt, memory, pos = None, query_pos = None):
        tgt2 = self.norm1(tgt)
        q = k = self.with_pos_embed(tgt2, query_pos)
        tgt2 = self.self_attn(q, k, value=tgt2)[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt2 = self.norm2(tgt)
        tgt2 = self.multihead_attn(query=self.with_pos_embed(tgt2, query_pos),
                                   key=self.with_pos_embed(memory, pos),
                                   value=memory)[0]
        tgt = tgt + self.dropout2(tgt2)
        tgt2 = self.norm3(tgt)
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt2))))
        tgt = tgt + self.dropout3(tgt2)
        return tgt


def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


def _get_activation_fn(activation):
    """Return an activation function given a string"""
    if activation == "relu":
        return F.relu
    if activation == "gelu":
        return F.gelu
    if activation == "glu":
        return F.glu
    raise RuntimeError(F"activation should be relu/gelu, not {activation}.")