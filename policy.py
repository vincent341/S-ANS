#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import math,torchvision,os
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from groupy.gconv.pytorch_gconv import P4MConvZ2, P4MConvP4M,P4ConvP4,P4ConvZ2
from groupy.gconv.pytorch_gconv.pooling import plane_group_spatial_max_pooling

from gym import spaces
from occant_utils.common import (
    add_pose,
    crop_map,
    subtract_pose,
    process_image,
    transpose_image,
    bottom_row_padding,
    bottom_row_cropping,
    spatial_transform_map,
)
from occant_utils.common import (
    FixedCategorical,
    Categorical,
    init,
)
from occant_baselines.rl.policy_utils import (
    CNNBase,
    Flatten,
    PoseEstimator,
)
from einops import rearrange
from PIL import Image,ImageDraw
from occant_utils import  visualization


EPS_MAPPER = 1e-8
#
class RTLayer(nn.Module):
    #rotate and translation layer
    def __init__(self):
        super(RTLayer,self).__init__()
    def get_rot_trans_mat(self,theta,col,row):
        #theta = torch.tensor(theta)
        return torch.tensor([[torch.cos(theta), -torch.sin(theta), col],
                             [torch.sin(theta), torch.cos(theta), row]]).to(theta.device)

    def rot_trans_img(self,x, theta, agent_pos,dtype):
        b=x.shape[0]#batch size
        rot_mat_batch = torch.zeros([b,2,3],dtype=torch.float32).to(x.device)
        #print("theta shape:",theta.shape)
        #rot_mat_batch = self.get_rot_trans_mat(theta,col=agent_pos[0],row=agent_pos[1])[None, ...].type(dtype).repeat(x.shape[0], 1, 1)
        for i in range(b):#compute rot_mat for all samples in the batch
            #rot_mat = self.get_rot_mat(theta[i])[None, ...].type(dtype).repeat(x.shape[0], 1, 1)
            rot_mat = self.get_rot_trans_mat(theta=theta[i,0],col=agent_pos[i,0],row=agent_pos[i,1]).unsqueeze(dim=0)
            rot_mat_batch[i] = rot_mat
        grid = F.affine_grid(rot_mat_batch, x.size()).type(dtype)
        x = F.grid_sample(x, grid,mode="nearest")#use nearest to avoid value that is neither 0 nor 1
        return x
    def forward(self,input,theta,agent_pos,dtype=torch.float32):
        b = input.shape[0]
        h = input.shape[2]
        w = input.shape[3]
        rotated_img = self.rot_trans_img(x=input, theta=theta.unsqueeze(dim=-1), agent_pos=torch.zeros((b, 2)),
                                         dtype=torch.float32)
        return rotated_img

class MaskedSoftmax(nn.Module):
    #https://gist.github.com/kaniblu/94f3ede72d1651b087a561cf80b306ca
    #masked softmax
    def __init__(self):
        super(MaskedSoftmax, self).__init__()
        self.softmax = nn.Softmax(1)

    def forward(self, x, mask=None):
        """
        Performs masked softmax, as simply masking post-softmax can be
        inaccurate
        :param x: [batch_size, num_items]
        :param mask: [batch_size, num_items]
        :return:
        """
        ind= (mask[-1]!=0)
        print("there is non-0 ele in mask[-1]:", ind.any())
        if mask is not None:
            mask = mask.float()
        if mask is not None:
            x_masked = x * mask + (1 - 1 / mask)
            print("x_masked[-1]:",x_masked[-1,0:10])
        else:
            x_masked = x
        x_max = x_masked.max(1)[0]
        x_exp = (x - x_max.unsqueeze(-1)).exp()
        print("x[-1]:", x[-1, 0:10])
        print("x_max[-1]:", x_max.unsqueeze(-1)[-1, 0:10])
        print("x_exp[-1] bf:", x_exp[-1, 0:10])
        print("mask[-1] bf:", mask[-1, 0:10])
        if mask is not None:
            x_exp = x_exp * mask.float()

        print("x_exp[-1] af:", x_exp[-1, 0:10])
        return x_exp / x_exp.sum(1).unsqueeze(-1)
class BlurPoolp4m(nn.Module):
    def __init__(self, channels, pad_type='reflect', filt_size=4, stride=2, pad_off=0,outchfactor=8):
        #input [b,c,8,h,w] or [b,c,h,w]
        #output [b,c,8,h,w] or [b,c,h,w]
        #outchfactor: outchannel factor. if previous layer is CP,it should be 1; if previous layer is p4m,it should be 8
        super(BlurPoolp4m, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)), int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2))]
        self.pad_sizes = [pad_size+pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride-1)/2.)
        self.channels = channels*outchfactor

        if(self.filt_size==1):
            a = np.array([1.,])
        elif(self.filt_size==2):
            a = np.array([1., 1.])
        elif(self.filt_size==3):
            a = np.array([1., 2., 1.])
        elif(self.filt_size==4):
            a = np.array([1., 3., 3., 1.])
        elif(self.filt_size==5):
            a = np.array([1., 4., 6., 4., 1.])
        elif(self.filt_size==6):
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif(self.filt_size==7):
            a = np.array([1., 6., 15., 20., 15., 6., 1.])
        #print("The size of filter size is:", str(self.filt_size), "filter is :", a, "\n")
        filt = torch.Tensor(a[:,None]*a[None,:])
        filt = filt/torch.sum(filt)
        self.register_buffer('filt', filt[None,None,:,:].repeat((self.channels,1,1,1)))

        self.pad = self.get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        print("The size of blur pool filter size is:", str(self.filt_size))
        xs = inp.size()
        if xs.__len__() == 5:
            inp = inp.view(xs[0], xs[1] * xs[2], xs[3], xs[4])
        if(self.filt_size==1):
            if(self.pad_off==0):
                inp = inp[:,:,::self.stride,::self.stride]
                if xs.__len__() == 5:
                    inp = inp.view(xs[0], xs[1], xs[2], inp.size()[2], inp.size()[3])
                return inp
            else:
                inp= self.pad(inp)[:,:,::self.stride,::self.stride]
                if xs.__len__() == 5:
                    inp = inp.view(xs[0], xs[1], xs[2], inp.size()[2], inp.size()[3])
                return inp
        else:
            inp= F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])
            if xs.__len__() == 5:
                inp = inp.view(xs[0], xs[1], xs[2], inp.size()[2], inp.size()[3])
            return  inp

    def get_pad_layer(self,pad_type):
        if(pad_type in ['refl','reflect']):
            PadLayer = nn.ReflectionPad2d
        elif(pad_type in ['repl','replicate']):
            PadLayer = nn.ReplicationPad2d
        elif(pad_type=='zero'):
            PadLayer = nn.ZeroPad2d
        else:
            print('Pad type [%s] not recognized'%pad_type)
        return PadLayer
class CyclicGroupPoolinglayer(nn.Module):
    def __init__(self,type="max"):
        #exploiting cyclic symmetry in cnns
        #in b,c,8,h,w
        #out b,c,h,w
        super(CyclicGroupPoolinglayer, self).__init__()
        self.type=type
    def forward(self, x):
        xs = x.size()
        if self.type=="max":
            x,_ = torch.max(x,dim=2)
        elif self.type=="mean":
            x = torch.mean(x, dim=2)
        else:
            raise ValueError("Invalid cyclicgroup pooling type!")
        # x = x.view(xs[0], xs[1] * xs[2], xs[3], xs[4])
        # x = self.mp(x)
        # x = x.view(xs[0], xs[1], xs[2], x.size()[2], x.size()[3])
        #x = torch.unsqueeze(x,dim=2)
        return x

class PlaneGroupPoolinglayer(nn.Module):
    def __init__(self,kernelsize,stride,pad):
        #in [b,c,8,h,w]
        #out [b,c,8,h/2,w/2]
        super(PlaneGroupPoolinglayer, self).__init__()
        self.mp=nn.MaxPool2d(kernel_size=kernelsize,stride=stride,padding=pad)
    def forward(self, x):
        xs = x.size()
        if xs.__len__() == 5:
            x = x.view(xs[0], xs[1] * xs[2], xs[3], xs[4])
        x = self.mp(x)
        if xs.__len__() == 5:
            x = x.view(xs[0], xs[1], xs[2], x.size()[2], x.size()[3])
        return x
class polarpoollayer(nn.Module):
    def __init__(self,type):
        super(polarpoollayer,self).__init__()
        self.type=type
    def polar_transform(self,images,polar_img_w, polar_img_h,transform_type='linearpolar'):
        b, c, h, w = images.shape
        row_range = torch.arange(0, polar_img_h).to(images.device)
        col_range = torch.arange(0, polar_img_h).to(images.device)
        grid_row, grid_col = (torch.meshgrid(row_range, col_range))
        grid_col, grid_row = self.polar2cart(rho=grid_col, phi=grid_row, w_c=w, h_c=h, w_p=polar_img_w, h_p=polar_img_h,
                                        polartype=transform_type)
        grid = torch.stack([grid_col, grid_row], dim=-1)  # [h,h,2], [h,h,0] is col; [h,h,1] is row
        grid = grid.unsqueeze(dim=0).repeat(b, 1, 1, 1)  # [b,h,h,2]
        grid = grid / ((h - 1) / 2) - 1
        out = torch.nn.functional.grid_sample(input=images, grid=grid, mode='bilinear', padding_mode='zeros')
        return out

    def polar2cart(self,rho,phi,w_c,h_c,w_p,h_p,polartype):
    #convert cor in polar to cor in cartetain
    #rho: radius, radius, col in polar space; phi, rad angle ,row in polar space
    #w_c,h_c; width and height of image in cartetain; here should be G
    # w_p,h_p; width and height of matrix/feature in polar space; here should be actor.outputsize
    #polartype either 'linearpolar' or 'logpolar'
    #return (col,row) in cartetain for (rho,phi) in polar space
    #reference https://docs.opencv.org/3.4/d4/d35/samples_2cpp_2polar_transforms_8cpp-example.html#a20
        if not (torch.is_tensor(w_c)):
            w_c = torch.FloatTensor([w_c]).to(rho.device)

        rho = rho.float()
        phi = phi.float()
        maxRadius = w_c/2
        Kangle = h_p/ (math.pi * 2)
        angleRad = phi / Kangle

        if polartype == "linearpolar":
            Klin = w_p/maxRadius
            magnitude = rho / Klin
        elif polartype == "logpolar":
            Klog = w_p/torch.log(maxRadius)
            magnitude = torch.exp(rho/Klog)
        else:
            raise ValueError('Invalide polar transformation type!')
        col = w_c/2 + magnitude*torch.cos(angleRad)
        row = h_c/2+magnitude*torch.sin(angleRad)
        #col = col.int()
        #row = row.int()
        return col,row
    def forward(self, input):
        #input:  [b,c,h,w] or [b,c,*,h,w]
        #out: [b,c,h]
        #first transform feature maps in cartesianspace to polar space
        #then maxpool along distance(r), to get rotation invariance
        if input.dim()==5:#input is [b,c,8 or 4,h,w]
            input = input.view(input.shape[0],input.shape[1]*input.shape[2],input.shape[3],input.shape[4])
        b,c,h,w = input.shape
        input_polar = self.polar_transform(images=input,polar_img_w=w,polar_img_h=h,transform_type="linearpolar")
        if self.type=="max":
            x,_ = torch.max(input_polar,dim=-2)#max or mean per col, getting [b,c,w]
        elif self.type=="mean":
            x = torch.mean(input_polar,dim=-2)
        else:
            raise ValueError("Invalid polarpool type!")
        return x

class AdaptiveAvgPool2dp4(nn.Module):
    def __init__(self):
        super(AdaptiveAvgPool2dp4, self).__init__()
        self.gap = nn.AdaptiveAvgPool2d(output_size=(1,1))
    def forward(self, input):
        xs = input.shape
        if input.dim() == 5:
            x = input.view(xs[0], xs[1] * xs[2], xs[3], xs[4])
        else:
            x = input
        x = self.gap(x)
        return x
class MyPadding(nn.Module):
    #used to fix bug for conv2d with 'circular padding'
    def __init__(self,padding_type,padding_num):
        super(MyPadding,self).__init__()
        self.padding_type = padding_type
        self.padding_num = padding_num
        self.pad_size = (padding_num,padding_num,padding_num,padding_num)
    def forward(self, x):
        xs = x.size()
        if xs.__len__() == 5:
            x = x.view(xs[0], xs[1] * xs[2], xs[3], xs[4])
            x = torch.nn.functional.pad(x,self.pad_size,mode=self.padding_type)
            return   x.view(xs[0],xs[1],xs[2],xs[3]+self.padding_num*2,xs[4]+self.padding_num*2)
        elif xs.__len__() == 4:#normal convolution padding
            return torch.nn.functional.pad(x,self.pad_size,mode=self.padding_type)
        else:
            raise ValueError("Invalid length!")
class P4MConvP4_withpad(nn.Module):
    def __init__(self,inch,outch,kernel_size,stride,padding_type,padding_num):
        #input:
        #inch,outch: in and outpuut channels for coolutionn
        #padding__type:'zero' or 'circular'
        #padding_num:x->(x,x,x,x,)
        super(P4MConvP4_withpad, self).__init__()
        self.padding = MyPadding(padding_type=padding_type, padding_num=padding_num)
        self.p4convp4 = P4MConvP4M(in_channels=inch, out_channels=outch, kernel_size=kernel_size, stride=stride,
                                 padding=0)
    def forward(self, input):
        #out = self.addcoords(input)
        out = self.padding(input)
        out = self.p4convp4(out)
        return out

class P4ConvP4_withpad(nn.Module):
    def __init__(self,inch,outch,kernel_size,stride,padding_type,padding_num):
        #input:
        #inch,outch: in and outpuut channels for coolutionn
        #padding__type:'zero' or 'circular'
        #padding_num:x->(x,x,x,x,)
        super(P4ConvP4_withpad, self).__init__()
        self.padding = MyPadding(padding_type=padding_type, padding_num=padding_num)
        self.p4convp4 = P4ConvP4(in_channels=inch, out_channels=outch, kernel_size=kernel_size, stride=stride,
                                 padding=0)
    def forward(self, input):
        #out = self.addcoords(input)
        out = self.padding(input)
        out = self.p4convp4(out)
        return out
class Ansexactp4flex(nn.Module):
    #this class is to create network exact same with MyANSnet when using vinila convolution
    def __init__(self, input_shape, global_config):
        #global_config.net_arc = [actor_arc,critic_arc]
        #shareconv: True or False to use shared conv layers for actor and critic
        super(Ansexactp4flex, self).__init__()
        input_ch = input_shape[2]
        actor_arc = global_config.net_arc[0]
        critic_arc = global_config.net_arc[1]
        self.G = int(input_shape[0])
        self.actor_resfactor = 2 ** (actor_arc.count("BP") + actor_arc.count("PP")+ actor_arc.count("M"))
        self.critic_resfactor = 2 ** (critic_arc.count("BP") + critic_arc.count("PP")+ critic_arc.count("M"))  # scale factor by BP and PP
        self.actor_outsize = 16
        self.hiddensize = 512
        self.batch_type = global_config.batch_type
        self.use_actorp4 = global_config.use_actorp4
        self.use_criticp4 = global_config.use_criticp4
        self.polarpooltype = global_config.polarpooltype
        self.Shareconv = global_config.shareconv
        cyclicpooltype = global_config.cyclicpooltype if hasattr(global_config, 'cyclicpooltype') else "max"
        self.blurpool_filtersize = int(global_config.blurpool_filtersize) if hasattr(global_config, 'blurpool_filtersize') else 4

        if self.Shareconv:#actor and critic share conv net
            assert (actor_arc==critic_arc or critic_arc==actor_arc+["GPP"] or critic_arc==actor_arc+["GAP"]),"Problem in shared actor and critic config!"
            self.main = self.make_layers(arc_cfg=actor_arc,in_channels=input_ch,cyclicpooltype=cyclicpooltype)
            #self.actor_main = self.main
            #self.critic_main = self.main
            if critic_arc[-1]=="GPP":
                self.critic_GP = polarpoollayer(type=self.polarpooltype)
            elif critic_arc[-1]=="GAP":
                self.critic_GP = AdaptiveAvgPool2dp4()
            else:
                self.critic_GP=nn.Identity()
        else:#actor and critic individual net
            self.actor_main = self.make_layers(arc_cfg=actor_arc, in_channels=input_ch,cyclicpooltype=cyclicpooltype)
            self.critic_main = self.make_layers(arc_cfg=critic_arc, in_channels=input_ch,cyclicpooltype=cyclicpooltype)
        #Compute first fc layer param for actor ==============================================================================================
        if actor_arc[-1]=="GAP":
            if "p4m" in str(actor_arc[-2]):
                convname, outch = actor_arc[-2].split("-")
                self.actorlinear1 = nn.Linear(in_features=int(outch) * 8, out_features=self.hiddensize)
            elif "p4g" in str(actor_arc[-2]):
                convname, outch = actor_arc[-2].split("-")
                self.actorlinear1 = nn.Linear(in_features=int(outch) * 4, out_features=self.hiddensize)
            elif str(actor_arc[-2]).isnumeric():
                outch = int(actor_arc[-2])
                self.actorlinear1 = nn.Linear(in_features=int(outch), out_features=self.hiddensize)
            else:
                raise ValueError("Invalid GAP type!")

        elif actor_arc[-1]=="GPP":
            if "p4m" in str(actor_arc[-2]):
                convname, outch = actor_arc[-2].split("-")
                self.actorlinear1 = nn.Linear(in_features=self.G * 8 * int(outch) // self.actor_resfactor, out_features=self.hiddensize)
            elif "p4g" in str(actor_arc[-2]):
                convname, outch = actor_arc[-2].split("-")
                self.actorlinear1 = nn.Linear(in_features=self.G * 4 * int(outch) // self.actor_resfactor, out_features=self.hiddensize)
            elif str(actor_arc[-2]).isnumeric():
                outch = int(actor_arc[-2])
                self.actorlinear1 = nn.Linear(in_features=self.G * int(outch) // self.actor_resfactor,out_features=self.hiddensize)
            else:
                raise ValueError("Invalid layer type before GPP!")
        elif str(actor_arc[-1]).isnumeric():#last layer is numerical,no global pooling
            self.actorlinear1 = nn.Linear(in_features=self.G*self.G*int(actor_arc[-1])//self.actor_resfactor//self.actor_resfactor,out_features=self.hiddensize)
        elif "p4m" in str(actor_arc[-1]):
            convname, outch = str(actor_arc[-1]).split("-")
            self.actorlinear1 = nn.Linear(in_features=self.G * self.G * int(outch) * 8//self.actor_resfactor//self.actor_resfactor,out_features=self.hiddensize)
        elif "p4g" in str(actor_arc[-1]):
            convname, outch = str(actor_arc[-1]).split("-")
            self.actorlinear1 = nn.Linear(in_features=self.G * self.G * int(outch) * 4//self.actor_resfactor//self.actor_resfactor,
                                          out_features=self.hiddensize)
        elif str(actor_arc[-1])=="CP":
            #find nearest p4m or p4g
            pointer = len(actor_arc)
            while (pointer >= 0):
                pointer = pointer - 1
                if "p4m" in actor_arc[pointer] or "p4g" in actor_arc[pointer]:
                    break
            if pointer >= 0:
                convname, outch = actor_arc[pointer].split("-")
            self.actorlinear1 = nn.Linear(
                in_features=self.G * self.G * int(outch)// self.actor_resfactor // self.actor_resfactor,
                out_features=self.hiddensize)
        else:
            raise ValueError("Invalid last layer type!")
        # Compute first fc layer param for critic ==============================================================================================
        if critic_arc[-1] == "GAP":
            if "p4m" in str(critic_arc[-2]):
                convname, outch = critic_arc[-2].split("-")
                self.criticlinear1 = nn.Linear(in_features=int(outch) * 8, out_features=self.hiddensize)
            elif "p4g" in str(critic_arc[-2]):
                convname, outch = critic_arc[-2].split("-")
                self.criticlinear1 = nn.Linear(in_features=int(outch) * 4, out_features=self.hiddensize)
            elif str(critic_arc[-2]).isnumeric():
                outch = int(critic_arc[-2])
                self.criticlinear1 = nn.Linear(in_features=int(outch), out_features=self.hiddensize)
            elif str(critic_arc[-2]) == "CP":
                pointer = len(critic_arc)-2
                while (pointer >= 0):
                    pointer = pointer - 1
                    if "p4m" in critic_arc[pointer] or "p4g" in critic_arc[pointer]:
                        break
                if pointer >= 0:
                    convname, outch = critic_arc[pointer].split("-")
                self.criticlinear1 = nn.Linear(in_features=int(outch),out_features=self.hiddensize)
            else:
                raise ValueError("Invalid GAP type!")

        elif critic_arc[-1] == "GPP":
            if "p4m" in str(critic_arc[-2]):
                convname, outch = critic_arc[-2].split("-")
                self.criticlinear1 = nn.Linear(in_features=self.G * 8 * int(outch) // self.critic_resfactor,
                                              out_features=self.hiddensize)
            elif "p4g" in str(critic_arc[-2]):
                convname, outch = critic_arc[-2].split("-")
                self.criticlinear1 = nn.Linear(in_features=self.G * 4 * int(outch) // self.critic_resfactor,
                                              out_features=self.hiddensize)
            elif str(critic_arc[-2]).isnumeric():
                outch = int(critic_arc[-2])
                self.criticlinear1 = nn.Linear(in_features=self.G * int(outch) // self.critic_resfactor,
                                              out_features=self.hiddensize)
            elif str(critic_arc[-2]) == "CP":
                pointer = len(critic_arc)-2
                while (pointer >= 0):
                    pointer = pointer - 1
                    if "p4m" in critic_arc[pointer] or "p4g" in critic_arc[pointer]:
                        break
                if pointer >= 0:
                    convname, outch = critic_arc[pointer].split("-")
                self.criticlinear1 = nn.Linear(in_features=self.G * int(outch) // self.critic_resfactor,out_features=self.hiddensize)
            else:
                raise ValueError("Invalid layer type before GPP!")
        elif str(critic_arc[-1]).isnumeric():  # last layer is numerical,no global pooling
            self.criticlinear1 = nn.Linear(
                in_features=self.G * self.G * int(critic_arc[-1]) // self.critic_resfactor // self.critic_resfactor,
                out_features=self.hiddensize)
        elif "p4m" in str(critic_arc[-1]):
            convname, outch = str(critic_arc[-1]).split("-")
            self.criticlinear1 = nn.Linear(
                in_features=self.G * self.G * int(outch) * 8 // self.critic_resfactor // self.critic_resfactor,
                out_features=self.hiddensize)
        elif "p4g" in str(critic_arc[-1]):
            convname, outch = str(critic_arc[-1]).split("-")
            self.criticlinear1 = nn.Linear(
                in_features=self.G * self.G * int(outch) * 4 // self.critic_resfactor // self.critic_resfactor,
                out_features=self.hiddensize)
        elif str(critic_arc[-1])=="CP":
            #find nearest p4m or p4g
            pointer = len(critic_arc)
            while (pointer >= 0):
                pointer = pointer - 1
                if "p4m" in critic_arc[pointer] or "p4g" in critic_arc[pointer]:
                    break
            if pointer >= 0:
                convname, outch = critic_arc[pointer].split("-")
            self.criticlinear1 = nn.Linear(
                in_features=self.G * self.G * int(outch)// self.critic_resfactor // self.critic_resfactor,
                out_features=self.hiddensize)
        else:
            raise ValueError("Invalid last layer type!")
        self.actorlinear2 = nn.Linear(in_features=self.hiddensize,out_features=self.actor_outsize**2)
        self.criticlinear2 = nn.Linear(in_features=self.hiddensize, out_features=self.actor_outsize ** 2)
        self.criticlinear3 = nn.Linear(in_features=self.actor_outsize ** 2,out_features=1)
        self.hooker = {}
        self.layer_features = {}
        if hasattr(global_config, "visualize_feature") and global_config.visualize_feature:
            #if save features by visualize_feature
            for name, layer in self.named_modules():
                #print(name,type(layer),"\n")
                #if isinstance(layer, torch.nn.Conv2d):
                if isinstance(layer,P4ConvZ2) or isinstance(layer,P4ConvP4):
                    print("hook layer:",name,type(layer),"\n")
                    # if 'layer' in name or 'conv' in name:
                    # if self.is_hook_layer(layer, name, self.hook_instance_type):
                    self.hooker[name] = layer.register_forward_hook(self.hook_fn)

    def forward(self,inputs):
        self.layer_features = {}
        if self.Shareconv:
            x = self.main(inputs)
            actor_x =x
            critic_x = self.critic_GP(x)
        else:
            actor_x = self.actor_main(inputs)
            critic_x = self.critic_main(inputs)
        actor_x = actor_x.view(actor_x.shape[0], -1)
        actor_x = self.actorlinear1(actor_x)
        actor_x = self.actorlinear2(actor_x)
        critic_x = critic_x.view(critic_x.shape[0], -1)
        critic_x = self.criticlinear1(critic_x)
        critic_x = self.criticlinear2(critic_x)
        critic_x = self.criticlinear3(critic_x)
        return actor_x, critic_x


    def make_layers(self, arc_cfg, in_channels,cyclicpooltype="max"):
        layers = []
        arc_cfg = arc_cfg

        for index, v in enumerate(arc_cfg):
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif v=="PP":#plane group pooling
                layers +=[PlaneGroupPoolinglayer(kernelsize=2,stride=2,pad=0)]
            elif v =="CP":#cyclic pooling
                layers +=[CyclicGroupPoolinglayer(type=cyclicpooltype)]
            elif v=="BP":#blur pooling
                #get num of out channels of prev layer; prev layer should be p4mz2 or p4mp4m
                #find the latest p4m layer, getting its out channels
                pointer = index
                while(pointer>=0):
                    pointer = pointer-1
                    if "p4m" in arc_cfg[pointer] or "p4g" in arc_cfg[pointer]:
                        break
                if pointer>=0:
                    convname, outch = arc_cfg[pointer].split("-")
                if "p4m" in arc_cfg[index-1]:#previous layer is p4m
                    layers += [PlaneGroupPoolinglayer(kernelsize=2, stride=1, pad=0),BlurPoolp4m(channels=int(outch),stride=2,outchfactor=8,filt_size=self.blurpool_filtersize)]
                elif "p4g" in arc_cfg[index-1]:#previous layer is p4g
                    layers += [PlaneGroupPoolinglayer(kernelsize=2, stride=1, pad=0),BlurPoolp4m(channels=int(outch),stride=2,outchfactor=4,filt_size=self.blurpool_filtersize)]
                else:# previous layer is CP
                    layers += [PlaneGroupPoolinglayer(kernelsize=2, stride=1, pad=0),BlurPoolp4m(channels=int(outch), stride=2, outchfactor=1,filt_size=self.blurpool_filtersize)]
            elif v=="GAP":
                layers += [AdaptiveAvgPool2dp4()]
            elif v=="GPP":
                layers += [polarpoollayer(type=self.polarpooltype)]
            elif "p4m" in str(v):
                convname,outch = v.split("-")
                if convname=="p4mz2":
                    conv2dp4 = P4MConvZ2(in_channels, int(outch), kernel_size=3, padding=1)#P4MConvZ2 produces [b,c,h,w]
                elif convname=="p4mp4m":
                    conv2dp4 = P4MConvP4M(in_channels, int(outch), kernel_size=3, padding=1)#P4MConvZ2 produces [b,c,8,h,w]
                else:
                    raise ValueError("Invalid p4 type1")
                layers += [conv2dp4,nn.ReLU(inplace=True)]#according to MyANSnet,every conv is followed by a relu
                # if index == len(arc_cfg) - 1:  # the last conv layer is not followed by relu and batchnorm
                #     layers += [conv2dp4]
                # else:
                #     layers += [conv2dp4, Batch2Dp4layer(inchannels=int(outch), type=self.batch_type,outfactor=8),
                #                nn.ReLU(inplace=True)]
                in_channels = int(outch)
            elif "p4g" in str(v):
                convname,outch = v.split("-")
                if convname=="p4gz2":
                    conv2dp4 = P4ConvZ2(in_channels, int(outch), kernel_size=3, padding=1)#P4MConvZ2 produces [b,c,h,w]
                elif convname=="p4gp4g":
                    conv2dp4 = P4ConvP4(in_channels, int(outch), kernel_size=3, padding=1)#P4MConvZ2 produces [b,c,8,h,w]
                else:
                    raise ValueError("Invalid p4 type1")
                layers += [conv2dp4, nn.ReLU(inplace=True)]  # according to MyANSnet,every conv is followed by a relu
                # if index == len(arc_cfg) - 1:  # the last conv layer is not followed by relu and batchnorm
                #     layers += [conv2dp4]
                # else:
                #     layers += [conv2dp4, Batch2Dp4layer(inchannels=int(outch), type=self.batch_type,outfactor=4),
                #                nn.ReLU(inplace=True)]
                in_channels = int(outch)
            else: #normal conv
                conv2d = nn.Conv2d(in_channels, int(v), kernel_size=3, padding=1)
                layers += [conv2d,nn.ReLU(inplace=True)]# according to MyANSnet,every conv is followed by a relu
                # if index == len(arc_cfg) - 1:  # the last conv layer is not followed by relu and batchnorm
                #     layers += [conv2d]
                # else:
                #    layers += [conv2d, nn.BatchNorm2d(int(v)), nn.ReLU(inplace=True)]
                in_channels = int(v)
        return nn.Sequential(*layers)
    def hook_fn(self, module, input, output):
        # image_grid = torchvision.utils.make_grid(tensor=output[0].cpu().unsqueeze(dim=1),normalize=True,scale_each=True,nrow=output.shape[1]//16,padding=2)
        # torchvision.transforms.ToPILImage()(image_grid).show()
        # find the name of the current module
        for name, layer in self.named_modules():
            current_name = name
            # print(current_name)
            # if isinstance(layer,torch.nn.Conv2d) and id(layer)==id(module) and name in self.save_featurename_list:
            # if id(layer) == id(module) and name in self.save_featurename_list:
            if id(layer) == id(module):
                self.layer_features[current_name] = output
                # print(current_name,"min:",torch.min(output[-1,-1]).cpu().data.numpy(),"\n")
                # print(current_name, "max:", torch.max(output[-1, -1]).cpu().data.numpy())
                # print(current_name,output[-1,-1].cpu().data.numpy())
                break
class MyANSnet(nn.Module):
    def __init__(self, input_shape, hidden_size=512,actor_outsize=16,actor_out_submean=False,acotor_out_softmax=True):
        # actor_out_submean: actor_out= actor_out - mean(actor_out) if true
        # actor_out_softmax: actor_out = softmax(acotr_out) if true
        super(MyANSnet,self).__init__()
        conv_out_size = int(input_shape[0] / 8. * input_shape[1] / 8.)# if input is 240, output is 15
        self.G = int(input_shape[0])
        self.actor_outsize = actor_outsize # out 16*16
        self.actor_out_submean = actor_out_submean
        self.acotor_out_softmax = acotor_out_softmax
        self.main = nn.Sequential(
            #nn.MaxPool2d(2),
            nn.Conv2d(input_shape[2], 32, 3, stride=1, padding=1),# out (G,G,32), 5=local map 4+local frontier mask
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, stride=1, padding=1),# (G,G,64)
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, stride=1, padding=1),#(G,G,128)
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(128, 64, 3, stride=1, padding=1), #(G,G,64)
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=1, padding=1), #(G,G,32)
            nn.ReLU(),
            Flatten()
        )
        self.actor_linear1 = nn.Linear(conv_out_size * 32, hidden_size)
        self.actor_linear2 = nn.Linear(hidden_size, actor_outsize*actor_outsize)# out 256

        self.critic_linear1 = nn.Linear(conv_out_size * 32, hidden_size)
        self.critic_linear2 = nn.Linear(hidden_size, actor_outsize*actor_outsize)

        self.critic_linear3 = nn.Linear(actor_outsize*actor_outsize, 1)
        self.masksoftmax = MaskedSoftmax()
        #hook for visualizing features
        self.hooker = {}
        self.layer_features = {}
        for name, layer in self.named_modules():
            if isinstance(layer, torch.nn.Conv2d):
            # if 'layer' in name or 'conv' in name:
            #if self.is_hook_layer(layer, name, self.hook_instance_type):
                self.hooker[name] = layer.register_forward_hook(self.hook_fn)
    def hook_fn(self,module, input, output):
        #image_grid = torchvision.utils.make_grid(tensor=output[0].cpu().unsqueeze(dim=1),normalize=True,scale_each=True,nrow=output.shape[1]//16,padding=2)
        #torchvision.transforms.ToPILImage()(image_grid).show()
        #find the name of the current module
        for name, layer in self.named_modules():
            current_name = name
            #print(current_name)
            #if isinstance(layer,torch.nn.Conv2d) and id(layer)==id(module) and name in self.save_featurename_list:
            #if id(layer) == id(module) and name in self.save_featurename_list:
            if id(layer) == id(module):
                self.layer_features[current_name]=output
                break
    def forward(self, inputs):
        #input [b,5,G,G] , 5th channel is frontier_mask; currently local map
        #frontier_mask = inputs[:,4,:,:].unsqueeze(dim=1)#expect [b,1,G,G]
        rand_name=str(np.random.randint(1,30,1,np.int)[0])
        self.layer_features = {}
        x = self.main(inputs)
        actor_x = nn.ReLU()(self.actor_linear1(x))
        actor_x = nn.ReLU()(self.actor_linear2(actor_x))#expected 256

        actor_x = actor_x.view(-1, self.actor_outsize*self.actor_outsize)  # expect [b,16*16]
        # actor_x = actor_x.view(-1, self.actor_outsize, self.actor_outsize)# expect [b,16,16]
        # actor_x = F.interpolate(actor_x.unsqueeze(dim=1), size=(self.G, self.G),
        #                         mode='nearest')  # rescale, expect [b,1,G,G]
        # #actor_x = actor_x+frontier_mask
        # #actor_x = torch.mul(actor_x, frontier_mask)  # elementwise multiply, expect [b,1,G,G]
        #
        #
        # # actor_x_pil_mul = torchvision.transforms.ToPILImage()(actor_x[0][0].cpu())
        # # actor_x_pil_mul.save("images_output/actor_aftermul_"+rand_name+".png")
        # # frontier_mask_pil = torchvision.transforms.ToPILImage()(frontier_mask[0][0].cpu())
        # # frontier_mask_pil.save("images_output/actor_frontier_mask_"+rand_name+".png")
        #
        # if self.actor_out_submean:
        #     actor_x = actor_x.view(-1,self.G*self.G)-torch.mean(actor_x.view(-1,self.G*self.G),dim=-1,keepdim=True)# expect [b,G*G]
        # if self.acotor_out_softmax: #if true, [0,1], using masked softmax
        #     #actor_x=actor_x.softmax(dim=-1)# expect [b,G*G]
        #     actor_x = self.masked_softmax(vector =actor_x.view(-1,self.G*self.G),mask=frontier_mask.view(-1,self.G*self.G),memory_efficient=True)
        #
        #     #actor_x = self.masksoftmax(x=actor_x.view(-1,self.G*self.G),mask=frontier_mask.view(-1,self.G*self.G))#casue problem when mask is all 0
        # actor_x = actor_x.view(-1, self.G*self.G)  # expect out [b,G,G]
        # ind = (frontier_mask[-1] != 0)
        # if ind.any()==False:
        #     print("All elements in frontier_mask[-1] is 0!")
        #print("frontier_mask.shape:",frontier_mask.shape)
        #actor_x = actor_x.view(-1,self.G,self.G)#expect out [b,G,G]
        #actor_x_pil30 = torchvision.transforms.ToPILImage()(actor_x[0].cpu())

        #actor_x = actor_x.squeeze(dim=1)#expect [b,G,G]
        #actor_x_pil240 = torchvision.transforms.ToPILImage()(actor_x[0].cpu())

        #actor_x_pil30.save("images_output/actorout_30.png")
        #actor_x_pil240.save("images_output/actor_aftersoftmax_"+rand_name+".png")

        #actor_x = actor_x.view(-1,self.G*self.G)#expect [b,G*G],range 0~1

        critic_x = self.critic_linear1(x)
        critic_x = self.critic_linear2(critic_x)
        #critic = self.critic_linear3(critic_x).squeeze(-1) # expected 1
        critic = self.critic_linear3(critic_x)  # expected 1
        return actor_x,critic

    def masked_softmax(self,vector: torch.Tensor,
                       mask: torch.Tensor,
                       dim: int = -1,
                       memory_efficient: bool = False,
                       mask_fill_value: float = -1e32) -> torch.Tensor:
        """
        https://github.com/allenai/allennlp/blob/b6cc9d39651273e8ec2a7e334908ffa9de5c2026/allennlp/nn/util.py#L272-L303
        my Problem: sum is not 1
        ``torch.nn.functional.softmax(vector)`` does not work if some elements of ``vector`` should be
        masked.  This performs a softmax on just the non-masked portions of ``vector``.  Passing
        ``None`` in for the mask is also acceptable; you'll just get a regular softmax.
        ``vector`` can have an arbitrary number of dimensions; the only requirement is that ``mask`` is
        broadcastable to ``vector's`` shape.  If ``mask`` has fewer dimensions than ``vector``, we will
        unsqueeze on dimension 1 until they match.  If you need a different unsqueezing of your mask,
        do it yourself before passing the mask into this function.
        If ``memory_efficient`` is set to true, we will simply use a very large negative number for those
        masked positions so that the probabilities of those positions would be approximately 0.
        This is not accurate in math, but works for most cases and consumes less memory.
        In the case that the input vector is completely masked and ``memory_efficient`` is false, this function
        returns an array of ``0.0``. This behavior may cause ``NaN`` if this is used as the last layer of
        a model that uses categorical cross-entropy loss. Instead, if ``memory_efficient`` is true, this function
        will treat every element as equal, and do softmax over equal numbers.
        """
        if mask is None:
            result = torch.nn.functional.softmax(vector, dim=dim)
        else:
            mask = mask.float()
            while mask.dim() < vector.dim():
                mask = mask.unsqueeze(1)
            if not memory_efficient:
                # To limit numerical errors from large vector elements outside the mask, we zero these out.
                result = torch.nn.functional.softmax(vector * mask, dim=dim)
                result = result * mask
                result = result / (result.sum(dim=dim, keepdim=True) + 1e-13)
            else:
                masked_vector = vector.masked_fill((1 - mask).bool(), mask_fill_value)
                result = torch.nn.functional.softmax(masked_vector, dim=dim)
        return result

class GlobalPolicy(nn.Module):
    def __init__(self, config,show_animation=False):
        super().__init__()

        self.show_animation=show_animation

        self.G = config.map_size
        self.config = config #me
        #---------------------------------------------------------------------------------------------------
        #me: the following is self-design network, similar to ANS  implementation
        assert hasattr(self.config, 'myego_localmap'), "myego_localmap is no defined in the configuration!"
        assert hasattr(self.config, 'myego_globalmap'), "myego_globalmap is no defined in the configuration!"
        #assert hasattr(self.config, 'myego_globalmap_type'), "myego_globalmap_type is no defined in the configuration!"
        assert config.myego_localmap == config.myego_globalmap, "Only support same value for myego_localmap and myego_globalmap"
        if config.myuse_globalmap and config.myuse_localmap:# both golbal local map
            if config.myego_localmap or config.myego_globalmap:
                input_shape = [self.G, self.G, 6]
            else:
                input_shape = [self.G, self.G, 7] #3 allo local + 4 allo global
        elif ( not config.myuse_globalmap) and config.myuse_localmap: #only input  local map
            input_shape = [self.G, self.G, 3]
        elif config.myuse_globalmap and (not config.myuse_localmap):#only input global map
            if config.myego_globalmap:
                input_shape = [self.G, self.G, 3]
            else:
                input_shape = [self.G, self.G, 4]
        else:
            #input_shape = [self.G, self.G, 7]
            raise Exception("please define myuse_globalmap and myuse_localmap!")
        if hasattr(config, "net_type") and config.net_type == "ansnet":
            self.actor_critic = MyANSnet(input_shape=input_shape, hidden_size=512, actor_out_submean=False,
                                         acotor_out_softmax=False)  # [w,h,channel]]. Here only try Local map
        elif hasattr(config, "net_type") and config.net_type == "ansnetexactp4":
            self.actor_critic = Ansexactp4flex(input_shape=input_shape, global_config=config)
        else:
            raise ValueError("Invalid net_type!")
        if config.use_data_parallel:
            #print("config.use_data_parallel:",config.use_data_parallel)
            self.actor_critic = nn.DataParallel(self.actor_critic, device_ids=config.gpu_ids, output_device=config.gpu_ids[0])


    def forward(self, inputs):
        raise NotImplementedError

    def get_rot_trans_mat(self,local_cor, theta, agent_pos, G,translate=False):
        #local_cor is in local map coordinate,[b,2]
        #theta: [b,1]
        #agent_pos:[b,2]
        #res is in global coordinate
        #translate : Translate cor if True
        #rotation:rotate cor if True

        # [local_col,local_row] = [local_cor]
        #local_cor = np.array(local_cor)
        b = theta.shape[0]  # batch size
        inv_rot_mat_batch = torch.zeros([b, 2, 2], dtype=torch.float32).to(theta.device)
        for bi in range(b):
            inv_rot_mat = torch.tensor([[torch.cos(theta[bi]), torch.sin(theta[bi])],
                                [-torch.sin(theta[bi]), torch.cos(theta[bi])]]).to(theta.device)
            inv_rot_mat_batch[bi] = inv_rot_mat.unsqueeze(dim=0)
        if translate:
            t = (agent_pos) - G // 2
            tmp = local_cor + t#[b,2]
        else:
            tmp = local_cor

        res = torch.matmul(inv_rot_mat_batch,tmp.unsqueeze(dim=-1))# [b,2,1]=[b,2,2]*[b,2,1]
        return res.squeeze(-1)[:,0],res.squeeze(-1)[:,1]#res.squeeze(-1):[b,2]; return col ,row

    def get_rot_trans_mat(self,theta,col,row):
        #theta = torch.tensor(theta)
        return torch.tensor([[torch.cos(theta), -torch.sin(theta), col],
                             [torch.sin(theta), torch.cos(theta), row]]).to(theta.device)

    def rot_trans_img(self,x, theta, agent_pos,dtype):
        b=x.shape[0]#batch size
        rot_mat_batch = torch.zeros([b,2,3],dtype=torch.float32).to(x.device)
        #print("theta shape:",theta.shape)
        for i in range(b):#compute rot_mat for all samples in the batch
            #rot_mat = self.get_rot_mat(theta[i])[None, ...].type(dtype).repeat(x.shape[0], 1, 1)
            rot_mat = self.get_rot_trans_mat(theta=theta[i,0],col=agent_pos[i,0],row=agent_pos[i,1]).unsqueeze(dim=0)
            rot_mat_batch[i] = rot_mat
        grid = F.affine_grid(rot_mat_batch, x.size()).type(dtype)
        x = F.grid_sample(x, grid)
        return x
    def mywrite_feature_tb(self,writer,feature_dict,step):
        #write feature maps to tb file, by writer
        #
        for i,j in feature_dict.items():
            if len(j.shape)<4 :#ndarray
                writer.add_image(tag=i,img_tensor=j,global_step=step)
            elif len(j.shape)==4:#tensor
                j_grid = torchvision.utils.make_grid(j[0].unsqueeze(dim=1), nrow=16,normalize=True)#j [b,c,w,h]
                writer.add_image(tag=i, img_tensor=j_grid, global_step=step)
            elif len(j.shape)==5:#output tensor of group conv
                #print("j shape:",j.shape,"j[0] shape",j[0].shape)
                feature_reshape=j[0].view(j[0].shape[0]*j[0].shape[1],j[0].shape[2],j[0].shape[3])
                #print("feature_reshape:", feature_reshape.shape)
                j_grid = torchvision.utils.make_grid(feature_reshape.unsqueeze(dim=1), nrow=4,normalize=True)#j [b,c,w,h]
                writer.add_image(tag=i, img_tensor=j_grid, global_step=step)
            else:
                raise ValueError('dim error.')
    def myrawmap2rgbmap(self,rawmap):
        #convert raw map to rgb map
        #raw map [b,2,w,h]
        #0 channel is occupancy prob map
        #1 channel is explored map
        #return a rgb map
        maps_dict = {}
        map_states=rawmap
        b,_,h,w = rawmap.shape
        maps_dict["explored_map"] = (map_states[:, 1] > 0.5).float()  # (bs, M, M)
        maps_dict["occ_space_map"] = (map_states[:, 0] > 0.5).float() * maps_dict[
            "explored_map"]  # (bs, M, M),value 1 is occupancy
        maps_dict["free_space_map"] = (map_states[:, 0] <= 0.5).float() * maps_dict[
            "explored_map"]  # value 1 is free space
        # convert to numpy cpu
        for key, value in maps_dict.items():
            maps_dict[key] = value.cpu().data.numpy()

        map_frontier_format = np.zeros([b, 3, h, w], dtype=np.uint8)  # 0~255

        # unknown region
        map_frontier_format[:, 0, :, :] += np.uint8((1 - maps_dict["explored_map"]) * 255)
        map_frontier_format[:, 1, :, :] += np.uint8((1 - maps_dict["explored_map"]) * 255)
        map_frontier_format[:, 2, :, :] += np.uint8((1 - maps_dict["explored_map"]) * 255)
        # free space
        map_frontier_format[:, 1, :, :] += np.uint8(maps_dict["free_space_map"] * 255)
        # map_frontier_format[:, 2, :, :] += np.uint8(maps_dict["free_space_map"] * 255)
        # occupied region
        map_frontier_format[:, 2, :, :] += np.uint8(maps_dict["occ_space_map"] * 255)
        return map_frontier_format

    def _get_h12(self, inputs,writer_dict=None):
        x = inputs["pose_in_map_at_t"]
        # me [b,4,M,M],ch 0:occupied region;ch 1:explored region;
        # ch 3:current position; ch 4 visited locations
        h = inputs["map_at_t"]
        theta = inputs["theta_pose_at_t"]#size
        #rotate original map if use ego local map or ego global map======================================
        #agent_before_rot = visualization.draw_triangle(h[0, 2, :].cpu().data.numpy(), position=x[0], theta=theta[0],
        #                                               color=(
        #                                               255, 0, 0))  # get (961,961) #draw agent on the 2nd channel
        #h[0, 0, :] = torch.tensor(agent_before_rot)
        if self.config.myego_localmap or self.config.myego_globalmap:
            x_norm = (x-h.shape[2]//2)*2/h.shape[2] # map [0,width-1] to [-1,1]
            #h_ego centerd at agent position, agent faces upward
            h_ego = self.rot_trans_img(h, theta,agent_pos=x_norm,
                                 dtype=torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor)#[b,4,M,M]
            # torchvision.transforms.ToPILImage()(h_rot[0,2,:].cpu().data.numpy()).convert('RGB').save("images_output/rotate.jpg")
        if self.config.myego_localmap:#me use ego local map as input
            #since cetern at agent, crop center
            h_1 = crop_map(h_ego[:,[0,1,-1],:], torch.zeros_like(x[:, :2]).to(x.device)+h.shape[2]//2, self.G)#h_ego.transpose(2,3)[0:-1] : 3 channels, current position channel is ruled out
        else:
            h_1 = crop_map(h[:,[0,1,-1],:], x[:, :2], self.G)#[b,3,G,G]
        if self.config.myego_globalmap:
            #assert hasattr(self.config, 'myego_globalmap_type'), "myego_globalmap_type is no defined in the configuration!"
            #if self.config.myego_globalmap_type=="egocenter":#centered at agent
            h_ego = crop_map(h_ego[:,[0,1,-1],:], torch.zeros_like(x[:, :2]).to(x.device)+h.shape[2]//2,
                                 h_ego.shape[-1])#[b,3,M,M] in egocenter, in globalcenter the size is [b,4,G,G]
            h_2 = F.adaptive_max_pool2d(h_ego, (self.G, self.G))  # [b,3 or 4,G,G] 3 for egocenter, 4 for globalcente
        else:
            h_2 = F.adaptive_max_pool2d(h, (self.G, self.G))  # [b,4,G,G]
        # h_1 = crop_map(h, x[:, :2], self.G)
        # h_2 = F.adaptive_max_pool2d(h, (self.G, self.G))#[b,4,G,G]

        if self.config.fbe == False and self.config.combine_fbe == True:
            f_mask = torch.unsqueeze(inputs["frontier_mask_at_t"], dim=1)#[b,1,M,M]
            #print("f_mask shape:",f_mask.shape)
            #print("inputs shape:",inputs["pose_in_map_at_t"][:, :2].shape)
            # f_mask = torch.stack([f_mask,f_mask,f_mask],dim=1)
            #f_mask = torch.unsqueeze(f_mask, dim=0)
            # f_mask_re = F.interpolate(f_mask,size=(240,240),mode='nearest')
            f_mask_re = crop_map(f_mask, x[:, :2], self.G)#[b,1,240,240]
            #f_mask_re = f_mask_re.view(f_mask_re.shape[0], -1)
            # action_logits_tmp = (action_logits - action_logits.min()) / (action_logits.max() - action_logits.min())  # minmax norm
            # me mask action_logits by f_mask_re;
            # sometimes action_logits is negative at frontiers,so abs is necessary,or prob is low at frontier, other region is high
            #action_logits = torch.mul(f_mask_re, torch.abs(action_logits))
        #h_12=torch.cat([h_1,f_mask_re],dim=1)#me: only local map related ino is used; global map try later
        elif self.config.myuse_globalmap and self.config.myuse_localmap:
            h_12 = torch.cat([h_1, h_2], dim=1)#6 for egocenter global map , 7 for allocenter gobal map
        elif self.config.myuse_globalmap and (not self.config.myuse_localmap):
            h_12 = h_2 #3 channels
        elif (not self.config.myuse_globalmap) and self.config.myuse_localmap:
            h_12 = h_1 # 4 channels
        else:
            raise Exception("h_12 not define!")

        #visualize h1 and h2 in tb file; only used when sample global target
        if writer_dict is not None:# in training writer_dict will be set None
            h_1_rgb_np = self.myrawmap2rgbmap(rawmap=h_1[:,0:2,:])
            h_2_rgb_np = self.myrawmap2rgbmap(rawmap=h_2[:, 0:2, :])#[b,3,h,w]
            _,sceneid = os.path.split(writer_dict["sceneid"])
            feature_dict={sceneid[0:-4]+"-epid_"+str(writer_dict["epid"])+"/_h1_finemap":h_1_rgb_np[0],
                          sceneid[0:-4]+"-epid_"+str(writer_dict["epid"])+"/_h2_coarsemap":h_2_rgb_np[0]}
            self.mywrite_feature_tb(feature_dict=feature_dict,writer=writer_dict["writer"],step=writer_dict["ep_step"])
        return h_12

    def act(self, inputs, rnn_hxs, prev_actions, masks, deterministic=False,writer_dict=None):
        """
        Note: inputs['pose_in_map_at_t'] must obey the following conventions:
              origin at top-left, downward Y and rightward X in the map coordinate system.
        """
        M = inputs["map_at_t"].shape[2]
        h_12 = self._get_h12(inputs,writer_dict=writer_dict)
        if writer_dict is None:#means in train phase;then remove hookers, do not record features
            for keyi,vali in self.actor_critic.hooker.items():
                self.actor_critic.hooker[keyi].remove()
        # action_logits expect [b,G*G],range 0~1
        # 210424 action_logits expect [b,actor_outputsieze*actor_outputsieze=16*16]
        action_logits,value = self.actor_critic(h_12)
        # save features to tb in eval phase ============================================================
        if writer_dict is not None:
            ##first add prefix of scene and epid
            for keyi, vali in self.actor_critic.layer_features.items():
                _, sceneid = os.path.split(writer_dict["sceneid"])
                newkey = sceneid[0:-4]+"-epid_"+str(writer_dict["epid"])+"/"+keyi
                self.actor_critic.layer_features[newkey] = self.actor_critic.layer_features.pop(keyi)
            ##then save
            self.mywrite_feature_tb(feature_dict=self.actor_critic.layer_features,
                                    writer=writer_dict["writer"],step=writer_dict["ep_step"])#write layer features to tb
        #================================================================================================
        # if self.config.fbe == False and self.config.combine_fbe == True:
        #     f_mask = torch.unsqueeze(inputs["frontier_mask"], dim=1)
        #     #print("f_mask shape:",f_mask.shape)
        #     #print("inputs shape:",inputs["pose_in_map_at_t"][:, :2].shape)
        #     # f_mask = torch.stack([f_mask,f_mask,f_mask],dim=1)
        #     #f_mask = torch.unsqueeze(f_mask, dim=0)
        #     # f_mask_re = F.interpolate(f_mask,size=(240,240),mode='nearest')
        #     f_mask_re = crop_map(f_mask, inputs["pose_in_map_at_t"][:, :2], self.G)
        #     f_mask_re = f_mask_re.view(f_mask_re.shape[0], -1)
        #     # action_logits_tmp = (action_logits - action_logits.min()) / (action_logits.max() - action_logits.min())  # minmax norm
        #     # me mask action_logits by f_mask_re;
        #     # sometimes action_logits is negative at frontiers,so abs is necessary,or prob is low at frontier, other region is high
        #     #action_logits = torch.mul(f_mask_re, torch.abs(action_logits))
        dist = FixedCategorical(logits=action_logits)
        # visualize g-map
        if writer_dict is not None:
            _, sceneid = os.path.split(writer_dict["sceneid"])
            #print("probs:",dist.probs.cpu().data.numpy())
            g_map = dist.probs.reshape([-1, 1, self.actor_critic.actor_outsize, self.actor_critic.actor_outsize])
            g_map = F.interpolate(g_map, size=(self.G, self.G), mode='nearest')[0]
            feature_dict = {sceneid[0:-4] + "-epid_" + str(writer_dict["epid"]) + "/g_map": g_map}
            self.mywrite_feature_tb(feature_dict=feature_dict, writer=writer_dict["writer"],
                                    step=writer_dict["ep_step"])
        #print("in act \n")
        #dist = FixedCategorical(probs=action_logits)# action_logits must be with size [b,G*G]
        # actionlogits_np = action_logits.cpu().data.numpy()
        # dist_probs_np = dist.probs.cpu().data.numpy()
        # ind = (actionlogits_np != 0)
        # ind2= (actionlogits_np == 0)
        # print("action-logits 1:",actionlogits_np[-1,20:30],
        #       "dist.probs 1:", dist_probs_np[-1, 20:30],
        #       "non 0 mean:",actionlogits_np[ind].mean(),
        #       #"0 element mean:", actionlogits_np[ind2].mean(),
        #       "\n entropy:",dist.entropy().cpu().data.numpy())
        #value = self.critic(h_12)

        #me save tmp map for debug---------------------------------------------------------
        if self.show_animation:
            tmp_combine_pil=Image.new('RGB',(self.G*4,self.G))
            #local map :
            tmp_local_map = h_12[:,0:3,:].cpu().data.numpy()#(b,2,G,G)
            tmp_global_map = h_12[:, 3:5, :].cpu().data.numpy()
            b, w, h0 = h_12.shape[0], h_12.shape[2], h_12.shape[3]
            for tmpi,tmp_local_map in enumerate([tmp_local_map,tmp_global_map]):
                local_maps_dict = {}
                local_maps_dict["explored_map"] = (tmp_local_map[:, 1] > 0.5)  # (bs, M, M)
                local_maps_dict["occ_space_map"] = (tmp_local_map[:, 0] > 0.5) * local_maps_dict[
                    "explored_map"]  # (bs, M, M),value 1 is occupancy
                local_maps_dict["free_space_map"] = (tmp_local_map[:, 0] <= 0.5) * local_maps_dict["explored_map"]  # value 1 is free space
                # convert to numpy cpu
                local_map_frontier_format = np.zeros([b, 3, h0, w], dtype=np.uint8)  # 0~255
                # unknown region
                local_map_frontier_format[:, 0, :, :] += np.uint8((1 - local_maps_dict["explored_map"]) * 255)
                local_map_frontier_format[:, 1, :, :] += np.uint8((1 - local_maps_dict["explored_map"]) * 255)
                local_map_frontier_format[:, 2, :, :] += np.uint8((1 - local_maps_dict["explored_map"]) * 255)
                # free space
                local_map_frontier_format[:, 1, :, :] += np.uint8(local_maps_dict["free_space_map"] * 255)
                # map_frontier_format[:, 2, :, :] += np.uint8(maps_dict["free_space_map"] * 255)
                # occupied region
                local_map_frontier_format[:, 2, :, :] += np.uint8(local_maps_dict["occ_space_map"] * 255)
                img_pil = Image.fromarray((local_map_frontier_format.transpose((0, 2, 3, 1))[0]))  # to [h,w,c]
                tmp_combine_pil.paste(img_pil, (tmpi * self.G, 0))
            tmp_local_visited_map = Image.fromarray(np.uint8(h_12.cpu().data.numpy()[0,2,:]*255)).convert('RGB')
            #global map:
            tmp_global_visitedorcurpos_map = Image.fromarray(np.uint8(h_12.cpu().data.numpy()[0,-1,:]*255)).convert('RGB')# visited map if egocenter, cur_pos if allocenter
            tmp_combine_pil.paste(tmp_local_visited_map, (2 * self.G, 0))
            tmp_combine_pil.paste(tmp_global_visitedorcurpos_map, (3 * self.G, 0))
            tmp_combine_pil.save("images_output/allomap-" + str(np.random.randint(1, 50, 1, np.int)[0]) + ".png")
            #-------------------------------------------------------------------------------------------


        if deterministic:
            action = dist.mode()
            action_sample = dist.sample()#me
            action_determin=action#me
        else:
            action = dist.sample()
            action_determin = dist.mode()
            action_sample=action

        action_log_probs = dist.log_probs(action)

        # if self.show_animation:# me: only recommanded for evaluation;
        #     #action_logits_tmp = (dist.probs - dist.probs.min()) / (dist.probs.max() - dist.probs.min())  # minmax norm
        #     action_logits_tmp = (action_logits - action_logits.min()) / (action_logits.max() - action_logits.min())  # minmax norm
        #     action_logits_pil = torchvision.transforms.ToPILImage()(action_logits_tmp[0].reshape((self.G, self.G)).cpu())
        #     action_logits_pil = action_logits_pil.convert('RGB')
        #     draw = ImageDraw.Draw(action_logits_pil)#draw(col,row)
        #     #draw.point((5,200), 'blue')#see coordinate contract
        #     draw.ellipse((action_determin.cpu().data.numpy() % self.G - 3,
        #                   action_determin.cpu().data.numpy() / self.G - 3,
        #                   action_determin.cpu().data.numpy() % self.G +3,
        #                   action_determin.cpu().data.numpy() / self.G +3),'red')
        #     draw.ellipse((action_sample.cpu().data.numpy() % self.G - 3,
        #                   action_sample.cpu().data.numpy() / self.G - 3,
        #                   action_sample.cpu().data.numpy() % self.G + 3,
        #                   action_sample.cpu().data.numpy() / self.G + 3), 'green')
        #
        #     tmp_combine_pil.paste(action_logits_pil, (self.G * 2, 0))
        #     #tmp_combine_pil.save("images_output/hmap-Gmap-entro"+str(dist.entropy().cpu().data.numpy()[0])+str(np.random.randint(1,30,1,np.int)[0])+".png")


        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, prev_actions, masks):
        h_12 = self._get_h12(inputs)
        #value = self.critic(h_12)
        action_logits, value = self.actor_critic(h_12)# me gai
        return value

    def evaluate_actions(self, inputs, rnn_hxs, prev_actions, masks, action):
        h_12 = self._get_h12(inputs)
        action_logits,_ = self.actor_critic(h_12)
        dist = FixedCategorical(logits=action_logits)
        #dist = FixedCategorical(probs=action_logits)
        _,value = self.actor_critic(h_12)

        action_log_probs = dist.log_probs(action)

        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


class LocalPolicy(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.nactions = config.nactions
        self.hidden_size = config.hidden_size
        embedding_buckets = config.EMBEDDING_BUCKETS

        self.base = CNNBase(
            True,
            embedding_buckets,
            hidden_size=self.hidden_size,
            img_mean=config.NORMALIZATION.img_mean,
            img_std=config.NORMALIZATION.img_std,
            input_shape=config.image_scale_hw,
        )

        self.dist = Categorical(self.hidden_size, self.nactions)
        init_ = lambda m: init(
            m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0)
        )

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, prev_actions, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, prev_actions, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, prev_actions, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, prev_actions, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


class HeuristicLocalPolicy(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

    def forward(self, inputs, rnn_hxs, prev_actions, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, prev_actions, masks, deterministic=False):
        goal_xy = inputs["goal_at_t"]
        goal_phi = torch.atan2(goal_xy[:, 1], goal_xy[:, 0])

        turn_angle = math.radians(self.config.AGENT_DYNAMICS.turn_angle)
        fwd_action_flag = torch.abs(goal_phi) <= 0.9 * turn_angle
        turn_left_flag = ~fwd_action_flag & (goal_phi < 0)
        turn_right_flag = ~fwd_action_flag & (goal_phi > 0)

        action = torch.zeros_like(goal_xy)[:, 0:1]
        action[fwd_action_flag] = 0
        action[turn_left_flag] = 1
        action[turn_right_flag] = 2
        action = action.long()

        return None, action, None, rnn_hxs

    def load_state_dict(self, *args, **kwargs):
        pass


class Mapper(nn.Module):
    def __init__(self, config, projection_unit):
        super().__init__()
        self.config = config
        self.map_config = {"size": config.map_size, "scale": config.map_scale}
        V = self.map_config["size"]
        s = self.map_config["scale"]
        self.img_mean_t = rearrange(
            torch.Tensor(self.config.NORMALIZATION.img_mean), "c -> () c () ()"
        )
        self.img_std_t = rearrange(
            torch.Tensor(self.config.NORMALIZATION.img_std), "c -> () c () ()"
        )
        self.pose_estimator = PoseEstimator(
            V,
            self.config.pose_predictor_inputs,
            n_pose_layers=self.config.n_pose_layers,
            n_ensemble_layers=self.config.n_ensemble_layers,
            input_shape=self.config.image_scale_hw,
        )
        self.projection_unit = projection_unit
        if self.config.freeze_projection_unit:
            for p in self.projection_unit.parameters():
                p.requires_grad = False

        # Cache to store pre-computed information
        self._cache = {}

    def forward(self, x, masks=None):
        outputs = self.predict_deltas(x, masks=masks)
        mt_1 = x["map_at_t_1"]
        if masks is not None:
            mt_1 = mt_1 * masks.view(-1, 1, 1, 1)
        with torch.no_grad():
            mt = self._register_map(mt_1, outputs["pt"], outputs["xt_hat"])
        outputs["mt"] = mt

        return outputs

    def predict_deltas(self, x, masks=None):
        # Transpose multichannel inputs
        st_1 = process_image(x["rgb_at_t_1"], self.img_mean_t, self.img_std_t)
        dt_1 = transpose_image(x["depth_at_t_1"])
        ego_map_gt_at_t_1 = transpose_image(x["ego_map_gt_at_t_1"])
        st = process_image(x["rgb_at_t"], self.img_mean_t, self.img_std_t)
        dt = transpose_image(x["depth_at_t"])
        ego_map_gt_at_t = transpose_image(x["ego_map_gt_at_t"])
        # This happens only for a baseline
        if (
            "ego_map_gt_anticipated_at_t_1" in x
            and x["ego_map_gt_anticipated_at_t_1"] is not None
        ):
            ego_map_gt_anticipated_at_t_1 = transpose_image(
                x["ego_map_gt_anticipated_at_t_1"]
            )
            ego_map_gt_anticipated_at_t = transpose_image(
                x["ego_map_gt_anticipated_at_t"]
            )
        else:
            ego_map_gt_anticipated_at_t_1 = None
            ego_map_gt_anticipated_at_t = None
        # Compute past and current egocentric maps
        bs = st_1.size(0)
        pu_inputs_t_1 = {
            "rgb": st_1,
            "depth": dt_1,
            "ego_map_gt": ego_map_gt_at_t_1,
            "ego_map_gt_anticipated": ego_map_gt_anticipated_at_t_1,
        }
        pu_inputs_t = {
            "rgb": st,
            "depth": dt,
            "ego_map_gt": ego_map_gt_at_t,
            "ego_map_gt_anticipated": ego_map_gt_anticipated_at_t,
        }
        pu_inputs = self._safe_cat(pu_inputs_t_1, pu_inputs_t)
        pu_outputs = self.projection_unit(pu_inputs)
        pu_outputs_t = {k: v[bs:] for k, v in pu_outputs.items()}
        pt_1, pt = pu_outputs["occ_estimate"][:bs], pu_outputs["occ_estimate"][bs:]
        # Compute relative pose
        dx = subtract_pose(x["pose_at_t_1"], x["pose_at_t"])
        # Estimate pose
        dx_hat = dx
        xt_hat = x["pose_at_t"]
        all_pose_outputs = None
        if not self.config.ignore_pose_estimator:
            all_pose_outputs = {}
            pose_inputs = {}
            if "rgb" in self.config.pose_predictor_inputs:
                pose_inputs["rgb_t_1"] = st_1
                pose_inputs["rgb_t"] = st
            if "depth" in self.config.pose_predictor_inputs:
                pose_inputs["depth_t_1"] = dt_1
                pose_inputs["depth_t"] = dt
            if "ego_map" in self.config.pose_predictor_inputs:
                pose_inputs["ego_map_t_1"] = pt_1
                pose_inputs["ego_map_t"] = pt
            if self.config.detach_map:
                for k in pose_inputs.keys():
                    pose_inputs[k] = pose_inputs[k].detach()
            n_pose_inputs = self._transform_observations(pose_inputs, dx)
            pose_outputs = self.pose_estimator(n_pose_inputs)
            dx_hat = add_pose(dx, pose_outputs["pose"])
            all_pose_outputs["pose_outputs"] = pose_outputs
            # Estimate global pose
            xt_hat = add_pose(x["pose_hat_at_t_1"], dx_hat)
        # Zero out pose prediction based on the mask
        if masks is not None:
            xt_hat = xt_hat * masks
            dx_hat = dx_hat * masks
        outputs = {
            "pt": pt,
            "dx_hat": dx_hat,
            "xt_hat": xt_hat,
            "all_pu_outputs": pu_outputs_t,
            "all_pose_outputs": all_pose_outputs,
        }
        if "ego_map_hat" in pu_outputs_t:
            outputs["ego_map_hat_at_t"] = pu_outputs_t["ego_map_hat"]
        return outputs

    def _bottom_row_spatial_transform(self, p, dx, invert=False):
        """
        Inputs:
            p - (bs, 2, V, V) local map
            dx - (bs, 3) egocentric transformation --- (dx, dy, dtheta)

        NOTE: The agent stands at the central column of the last row in the
        ego-centric map and looks forward. But the rotation happens about the
        center of the map.  To handle this, first zero-pad pt_1 and then crop
        it after transforming.

        Conventions:
            The origin is at the bottom-center of the map.
            X is upward with agent's forward direction
            Y is rightward with agent's rightward direction
        """
        V = p.shape[2]
        p_pad = bottom_row_padding(p)
        p_trans_pad = self._spatial_transform(p_pad, dx, invert=invert)
        # Crop out the original part
        p_trans = bottom_row_cropping(p_trans_pad, V)

        return p_trans

    def _spatial_transform(self, p, dx, invert=False):
        """
        Applies the transformation dx to image p.
        Inputs:
            p - (bs, 2, H, W) map
            dx - (bs, 3) egocentric transformation --- (dx, dy, dtheta)

        Conventions:
            The origin is at the center of the map.
            X is upward with agent's forward direction
            Y is rightward with agent's rightward direction

        Note: These denote transforms in an agent's position. Not the image directly.
        For example, if an agent is moving upward, then the map will be moving downward.
        To disable this behavior, set invert=False.
        """
        s = self.map_config["scale"]
        # Convert dx to map image coordinate system with X as rightward and Y as downward
        dx_map = torch.stack(
            [(dx[:, 1] / s), -(dx[:, 0] / s), dx[:, 2]], dim=1
        )  # anti-clockwise rotation
        p_trans = spatial_transform_map(p, dx_map, invert=invert)

        return p_trans

    def _register_map(self, m, p, x):
        """
        Given the locally computed map, register it to the global map based
        on the current position.

        Inputs:
            m - (bs, F, M, M) global map
            p - (bs, F, V, V) local map
            x - (bs, 3) in global coordinates
        """
        V = self.map_config["size"]
        s = self.map_config["scale"]
        M = m.shape[2]
        Vby2 = (V - 1) // 2 if V % 2 == 1 else V // 2
        Mby2 = (M - 1) // 2 if M % 2 == 1 else M // 2
        # The agent stands at the bottom-center of the egomap and looks upward
        left_h_pad = Mby2 - V + 1
        right_h_pad = M - V - left_h_pad
        left_w_pad = Mby2 - Vby2
        right_w_pad = M - V - left_w_pad
        # Add zero padding to p so that it matches size of global map
        p_pad = F.pad(
            p, (left_w_pad, right_w_pad, left_h_pad, right_h_pad), "constant", 0
        )
        # Register the local map
        p_reg = self._spatial_transform(p_pad, x)
        # Aggregate
        m_updated = self._aggregate(m, p_reg)

        return m_updated

    def _aggregate(self, m, p_reg):
        """
        Inputs:
            m - (bs, 2, M, M) - global map
            p_reg - (bs, 2, M, M) - registered egomap
        """
        reg_type = self.config.registration_type
        beta = self.config.map_registration_momentum
        if reg_type == "max":
            m_updated = torch.max(m, p_reg)
        elif reg_type == "overwrite":
            # Overwrite only the currently explored regions
            mask = (p_reg[:, 1] > self.config.thresh_explored).float()
            mask = mask.unsqueeze(1)
            m_updated = m * (1 - mask) + p_reg * mask
        elif reg_type == "moving_average":
            mask_unexplored = (
                (p_reg[:, 1] <= self.config.thresh_explored).float().unsqueeze(1)
            )
            mask_unfilled = (m[:, 1] == 0).float().unsqueeze(1)
            m_ma = p_reg * (1 - beta) + m * beta
            m_updated = (
                m * mask_unexplored
                + m_ma * (1.0 - mask_unexplored) * (1.0 - mask_unfilled)
                + p_reg * (1.0 - mask_unexplored) * mask_unfilled
            )
        elif reg_type == "entropy_moving_average":
            explored_mask = (p_reg[:, 1] > self.config.thresh_explored).float()
            log_p_reg = torch.log(p_reg + EPS_MAPPER)
            log_1_p_reg = torch.log(1 - p_reg + EPS_MAPPER)
            entropy = -p_reg * log_p_reg - (1 - p_reg) * log_1_p_reg
            entropy_mask = (entropy.mean(dim=1) < self.config.thresh_entropy).float()
            explored_mask = explored_mask * entropy_mask
            unfilled_mask = (m[:, 1] == 0).float()
            m_updated = m
            # For regions that are unfilled, write as it is
            mask = unfilled_mask * explored_mask
            mask = mask.unsqueeze(1)
            m_updated = m_updated * (1 - mask) + p_reg * mask
            # For regions that are filled, do a moving average
            mask = (1 - unfilled_mask) * explored_mask
            mask = mask.unsqueeze(1)
            p_reg_ma = (p_reg * (1 - beta) + m_updated * beta) * mask
            m_updated = m_updated * (1 - mask) + p_reg_ma * mask
        else:
            raise ValueError(
                f"Mapper: registration_type: {self.config.registration_type} not defined!"
            )

        return m_updated

    def ext_register_map(self, m, p, x):
        return self._register_map(m, p, x)

    def _transform_observations(self, inputs, dx):
        """Converts observations from t-1 to coordinate frame for t.
        """
        # ====================== Transform past egocentric map ========================
        if "ego_map_t_1" in inputs:
            ego_map_t_1 = inputs["ego_map_t_1"]
            ego_map_t_1_trans = self._bottom_row_spatial_transform(
                ego_map_t_1, dx, invert=True
            )
            inputs["ego_map_t_1"] = ego_map_t_1_trans
        occ_cfg = self.projection_unit.main.config
        # ========================= Transform rgb and depth ===========================
        if "depth_t_1" in inputs:
            device = inputs["depth_t_1"].device
            depth_t_1 = inputs["depth_t_1"]
            if "K" not in self._cache.keys():
                # Project images from previous camera pose to current camera pose
                # Compute intrinsic camera matrix
                hfov = math.radians(occ_cfg.EGO_PROJECTION.hfov)
                vfov = math.radians(occ_cfg.EGO_PROJECTION.vfov)
                K = torch.Tensor(
                    [
                        [1 / math.tan(hfov / 2.0), 0.0, 0.0, 0.0],
                        [0.0, 1 / math.tan(vfov / 2.0), 0.0, 0.0],
                        [0.0, 0.0, 1.0, 0.0],
                        [0.0, 0.0, 0.0, 1.0],
                    ]
                ).to(
                    device
                )  # (4, 4)
                self._cache["K"] = K.cpu()
            else:
                K = self._cache["K"].to(device)
            H, W = depth_t_1.shape[2:]
            min_depth = occ_cfg.EGO_PROJECTION.min_depth
            max_depth = occ_cfg.EGO_PROJECTION.max_depth
            depth_t_1_unnorm = depth_t_1 * (max_depth - min_depth) + min_depth
            if "xs" not in self._cache.keys():
                xs, ys = np.meshgrid(np.linspace(-1, 1, W), np.linspace(1, -1, H))
                xs = torch.Tensor(xs.reshape(1, H, W)).to(device).unsqueeze(0)
                ys = torch.Tensor(ys.reshape(1, H, W)).to(device).unsqueeze(0)
                self._cache["xs"] = xs.cpu()
                self._cache["ys"] = ys.cpu()
            else:
                xs = self._cache["xs"].to(device)
                ys = self._cache["ys"].to(device)
            # Unproject
            # negate depth as the camera looks along -Z
            xys = torch.stack(
                [
                    xs * depth_t_1_unnorm,
                    ys * depth_t_1_unnorm,
                    -depth_t_1_unnorm,
                    torch.ones_like(depth_t_1_unnorm),
                ],
                dim=4,
            )  # (bs, 1, H, W, 4)
            # Points in the target (camera 2)
            xys = rearrange(xys, "b () h w f -> b (h w) f")
            if "invK" not in self._cache.keys():
                invK = torch.inverse(K)
                self._cache["invK"] = invK.cpu()
            else:
                invK = self._cache["invK"].to(device)
            xy_c2 = torch.matmul(xys, invK.unsqueeze(0))
            # ================ Camera 2 --> Camera 1 transformation ===============
            # We need the target to source transformation to warp from camera 1
            # to camera 2. In dx, dx[:, 0] is -Z, dx[:, 1] is X and dx[:, 2] is
            # rotation from -Z to X.
            translation = torch.stack(
                [dx[:, 1], torch.zeros_like(dx[:, 1]), -dx[:, 0]], dim=1
            )  # (bs, 3)
            T_world_camera2 = torch.zeros(xy_c2.shape[0], 4, 4).to(device)
            # Right-hand-rule rotation about Y axis
            cos_theta = torch.cos(-dx[:, 2])
            sin_theta = torch.sin(-dx[:, 2])
            T_world_camera2[:, 0, 0].copy_(cos_theta)
            T_world_camera2[:, 0, 2].copy_(sin_theta)
            T_world_camera2[:, 1, 1].fill_(1.0)
            T_world_camera2[:, 2, 0].copy_(-sin_theta)
            T_world_camera2[:, 2, 2].copy_(cos_theta)
            T_world_camera2[:, :3, 3].copy_(translation)
            T_world_camera2[:, 3, 3].fill_(1.0)
            # Transformation matrix from camera 2 --> world.
            T_camera1_camera2 = T_world_camera2  # (bs, 4, 4)
            xy_c1 = torch.matmul(
                T_camera1_camera2, xy_c2.transpose(1, 2)
            )  # (bs, 4, HW)
            # Convert camera coordinates to image coordinates
            xy_newimg = torch.matmul(K, xy_c1)  # (bs, 4, HW)
            xy_newimg = xy_newimg.transpose(1, 2)  # (bs, HW, 4)
            xys_newimg = xy_newimg[:, :, :2] / (
                -xy_newimg[:, :, 2:3] + 1e-8
            )  # (bs, HW, 2)
            # Flip back to y-down to match array indexing
            xys_newimg[:, :, 1] *= -1  # (bs, HW, 2)
            # ================== Apply warp to RGB, Depth images ==================
            sampler = rearrange(xys_newimg, "b (h w) f -> b h w f", h=H, w=W)
            depth_t_1_trans = F.grid_sample(depth_t_1, sampler, padding_mode="zeros")
            inputs["depth_t_1"] = depth_t_1_trans
            if "rgb_t_1" in inputs:
                rgb_t_1 = inputs["rgb_t_1"]
                rgb_t_1_trans = F.grid_sample(rgb_t_1, sampler, padding_mode="zeros")
                inputs["rgb_t_1"] = rgb_t_1_trans

        return inputs

    def _safe_cat(self, d1, d2):
        """Given two dicts of tensors with same keys, the values are
        concatenated if not None.
        """
        d = {}
        for k, v1 in d1.items():
            d[k] = None if v1 is None else torch.cat([v1, d2[k]], 0)
        return d


class MapperDataParallelWrapper(Mapper):
    def forward(self, *args, method_name="predict_deltas", **kwargs):
        if method_name == "predict_deltas":
            outputs = self.predict_deltas(*args, **kwargs)
        elif method_name == "estimate_ego_map":
            outputs = self._estimate_ego_map(*args, **kwargs)

        return outputs

    def _estimate_ego_map(self, x):
        return self.projection_unit(x)
