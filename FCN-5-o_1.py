from __future__ import print_function, division
from io import open
import argparse, os, glob, math, random, time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as tudata

import numpy as np

from matplotlib import pyplot

# require 'nn'
# require 'nngraph'
# require 'optim'
# require 'cutorch'
# require 'cunn'
# require 'image'
# require 'cudnn'


print("start...")
parser = argparse.ArgumentParser(description='Options.')
parser.add_argument('#--ext', default='.png', type=str, help='only load a specific type of data')
parser.add_argument('#--maxIter_INQ', default=0, type=int, help='the number of iterations for each step in INQ')
parser.add_argument('#--maxIter_train', default=0, type=int, help='the number of iterations for general training')
parser.add_argument('#--learningRate', default=0.0005, type=float, help='initial learning rate')
parser.add_argument('#--dropoutProb', default=0.5, type=float, help='probability of zeroing a neuron (dropout probability)')
parser.add_argument('#--CheckPointDir', default='results', type=str, help='directory to save network files')
parser.add_argument('#--checkpoint', default=1000, type=int, help='save checkpoints')
parser.add_argument('#--momentum', default=0.9, type=float, help='initial momentum for training')
parser.add_argument('#--KS', default=6, dype=int, help='the key parameter to determine the size of image, max is 54')
parser.add_argument('#--gpu', default=1, type=int, help='gpu device to use')
parser.add_argument('#--model', default='', type=str, help='model')
parser.add_argument('#--imageType', default=3, type=int, help='1: grayscale, 3: RGB')
parser.add_argument('#--batch_size', default=4, type=int, help='batch size')
parser.add_argument('#--datadir', default='./', type=str, help='the directory for dataset')

parser.add_argument('#--nclass', defaul=2, type=int, help='Number of classes')

opt = parser.parse_args()

opt.imgdir = os.path.join(opt.datadir, 'img')
opt.maskdir = os.path.join(opt.datadir, 'mask')
opt.validir = os.path.join(opt.datadir, 'valid')
opt.testdir = os.path.join(opt.datadir, 'test')

# print args.accumulate(args.integers)
    # cmd = torch.CmdLine()
    # cmd:text()
    # cmd:text('Options:')
    # cmd:option('#--ext','.png','only load a specific type of data')
    # cmd:option('#--maxIter_INQ',0,'the number of iterations for each step in INQ')
    # cmd:option('#--maxIter_train',0,'the number of iterations for general training')
    # cmd:option('#--learningRate',0.0005,'initial learning rate')
    # cmd:option('#--dropoutProb', 0.5, 'probability of zeroing a neuron (dropout probability)')
    # cmd:option('#--CheckPointDir', 'results','directory to save network files')
    # cmd:option('#--checkpoint',1000,'save checkpoints')
    # cmd:option('#--momentum',0.9,'initial momentum for training')
    # cmd:option('#--XX',6,'the key parameter to determine the size of image, max is 54')
    # cmd:option('#--gpu',1,'gpu device to use')
    # cmd:option('#--model','','model')
    # cmd:option('#--imageType',3,'1: grayscale, 3: RGB')
    # cmd:option('#--batch_size',4,'batch size')
    # cmd:option('#--imgdir', 'null', 'the directory to load')
    # cmd:option('#--maskdir', 'null', 'the directory to load')
    # cmd:option('#--validir', 'test valid', 'the directory to load')
    # cmd:option('#--testdir', 'testAll', 'the directory to load')
    # cmd:option('#--nclass',2,'Number of classes')
    # cmd:text()
    # opt = cmd:parse(arg or {})

if not os.path.exists(opt.CheckPointDir):
    os.mkdirs(opt.CheckPointDir)

INQ_train_path = os.path.join(opt.CheckPointDir, 'INQ_train')
final_INQ_path = os.path.join(opt.CheckPointDir, 'final_INQ_results')
if not os.path.exists(INQ_train_path):
    os.mkdirs(INQ_train_path)
if not os.path.exists(final_INQ_path):
    os.mkdirs(final_INQ_path)

# os.execute("rm -r " .. opt.CheckPointDir)
    # os.execute("mkdir " .. opt.CheckPointDir)
    # #--os.execute("mkdir " .. opt.CheckPointDir..'/general_train')
    # os.execute("mkdir " .. opt.CheckPointDir..'/INQ_train')
    # os.execute("mkdir " .. opt.CheckPointDir..'/final_INQ_results')
    # #--general_train_path = opt.CheckPointDir..'/general_train'
    # INQ_train_path = opt.CheckPointDir..'/INQ_train'
    # final_INQ_path = opt.CheckPointDir..'/final_INQ_results'

print("CheckPointDir=" + opt.CheckPointDir)   
print("datadir=" + opt.datadir)
# print("maskdir="..opt.maskdir)
# print("validir="..opt.validir)
# #-- for debug setting
opt.maxIter_INQ = 12000 ##--8000  40
opt.checkpoint = 1000  ##--1000  10
opt.batch_size = 3 ##--1

# #--create some folders to store trained "new train samples"
    # #--local pbatch=8
    # #--local uc_th=3000 #--3000
    # #--opt.checkpoint = 10
    # #--opt.addpoint = 10
    # #--opt.maxiter = 1200
    # #--print("debug mode: uc_th="..uc_th..", checkpoint="..opt.checkpoint..",addpoint="..opt.addpoint)

    # local interv=100;

    # #-- set up gpu
    # cutorch.setDevice(opt.gpu)
# cudnn.benchmark = true
# cudnn.fastest = true
    # #--cudnn.verbose = true

# #-- parameters for the size of the sample
    # KS=opt.KS
    # local ous=32*KS
    # local ins=32*KS
    # local Iname
    # print(ins..' '..ous)

    # #-- Get the list of files in the given directory
    # train_files = {} #-- store the file names of training images
    # label_files = {} #-- store the file names of the label (or ideal output) of training images
    # vali_files = {}
    # test_files = {}

interv=100
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("device using: ")
print(device)

KS=opt.KS
ous=32*KS
ins=32*KS
print(ins, ous)

train_files = glob.glob(os.path.join(opt.imgdir, '*'+opt.ext)) 
train_files.sort()
##-- store the file names of training images
label_files = glob.glob(os.path.join(opt.maskdir, '*'+opt.ext)) 
label_files.sort()
##-- store the file names of the label (or ideal output) of training images
vali_files = glob.glob(os.path.join(opt.validir, '*'+opt.ext))
test_files = glob.glob(os.path.join(opt.testdir, '*'+opt.ext))

print("#train_files:" + str(len(train_files)))
print("#label_files" + str(len(label_files))
print("#vali_files" + str(len(vali_files))
print("#test_files" + str(len(test_files))

# for file in paths.files(opt.imgdir) do
    #    if file:find(opt.ext .. '$') then
    #       table.insert(train_files, paths.concat(opt.imgdir,file))
    #    end
    # end

    # for file in paths.files(opt.maskdir) do
    #    if file:find(opt.ext .. '$') then
    #       table.insert(label_files, paths.concat(opt.maskdir,file))
    #    end
    # end


    # for file in paths.files(opt.validir) do
    #    if file:find(opt.ext .. '$') then
    #       table.insert(vali_files, paths.concat(opt.validir,file))
    #    end
    # end

    # for file in paths.files(opt.testdir) do
    #     if file:find(opt.ext .. '$') then
    #         table.insert(test_files, paths.concat(opt.testdir,file))
    #     end
    # end

    # table.sort(train_files, function (a,b) return a < b end)

    # table.sort(label_files, function (a,b) return a < b end)

    # print("#train_files")
    # print(#train_files)
    # print("#label_files")
    # print(#label_files)
    # print("#vali_files")
    # print(#vali_files)
    # print("#test_files")
    # print(#test_files)

train_images = []
train_org_images = []
label_images = []
# local pi=3.14159265359
big_ous=ous*2
big_ins=ins*2
pad_size = big_ous-big_ins

for i in range(len(train_files)):
    process_train_files[i]_to_fill_train_images_train_org_images_and_label_images

# for i=1,#train_files do
   
    #    local train_image = image.load(train_files[i],opt.imageType,'double')

    #    #--train_image = image.scale(train_image,'*1/2','simple');

    #    local totdim = train_image:size():size()
    #    local mirrored_train_image = torch.DoubleTensor(opt.imageType,train_image:size(totdim-1)+pad_size,train_image:size(totdim)+pad_size):fill(0)
    #    local center_train_image = mirrored_train_image:sub(1,opt.imageType,pad_size/2+1,pad_size/2+train_image:size(totdim-1),pad_size/2+1,pad_size/2+train_image:size(totdim))
    #    center_train_image:copy(train_image)
        
    #    for j=1,pad_size/2 do
        #   for c=1,opt.imageType do
            # mirrored_train_image[{c,j,{}}]=mirrored_train_image[{c,pad_size+1-j,{}}]
            # mirrored_train_image[{c,pad_size/2+train_image:size(totdim-1)+j,{}}]=mirrored_train_image[{c,pad_size/2+train_image:size(totdim-1)+1-j,{}}]
            # #--mirrored_train_image[{c,j,{}}]:fill(0)
            # #--mirrored_train_image[{c,pad_size/2+train_image:size(totdim-1)+j,{}}]:fill(0)
        #   end
    #    end
    #    for j=1,pad_size/2 do
    #       for c=1,opt.imageType do
    #          mirrored_train_image[{c,{},j}]=mirrored_train_image[{c,{},pad_size+1-j}]
    #          mirrored_train_image[{c,{},pad_size/2+train_image:size(totdim)+j}]=mirrored_train_image[{c,{},pad_size/2+train_image:size(totdim)+1-j}]
    #          #--mirrored_train_image[{c,{},j}]:fill(0)
    #          #--mirrored_train_image[{c,{},pad_size/2+train_image:size(totdim)+j}]:fill(0)
    #       end
    #    end
    
    #    local label_image= image.load(label_files[i],1,'byte');

    #    #--label_image = image.scale(label_image,'*1/2','simple');

    #    table.insert(train_org_images,train_image)
    #    table.insert(train_images,mirrored_train_image)
    #    table.insert(label_images,label_image)
    # end

# math.randomseed(os.time())

def get_image():
    randomly_select_a_pair_of_train_image_and_label_image
    process_them_with_randoml_cut_and_rotation
    return pair_of_processed_train_image_and_label_image

# local function get_image()
    #    local i_idx=math.random(1,#train_files)
    #    local totdim=label_images[i_idx]:size():size()
    #    local xs = label_images[i_idx]:size(totdim)
    #    local ys = label_images[i_idx]:size(totdim-1)
   
    #    local deci=math.random(1,2)
    #    local flipi=math.random(1,2)
    #    if (xs<big_ous) or (ys<big_ous) then
    #      deci = 1
    #    end
    #    if (deci==2) then
    #      local rotate=math.random()
    #      local stx = math.random(0,xs-big_ous)
    #      local sty = math.random(0,ys-big_ous)
    #      train_sample=image.crop(train_org_images[i_idx],stx,sty,stx+big_ous,sty+big_ous);
    #      train_sample=image.rotate(train_sample,-2*pi*rotate)
    #      label_sample=image.crop(label_images[i_idx],stx,sty,stx+big_ous,sty+big_ous)
    #      label_sample=image.rotate(label_sample,-2*pi*rotate)
    #      train_sample=image.crop(train_sample,'c',ous,ous)
    #      label_sample=image.crop(label_sample,'c',ins,ins)
    #      if flipi==2 then
    #        train_sample=image.hflip(train_sample)
    #        label_sample=image.hflip(label_sample)
    #        #--print('flip1');
    #      end
    #      return train_sample, label_sample
    #    end
    #    local rotate=math.random(1,4)
    #    local stx = math.random(0,xs-ins)
    #    local sty = math.random(0,ys-ins)
    #    train_sample=image.crop(train_images[i_idx],stx+pad_size/4,sty+pad_size/4,stx+pad_size/4+ous,sty+pad_size/4+ous);
    #    train_sample=image.rotate(train_sample,-pi/2*(rotate-1))
    #    label_sample=image.crop(label_images[i_idx],stx,sty,stx+ins,sty+ins)
    #    label_sample=image.rotate(label_sample,-pi/2*(rotate-1))
    #    if flipi==2 then
    #      train_sample=image.hflip(train_sample)
    #      label_sample=image.hflip(label_sample)
    #      #--print('flip2');
    #    end
    #    return train_sample, label_sample
    # end

# local n_classifer
# local nc = 32
nc = 32

do 
input1 = nn.Identity()()
input2 = nn.Identity()()
input3 = nn.Identity()()
input4 = nn.Identity()()
input5 = nn.Identity()()

n_classifer=5

p1L1b=cudnn.SpatialConvolution(opt.imageType, nc, 3, 3, 1, 1, 1, 1)(input1) ##--4XX+28
p1L1b_bn=nn.SpatialBatchNormalization(nc)(p1L1b)
p1L1b_relu=cudnn.ReLU(true)(p1L1b_bn)
p1L1c=cudnn.SpatialConvolution(nc, nc, 3, 3, 1, 1, 1, 1)(p1L1b_relu) ##--4XX+28
p1L1c_bn=nn.SpatialBatchNormalization(nc)(p1L1c)
p1L1c_relu=cudnn.ReLU(true)(p1L1c_bn)
p1L2a=cudnn.SpatialMaxPooling(2, 2, 2, 2)(p1L1c_relu)
p1L2res1_input0=nn.Padding(1,7*nc,3)(p1L2a)
p1L2res1c2=cudnn.SpatialConvolution(nc, 2*nc, 1, 1, 1, 1, 0, 0)(p1L2a)
p1L2res1c2_bn=nn.SpatialBatchNormalization(2*nc)(p1L2res1c2)
p1L2res1c2_relu=cudnn.ReLU(true)(p1L2res1c2_bn)
p1L2res1c3=cudnn.SpatialConvolution(2*nc, 2*nc, 3, 3, 1, 1, 1, 1)(p1L2res1c2_relu)
p1L2res1c3_bn=nn.SpatialBatchNormalization(2*nc)(p1L2res1c3)
p1L2res1c3_relu=cudnn.ReLU(true)(p1L2res1c3_bn)
p1L2res1c4=cudnn.SpatialConvolution(2*nc, 8*nc, 1, 1, 1, 1, 0, 0)(p1L2res1c3_relu)
p1L2res1=nn.CAddTable(false)({p1L2res1_input0,p1L2res1c4})
p1L2res1_bn=nn.SpatialBatchNormalization(8*nc)(p1L2res1)
p1L2res1_relu=cudnn.ReLU(true)(p1L2res1_bn)
p1L2res2c2=cudnn.SpatialConvolution(8*nc, 2*nc, 1, 1, 1, 1, 0, 0)(p1L2res1_relu)
p1L2res2c2_bn=nn.SpatialBatchNormalization(2*nc)(p1L2res2c2)
p1L2res2c2_relu=cudnn.ReLU(true)(p1L2res2c2_bn)
p1L2res2c3=cudnn.SpatialConvolution(2*nc, 2*nc, 3, 3, 1, 1, 1, 1)(p1L2res2c2_relu)
p1L2res2c3_bn=nn.SpatialBatchNormalization(2*nc)(p1L2res2c3)
p1L2res2c3_relu=cudnn.ReLU(true)(p1L2res2c3_bn)
p1L2res2c4=cudnn.SpatialConvolution(2*nc, 8*nc, 1, 1, 1, 1, 0, 0)(p1L2res2c3_relu)
p1L2res2=nn.CAddTable(false)({p1L2res1_relu,p1L2res2c4})
p1L2res2_bn=nn.SpatialBatchNormalization(8*nc)(p1L2res2)
p1L2res2_relu=cudnn.ReLU(true)(p1L2res2_bn)
p1L3a=cudnn.SpatialMaxPooling(2, 2, 2, 2)(p1L2res2_relu)
p1L3res0_input0=nn.Padding(1,8*nc,3)(p1L3a)
p1L3res0c2=cudnn.SpatialConvolution(8*nc, 4*nc, 1, 1, 1, 1, 0, 0)(p1L3a)
p1L3res0c2_bn=nn.SpatialBatchNormalization(4*nc)(p1L3res0c2)
p1L3res0c2_relu=cudnn.ReLU(true)(p1L3res0c2_bn)
p1L3res0c3=cudnn.SpatialConvolution(4*nc, 4*nc, 3, 3, 1, 1, 1, 1)(p1L3res0c2_relu)
p1L3res0c3_bn=nn.SpatialBatchNormalization(4*nc)(p1L3res0c3)
p1L3res0c3_relu=cudnn.ReLU(true)(p1L3res0c3_bn)
p1L3res0c4=cudnn.SpatialConvolution(4*nc, 16*nc, 1, 1, 1, 1, 0, 0)(p1L3res0c3_relu)
p1L3res0=nn.CAddTable(false)({p1L3res0_input0,p1L3res0c4})
p1L3res0_bn=nn.SpatialBatchNormalization(16*nc)(p1L3res0)
p1L3res0_relu=cudnn.ReLU(true)(p1L3res0_bn)
p1L3res1c2=cudnn.SpatialConvolution(16*nc, 4*nc, 1, 1, 1, 1, 0, 0)(p1L3res0_relu)
p1L3res1c2_bn=nn.SpatialBatchNormalization(4*nc)(p1L3res1c2)
p1L3res1c2_relu=cudnn.ReLU(true)(p1L3res1c2_bn)
p1L3res1c3=cudnn.SpatialConvolution(4*nc, 4*nc, 3, 3, 1, 1, 1, 1)(p1L3res1c2_relu)
p1L3res1c3_bn=nn.SpatialBatchNormalization(4*nc)(p1L3res1c3)
p1L3res1c3_relu=cudnn.ReLU(true)(p1L3res1c3_bn)
p1L3res1c4=cudnn.SpatialConvolution(4*nc, 16*nc, 1, 1, 1, 1, 0, 0)(p1L3res1c3_relu)
p1L3res1=nn.CAddTable(false)({p1L3res0_relu,p1L3res1c4})
p1L3res1_bn=nn.SpatialBatchNormalization(16*nc)(p1L3res1)
p1L3res1_relu=cudnn.ReLU(true)(p1L3res1_bn)
p1L4a=cudnn.SpatialMaxPooling(2, 2, 2, 2)(p1L3res1_relu)
p1L4res0_input0=nn.Padding(1,16*nc,3)(p1L4a)
p1L4res0c2=cudnn.SpatialConvolution(16*nc, 8*nc, 1, 1, 1, 1, 0, 0)(p1L4a)
p1L4res0c2_bn=nn.SpatialBatchNormalization(8*nc)(p1L4res0c2)
p1L4res0c2_relu=cudnn.ReLU(true)(p1L4res0c2_bn)
p1L4res0c3=cudnn.SpatialConvolution(8*nc, 8*nc, 3, 3, 1, 1, 1, 1)(p1L4res0c2_relu)
p1L4res0c3_bn=nn.SpatialBatchNormalization(8*nc)(p1L4res0c3)
p1L4res0c3_relu=cudnn.ReLU(true)(p1L4res0c3_bn)
p1L4res0c4=cudnn.SpatialConvolution(8*nc, 32*nc, 1, 1, 1, 1, 0, 0)(p1L4res0c3_relu)
p1L4res0=nn.CAddTable(false)({p1L4res0_input0,p1L4res0c4})
p1L4res0_bn=nn.SpatialBatchNormalization(32*nc)(p1L4res0)
p1L4res0_relu=cudnn.ReLU(true)(p1L4res0_bn)
p1L4res1c2=cudnn.SpatialConvolution(32*nc, 8*nc, 1, 1, 1, 1, 0, 0)(p1L4res0_relu)
p1L4res1c2_bn=nn.SpatialBatchNormalization(8*nc)(p1L4res1c2)
p1L4res1c2_relu=cudnn.ReLU(true)(p1L4res1c2_bn)
p1L4res1c3=cudnn.SpatialConvolution(8*nc, 8*nc, 3, 3, 1, 1, 1, 1)(p1L4res1c2_relu)
p1L4res1c3_bn=nn.SpatialBatchNormalization(8*nc)(p1L4res1c3)
p1L4res1c3_relu=cudnn.ReLU(true)(p1L4res1c3_bn)
p1L4res1c4=cudnn.SpatialConvolution(8*nc, 32*nc, 1, 1, 1, 1, 0, 0)(p1L4res1c3_relu)
p1L4res1=nn.CAddTable(false)({p1L4res0_relu,p1L4res1c4})
p1L4res1_bn=nn.SpatialBatchNormalization(32*nc)(p1L4res1)
p1L4res1_relu=cudnn.ReLU(true)(p1L4res1_bn)
p1L4res1_drop=nn.SpatialDropout(opt.dropoutProb)(p1L4res1_relu)
p1L5a=cudnn.SpatialMaxPooling(2, 2, 2, 2)(p1L4res1_drop)
p1L5res0c2=cudnn.SpatialConvolution(32*nc, 8*nc, 1, 1, 1, 1, 0, 0)(p1L5a)
p1L5res0c2_bn=nn.SpatialBatchNormalization(8*nc)(p1L5res0c2)
p1L5res0c2_relu=cudnn.ReLU(true)(p1L5res0c2_bn)
p1L5res0c3=cudnn.SpatialConvolution(8*nc, 8*nc, 3, 3, 1, 1, 1, 1)(p1L5res0c2_relu)
p1L5res0c3_bn=nn.SpatialBatchNormalization(8*nc)(p1L5res0c3)
p1L5res0c3_relu=cudnn.ReLU(true)(p1L5res0c3_bn)
p1L5res0c4=cudnn.SpatialConvolution(8*nc, 32*nc, 1, 1, 1, 1, 0, 0)(p1L5res0c3_relu)
p1L5res0=nn.CAddTable(false)({p1L5a,p1L5res0c4})
p1L5res0_bn=nn.SpatialBatchNormalization(32*nc)(p1L5res0)
p1L5res0_relu=cudnn.ReLU(true)(p1L5res0_bn)
p1L5res1c2=cudnn.SpatialConvolution(32*nc, 8*nc, 1, 1, 1, 1, 0, 0)(p1L5res0_relu)
p1L5res1c2_bn=nn.SpatialBatchNormalization(8*nc)(p1L5res1c2)
p1L5res1c2_relu=cudnn.ReLU(true)(p1L5res1c2_bn)
p1L5res1c3=cudnn.SpatialConvolution(8*nc, 8*nc, 3, 3, 1, 1, 1, 1)(p1L5res1c2_relu)
p1L5res1c3_bn=nn.SpatialBatchNormalization(8*nc)(p1L5res1c3)
p1L5res1c3_relu=cudnn.ReLU(true)(p1L5res1c3_bn)
p1L5res1c4=cudnn.SpatialConvolution(8*nc, 32*nc, 1, 1, 1, 1, 0, 0)(p1L5res1c3_relu)
p1L5res1=nn.CAddTable(false)({p1L5res0_relu,p1L5res1c4})
p1L5res1_bn=nn.SpatialBatchNormalization(32*nc)(p1L5res1)
p1L5res1_relu=cudnn.ReLU(true)(p1L5res1_bn)
p1L5res1_drop=nn.SpatialDropout(opt.dropoutProb)(p1L5res1_relu)
p1L6a=cudnn.SpatialMaxPooling(2, 2, 2, 2)(p1L5res1_drop)
p1L6res0c2=cudnn.SpatialConvolution(32*nc, 8*nc, 1, 1, 1, 1, 0, 0)(p1L6a)
p1L6res0c2_bn=nn.SpatialBatchNormalization(8*nc)(p1L6res0c2)
p1L6res0c2_relu=cudnn.ReLU(true)(p1L6res0c2_bn)
p1L6res0c3=cudnn.SpatialConvolution(8*nc, 8*nc, 3, 3, 1, 1, 1, 1)(p1L6res0c2_relu)
p1L6res0c3_bn=nn.SpatialBatchNormalization(8*nc)(p1L6res0c3)
p1L6res0c3_relu=cudnn.ReLU(true)(p1L6res0c3_bn)
p1L6res0c4=cudnn.SpatialConvolution(8*nc, 32*nc, 1, 1, 1, 1, 0, 0)(p1L6res0c3_relu)
p1L6res0=nn.CAddTable(false)({p1L6a,p1L6res0c4})
p1L6res0_bn=nn.SpatialBatchNormalization(32*nc)(p1L6res0)
p1L6res0_relu=cudnn.ReLU(true)(p1L6res0_bn)
p1L6res1c2=cudnn.SpatialConvolution(32*nc, 8*nc, 1, 1, 1, 1, 0, 0)(p1L6res0_relu)
p1L6res1c2_bn=nn.SpatialBatchNormalization(8*nc)(p1L6res1c2)
p1L6res1c2_relu=cudnn.ReLU(true)(p1L6res1c2_bn)
p1L6res1c3=cudnn.SpatialConvolution(8*nc, 8*nc, 3, 3, 1, 1, 1, 1)(p1L6res1c2_relu)
p1L6res1c3_bn=nn.SpatialBatchNormalization(8*nc)(p1L6res1c3)
p1L6res1c3_relu=cudnn.ReLU(true)(p1L6res1c3_bn)
p1L6res1c4=cudnn.SpatialConvolution(8*nc, 32*nc, 1, 1, 1, 1, 0, 0)(p1L6res1c3_relu)
p1L6res1=nn.CAddTable(false)({p1L6res0_relu,p1L6res1c4})
p1L6res1_bn=nn.SpatialBatchNormalization(32*nc)(p1L6res1)
p1L6res1_relu=cudnn.ReLU(true)(p1L6res1_bn)
p1L6res1_drop=nn.SpatialDropout(opt.dropoutProb)(p1L6res1_relu)
p1L1up0=nn.SpatialConvolution(nc, opt.nclass, 3, 3, 1, 1, 1, 1)(p1L1c_relu)
p1L1up0_bn=nn.SpatialBatchNormalization(opt.nclass)(p1L1up0)
p1L1up0_relu=cudnn.ReLU(true)(p1L1up0_bn)
p1L2up=nn.SpatialFullConvolution(8*nc,opt.nclass,4,4,2,2,1,1)(p1L2res2_relu)
p1L2up_bn=nn.SpatialBatchNormalization(opt.nclass)(p1L2up)
p1L2up_relu=cudnn.ReLU(true)(p1L2up_bn)
p1L3ups1=nn.SpatialFullConvolution(16*nc,2*opt.nclass,4,4,2,2,1,1)(p1L3res1_relu)
p1L3ups1_bn=nn.SpatialBatchNormalization(2*opt.nclass)(p1L3ups1)
p1L3ups1_relu=cudnn.ReLU(true)(p1L3ups1_bn)
p1L3up=nn.SpatialFullConvolution(2*opt.nclass,opt.nclass,4,4,2,2,1,1)(p1L3ups1_relu)
p1L3up_bn=nn.SpatialBatchNormalization(opt.nclass)(p1L3up)
p1L3up_relu=cudnn.ReLU(true)(p1L3up_bn)
     ##-- XX -> 2XX-2 4XX-6 8XX-14 16XX-30-> 16XX-32
p1L4ups1=nn.SpatialFullConvolution(32*nc,4*opt.nclass,4,4,2,2,1,1)(p1L4res1_drop) 
p1L4ups1_bn=nn.SpatialBatchNormalization(4*opt.nclass)(p1L4ups1)
p1L4ups1_relu=cudnn.ReLU(true)(p1L4ups1_bn)
p1L4ups2=nn.SpatialFullConvolution(4*opt.nclass,2*opt.nclass,4,4,2,2,1,1)(p1L4ups1_relu)
p1L4ups2_bn=nn.SpatialBatchNormalization(2*opt.nclass)(p1L4ups2)
p1L4ups2_relu=cudnn.ReLU(true)(p1L4ups2_bn)
p1L4up=nn.SpatialFullConvolution(2*opt.nclass,opt.nclass,4,4,2,2,1,1)(p1L4ups2_relu)
p1L4up_bn=nn.SpatialBatchNormalization(opt.nclass)(p1L4up)
p1L4up_relu=cudnn.ReLU(true)(p1L4up_bn)
p1L5ups1=nn.SpatialFullConvolution(32*nc,8*opt.nclass,4,4,2,2,1,1)(p1L5res1_drop)
p1L5ups1_bn=nn.SpatialBatchNormalization(8*opt.nclass)(p1L5ups1)
p1L5ups1_relu=cudnn.ReLU(true)(p1L5ups1_bn)
p1L5ups2=nn.SpatialFullConvolution(8*opt.nclass,4*opt.nclass,4,4,2,2,1,1)(p1L5ups1_relu)
p1L5ups2_bn=nn.SpatialBatchNormalization(4*opt.nclass)(p1L5ups2)
p1L5ups2_relu=cudnn.ReLU(true)(p1L5ups2_bn)
p1L5ups3=nn.SpatialFullConvolution(4*opt.nclass,2*opt.nclass,4,4,2,2,1,1)(p1L5ups2_relu)
p1L5ups3_bn=nn.SpatialBatchNormalization(2*opt.nclass)(p1L5ups3)
p1L5ups3_relu=cudnn.ReLU(true)(p1L5ups3_bn)
p1L5up=nn.SpatialFullConvolution(2*opt.nclass,opt.nclass,4,4,2,2,1,1)(p1L5ups3_relu)
p1L5up_bn=nn.SpatialBatchNormalization(opt.nclass)(p1L5up)
p1L5up_relu=cudnn.ReLU(true)(p1L5up_bn)
p1L6ups1=nn.SpatialFullConvolution(32*nc,16*opt.nclass,4,4,2,2,1,1)(p1L6res1_drop)
p1L6ups1_bn=nn.SpatialBatchNormalization(16*opt.nclass)(p1L6ups1)
p1L6ups1_relu=cudnn.ReLU(true)(p1L6ups1_bn)
p1L6ups2=nn.SpatialFullConvolution(16*opt.nclass,8*opt.nclass,4,4,2,2,1,1)(p1L6ups1_relu)
p1L6ups2_bn=nn.SpatialBatchNormalization(8*opt.nclass)(p1L6ups2)
p1L6ups2_relu=cudnn.ReLU(true)(p1L6ups2_bn)
p1L6ups3=nn.SpatialFullConvolution(8*opt.nclass,4*opt.nclass,4,4,2,2,1,1)(p1L6ups2_relu)
p1L6ups3_bn=nn.SpatialBatchNormalization(4*opt.nclass)(p1L6ups3)
p1L6ups3_relu=cudnn.ReLU(true)(p1L6ups3_bn)
p1L6ups4=nn.SpatialFullConvolution(4*opt.nclass,2*opt.nclass,4,4,2,2,1,1)(p1L6ups3_relu)
p1L6ups4_bn=nn.SpatialBatchNormalization(2*opt.nclass)(p1L6ups4)
p1L6ups4_relu=cudnn.ReLU(true)(p1L6ups4_bn)
p1L6up=nn.SpatialFullConvolution(2*opt.nclass,opt.nclass,4,4,2,2,1,1)(p1L6ups4_relu)
p1L6up_bn=nn.SpatialBatchNormalization(opt.nclass)(p1L6up)
p1L6up_relu=cudnn.ReLU(true)(p1L6up_bn)
p1Lallup1 = nn.JoinTable(1,3)({p1L1up0_relu,p1L2up_relu,p1L3up_relu,p1L4up_relu,p1L5up_relu,p1L6up_relu})
p1Lallup2 = nn.SpatialConvolution(6*opt.nclass, opt.nclass, 3, 3, 1, 1, 1, 1)(p1Lallup1)
p1Lallup2_bn = nn.SpatialBatchNormalization(opt.nclass)(p1Lallup2)
p1Lallup3 = cudnn.ReLU(true)(p1Lallup2_bn)
p1Lallup4 = nn.SpatialConvolution(opt.nclass, opt.nclass, 1, 1, 1, 1, 0, 0)(p1Lallup3)
p1L1up20=nn.SpatialConvolution(nc, opt.nclass, 3, 3, 1, 1, 1, 1)(p1L1c_relu)
p1L1up20_bn=nn.SpatialBatchNormalization(opt.nclass)(p1L1up20)
p1L1up20_relu=cudnn.ReLU(true)(p1L1up20_bn)
p1L2up2=nn.SpatialFullConvolution(8*nc,opt.nclass,4,4,2,2,1,1)(p1L2res2_relu)
p1L2up2_bn=nn.SpatialBatchNormalization(opt.nclass)(p1L2up2)
p1L2up2_relu=cudnn.ReLU(true)(p1L2up2_bn)
p1L3up2s1=nn.SpatialFullConvolution(16*nc,2*opt.nclass,4,4,2,2,1,1)(p1L3res1_relu)
p1L3up2s1_bn=nn.SpatialBatchNormalization(2*opt.nclass)(p1L3up2s1)
p1L3up2s1_relu=cudnn.ReLU(true)(p1L3up2s1_bn)
p1L3up2=nn.SpatialFullConvolution(2*opt.nclass,opt.nclass,4,4,2,2,1,1)(p1L3up2s1_relu)
p1L3up2_bn=nn.SpatialBatchNormalization(opt.nclass)(p1L3up2)
p1L3up2_relu=cudnn.ReLU(true)(p1L3up2_bn)
p1L4up2s1=nn.SpatialFullConvolution(32*nc,4*opt.nclass,4,4,2,2,1,1)(p1L4res1_drop) 
p1L4up2s1_bn=nn.SpatialBatchNormalization(4*opt.nclass)(p1L4up2s1)
p1L4up2s1_relu=cudnn.ReLU(true)(p1L4up2s1_bn)
p1L4up2s2=nn.SpatialFullConvolution(4*opt.nclass,2*opt.nclass,4,4,2,2,1,1)(p1L4up2s1_relu)
p1L4up2s2_bn=nn.SpatialBatchNormalization(2*opt.nclass)(p1L4up2s2)
p1L4up2s2_relu=cudnn.ReLU(true)(p1L4up2s2_bn)
p1L4up2=nn.SpatialFullConvolution(2*opt.nclass,opt.nclass,4,4,2,2,1,1)(p1L4up2s2_relu)
p1L4up2_bn=nn.SpatialBatchNormalization(opt.nclass)(p1L4up2)
p1L4up2_relu=cudnn.ReLU(true)(p1L4up2_bn)
p1L5up2s1=nn.SpatialFullConvolution(32*nc,8*opt.nclass,4,4,2,2,1,1)(p1L5res1_drop)
p1L5up2s1_bn=nn.SpatialBatchNormalization(8*opt.nclass)(p1L5up2s1)
p1L5up2s1_relu=cudnn.ReLU(true)(p1L5up2s1_bn)
p1L5up2s2=nn.SpatialFullConvolution(8*opt.nclass,4*opt.nclass,4,4,2,2,1,1)(p1L5up2s1_relu)
p1L5up2s2_bn=nn.SpatialBatchNormalization(4*opt.nclass)(p1L5up2s2)
p1L5up2s2_relu=cudnn.ReLU(true)(p1L5up2s2_bn)
p1L5up2s3=nn.SpatialFullConvolution(4*opt.nclass,2*opt.nclass,4,4,2,2,1,1)(p1L5up2s2_relu)
p1L5up2s3_bn=nn.SpatialBatchNormalization(2*opt.nclass)(p1L5up2s3)
p1L5up2s3_relu=cudnn.ReLU(true)(p1L5up2s3_bn)
p1L5up2=nn.SpatialFullConvolution(2*opt.nclass,opt.nclass,4,4,2,2,1,1)(p1L5up2s3_relu)
p1L5up2_bn=nn.SpatialBatchNormalization(opt.nclass)(p1L5up2)
p1L5up2_relu=cudnn.ReLU(true)(p1L5up2_bn)
p1L6up2s1=nn.SpatialFullConvolution(32*nc,16*opt.nclass,4,4,2,2,1,1)(p1L6res1_drop)
p1L6up2s1_bn=nn.SpatialBatchNormalization(16*opt.nclass)(p1L6up2s1)
p1L6up2s1_relu=cudnn.ReLU(true)(p1L6up2s1_bn)
p1L6up2s2=nn.SpatialFullConvolution(16*opt.nclass,8*opt.nclass,4,4,2,2,1,1)(p1L6up2s1_relu)
p1L6up2s2_bn=nn.SpatialBatchNormalization(8*opt.nclass)(p1L6up2s2)
p1L6up2s2_relu=cudnn.ReLU(true)(p1L6up2s2_bn)
p1L6up2s3=nn.SpatialFullConvolution(8*opt.nclass,4*opt.nclass,4,4,2,2,1,1)(p1L6up2s2_relu)
p1L6up2s3_bn=nn.SpatialBatchNormalization(4*opt.nclass)(p1L6up2s3)
p1L6up2s3_relu=cudnn.ReLU(true)(p1L6up2s3_bn)
p1L6up2s4=nn.SpatialFullConvolution(4*opt.nclass,2*opt.nclass,4,4,2,2,1,1)(p1L6up2s3_relu)
p1L6up2s4_bn=nn.SpatialBatchNormalization(2*opt.nclass)(p1L6up2s4)
p1L6up2s4_relu=cudnn.ReLU(true)(p1L6up2s4_bn)
p1L6up2=nn.SpatialFullConvolution(2*opt.nclass,opt.nclass,4,4,2,2,1,1)(p1L6up2s4_relu)
p1L6up2_bn=nn.SpatialBatchNormalization(opt.nclass)(p1L6up2)
p1L6up2_relu=cudnn.ReLU(true)(p1L6up2_bn)
p1Lallup21 = nn.JoinTable(1,3)({p1L1up20_relu,p1L2up2_relu,p1L3up2_relu,p1L4up2_relu,p1L5up2_relu,p1L6up2_relu})
p1Lallup22 = nn.SpatialConvolution(6*opt.nclass, opt.nclass, 3, 3, 1, 1, 1, 1)(p1Lallup21)
p1Lallup22_bn = nn.SpatialBatchNormalization(opt.nclass)(p1Lallup22)
p1Lallup23 = cudnn.ReLU(true)(p1Lallup22_bn)
p1Lallup24 = nn.SpatialConvolution(opt.nclass, opt.nclass, 1, 1, 1, 1, 0, 0)(p1Lallup23)

p2L1b=cudnn.SpatialConvolution(opt.imageType, nc, 3, 3, 1, 1, 1, 1)(input2) ##--4XX+28
p2L1b_bn=nn.SpatialBatchNormalization(nc)(p2L1b)
p2L1b_relu=cudnn.ReLU(true)(p2L1b_bn)
p2L1c=cudnn.SpatialConvolution(nc, nc, 3, 3, 1, 1, 1, 1)(p2L1b_relu) ##--4XX+28
p2L1c_bn=nn.SpatialBatchNormalization(nc)(p2L1c)
p2L1c_relu=cudnn.ReLU(true)(p2L1c_bn)
p2L2a=cudnn.SpatialMaxPooling(2, 2, 2, 2)(p2L1c_relu)
p2L2res1_input0=nn.Padding(1,7*nc,3)(p2L2a)
p2L2res1c2=cudnn.SpatialConvolution(nc, 2*nc, 1, 1, 1, 1, 0, 0)(p2L2a)
p2L2res1c2_bn=nn.SpatialBatchNormalization(2*nc)(p2L2res1c2)
p2L2res1c2_relu=cudnn.ReLU(true)(p2L2res1c2_bn)
p2L2res1c3=cudnn.SpatialConvolution(2*nc, 2*nc, 3, 3, 1, 1, 1, 1)(p2L2res1c2_relu)
p2L2res1c3_bn=nn.SpatialBatchNormalization(2*nc)(p2L2res1c3)
p2L2res1c3_relu=cudnn.ReLU(true)(p2L2res1c3_bn)
p2L2res1c4=cudnn.SpatialConvolution(2*nc, 8*nc, 1, 1, 1, 1, 0, 0)(p2L2res1c3_relu)
p2L2res1=nn.CAddTable(false)({p2L2res1_input0,p2L2res1c4})
p2L2res1_bn=nn.SpatialBatchNormalization(8*nc)(p2L2res1)
p2L2res1_relu=cudnn.ReLU(true)(p2L2res1_bn)
p2L2res2c2=cudnn.SpatialConvolution(8*nc, 2*nc, 1, 1, 1, 1, 0, 0)(p2L2res1_relu)
p2L2res2c2_bn=nn.SpatialBatchNormalization(2*nc)(p2L2res2c2)
p2L2res2c2_relu=cudnn.ReLU(true)(p2L2res2c2_bn)
p2L2res2c3=cudnn.SpatialConvolution(2*nc, 2*nc, 3, 3, 1, 1, 1, 1)(p2L2res2c2_relu)
p2L2res2c3_bn=nn.SpatialBatchNormalization(2*nc)(p2L2res2c3)
p2L2res2c3_relu=cudnn.ReLU(true)(p2L2res2c3_bn)
p2L2res2c4=cudnn.SpatialConvolution(2*nc, 8*nc, 1, 1, 1, 1, 0, 0)(p2L2res2c3_relu)
p2L2res2=nn.CAddTable(false)({p2L2res1_relu,p2L2res2c4})
p2L2res2_bn=nn.SpatialBatchNormalization(8*nc)(p2L2res2)
p2L2res2_relu=cudnn.ReLU(true)(p2L2res2_bn)
p2L3a=cudnn.SpatialMaxPooling(2, 2, 2, 2)(p2L2res2_relu)
p2L3res0_input0=nn.Padding(1,8*nc,3)(p2L3a)
p2L3res0c2=cudnn.SpatialConvolution(8*nc, 4*nc, 1, 1, 1, 1, 0, 0)(p2L3a)
p2L3res0c2_bn=nn.SpatialBatchNormalization(4*nc)(p2L3res0c2)
p2L3res0c2_relu=cudnn.ReLU(true)(p2L3res0c2_bn)
p2L3res0c3=cudnn.SpatialConvolution(4*nc, 4*nc, 3, 3, 1, 1, 1, 1)(p2L3res0c2_relu)
p2L3res0c3_bn=nn.SpatialBatchNormalization(4*nc)(p2L3res0c3)
p2L3res0c3_relu=cudnn.ReLU(true)(p2L3res0c3_bn)
p2L3res0c4=cudnn.SpatialConvolution(4*nc, 16*nc, 1, 1, 1, 1, 0, 0)(p2L3res0c3_relu)
p2L3res0=nn.CAddTable(false)({p2L3res0_input0,p2L3res0c4})
p2L3res0_bn=nn.SpatialBatchNormalization(16*nc)(p2L3res0)
p2L3res0_relu=cudnn.ReLU(true)(p2L3res0_bn)
p2L3res1c2=cudnn.SpatialConvolution(16*nc, 4*nc, 1, 1, 1, 1, 0, 0)(p2L3res0_relu)
p2L3res1c2_bn=nn.SpatialBatchNormalization(4*nc)(p2L3res1c2)
p2L3res1c2_relu=cudnn.ReLU(true)(p2L3res1c2_bn)
p2L3res1c3=cudnn.SpatialConvolution(4*nc, 4*nc, 3, 3, 1, 1, 1, 1)(p2L3res1c2_relu)
p2L3res1c3_bn=nn.SpatialBatchNormalization(4*nc)(p2L3res1c3)
p2L3res1c3_relu=cudnn.ReLU(true)(p2L3res1c3_bn)
p2L3res1c4=cudnn.SpatialConvolution(4*nc, 16*nc, 1, 1, 1, 1, 0, 0)(p2L3res1c3_relu)
p2L3res1=nn.CAddTable(false)({p2L3res0_relu,p2L3res1c4})
p2L3res1_bn=nn.SpatialBatchNormalization(16*nc)(p2L3res1)
p2L3res1_relu=cudnn.ReLU(true)(p2L3res1_bn)
p2L4a=cudnn.SpatialMaxPooling(2, 2, 2, 2)(p2L3res1_relu)
p2L4res0_input0=nn.Padding(1,16*nc,3)(p2L4a)
p2L4res0c2=cudnn.SpatialConvolution(16*nc, 8*nc, 1, 1, 1, 1, 0, 0)(p2L4a)
p2L4res0c2_bn=nn.SpatialBatchNormalization(8*nc)(p2L4res0c2)
p2L4res0c2_relu=cudnn.ReLU(true)(p2L4res0c2_bn)
p2L4res0c3=cudnn.SpatialConvolution(8*nc, 8*nc, 3, 3, 1, 1, 1, 1)(p2L4res0c2_relu)
p2L4res0c3_bn=nn.SpatialBatchNormalization(8*nc)(p2L4res0c3)
p2L4res0c3_relu=cudnn.ReLU(true)(p2L4res0c3_bn)
p2L4res0c4=cudnn.SpatialConvolution(8*nc, 32*nc, 1, 1, 1, 1, 0, 0)(p2L4res0c3_relu)
p2L4res0=nn.CAddTable(false)({p2L4res0_input0,p2L4res0c4})
p2L4res0_bn=nn.SpatialBatchNormalization(32*nc)(p2L4res0)
p2L4res0_relu=cudnn.ReLU(true)(p2L4res0_bn)
p2L4res1c2=cudnn.SpatialConvolution(32*nc, 8*nc, 1, 1, 1, 1, 0, 0)(p2L4res0_relu)
p2L4res1c2_bn=nn.SpatialBatchNormalization(8*nc)(p2L4res1c2)
p2L4res1c2_relu=cudnn.ReLU(true)(p2L4res1c2_bn)
p2L4res1c3=cudnn.SpatialConvolution(8*nc, 8*nc, 3, 3, 1, 1, 1, 1)(p2L4res1c2_relu)
p2L4res1c3_bn=nn.SpatialBatchNormalization(8*nc)(p2L4res1c3)
p2L4res1c3_relu=cudnn.ReLU(true)(p2L4res1c3_bn)
p2L4res1c4=cudnn.SpatialConvolution(8*nc, 32*nc, 1, 1, 1, 1, 0, 0)(p2L4res1c3_relu)
p2L4res1=nn.CAddTable(false)({p2L4res0_relu,p2L4res1c4})
p2L4res1_bn=nn.SpatialBatchNormalization(32*nc)(p2L4res1)
p2L4res1_relu=cudnn.ReLU(true)(p2L4res1_bn)
p2L4res1_drop=nn.SpatialDropout(opt.dropoutProb)(p2L4res1_relu)
p2L5a=cudnn.SpatialMaxPooling(2, 2, 2, 2)(p2L4res1_drop)
p2L5res0c2=cudnn.SpatialConvolution(32*nc, 8*nc, 1, 1, 1, 1, 0, 0)(p2L5a)
p2L5res0c2_bn=nn.SpatialBatchNormalization(8*nc)(p2L5res0c2)
p2L5res0c2_relu=cudnn.ReLU(true)(p2L5res0c2_bn)
p2L5res0c3=cudnn.SpatialConvolution(8*nc, 8*nc, 3, 3, 1, 1, 1, 1)(p2L5res0c2_relu)
p2L5res0c3_bn=nn.SpatialBatchNormalization(8*nc)(p2L5res0c3)
p2L5res0c3_relu=cudnn.ReLU(true)(p2L5res0c3_bn)
p2L5res0c4=cudnn.SpatialConvolution(8*nc, 32*nc, 1, 1, 1, 1, 0, 0)(p2L5res0c3_relu)
p2L5res0=nn.CAddTable(false)({p2L5a,p2L5res0c4})
p2L5res0_bn=nn.SpatialBatchNormalization(32*nc)(p2L5res0)
p2L5res0_relu=cudnn.ReLU(true)(p2L5res0_bn)
p2L5res1c2=cudnn.SpatialConvolution(32*nc, 8*nc, 1, 1, 1, 1, 0, 0)(p2L5res0_relu)
p2L5res1c2_bn=nn.SpatialBatchNormalization(8*nc)(p2L5res1c2)
p2L5res1c2_relu=cudnn.ReLU(true)(p2L5res1c2_bn)
p2L5res1c3=cudnn.SpatialConvolution(8*nc, 8*nc, 3, 3, 1, 1, 1, 1)(p2L5res1c2_relu)
p2L5res1c3_bn=nn.SpatialBatchNormalization(8*nc)(p2L5res1c3)
p2L5res1c3_relu=cudnn.ReLU(true)(p2L5res1c3_bn)
p2L5res1c4=cudnn.SpatialConvolution(8*nc, 32*nc, 1, 1, 1, 1, 0, 0)(p2L5res1c3_relu)
p2L5res1=nn.CAddTable(false)({p2L5res0_relu,p2L5res1c4})
p2L5res1_bn=nn.SpatialBatchNormalization(32*nc)(p2L5res1)
p2L5res1_relu=cudnn.ReLU(true)(p2L5res1_bn)
p2L5res1_drop=nn.SpatialDropout(opt.dropoutProb)(p2L5res1_relu)
p2L6a=cudnn.SpatialMaxPooling(2, 2, 2, 2)(p2L5res1_drop)
p2L6res0c2=cudnn.SpatialConvolution(32*nc, 8*nc, 1, 1, 1, 1, 0, 0)(p2L6a)
p2L6res0c2_bn=nn.SpatialBatchNormalization(8*nc)(p2L6res0c2)
p2L6res0c2_relu=cudnn.ReLU(true)(p2L6res0c2_bn)
p2L6res0c3=cudnn.SpatialConvolution(8*nc, 8*nc, 3, 3, 1, 1, 1, 1)(p2L6res0c2_relu)
p2L6res0c3_bn=nn.SpatialBatchNormalization(8*nc)(p2L6res0c3)
p2L6res0c3_relu=cudnn.ReLU(true)(p2L6res0c3_bn)
p2L6res0c4=cudnn.SpatialConvolution(8*nc, 32*nc, 1, 1, 1, 1, 0, 0)(p2L6res0c3_relu)
p2L6res0=nn.CAddTable(false)({p2L6a,p2L6res0c4})
p2L6res0_bn=nn.SpatialBatchNormalization(32*nc)(p2L6res0)
p2L6res0_relu=cudnn.ReLU(true)(p2L6res0_bn)
p2L6res1c2=cudnn.SpatialConvolution(32*nc, 8*nc, 1, 1, 1, 1, 0, 0)(p2L6res0_relu)
p2L6res1c2_bn=nn.SpatialBatchNormalization(8*nc)(p2L6res1c2)
p2L6res1c2_relu=cudnn.ReLU(true)(p2L6res1c2_bn)
p2L6res1c3=cudnn.SpatialConvolution(8*nc, 8*nc, 3, 3, 1, 1, 1, 1)(p2L6res1c2_relu)
p2L6res1c3_bn=nn.SpatialBatchNormalization(8*nc)(p2L6res1c3)
p2L6res1c3_relu=cudnn.ReLU(true)(p2L6res1c3_bn)
p2L6res1c4=cudnn.SpatialConvolution(8*nc, 32*nc, 1, 1, 1, 1, 0, 0)(p2L6res1c3_relu)
p2L6res1=nn.CAddTable(false)({p2L6res0_relu,p2L6res1c4})
p2L6res1_bn=nn.SpatialBatchNormalization(32*nc)(p2L6res1)
p2L6res1_relu=cudnn.ReLU(true)(p2L6res1_bn)
p2L6res1_drop=nn.SpatialDropout(opt.dropoutProb)(p2L6res1_relu)
p2L1up0=nn.SpatialConvolution(nc, opt.nclass, 3, 3, 1, 1, 1, 1)(p2L1c_relu)
p2L1up0_bn=nn.SpatialBatchNormalization(opt.nclass)(p2L1up0)
p2L1up0_relu=cudnn.ReLU(true)(p2L1up0_bn)
p2L2up=nn.SpatialFullConvolution(8*nc,opt.nclass,4,4,2,2,1,1)(p2L2res2_relu)
p2L2up_bn=nn.SpatialBatchNormalization(opt.nclass)(p2L2up)
p2L2up_relu=cudnn.ReLU(true)(p2L2up_bn)
p2L3ups1=nn.SpatialFullConvolution(16*nc,2*opt.nclass,4,4,2,2,1,1)(p2L3res1_relu)
p2L3ups1_bn=nn.SpatialBatchNormalization(2*opt.nclass)(p2L3ups1)
p2L3ups1_relu=cudnn.ReLU(true)(p2L3ups1_bn)
p2L3up=nn.SpatialFullConvolution(2*opt.nclass,opt.nclass,4,4,2,2,1,1)(p2L3ups1_relu)
p2L3up_bn=nn.SpatialBatchNormalization(opt.nclass)(p2L3up)
p2L3up_relu=cudnn.ReLU(true)(p2L3up_bn)
##-- XX -> 2XX-2 4XX-6 8XX-14 16XX-30-> 16XX-32
p2L4ups1=nn.SpatialFullConvolution(32*nc,4*opt.nclass,4,4,2,2,1,1)(p2L4res1_drop) 
p2L4ups1_bn=nn.SpatialBatchNormalization(4*opt.nclass)(p2L4ups1)
p2L4ups1_relu=cudnn.ReLU(true)(p2L4ups1_bn)
p2L4ups2=nn.SpatialFullConvolution(4*opt.nclass,2*opt.nclass,4,4,2,2,1,1)(p2L4ups1_relu)
p2L4ups2_bn=nn.SpatialBatchNormalization(2*opt.nclass)(p2L4ups2)
p2L4ups2_relu=cudnn.ReLU(true)(p2L4ups2_bn)
p2L4up=nn.SpatialFullConvolution(2*opt.nclass,opt.nclass,4,4,2,2,1,1)(p2L4ups2_relu)
p2L4up_bn=nn.SpatialBatchNormalization(opt.nclass)(p2L4up)
p2L4up_relu=cudnn.ReLU(true)(p2L4up_bn)
p2L5ups1=nn.SpatialFullConvolution(32*nc,8*opt.nclass,4,4,2,2,1,1)(p2L5res1_drop)
p2L5ups1_bn=nn.SpatialBatchNormalization(8*opt.nclass)(p2L5ups1)
p2L5ups1_relu=cudnn.ReLU(true)(p2L5ups1_bn)
p2L5ups2=nn.SpatialFullConvolution(8*opt.nclass,4*opt.nclass,4,4,2,2,1,1)(p2L5ups1_relu)
p2L5ups2_bn=nn.SpatialBatchNormalization(4*opt.nclass)(p2L5ups2)
p2L5ups2_relu=cudnn.ReLU(true)(p2L5ups2_bn)
p2L5ups3=nn.SpatialFullConvolution(4*opt.nclass,2*opt.nclass,4,4,2,2,1,1)(p2L5ups2_relu)
p2L5ups3_bn=nn.SpatialBatchNormalization(2*opt.nclass)(p2L5ups3)
p2L5ups3_relu=cudnn.ReLU(true)(p2L5ups3_bn)
p2L5up=nn.SpatialFullConvolution(2*opt.nclass,opt.nclass,4,4,2,2,1,1)(p2L5ups3_relu)
p2L5up_bn=nn.SpatialBatchNormalization(opt.nclass)(p2L5up)
p2L5up_relu=cudnn.ReLU(true)(p2L5up_bn)
p2L6ups1=nn.SpatialFullConvolution(32*nc,16*opt.nclass,4,4,2,2,1,1)(p2L6res1_drop)
p2L6ups1_bn=nn.SpatialBatchNormalization(16*opt.nclass)(p2L6ups1)
p2L6ups1_relu=cudnn.ReLU(true)(p2L6ups1_bn)
p2L6ups2=nn.SpatialFullConvolution(16*opt.nclass,8*opt.nclass,4,4,2,2,1,1)(p2L6ups1_relu)
p2L6ups2_bn=nn.SpatialBatchNormalization(8*opt.nclass)(p2L6ups2)
p2L6ups2_relu=cudnn.ReLU(true)(p2L6ups2_bn)
p2L6ups3=nn.SpatialFullConvolution(8*opt.nclass,4*opt.nclass,4,4,2,2,1,1)(p2L6ups2_relu)
p2L6ups3_bn=nn.SpatialBatchNormalization(4*opt.nclass)(p2L6ups3)
p2L6ups3_relu=cudnn.ReLU(true)(p2L6ups3_bn)
p2L6ups4=nn.SpatialFullConvolution(4*opt.nclass,2*opt.nclass,4,4,2,2,1,1)(p2L6ups3_relu)
p2L6ups4_bn=nn.SpatialBatchNormalization(2*opt.nclass)(p2L6ups4)
p2L6ups4_relu=cudnn.ReLU(true)(p2L6ups4_bn)
p2L6up=nn.SpatialFullConvolution(2*opt.nclass,opt.nclass,4,4,2,2,1,1)(p2L6ups4_relu)
p2L6up_bn=nn.SpatialBatchNormalization(opt.nclass)(p2L6up)
p2L6up_relu=cudnn.ReLU(true)(p2L6up_bn)
p2Lallup1 = nn.JoinTable(1,3)({p2L1up0_relu,p2L2up_relu,p2L3up_relu,p2L4up_relu,p2L5up_relu,p2L6up_relu})
p2Lallup2 = nn.SpatialConvolution(6*opt.nclass, opt.nclass, 3, 3, 1, 1, 1, 1)(p2Lallup1)
p2Lallup2_bn = nn.SpatialBatchNormalization(opt.nclass)(p2Lallup2)
p2Lallup3 = cudnn.ReLU(true)(p2Lallup2_bn)
p2Lallup4 = nn.SpatialConvolution(opt.nclass, opt.nclass, 1, 1, 1, 1, 0, 0)(p2Lallup3)
p2L1up20=nn.SpatialConvolution(nc, opt.nclass, 3, 3, 1, 1, 1, 1)(p2L1c_relu)
p2L1up20_bn=nn.SpatialBatchNormalization(opt.nclass)(p2L1up20)
p2L1up20_relu=cudnn.ReLU(true)(p2L1up20_bn)
p2L2up2=nn.SpatialFullConvolution(8*nc,opt.nclass,4,4,2,2,1,1)(p2L2res2_relu)
p2L2up2_bn=nn.SpatialBatchNormalization(opt.nclass)(p2L2up2)
p2L2up2_relu=cudnn.ReLU(true)(p2L2up2_bn)
p2L3up2s1=nn.SpatialFullConvolution(16*nc,2*opt.nclass,4,4,2,2,1,1)(p2L3res1_relu)
p2L3up2s1_bn=nn.SpatialBatchNormalization(2*opt.nclass)(p2L3up2s1)
p2L3up2s1_relu=cudnn.ReLU(true)(p2L3up2s1_bn)
p2L3up2=nn.SpatialFullConvolution(2*opt.nclass,opt.nclass,4,4,2,2,1,1)(p2L3up2s1_relu)
p2L3up2_bn=nn.SpatialBatchNormalization(opt.nclass)(p2L3up2)
p2L3up2_relu=cudnn.ReLU(true)(p2L3up2_bn)
p2L4up2s1=nn.SpatialFullConvolution(32*nc,4*opt.nclass,4,4,2,2,1,1)(p2L4res1_drop) 
p2L4up2s1_bn=nn.SpatialBatchNormalization(4*opt.nclass)(p2L4up2s1)
p2L4up2s1_relu=cudnn.ReLU(true)(p2L4up2s1_bn)
p2L4up2s2=nn.SpatialFullConvolution(4*opt.nclass,2*opt.nclass,4,4,2,2,1,1)(p2L4up2s1_relu)
p2L4up2s2_bn=nn.SpatialBatchNormalization(2*opt.nclass)(p2L4up2s2)
p2L4up2s2_relu=cudnn.ReLU(true)(p2L4up2s2_bn)
p2L4up2=nn.SpatialFullConvolution(2*opt.nclass,opt.nclass,4,4,2,2,1,1)(p2L4up2s2_relu)
p2L4up2_bn=nn.SpatialBatchNormalization(opt.nclass)(p2L4up2)
p2L4up2_relu=cudnn.ReLU(true)(p2L4up2_bn)
p2L5up2s1=nn.SpatialFullConvolution(32*nc,8*opt.nclass,4,4,2,2,1,1)(p2L5res1_drop)
p2L5up2s1_bn=nn.SpatialBatchNormalization(8*opt.nclass)(p2L5up2s1)
p2L5up2s1_relu=cudnn.ReLU(true)(p2L5up2s1_bn)
p2L5up2s2=nn.SpatialFullConvolution(8*opt.nclass,4*opt.nclass,4,4,2,2,1,1)(p2L5up2s1_relu)
p2L5up2s2_bn=nn.SpatialBatchNormalization(4*opt.nclass)(p2L5up2s2)
p2L5up2s2_relu=cudnn.ReLU(true)(p2L5up2s2_bn)
p2L5up2s3=nn.SpatialFullConvolution(4*opt.nclass,2*opt.nclass,4,4,2,2,1,1)(p2L5up2s2_relu)
p2L5up2s3_bn=nn.SpatialBatchNormalization(2*opt.nclass)(p2L5up2s3)
p2L5up2s3_relu=cudnn.ReLU(true)(p2L5up2s3_bn)
p2L5up2=nn.SpatialFullConvolution(2*opt.nclass,opt.nclass,4,4,2,2,1,1)(p2L5up2s3_relu)
p2L5up2_bn=nn.SpatialBatchNormalization(opt.nclass)(p2L5up2)
p2L5up2_relu=cudnn.ReLU(true)(p2L5up2_bn)
p2L6up2s1=nn.SpatialFullConvolution(32*nc,16*opt.nclass,4,4,2,2,1,1)(p2L6res1_drop)
p2L6up2s1_bn=nn.SpatialBatchNormalization(16*opt.nclass)(p2L6up2s1)
p2L6up2s1_relu=cudnn.ReLU(true)(p2L6up2s1_bn)
p2L6up2s2=nn.SpatialFullConvolution(16*opt.nclass,8*opt.nclass,4,4,2,2,1,1)(p2L6up2s1_relu)
p2L6up2s2_bn=nn.SpatialBatchNormalization(8*opt.nclass)(p2L6up2s2)
p2L6up2s2_relu=cudnn.ReLU(true)(p2L6up2s2_bn)
p2L6up2s3=nn.SpatialFullConvolution(8*opt.nclass,4*opt.nclass,4,4,2,2,1,1)(p2L6up2s2_relu)
p2L6up2s3_bn=nn.SpatialBatchNormalization(4*opt.nclass)(p2L6up2s3)
p2L6up2s3_relu=cudnn.ReLU(true)(p2L6up2s3_bn)
p2L6up2s4=nn.SpatialFullConvolution(4*opt.nclass,2*opt.nclass,4,4,2,2,1,1)(p2L6up2s3_relu)
p2L6up2s4_bn=nn.SpatialBatchNormalization(2*opt.nclass)(p2L6up2s4)
p2L6up2s4_relu=cudnn.ReLU(true)(p2L6up2s4_bn)
p2L6up2=nn.SpatialFullConvolution(2*opt.nclass,opt.nclass,4,4,2,2,1,1)(p2L6up2s4_relu)
p2L6up2_bn=nn.SpatialBatchNormalization(opt.nclass)(p2L6up2)
p2L6up2_relu=cudnn.ReLU(true)(p2L6up2_bn)
p2Lallup21 = nn.JoinTable(1,3)({p2L1up20_relu,p2L2up2_relu,p2L3up2_relu,p2L4up2_relu,p2L5up2_relu,p2L6up2_relu})
p2Lallup22 = nn.SpatialConvolution(6*opt.nclass, opt.nclass, 3, 3, 1, 1, 1, 1)(p2Lallup21)
p2Lallup22_bn = nn.SpatialBatchNormalization(opt.nclass)(p2Lallup22)
p2Lallup23 = cudnn.ReLU(true)(p2Lallup22_bn)
p2Lallup24 = nn.SpatialConvolution(opt.nclass, opt.nclass, 1, 1, 1, 1, 0, 0)(p2Lallup23)

p3L1b=cudnn.SpatialConvolution(opt.imageType, nc, 3, 3, 1, 1, 1, 1)(input3) ##--4XX+28
p3L1b_bn=nn.SpatialBatchNormalization(nc)(p3L1b)
p3L1b_relu=cudnn.ReLU(true)(p3L1b_bn)
p3L1c=cudnn.SpatialConvolution(nc, nc, 3, 3, 1, 1, 1, 1)(p3L1b_relu) ##--4XX+28
p3L1c_bn=nn.SpatialBatchNormalization(nc)(p3L1c)
p3L1c_relu=cudnn.ReLU(true)(p3L1c_bn)
p3L2a=cudnn.SpatialMaxPooling(2, 2, 2, 2)(p3L1c_relu)
p3L2res1_input0=nn.Padding(1,7*nc,3)(p3L2a)
p3L2res1c2=cudnn.SpatialConvolution(nc, 2*nc, 1, 1, 1, 1, 0, 0)(p3L2a)
p3L2res1c2_bn=nn.SpatialBatchNormalization(2*nc)(p3L2res1c2)
p3L2res1c2_relu=cudnn.ReLU(true)(p3L2res1c2_bn)
p3L2res1c3=cudnn.SpatialConvolution(2*nc, 2*nc, 3, 3, 1, 1, 1, 1)(p3L2res1c2_relu)
p3L2res1c3_bn=nn.SpatialBatchNormalization(2*nc)(p3L2res1c3)
p3L2res1c3_relu=cudnn.ReLU(true)(p3L2res1c3_bn)
p3L2res1c4=cudnn.SpatialConvolution(2*nc, 8*nc, 1, 1, 1, 1, 0, 0)(p3L2res1c3_relu)
p3L2res1=nn.CAddTable(false)({p3L2res1_input0,p3L2res1c4})
p3L2res1_bn=nn.SpatialBatchNormalization(8*nc)(p3L2res1)
p3L2res1_relu=cudnn.ReLU(true)(p3L2res1_bn)
p3L2res2c2=cudnn.SpatialConvolution(8*nc, 2*nc, 1, 1, 1, 1, 0, 0)(p3L2res1_relu)
p3L2res2c2_bn=nn.SpatialBatchNormalization(2*nc)(p3L2res2c2)
p3L2res2c2_relu=cudnn.ReLU(true)(p3L2res2c2_bn)
p3L2res2c3=cudnn.SpatialConvolution(2*nc, 2*nc, 3, 3, 1, 1, 1, 1)(p3L2res2c2_relu)
p3L2res2c3_bn=nn.SpatialBatchNormalization(2*nc)(p3L2res2c3)
p3L2res2c3_relu=cudnn.ReLU(true)(p3L2res2c3_bn)
p3L2res2c4=cudnn.SpatialConvolution(2*nc, 8*nc, 1, 1, 1, 1, 0, 0)(p3L2res2c3_relu)
p3L2res2=nn.CAddTable(false)({p3L2res1_relu,p3L2res2c4})
p3L2res2_bn=nn.SpatialBatchNormalization(8*nc)(p3L2res2)
p3L2res2_relu=cudnn.ReLU(true)(p3L2res2_bn)
p3L3a=cudnn.SpatialMaxPooling(2, 2, 2, 2)(p3L2res2_relu)
p3L3res0_input0=nn.Padding(1,8*nc,3)(p3L3a)
p3L3res0c2=cudnn.SpatialConvolution(8*nc, 4*nc, 1, 1, 1, 1, 0, 0)(p3L3a)
p3L3res0c2_bn=nn.SpatialBatchNormalization(4*nc)(p3L3res0c2)
p3L3res0c2_relu=cudnn.ReLU(true)(p3L3res0c2_bn)
p3L3res0c3=cudnn.SpatialConvolution(4*nc, 4*nc, 3, 3, 1, 1, 1, 1)(p3L3res0c2_relu)
p3L3res0c3_bn=nn.SpatialBatchNormalization(4*nc)(p3L3res0c3)
p3L3res0c3_relu=cudnn.ReLU(true)(p3L3res0c3_bn)
p3L3res0c4=cudnn.SpatialConvolution(4*nc, 16*nc, 1, 1, 1, 1, 0, 0)(p3L3res0c3_relu)
p3L3res0=nn.CAddTable(false)({p3L3res0_input0,p3L3res0c4})
p3L3res0_bn=nn.SpatialBatchNormalization(16*nc)(p3L3res0)
p3L3res0_relu=cudnn.ReLU(true)(p3L3res0_bn)
p3L3res1c2=cudnn.SpatialConvolution(16*nc, 4*nc, 1, 1, 1, 1, 0, 0)(p3L3res0_relu)
p3L3res1c2_bn=nn.SpatialBatchNormalization(4*nc)(p3L3res1c2)
p3L3res1c2_relu=cudnn.ReLU(true)(p3L3res1c2_bn)
p3L3res1c3=cudnn.SpatialConvolution(4*nc, 4*nc, 3, 3, 1, 1, 1, 1)(p3L3res1c2_relu)
p3L3res1c3_bn=nn.SpatialBatchNormalization(4*nc)(p3L3res1c3)
p3L3res1c3_relu=cudnn.ReLU(true)(p3L3res1c3_bn)
p3L3res1c4=cudnn.SpatialConvolution(4*nc, 16*nc, 1, 1, 1, 1, 0, 0)(p3L3res1c3_relu)
p3L3res1=nn.CAddTable(false)({p3L3res0_relu,p3L3res1c4})
p3L3res1_bn=nn.SpatialBatchNormalization(16*nc)(p3L3res1)
p3L3res1_relu=cudnn.ReLU(true)(p3L3res1_bn)
p3L4a=cudnn.SpatialMaxPooling(2, 2, 2, 2)(p3L3res1_relu)
p3L4res0_input0=nn.Padding(1,16*nc,3)(p3L4a)
p3L4res0c2=cudnn.SpatialConvolution(16*nc, 8*nc, 1, 1, 1, 1, 0, 0)(p3L4a)
p3L4res0c2_bn=nn.SpatialBatchNormalization(8*nc)(p3L4res0c2)
p3L4res0c2_relu=cudnn.ReLU(true)(p3L4res0c2_bn)
p3L4res0c3=cudnn.SpatialConvolution(8*nc, 8*nc, 3, 3, 1, 1, 1, 1)(p3L4res0c2_relu)
p3L4res0c3_bn=nn.SpatialBatchNormalization(8*nc)(p3L4res0c3)
p3L4res0c3_relu=cudnn.ReLU(true)(p3L4res0c3_bn)
p3L4res0c4=cudnn.SpatialConvolution(8*nc, 32*nc, 1, 1, 1, 1, 0, 0)(p3L4res0c3_relu)
p3L4res0=nn.CAddTable(false)({p3L4res0_input0,p3L4res0c4})
p3L4res0_bn=nn.SpatialBatchNormalization(32*nc)(p3L4res0)
p3L4res0_relu=cudnn.ReLU(true)(p3L4res0_bn)
p3L4res1c2=cudnn.SpatialConvolution(32*nc, 8*nc, 1, 1, 1, 1, 0, 0)(p3L4res0_relu)
p3L4res1c2_bn=nn.SpatialBatchNormalization(8*nc)(p3L4res1c2)
p3L4res1c2_relu=cudnn.ReLU(true)(p3L4res1c2_bn)
p3L4res1c3=cudnn.SpatialConvolution(8*nc, 8*nc, 3, 3, 1, 1, 1, 1)(p3L4res1c2_relu)
p3L4res1c3_bn=nn.SpatialBatchNormalization(8*nc)(p3L4res1c3)
p3L4res1c3_relu=cudnn.ReLU(true)(p3L4res1c3_bn)
p3L4res1c4=cudnn.SpatialConvolution(8*nc, 32*nc, 1, 1, 1, 1, 0, 0)(p3L4res1c3_relu)
p3L4res1=nn.CAddTable(false)({p3L4res0_relu,p3L4res1c4})
p3L4res1_bn=nn.SpatialBatchNormalization(32*nc)(p3L4res1)
p3L4res1_relu=cudnn.ReLU(true)(p3L4res1_bn)
p3L4res1_drop=nn.SpatialDropout(opt.dropoutProb)(p3L4res1_relu)
p3L5a=cudnn.SpatialMaxPooling(2, 2, 2, 2)(p3L4res1_drop)
p3L5res0c2=cudnn.SpatialConvolution(32*nc, 8*nc, 1, 1, 1, 1, 0, 0)(p3L5a)
p3L5res0c2_bn=nn.SpatialBatchNormalization(8*nc)(p3L5res0c2)
p3L5res0c2_relu=cudnn.ReLU(true)(p3L5res0c2_bn)
p3L5res0c3=cudnn.SpatialConvolution(8*nc, 8*nc, 3, 3, 1, 1, 1, 1)(p3L5res0c2_relu)
p3L5res0c3_bn=nn.SpatialBatchNormalization(8*nc)(p3L5res0c3)
p3L5res0c3_relu=cudnn.ReLU(true)(p3L5res0c3_bn)
p3L5res0c4=cudnn.SpatialConvolution(8*nc, 32*nc, 1, 1, 1, 1, 0, 0)(p3L5res0c3_relu)
p3L5res0=nn.CAddTable(false)({p3L5a,p3L5res0c4})
p3L5res0_bn=nn.SpatialBatchNormalization(32*nc)(p3L5res0)
p3L5res0_relu=cudnn.ReLU(true)(p3L5res0_bn)
p3L5res1c2=cudnn.SpatialConvolution(32*nc, 8*nc, 1, 1, 1, 1, 0, 0)(p3L5res0_relu)
p3L5res1c2_bn=nn.SpatialBatchNormalization(8*nc)(p3L5res1c2)
p3L5res1c2_relu=cudnn.ReLU(true)(p3L5res1c2_bn)
p3L5res1c3=cudnn.SpatialConvolution(8*nc, 8*nc, 3, 3, 1, 1, 1, 1)(p3L5res1c2_relu)
p3L5res1c3_bn=nn.SpatialBatchNormalization(8*nc)(p3L5res1c3)
p3L5res1c3_relu=cudnn.ReLU(true)(p3L5res1c3_bn)
p3L5res1c4=cudnn.SpatialConvolution(8*nc, 32*nc, 1, 1, 1, 1, 0, 0)(p3L5res1c3_relu)
p3L5res1=nn.CAddTable(false)({p3L5res0_relu,p3L5res1c4})
p3L5res1_bn=nn.SpatialBatchNormalization(32*nc)(p3L5res1)
p3L5res1_relu=cudnn.ReLU(true)(p3L5res1_bn)
p3L5res1_drop=nn.SpatialDropout(opt.dropoutProb)(p3L5res1_relu)
p3L6a=cudnn.SpatialMaxPooling(2, 2, 2, 2)(p3L5res1_drop)
p3L6res0c2=cudnn.SpatialConvolution(32*nc, 8*nc, 1, 1, 1, 1, 0, 0)(p3L6a)
p3L6res0c2_bn=nn.SpatialBatchNormalization(8*nc)(p3L6res0c2)
p3L6res0c2_relu=cudnn.ReLU(true)(p3L6res0c2_bn)
p3L6res0c3=cudnn.SpatialConvolution(8*nc, 8*nc, 3, 3, 1, 1, 1, 1)(p3L6res0c2_relu)
p3L6res0c3_bn=nn.SpatialBatchNormalization(8*nc)(p3L6res0c3)
p3L6res0c3_relu=cudnn.ReLU(true)(p3L6res0c3_bn)
p3L6res0c4=cudnn.SpatialConvolution(8*nc, 32*nc, 1, 1, 1, 1, 0, 0)(p3L6res0c3_relu)
p3L6res0=nn.CAddTable(false)({p3L6a,p3L6res0c4})
p3L6res0_bn=nn.SpatialBatchNormalization(32*nc)(p3L6res0)
p3L6res0_relu=cudnn.ReLU(true)(p3L6res0_bn)
p3L6res1c2=cudnn.SpatialConvolution(32*nc, 8*nc, 1, 1, 1, 1, 0, 0)(p3L6res0_relu)
p3L6res1c2_bn=nn.SpatialBatchNormalization(8*nc)(p3L6res1c2)
p3L6res1c2_relu=cudnn.ReLU(true)(p3L6res1c2_bn)
p3L6res1c3=cudnn.SpatialConvolution(8*nc, 8*nc, 3, 3, 1, 1, 1, 1)(p3L6res1c2_relu)
p3L6res1c3_bn=nn.SpatialBatchNormalization(8*nc)(p3L6res1c3)
p3L6res1c3_relu=cudnn.ReLU(true)(p3L6res1c3_bn)
p3L6res1c4=cudnn.SpatialConvolution(8*nc, 32*nc, 1, 1, 1, 1, 0, 0)(p3L6res1c3_relu)
p3L6res1=nn.CAddTable(false)({p3L6res0_relu,p3L6res1c4})
p3L6res1_bn=nn.SpatialBatchNormalization(32*nc)(p3L6res1)
p3L6res1_relu=cudnn.ReLU(true)(p3L6res1_bn)
p3L6res1_drop=nn.SpatialDropout(opt.dropoutProb)(p3L6res1_relu)
p3L1up0=nn.SpatialConvolution(nc, opt.nclass, 3, 3, 1, 1, 1, 1)(p3L1c_relu)
p3L1up0_bn=nn.SpatialBatchNormalization(opt.nclass)(p3L1up0)
p3L1up0_relu=cudnn.ReLU(true)(p3L1up0_bn)
p3L2up=nn.SpatialFullConvolution(8*nc,opt.nclass,4,4,2,2,1,1)(p3L2res2_relu)
p3L2up_bn=nn.SpatialBatchNormalization(opt.nclass)(p3L2up)
p3L2up_relu=cudnn.ReLU(true)(p3L2up_bn)
p3L3ups1=nn.SpatialFullConvolution(16*nc,2*opt.nclass,4,4,2,2,1,1)(p3L3res1_relu)
p3L3ups1_bn=nn.SpatialBatchNormalization(2*opt.nclass)(p3L3ups1)
p3L3ups1_relu=cudnn.ReLU(true)(p3L3ups1_bn)
p3L3up=nn.SpatialFullConvolution(2*opt.nclass,opt.nclass,4,4,2,2,1,1)(p3L3ups1_relu)
p3L3up_bn=nn.SpatialBatchNormalization(opt.nclass)(p3L3up)
p3L3up_relu=cudnn.ReLU(true)(p3L3up_bn)
#   #-- XX -> 2XX-2 4XX-6 8XX-14 16XX-30-> 16XX-32
p3L4ups1=nn.SpatialFullConvolution(32*nc,4*opt.nclass,4,4,2,2,1,1)(p3L4res1_drop) 
p3L4ups1_bn=nn.SpatialBatchNormalization(4*opt.nclass)(p3L4ups1)
p3L4ups1_relu=cudnn.ReLU(true)(p3L4ups1_bn)
p3L4ups2=nn.SpatialFullConvolution(4*opt.nclass,2*opt.nclass,4,4,2,2,1,1)(p3L4ups1_relu)
p3L4ups2_bn=nn.SpatialBatchNormalization(2*opt.nclass)(p3L4ups2)
p3L4ups2_relu=cudnn.ReLU(true)(p3L4ups2_bn)
p3L4up=nn.SpatialFullConvolution(2*opt.nclass,opt.nclass,4,4,2,2,1,1)(p3L4ups2_relu)
p3L4up_bn=nn.SpatialBatchNormalization(opt.nclass)(p3L4up)
p3L4up_relu=cudnn.ReLU(true)(p3L4up_bn)
p3L5ups1=nn.SpatialFullConvolution(32*nc,8*opt.nclass,4,4,2,2,1,1)(p3L5res1_drop)
p3L5ups1_bn=nn.SpatialBatchNormalization(8*opt.nclass)(p3L5ups1)
p3L5ups1_relu=cudnn.ReLU(true)(p3L5ups1_bn)
p3L5ups2=nn.SpatialFullConvolution(8*opt.nclass,4*opt.nclass,4,4,2,2,1,1)(p3L5ups1_relu)
p3L5ups2_bn=nn.SpatialBatchNormalization(4*opt.nclass)(p3L5ups2)
p3L5ups2_relu=cudnn.ReLU(true)(p3L5ups2_bn)
p3L5ups3=nn.SpatialFullConvolution(4*opt.nclass,2*opt.nclass,4,4,2,2,1,1)(p3L5ups2_relu)
p3L5ups3_bn=nn.SpatialBatchNormalization(2*opt.nclass)(p3L5ups3)
p3L5ups3_relu=cudnn.ReLU(true)(p3L5ups3_bn)
p3L5up=nn.SpatialFullConvolution(2*opt.nclass,opt.nclass,4,4,2,2,1,1)(p3L5ups3_relu)
p3L5up_bn=nn.SpatialBatchNormalization(opt.nclass)(p3L5up)
p3L5up_relu=cudnn.ReLU(true)(p3L5up_bn)
p3L6ups1=nn.SpatialFullConvolution(32*nc,16*opt.nclass,4,4,2,2,1,1)(p3L6res1_drop)
p3L6ups1_bn=nn.SpatialBatchNormalization(16*opt.nclass)(p3L6ups1)
p3L6ups1_relu=cudnn.ReLU(true)(p3L6ups1_bn)
p3L6ups2=nn.SpatialFullConvolution(16*opt.nclass,8*opt.nclass,4,4,2,2,1,1)(p3L6ups1_relu)
p3L6ups2_bn=nn.SpatialBatchNormalization(8*opt.nclass)(p3L6ups2)
p3L6ups2_relu=cudnn.ReLU(true)(p3L6ups2_bn)
p3L6ups3=nn.SpatialFullConvolution(8*opt.nclass,4*opt.nclass,4,4,2,2,1,1)(p3L6ups2_relu)
p3L6ups3_bn=nn.SpatialBatchNormalization(4*opt.nclass)(p3L6ups3)
p3L6ups3_relu=cudnn.ReLU(true)(p3L6ups3_bn)
p3L6ups4=nn.SpatialFullConvolution(4*opt.nclass,2*opt.nclass,4,4,2,2,1,1)(p3L6ups3_relu)
p3L6ups4_bn=nn.SpatialBatchNormalization(2*opt.nclass)(p3L6ups4)
p3L6ups4_relu=cudnn.ReLU(true)(p3L6ups4_bn)
p3L6up=nn.SpatialFullConvolution(2*opt.nclass,opt.nclass,4,4,2,2,1,1)(p3L6ups4_relu)
p3L6up_bn=nn.SpatialBatchNormalization(opt.nclass)(p3L6up)
p3L6up_relu=cudnn.ReLU(true)(p3L6up_bn)
p3Lallup1 = nn.JoinTable(1,3)({p3L1up0_relu,p3L2up_relu,p3L3up_relu,p3L4up_relu,p3L5up_relu,p3L6up_relu})
p3Lallup2 = nn.SpatialConvolution(6*opt.nclass, opt.nclass, 3, 3, 1, 1, 1, 1)(p3Lallup1)
p3Lallup2_bn = nn.SpatialBatchNormalization(opt.nclass)(p3Lallup2)
p3Lallup3 = cudnn.ReLU(true)(p3Lallup2_bn)
p3Lallup4 = nn.SpatialConvolution(opt.nclass, opt.nclass, 1, 1, 1, 1, 0, 0)(p3Lallup3)
p3L1up20=nn.SpatialConvolution(nc, opt.nclass, 3, 3, 1, 1, 1, 1)(p3L1c_relu)
p3L1up20_bn=nn.SpatialBatchNormalization(opt.nclass)(p3L1up20)
p3L1up20_relu=cudnn.ReLU(true)(p3L1up20_bn)
p3L2up2=nn.SpatialFullConvolution(8*nc,opt.nclass,4,4,2,2,1,1)(p3L2res2_relu)
p3L2up2_bn=nn.SpatialBatchNormalization(opt.nclass)(p3L2up2)
p3L2up2_relu=cudnn.ReLU(true)(p3L2up2_bn)
p3L3up2s1=nn.SpatialFullConvolution(16*nc,2*opt.nclass,4,4,2,2,1,1)(p3L3res1_relu)
p3L3up2s1_bn=nn.SpatialBatchNormalization(2*opt.nclass)(p3L3up2s1)
p3L3up2s1_relu=cudnn.ReLU(true)(p3L3up2s1_bn)
p3L3up2=nn.SpatialFullConvolution(2*opt.nclass,opt.nclass,4,4,2,2,1,1)(p3L3up2s1_relu)
p3L3up2_bn=nn.SpatialBatchNormalization(opt.nclass)(p3L3up2)
p3L3up2_relu=cudnn.ReLU(true)(p3L3up2_bn)
p3L4up2s1=nn.SpatialFullConvolution(32*nc,4*opt.nclass,4,4,2,2,1,1)(p3L4res1_drop) 
p3L4up2s1_bn=nn.SpatialBatchNormalization(4*opt.nclass)(p3L4up2s1)
p3L4up2s1_relu=cudnn.ReLU(true)(p3L4up2s1_bn)
p3L4up2s2=nn.SpatialFullConvolution(4*opt.nclass,2*opt.nclass,4,4,2,2,1,1)(p3L4up2s1_relu)
p3L4up2s2_bn=nn.SpatialBatchNormalization(2*opt.nclass)(p3L4up2s2)
p3L4up2s2_relu=cudnn.ReLU(true)(p3L4up2s2_bn)
p3L4up2=nn.SpatialFullConvolution(2*opt.nclass,opt.nclass,4,4,2,2,1,1)(p3L4up2s2_relu)
p3L4up2_bn=nn.SpatialBatchNormalization(opt.nclass)(p3L4up2)
p3L4up2_relu=cudnn.ReLU(true)(p3L4up2_bn)
p3L5up2s1=nn.SpatialFullConvolution(32*nc,8*opt.nclass,4,4,2,2,1,1)(p3L5res1_drop)
p3L5up2s1_bn=nn.SpatialBatchNormalization(8*opt.nclass)(p3L5up2s1)
p3L5up2s1_relu=cudnn.ReLU(true)(p3L5up2s1_bn)
p3L5up2s2=nn.SpatialFullConvolution(8*opt.nclass,4*opt.nclass,4,4,2,2,1,1)(p3L5up2s1_relu)
p3L5up2s2_bn=nn.SpatialBatchNormalization(4*opt.nclass)(p3L5up2s2)
p3L5up2s2_relu=cudnn.ReLU(true)(p3L5up2s2_bn)
p3L5up2s3=nn.SpatialFullConvolution(4*opt.nclass,2*opt.nclass,4,4,2,2,1,1)(p3L5up2s2_relu)
p3L5up2s3_bn=nn.SpatialBatchNormalization(2*opt.nclass)(p3L5up2s3)
p3L5up2s3_relu=cudnn.ReLU(true)(p3L5up2s3_bn)
p3L5up2=nn.SpatialFullConvolution(2*opt.nclass,opt.nclass,4,4,2,2,1,1)(p3L5up2s3_relu)
p3L5up2_bn=nn.SpatialBatchNormalization(opt.nclass)(p3L5up2)
p3L5up2_relu=cudnn.ReLU(true)(p3L5up2_bn)
p3L6up2s1=nn.SpatialFullConvolution(32*nc,16*opt.nclass,4,4,2,2,1,1)(p3L6res1_drop)
p3L6up2s1_bn=nn.SpatialBatchNormalization(16*opt.nclass)(p3L6up2s1)
p3L6up2s1_relu=cudnn.ReLU(true)(p3L6up2s1_bn)
p3L6up2s2=nn.SpatialFullConvolution(16*opt.nclass,8*opt.nclass,4,4,2,2,1,1)(p3L6up2s1_relu)
p3L6up2s2_bn=nn.SpatialBatchNormalization(8*opt.nclass)(p3L6up2s2)
p3L6up2s2_relu=cudnn.ReLU(true)(p3L6up2s2_bn)
p3L6up2s3=nn.SpatialFullConvolution(8*opt.nclass,4*opt.nclass,4,4,2,2,1,1)(p3L6up2s2_relu)
p3L6up2s3_bn=nn.SpatialBatchNormalization(4*opt.nclass)(p3L6up2s3)
p3L6up2s3_relu=cudnn.ReLU(true)(p3L6up2s3_bn)
p3L6up2s4=nn.SpatialFullConvolution(4*opt.nclass,2*opt.nclass,4,4,2,2,1,1)(p3L6up2s3_relu)
p3L6up2s4_bn=nn.SpatialBatchNormalization(2*opt.nclass)(p3L6up2s4)
p3L6up2s4_relu=cudnn.ReLU(true)(p3L6up2s4_bn)
p3L6up2=nn.SpatialFullConvolution(2*opt.nclass,opt.nclass,4,4,2,2,1,1)(p3L6up2s4_relu)
p3L6up2_bn=nn.SpatialBatchNormalization(opt.nclass)(p3L6up2)
p3L6up2_relu=cudnn.ReLU(true)(p3L6up2_bn)
p3Lallup21 = nn.JoinTable(1,3)({p3L1up20_relu,p3L2up2_relu,p3L3up2_relu,p3L4up2_relu,p3L5up2_relu,p3L6up2_relu})
p3Lallup22 = nn.SpatialConvolution(6*opt.nclass, opt.nclass, 3, 3, 1, 1, 1, 1)(p3Lallup21)
p3Lallup22_bn = nn.SpatialBatchNormalization(opt.nclass)(p3Lallup22)
p3Lallup23 = cudnn.ReLU(true)(p3Lallup22_bn)
p3Lallup24 = nn.SpatialConvolution(opt.nclass, opt.nclass, 1, 1, 1, 1, 0, 0)(p3Lallup23)

p4L1b=cudnn.SpatialConvolution(opt.imageType, nc, 3, 3, 1, 1, 1, 1)(input4) ##--4XX+28
p4L1b_bn=nn.SpatialBatchNormalization(nc)(p4L1b)
p4L1b_relu=cudnn.ReLU(true)(p4L1b_bn)
p4L1c=cudnn.SpatialConvolution(nc, nc, 3, 3, 1, 1, 1, 1)(p4L1b_relu) ##--4XX+28
p4L1c_bn=nn.SpatialBatchNormalization(nc)(p4L1c)
p4L1c_relu=cudnn.ReLU(true)(p4L1c_bn)
p4L2a=cudnn.SpatialMaxPooling(2, 2, 2, 2)(p4L1c_relu)
p4L2res1_input0=nn.Padding(1,7*nc,3)(p4L2a)
p4L2res1c2=cudnn.SpatialConvolution(nc, 2*nc, 1, 1, 1, 1, 0, 0)(p4L2a)
p4L2res1c2_bn=nn.SpatialBatchNormalization(2*nc)(p4L2res1c2)
p4L2res1c2_relu=cudnn.ReLU(true)(p4L2res1c2_bn)
p4L2res1c3=cudnn.SpatialConvolution(2*nc, 2*nc, 3, 3, 1, 1, 1, 1)(p4L2res1c2_relu)
p4L2res1c3_bn=nn.SpatialBatchNormalization(2*nc)(p4L2res1c3)
p4L2res1c3_relu=cudnn.ReLU(true)(p4L2res1c3_bn)
p4L2res1c4=cudnn.SpatialConvolution(2*nc, 8*nc, 1, 1, 1, 1, 0, 0)(p4L2res1c3_relu)
p4L2res1=nn.CAddTable(false)({p4L2res1_input0,p4L2res1c4})
p4L2res1_bn=nn.SpatialBatchNormalization(8*nc)(p4L2res1)
p4L2res1_relu=cudnn.ReLU(true)(p4L2res1_bn)
p4L2res2c2=cudnn.SpatialConvolution(8*nc, 2*nc, 1, 1, 1, 1, 0, 0)(p4L2res1_relu)
p4L2res2c2_bn=nn.SpatialBatchNormalization(2*nc)(p4L2res2c2)
p4L2res2c2_relu=cudnn.ReLU(true)(p4L2res2c2_bn)
p4L2res2c3=cudnn.SpatialConvolution(2*nc, 2*nc, 3, 3, 1, 1, 1, 1)(p4L2res2c2_relu)
p4L2res2c3_bn=nn.SpatialBatchNormalization(2*nc)(p4L2res2c3)
p4L2res2c3_relu=cudnn.ReLU(true)(p4L2res2c3_bn)
p4L2res2c4=cudnn.SpatialConvolution(2*nc, 8*nc, 1, 1, 1, 1, 0, 0)(p4L2res2c3_relu)
p4L2res2=nn.CAddTable(false)({p4L2res1_relu,p4L2res2c4})
p4L2res2_bn=nn.SpatialBatchNormalization(8*nc)(p4L2res2)
p4L2res2_relu=cudnn.ReLU(true)(p4L2res2_bn)
p4L3a=cudnn.SpatialMaxPooling(2, 2, 2, 2)(p4L2res2_relu)
p4L3res0_input0=nn.Padding(1,8*nc,3)(p4L3a)
p4L3res0c2=cudnn.SpatialConvolution(8*nc, 4*nc, 1, 1, 1, 1, 0, 0)(p4L3a)
p4L3res0c2_bn=nn.SpatialBatchNormalization(4*nc)(p4L3res0c2)
p4L3res0c2_relu=cudnn.ReLU(true)(p4L3res0c2_bn)
p4L3res0c3=cudnn.SpatialConvolution(4*nc, 4*nc, 3, 3, 1, 1, 1, 1)(p4L3res0c2_relu)
p4L3res0c3_bn=nn.SpatialBatchNormalization(4*nc)(p4L3res0c3)
p4L3res0c3_relu=cudnn.ReLU(true)(p4L3res0c3_bn)
p4L3res0c4=cudnn.SpatialConvolution(4*nc, 16*nc, 1, 1, 1, 1, 0, 0)(p4L3res0c3_relu)
p4L3res0=nn.CAddTable(false)({p4L3res0_input0,p4L3res0c4})
p4L3res0_bn=nn.SpatialBatchNormalization(16*nc)(p4L3res0)
p4L3res0_relu=cudnn.ReLU(true)(p4L3res0_bn)
p4L3res1c2=cudnn.SpatialConvolution(16*nc, 4*nc, 1, 1, 1, 1, 0, 0)(p4L3res0_relu)
p4L3res1c2_bn=nn.SpatialBatchNormalization(4*nc)(p4L3res1c2)
p4L3res1c2_relu=cudnn.ReLU(true)(p4L3res1c2_bn)
p4L3res1c3=cudnn.SpatialConvolution(4*nc, 4*nc, 3, 3, 1, 1, 1, 1)(p4L3res1c2_relu)
p4L3res1c3_bn=nn.SpatialBatchNormalization(4*nc)(p4L3res1c3)
p4L3res1c3_relu=cudnn.ReLU(true)(p4L3res1c3_bn)
p4L3res1c4=cudnn.SpatialConvolution(4*nc, 16*nc, 1, 1, 1, 1, 0, 0)(p4L3res1c3_relu)
p4L3res1=nn.CAddTable(false)({p4L3res0_relu,p4L3res1c4})
p4L3res1_bn=nn.SpatialBatchNormalization(16*nc)(p4L3res1)
p4L3res1_relu=cudnn.ReLU(true)(p4L3res1_bn)
p4L4a=cudnn.SpatialMaxPooling(2, 2, 2, 2)(p4L3res1_relu)
p4L4res0_input0=nn.Padding(1,16*nc,3)(p4L4a)
p4L4res0c2=cudnn.SpatialConvolution(16*nc, 8*nc, 1, 1, 1, 1, 0, 0)(p4L4a)
p4L4res0c2_bn=nn.SpatialBatchNormalization(8*nc)(p4L4res0c2)
p4L4res0c2_relu=cudnn.ReLU(true)(p4L4res0c2_bn)
p4L4res0c3=cudnn.SpatialConvolution(8*nc, 8*nc, 3, 3, 1, 1, 1, 1)(p4L4res0c2_relu)
p4L4res0c3_bn=nn.SpatialBatchNormalization(8*nc)(p4L4res0c3)
p4L4res0c3_relu=cudnn.ReLU(true)(p4L4res0c3_bn)
p4L4res0c4=cudnn.SpatialConvolution(8*nc, 32*nc, 1, 1, 1, 1, 0, 0)(p4L4res0c3_relu)
p4L4res0=nn.CAddTable(false)({p4L4res0_input0,p4L4res0c4})
p4L4res0_bn=nn.SpatialBatchNormalization(32*nc)(p4L4res0)
p4L4res0_relu=cudnn.ReLU(true)(p4L4res0_bn)
p4L4res1c2=cudnn.SpatialConvolution(32*nc, 8*nc, 1, 1, 1, 1, 0, 0)(p4L4res0_relu)
p4L4res1c2_bn=nn.SpatialBatchNormalization(8*nc)(p4L4res1c2)
p4L4res1c2_relu=cudnn.ReLU(true)(p4L4res1c2_bn)
p4L4res1c3=cudnn.SpatialConvolution(8*nc, 8*nc, 3, 3, 1, 1, 1, 1)(p4L4res1c2_relu)
p4L4res1c3_bn=nn.SpatialBatchNormalization(8*nc)(p4L4res1c3)
p4L4res1c3_relu=cudnn.ReLU(true)(p4L4res1c3_bn)
p4L4res1c4=cudnn.SpatialConvolution(8*nc, 32*nc, 1, 1, 1, 1, 0, 0)(p4L4res1c3_relu)
p4L4res1=nn.CAddTable(false)({p4L4res0_relu,p4L4res1c4})
p4L4res1_bn=nn.SpatialBatchNormalization(32*nc)(p4L4res1)
p4L4res1_relu=cudnn.ReLU(true)(p4L4res1_bn)
p4L4res1_drop=nn.SpatialDropout(opt.dropoutProb)(p4L4res1_relu)
p4L5a=cudnn.SpatialMaxPooling(2, 2, 2, 2)(p4L4res1_drop)
p4L5res0c2=cudnn.SpatialConvolution(32*nc, 8*nc, 1, 1, 1, 1, 0, 0)(p4L5a)
p4L5res0c2_bn=nn.SpatialBatchNormalization(8*nc)(p4L5res0c2)
p4L5res0c2_relu=cudnn.ReLU(true)(p4L5res0c2_bn)
p4L5res0c3=cudnn.SpatialConvolution(8*nc, 8*nc, 3, 3, 1, 1, 1, 1)(p4L5res0c2_relu)
p4L5res0c3_bn=nn.SpatialBatchNormalization(8*nc)(p4L5res0c3)
p4L5res0c3_relu=cudnn.ReLU(true)(p4L5res0c3_bn)
p4L5res0c4=cudnn.SpatialConvolution(8*nc, 32*nc, 1, 1, 1, 1, 0, 0)(p4L5res0c3_relu)
p4L5res0=nn.CAddTable(false)({p4L5a,p4L5res0c4})
p4L5res0_bn=nn.SpatialBatchNormalization(32*nc)(p4L5res0)
p4L5res0_relu=cudnn.ReLU(true)(p4L5res0_bn)
p4L5res1c2=cudnn.SpatialConvolution(32*nc, 8*nc, 1, 1, 1, 1, 0, 0)(p4L5res0_relu)
p4L5res1c2_bn=nn.SpatialBatchNormalization(8*nc)(p4L5res1c2)
p4L5res1c2_relu=cudnn.ReLU(true)(p4L5res1c2_bn)
p4L5res1c3=cudnn.SpatialConvolution(8*nc, 8*nc, 3, 3, 1, 1, 1, 1)(p4L5res1c2_relu)
p4L5res1c3_bn=nn.SpatialBatchNormalization(8*nc)(p4L5res1c3)
p4L5res1c3_relu=cudnn.ReLU(true)(p4L5res1c3_bn)
p4L5res1c4=cudnn.SpatialConvolution(8*nc, 32*nc, 1, 1, 1, 1, 0, 0)(p4L5res1c3_relu)
p4L5res1=nn.CAddTable(false)({p4L5res0_relu,p4L5res1c4})
p4L5res1_bn=nn.SpatialBatchNormalization(32*nc)(p4L5res1)
p4L5res1_relu=cudnn.ReLU(true)(p4L5res1_bn)
p4L5res1_drop=nn.SpatialDropout(opt.dropoutProb)(p4L5res1_relu)
p4L6a=cudnn.SpatialMaxPooling(2, 2, 2, 2)(p4L5res1_drop)
p4L6res0c2=cudnn.SpatialConvolution(32*nc, 8*nc, 1, 1, 1, 1, 0, 0)(p4L6a)
p4L6res0c2_bn=nn.SpatialBatchNormalization(8*nc)(p4L6res0c2)
p4L6res0c2_relu=cudnn.ReLU(true)(p4L6res0c2_bn)
p4L6res0c3=cudnn.SpatialConvolution(8*nc, 8*nc, 3, 3, 1, 1, 1, 1)(p4L6res0c2_relu)
p4L6res0c3_bn=nn.SpatialBatchNormalization(8*nc)(p4L6res0c3)
p4L6res0c3_relu=cudnn.ReLU(true)(p4L6res0c3_bn)
p4L6res0c4=cudnn.SpatialConvolution(8*nc, 32*nc, 1, 1, 1, 1, 0, 0)(p4L6res0c3_relu)
p4L6res0=nn.CAddTable(false)({p4L6a,p4L6res0c4})
p4L6res0_bn=nn.SpatialBatchNormalization(32*nc)(p4L6res0)
p4L6res0_relu=cudnn.ReLU(true)(p4L6res0_bn)
p4L6res1c2=cudnn.SpatialConvolution(32*nc, 8*nc, 1, 1, 1, 1, 0, 0)(p4L6res0_relu)
p4L6res1c2_bn=nn.SpatialBatchNormalization(8*nc)(p4L6res1c2)
p4L6res1c2_relu=cudnn.ReLU(true)(p4L6res1c2_bn)
p4L6res1c3=cudnn.SpatialConvolution(8*nc, 8*nc, 3, 3, 1, 1, 1, 1)(p4L6res1c2_relu)
p4L6res1c3_bn=nn.SpatialBatchNormalization(8*nc)(p4L6res1c3)
p4L6res1c3_relu=cudnn.ReLU(true)(p4L6res1c3_bn)
p4L6res1c4=cudnn.SpatialConvolution(8*nc, 32*nc, 1, 1, 1, 1, 0, 0)(p4L6res1c3_relu)
p4L6res1=nn.CAddTable(false)({p4L6res0_relu,p4L6res1c4})
p4L6res1_bn=nn.SpatialBatchNormalization(32*nc)(p4L6res1)
p4L6res1_relu=cudnn.ReLU(true)(p4L6res1_bn)
p4L6res1_drop=nn.SpatialDropout(opt.dropoutProb)(p4L6res1_relu)
p4L1up0=nn.SpatialConvolution(nc, opt.nclass, 3, 3, 1, 1, 1, 1)(p4L1c_relu)
p4L1up0_bn=nn.SpatialBatchNormalization(opt.nclass)(p4L1up0)
p4L1up0_relu=cudnn.ReLU(true)(p4L1up0_bn)
p4L2up=nn.SpatialFullConvolution(8*nc,opt.nclass,4,4,2,2,1,1)(p4L2res2_relu)
p4L2up_bn=nn.SpatialBatchNormalization(opt.nclass)(p4L2up)
p4L2up_relu=cudnn.ReLU(true)(p4L2up_bn)
p4L3ups1=nn.SpatialFullConvolution(16*nc,2*opt.nclass,4,4,2,2,1,1)(p4L3res1_relu)
p4L3ups1_bn=nn.SpatialBatchNormalization(2*opt.nclass)(p4L3ups1)
p4L3ups1_relu=cudnn.ReLU(true)(p4L3ups1_bn)
p4L3up=nn.SpatialFullConvolution(2*opt.nclass,opt.nclass,4,4,2,2,1,1)(p4L3ups1_relu)
p4L3up_bn=nn.SpatialBatchNormalization(opt.nclass)(p4L3up)
p4L3up_relu=cudnn.ReLU(true)(p4L3up_bn)
#   #-- XX -> 2XX-2 4XX-6 8XX-14 16XX-30-> 16XX-32
p4L4ups1=nn.SpatialFullConvolution(32*nc,4*opt.nclass,4,4,2,2,1,1)(p4L4res1_drop) 
p4L4ups1_bn=nn.SpatialBatchNormalization(4*opt.nclass)(p4L4ups1)
p4L4ups1_relu=cudnn.ReLU(true)(p4L4ups1_bn)
p4L4ups2=nn.SpatialFullConvolution(4*opt.nclass,2*opt.nclass,4,4,2,2,1,1)(p4L4ups1_relu)
p4L4ups2_bn=nn.SpatialBatchNormalization(2*opt.nclass)(p4L4ups2)
p4L4ups2_relu=cudnn.ReLU(true)(p4L4ups2_bn)
p4L4up=nn.SpatialFullConvolution(2*opt.nclass,opt.nclass,4,4,2,2,1,1)(p4L4ups2_relu)
p4L4up_bn=nn.SpatialBatchNormalization(opt.nclass)(p4L4up)
p4L4up_relu=cudnn.ReLU(true)(p4L4up_bn)
p4L5ups1=nn.SpatialFullConvolution(32*nc,8*opt.nclass,4,4,2,2,1,1)(p4L5res1_drop)
p4L5ups1_bn=nn.SpatialBatchNormalization(8*opt.nclass)(p4L5ups1)
p4L5ups1_relu=cudnn.ReLU(true)(p4L5ups1_bn)
p4L5ups2=nn.SpatialFullConvolution(8*opt.nclass,4*opt.nclass,4,4,2,2,1,1)(p4L5ups1_relu)
p4L5ups2_bn=nn.SpatialBatchNormalization(4*opt.nclass)(p4L5ups2)
p4L5ups2_relu=cudnn.ReLU(true)(p4L5ups2_bn)
p4L5ups3=nn.SpatialFullConvolution(4*opt.nclass,2*opt.nclass,4,4,2,2,1,1)(p4L5ups2_relu)
p4L5ups3_bn=nn.SpatialBatchNormalization(2*opt.nclass)(p4L5ups3)
p4L5ups3_relu=cudnn.ReLU(true)(p4L5ups3_bn)
p4L5up=nn.SpatialFullConvolution(2*opt.nclass,opt.nclass,4,4,2,2,1,1)(p4L5ups3_relu)
p4L5up_bn=nn.SpatialBatchNormalization(opt.nclass)(p4L5up)
p4L5up_relu=cudnn.ReLU(true)(p4L5up_bn)
p4L6ups1=nn.SpatialFullConvolution(32*nc,16*opt.nclass,4,4,2,2,1,1)(p4L6res1_drop)
p4L6ups1_bn=nn.SpatialBatchNormalization(16*opt.nclass)(p4L6ups1)
p4L6ups1_relu=cudnn.ReLU(true)(p4L6ups1_bn)
p4L6ups2=nn.SpatialFullConvolution(16*opt.nclass,8*opt.nclass,4,4,2,2,1,1)(p4L6ups1_relu)
p4L6ups2_bn=nn.SpatialBatchNormalization(8*opt.nclass)(p4L6ups2)
p4L6ups2_relu=cudnn.ReLU(true)(p4L6ups2_bn)
p4L6ups3=nn.SpatialFullConvolution(8*opt.nclass,4*opt.nclass,4,4,2,2,1,1)(p4L6ups2_relu)
p4L6ups3_bn=nn.SpatialBatchNormalization(4*opt.nclass)(p4L6ups3)
p4L6ups3_relu=cudnn.ReLU(true)(p4L6ups3_bn)
p4L6ups4=nn.SpatialFullConvolution(4*opt.nclass,2*opt.nclass,4,4,2,2,1,1)(p4L6ups3_relu)
p4L6ups4_bn=nn.SpatialBatchNormalization(2*opt.nclass)(p4L6ups4)
p4L6ups4_relu=cudnn.ReLU(true)(p4L6ups4_bn)
p4L6up=nn.SpatialFullConvolution(2*opt.nclass,opt.nclass,4,4,2,2,1,1)(p4L6ups4_relu)
p4L6up_bn=nn.SpatialBatchNormalization(opt.nclass)(p4L6up)
p4L6up_relu=cudnn.ReLU(true)(p4L6up_bn)
p4Lallup1 = nn.JoinTable(1,3)({p4L1up0_relu,p4L2up_relu,p4L3up_relu,p4L4up_relu,p4L5up_relu,p4L6up_relu})
p4Lallup2 = nn.SpatialConvolution(6*opt.nclass, opt.nclass, 3, 3, 1, 1, 1, 1)(p4Lallup1)
p4Lallup2_bn = nn.SpatialBatchNormalization(opt.nclass)(p4Lallup2)
p4Lallup3 = cudnn.ReLU(true)(p4Lallup2_bn)
p4Lallup4 = nn.SpatialConvolution(opt.nclass, opt.nclass, 1, 1, 1, 1, 0, 0)(p4Lallup3)
p4L1up20=nn.SpatialConvolution(nc, opt.nclass, 3, 3, 1, 1, 1, 1)(p4L1c_relu)
p4L1up20_bn=nn.SpatialBatchNormalization(opt.nclass)(p4L1up20)
p4L1up20_relu=cudnn.ReLU(true)(p4L1up20_bn)
p4L2up2=nn.SpatialFullConvolution(8*nc,opt.nclass,4,4,2,2,1,1)(p4L2res2_relu)
p4L2up2_bn=nn.SpatialBatchNormalization(opt.nclass)(p4L2up2)
p4L2up2_relu=cudnn.ReLU(true)(p4L2up2_bn)
p4L3up2s1=nn.SpatialFullConvolution(16*nc,2*opt.nclass,4,4,2,2,1,1)(p4L3res1_relu)
p4L3up2s1_bn=nn.SpatialBatchNormalization(2*opt.nclass)(p4L3up2s1)
p4L3up2s1_relu=cudnn.ReLU(true)(p4L3up2s1_bn)
p4L3up2=nn.SpatialFullConvolution(2*opt.nclass,opt.nclass,4,4,2,2,1,1)(p4L3up2s1_relu)
p4L3up2_bn=nn.SpatialBatchNormalization(opt.nclass)(p4L3up2)
p4L3up2_relu=cudnn.ReLU(true)(p4L3up2_bn)
p4L4up2s1=nn.SpatialFullConvolution(32*nc,4*opt.nclass,4,4,2,2,1,1)(p4L4res1_drop) 
p4L4up2s1_bn=nn.SpatialBatchNormalization(4*opt.nclass)(p4L4up2s1)
p4L4up2s1_relu=cudnn.ReLU(true)(p4L4up2s1_bn)
p4L4up2s2=nn.SpatialFullConvolution(4*opt.nclass,2*opt.nclass,4,4,2,2,1,1)(p4L4up2s1_relu)
p4L4up2s2_bn=nn.SpatialBatchNormalization(2*opt.nclass)(p4L4up2s2)
p4L4up2s2_relu=cudnn.ReLU(true)(p4L4up2s2_bn)
p4L4up2=nn.SpatialFullConvolution(2*opt.nclass,opt.nclass,4,4,2,2,1,1)(p4L4up2s2_relu)
p4L4up2_bn=nn.SpatialBatchNormalization(opt.nclass)(p4L4up2)
p4L4up2_relu=cudnn.ReLU(true)(p4L4up2_bn)
p4L5up2s1=nn.SpatialFullConvolution(32*nc,8*opt.nclass,4,4,2,2,1,1)(p4L5res1_drop)
p4L5up2s1_bn=nn.SpatialBatchNormalization(8*opt.nclass)(p4L5up2s1)
p4L5up2s1_relu=cudnn.ReLU(true)(p4L5up2s1_bn)
p4L5up2s2=nn.SpatialFullConvolution(8*opt.nclass,4*opt.nclass,4,4,2,2,1,1)(p4L5up2s1_relu)
p4L5up2s2_bn=nn.SpatialBatchNormalization(4*opt.nclass)(p4L5up2s2)
p4L5up2s2_relu=cudnn.ReLU(true)(p4L5up2s2_bn)
p4L5up2s3=nn.SpatialFullConvolution(4*opt.nclass,2*opt.nclass,4,4,2,2,1,1)(p4L5up2s2_relu)
p4L5up2s3_bn=nn.SpatialBatchNormalization(2*opt.nclass)(p4L5up2s3)
p4L5up2s3_relu=cudnn.ReLU(true)(p4L5up2s3_bn)
p4L5up2=nn.SpatialFullConvolution(2*opt.nclass,opt.nclass,4,4,2,2,1,1)(p4L5up2s3_relu)
p4L5up2_bn=nn.SpatialBatchNormalization(opt.nclass)(p4L5up2)
p4L5up2_relu=cudnn.ReLU(true)(p4L5up2_bn)
p4L6up2s1=nn.SpatialFullConvolution(32*nc,16*opt.nclass,4,4,2,2,1,1)(p4L6res1_drop)
p4L6up2s1_bn=nn.SpatialBatchNormalization(16*opt.nclass)(p4L6up2s1)
p4L6up2s1_relu=cudnn.ReLU(true)(p4L6up2s1_bn)
p4L6up2s2=nn.SpatialFullConvolution(16*opt.nclass,8*opt.nclass,4,4,2,2,1,1)(p4L6up2s1_relu)
p4L6up2s2_bn=nn.SpatialBatchNormalization(8*opt.nclass)(p4L6up2s2)
p4L6up2s2_relu=cudnn.ReLU(true)(p4L6up2s2_bn)
p4L6up2s3=nn.SpatialFullConvolution(8*opt.nclass,4*opt.nclass,4,4,2,2,1,1)(p4L6up2s2_relu)
p4L6up2s3_bn=nn.SpatialBatchNormalization(4*opt.nclass)(p4L6up2s3)
p4L6up2s3_relu=cudnn.ReLU(true)(p4L6up2s3_bn)
p4L6up2s4=nn.SpatialFullConvolution(4*opt.nclass,2*opt.nclass,4,4,2,2,1,1)(p4L6up2s3_relu)
p4L6up2s4_bn=nn.SpatialBatchNormalization(2*opt.nclass)(p4L6up2s4)
p4L6up2s4_relu=cudnn.ReLU(true)(p4L6up2s4_bn)
p4L6up2=nn.SpatialFullConvolution(2*opt.nclass,opt.nclass,4,4,2,2,1,1)(p4L6up2s4_relu)
p4L6up2_bn=nn.SpatialBatchNormalization(opt.nclass)(p4L6up2)
p4L6up2_relu=cudnn.ReLU(true)(p4L6up2_bn)
p4Lallup21 = nn.JoinTable(1,3)({p4L1up20_relu,p4L2up2_relu,p4L3up2_relu,p4L4up2_relu,p4L5up2_relu,p4L6up2_relu})
p4Lallup22 = nn.SpatialConvolution(6*opt.nclass, opt.nclass, 3, 3, 1, 1, 1, 1)(p4Lallup21)
p4Lallup22_bn = nn.SpatialBatchNormalization(opt.nclass)(p4Lallup22)
p4Lallup23 = cudnn.ReLU(true)(p4Lallup22_bn)
p4Lallup24 = nn.SpatialConvolution(opt.nclass, opt.nclass, 1, 1, 1, 1, 0, 0)(p4Lallup23)

p5L1b=cudnn.SpatialConvolution(opt.imageType, nc, 3, 3, 1, 1, 1, 1)(input5) ##--4XX+28
p5L1b_bn=nn.SpatialBatchNormalization(nc)(p5L1b)
p5L1b_relu=cudnn.ReLU(true)(p5L1b_bn)
p5L1c=cudnn.SpatialConvolution(nc, nc, 3, 3, 1, 1, 1, 1)(p5L1b_relu) ##--4XX+28
p5L1c_bn=nn.SpatialBatchNormalization(nc)(p5L1c)
p5L1c_relu=cudnn.ReLU(true)(p5L1c_bn)
p5L2a=cudnn.SpatialMaxPooling(2, 2, 2, 2)(p5L1c_relu)
p5L2res1_input0=nn.Padding(1,7*nc,3)(p5L2a)
p5L2res1c2=cudnn.SpatialConvolution(nc, 2*nc, 1, 1, 1, 1, 0, 0)(p5L2a)
p5L2res1c2_bn=nn.SpatialBatchNormalization(2*nc)(p5L2res1c2)
p5L2res1c2_relu=cudnn.ReLU(true)(p5L2res1c2_bn)
p5L2res1c3=cudnn.SpatialConvolution(2*nc, 2*nc, 3, 3, 1, 1, 1, 1)(p5L2res1c2_relu)
p5L2res1c3_bn=nn.SpatialBatchNormalization(2*nc)(p5L2res1c3)
p5L2res1c3_relu=cudnn.ReLU(true)(p5L2res1c3_bn)
p5L2res1c4=cudnn.SpatialConvolution(2*nc, 8*nc, 1, 1, 1, 1, 0, 0)(p5L2res1c3_relu)
p5L2res1=nn.CAddTable(false)({p5L2res1_input0,p5L2res1c4})
p5L2res1_bn=nn.SpatialBatchNormalization(8*nc)(p5L2res1)
p5L2res1_relu=cudnn.ReLU(true)(p5L2res1_bn)
p5L2res2c2=cudnn.SpatialConvolution(8*nc, 2*nc, 1, 1, 1, 1, 0, 0)(p5L2res1_relu)
p5L2res2c2_bn=nn.SpatialBatchNormalization(2*nc)(p5L2res2c2)
p5L2res2c2_relu=cudnn.ReLU(true)(p5L2res2c2_bn)
p5L2res2c3=cudnn.SpatialConvolution(2*nc, 2*nc, 3, 3, 1, 1, 1, 1)(p5L2res2c2_relu)
p5L2res2c3_bn=nn.SpatialBatchNormalization(2*nc)(p5L2res2c3)
p5L2res2c3_relu=cudnn.ReLU(true)(p5L2res2c3_bn)
p5L2res2c4=cudnn.SpatialConvolution(2*nc, 8*nc, 1, 1, 1, 1, 0, 0)(p5L2res2c3_relu)
p5L2res2=nn.CAddTable(false)({p5L2res1_relu,p5L2res2c4})
p5L2res2_bn=nn.SpatialBatchNormalization(8*nc)(p5L2res2)
p5L2res2_relu=cudnn.ReLU(true)(p5L2res2_bn)
p5L3a=cudnn.SpatialMaxPooling(2, 2, 2, 2)(p5L2res2_relu)
p5L3res0_input0=nn.Padding(1,8*nc,3)(p5L3a)
p5L3res0c2=cudnn.SpatialConvolution(8*nc, 4*nc, 1, 1, 1, 1, 0, 0)(p5L3a)
p5L3res0c2_bn=nn.SpatialBatchNormalization(4*nc)(p5L3res0c2)
p5L3res0c2_relu=cudnn.ReLU(true)(p5L3res0c2_bn)
p5L3res0c3=cudnn.SpatialConvolution(4*nc, 4*nc, 3, 3, 1, 1, 1, 1)(p5L3res0c2_relu)
p5L3res0c3_bn=nn.SpatialBatchNormalization(4*nc)(p5L3res0c3)
p5L3res0c3_relu=cudnn.ReLU(true)(p5L3res0c3_bn)
p5L3res0c4=cudnn.SpatialConvolution(4*nc, 16*nc, 1, 1, 1, 1, 0, 0)(p5L3res0c3_relu)
p5L3res0=nn.CAddTable(false)({p5L3res0_input0,p5L3res0c4})
p5L3res0_bn=nn.SpatialBatchNormalization(16*nc)(p5L3res0)
p5L3res0_relu=cudnn.ReLU(true)(p5L3res0_bn)
p5L3res1c2=cudnn.SpatialConvolution(16*nc, 4*nc, 1, 1, 1, 1, 0, 0)(p5L3res0_relu)
p5L3res1c2_bn=nn.SpatialBatchNormalization(4*nc)(p5L3res1c2)
p5L3res1c2_relu=cudnn.ReLU(true)(p5L3res1c2_bn)
p5L3res1c3=cudnn.SpatialConvolution(4*nc, 4*nc, 3, 3, 1, 1, 1, 1)(p5L3res1c2_relu)
p5L3res1c3_bn=nn.SpatialBatchNormalization(4*nc)(p5L3res1c3)
p5L3res1c3_relu=cudnn.ReLU(true)(p5L3res1c3_bn)
p5L3res1c4=cudnn.SpatialConvolution(4*nc, 16*nc, 1, 1, 1, 1, 0, 0)(p5L3res1c3_relu)
p5L3res1=nn.CAddTable(false)({p5L3res0_relu,p5L3res1c4})
p5L3res1_bn=nn.SpatialBatchNormalization(16*nc)(p5L3res1)
p5L3res1_relu=cudnn.ReLU(true)(p5L3res1_bn)
p5L4a=cudnn.SpatialMaxPooling(2, 2, 2, 2)(p5L3res1_relu)
p5L4res0_input0=nn.Padding(1,16*nc,3)(p5L4a)
p5L4res0c2=cudnn.SpatialConvolution(16*nc, 8*nc, 1, 1, 1, 1, 0, 0)(p5L4a)
p5L4res0c2_bn=nn.SpatialBatchNormalization(8*nc)(p5L4res0c2)
p5L4res0c2_relu=cudnn.ReLU(true)(p5L4res0c2_bn)
p5L4res0c3=cudnn.SpatialConvolution(8*nc, 8*nc, 3, 3, 1, 1, 1, 1)(p5L4res0c2_relu)
p5L4res0c3_bn=nn.SpatialBatchNormalization(8*nc)(p5L4res0c3)
p5L4res0c3_relu=cudnn.ReLU(true)(p5L4res0c3_bn)
p5L4res0c4=cudnn.SpatialConvolution(8*nc, 32*nc, 1, 1, 1, 1, 0, 0)(p5L4res0c3_relu)
p5L4res0=nn.CAddTable(false)({p5L4res0_input0,p5L4res0c4})
p5L4res0_bn=nn.SpatialBatchNormalization(32*nc)(p5L4res0)
p5L4res0_relu=cudnn.ReLU(true)(p5L4res0_bn)
p5L4res1c2=cudnn.SpatialConvolution(32*nc, 8*nc, 1, 1, 1, 1, 0, 0)(p5L4res0_relu)
p5L4res1c2_bn=nn.SpatialBatchNormalization(8*nc)(p5L4res1c2)
p5L4res1c2_relu=cudnn.ReLU(true)(p5L4res1c2_bn)
p5L4res1c3=cudnn.SpatialConvolution(8*nc, 8*nc, 3, 3, 1, 1, 1, 1)(p5L4res1c2_relu)
p5L4res1c3_bn=nn.SpatialBatchNormalization(8*nc)(p5L4res1c3)
p5L4res1c3_relu=cudnn.ReLU(true)(p5L4res1c3_bn)
p5L4res1c4=cudnn.SpatialConvolution(8*nc, 32*nc, 1, 1, 1, 1, 0, 0)(p5L4res1c3_relu)
p5L4res1=nn.CAddTable(false)({p5L4res0_relu,p5L4res1c4})
p5L4res1_bn=nn.SpatialBatchNormalization(32*nc)(p5L4res1)
p5L4res1_relu=cudnn.ReLU(true)(p5L4res1_bn)
p5L4res1_drop=nn.SpatialDropout(opt.dropoutProb)(p5L4res1_relu)
p5L5a=cudnn.SpatialMaxPooling(2, 2, 2, 2)(p5L4res1_drop)
p5L5res0c2=cudnn.SpatialConvolution(32*nc, 8*nc, 1, 1, 1, 1, 0, 0)(p5L5a)
p5L5res0c2_bn=nn.SpatialBatchNormalization(8*nc)(p5L5res0c2)
p5L5res0c2_relu=cudnn.ReLU(true)(p5L5res0c2_bn)
p5L5res0c3=cudnn.SpatialConvolution(8*nc, 8*nc, 3, 3, 1, 1, 1, 1)(p5L5res0c2_relu)
p5L5res0c3_bn=nn.SpatialBatchNormalization(8*nc)(p5L5res0c3)
p5L5res0c3_relu=cudnn.ReLU(true)(p5L5res0c3_bn)
p5L5res0c4=cudnn.SpatialConvolution(8*nc, 32*nc, 1, 1, 1, 1, 0, 0)(p5L5res0c3_relu)
p5L5res0=nn.CAddTable(false)({p5L5a,p5L5res0c4})
p5L5res0_bn=nn.SpatialBatchNormalization(32*nc)(p5L5res0)
p5L5res0_relu=cudnn.ReLU(true)(p5L5res0_bn)
p5L5res1c2=cudnn.SpatialConvolution(32*nc, 8*nc, 1, 1, 1, 1, 0, 0)(p5L5res0_relu)
p5L5res1c2_bn=nn.SpatialBatchNormalization(8*nc)(p5L5res1c2)
p5L5res1c2_relu=cudnn.ReLU(true)(p5L5res1c2_bn)
p5L5res1c3=cudnn.SpatialConvolution(8*nc, 8*nc, 3, 3, 1, 1, 1, 1)(p5L5res1c2_relu)
p5L5res1c3_bn=nn.SpatialBatchNormalization(8*nc)(p5L5res1c3)
p5L5res1c3_relu=cudnn.ReLU(true)(p5L5res1c3_bn)
p5L5res1c4=cudnn.SpatialConvolution(8*nc, 32*nc, 1, 1, 1, 1, 0, 0)(p5L5res1c3_relu)
p5L5res1=nn.CAddTable(false)({p5L5res0_relu,p5L5res1c4})
p5L5res1_bn=nn.SpatialBatchNormalization(32*nc)(p5L5res1)
p5L5res1_relu=cudnn.ReLU(true)(p5L5res1_bn)
p5L5res1_drop=nn.SpatialDropout(opt.dropoutProb)(p5L5res1_relu)
p5L6a=cudnn.SpatialMaxPooling(2, 2, 2, 2)(p5L5res1_drop)
p5L6res0c2=cudnn.SpatialConvolution(32*nc, 8*nc, 1, 1, 1, 1, 0, 0)(p5L6a)
p5L6res0c2_bn=nn.SpatialBatchNormalization(8*nc)(p5L6res0c2)
p5L6res0c2_relu=cudnn.ReLU(true)(p5L6res0c2_bn)
p5L6res0c3=cudnn.SpatialConvolution(8*nc, 8*nc, 3, 3, 1, 1, 1, 1)(p5L6res0c2_relu)
p5L6res0c3_bn=nn.SpatialBatchNormalization(8*nc)(p5L6res0c3)
p5L6res0c3_relu=cudnn.ReLU(true)(p5L6res0c3_bn)
p5L6res0c4=cudnn.SpatialConvolution(8*nc, 32*nc, 1, 1, 1, 1, 0, 0)(p5L6res0c3_relu)
p5L6res0=nn.CAddTable(false)({p5L6a,p5L6res0c4})
p5L6res0_bn=nn.SpatialBatchNormalization(32*nc)(p5L6res0)
p5L6res0_relu=cudnn.ReLU(true)(p5L6res0_bn)
p5L6res1c2=cudnn.SpatialConvolution(32*nc, 8*nc, 1, 1, 1, 1, 0, 0)(p5L6res0_relu)
p5L6res1c2_bn=nn.SpatialBatchNormalization(8*nc)(p5L6res1c2)
p5L6res1c2_relu=cudnn.ReLU(true)(p5L6res1c2_bn)
p5L6res1c3=cudnn.SpatialConvolution(8*nc, 8*nc, 3, 3, 1, 1, 1, 1)(p5L6res1c2_relu)
p5L6res1c3_bn=nn.SpatialBatchNormalization(8*nc)(p5L6res1c3)
p5L6res1c3_relu=cudnn.ReLU(true)(p5L6res1c3_bn)
p5L6res1c4=cudnn.SpatialConvolution(8*nc, 32*nc, 1, 1, 1, 1, 0, 0)(p5L6res1c3_relu)
p5L6res1=nn.CAddTable(false)({p5L6res0_relu,p5L6res1c4})
p5L6res1_bn=nn.SpatialBatchNormalization(32*nc)(p5L6res1)
p5L6res1_relu=cudnn.ReLU(true)(p5L6res1_bn)
p5L6res1_drop=nn.SpatialDropout(opt.dropoutProb)(p5L6res1_relu)
p5L1up0=nn.SpatialConvolution(nc, opt.nclass, 3, 3, 1, 1, 1, 1)(p5L1c_relu)
p5L1up0_bn=nn.SpatialBatchNormalization(opt.nclass)(p5L1up0)
p5L1up0_relu=cudnn.ReLU(true)(p5L1up0_bn)
p5L2up=nn.SpatialFullConvolution(8*nc,opt.nclass,4,4,2,2,1,1)(p5L2res2_relu)
p5L2up_bn=nn.SpatialBatchNormalization(opt.nclass)(p5L2up)
p5L2up_relu=cudnn.ReLU(true)(p5L2up_bn)
p5L3ups1=nn.SpatialFullConvolution(16*nc,2*opt.nclass,4,4,2,2,1,1)(p5L3res1_relu)
p5L3ups1_bn=nn.SpatialBatchNormalization(2*opt.nclass)(p5L3ups1)
p5L3ups1_relu=cudnn.ReLU(true)(p5L3ups1_bn)
p5L3up=nn.SpatialFullConvolution(2*opt.nclass,opt.nclass,4,4,2,2,1,1)(p5L3ups1_relu)
p5L3up_bn=nn.SpatialBatchNormalization(opt.nclass)(p5L3up)
p5L3up_relu=cudnn.ReLU(true)(p5L3up_bn)
#    #-- XX -> 2XX-2 4XX-6 8XX-14 16XX-30-> 16XX-32
p5L4ups1=nn.SpatialFullConvolution(32*nc,4*opt.nclass,4,4,2,2,1,1)(p5L4res1_drop) 
p5L4ups1_bn=nn.SpatialBatchNormalization(4*opt.nclass)(p5L4ups1)
p5L4ups1_relu=cudnn.ReLU(true)(p5L4ups1_bn)
p5L4ups2=nn.SpatialFullConvolution(4*opt.nclass,2*opt.nclass,4,4,2,2,1,1)(p5L4ups1_relu)
p5L4ups2_bn=nn.SpatialBatchNormalization(2*opt.nclass)(p5L4ups2)
p5L4ups2_relu=cudnn.ReLU(true)(p5L4ups2_bn)
p5L4up=nn.SpatialFullConvolution(2*opt.nclass,opt.nclass,4,4,2,2,1,1)(p5L4ups2_relu)
p5L4up_bn=nn.SpatialBatchNormalization(opt.nclass)(p5L4up)
p5L4up_relu=cudnn.ReLU(true)(p5L4up_bn)
p5L5ups1=nn.SpatialFullConvolution(32*nc,8*opt.nclass,4,4,2,2,1,1)(p5L5res1_drop)
p5L5ups1_bn=nn.SpatialBatchNormalization(8*opt.nclass)(p5L5ups1)
p5L5ups1_relu=cudnn.ReLU(true)(p5L5ups1_bn)
p5L5ups2=nn.SpatialFullConvolution(8*opt.nclass,4*opt.nclass,4,4,2,2,1,1)(p5L5ups1_relu)
p5L5ups2_bn=nn.SpatialBatchNormalization(4*opt.nclass)(p5L5ups2)
p5L5ups2_relu=cudnn.ReLU(true)(p5L5ups2_bn)
p5L5ups3=nn.SpatialFullConvolution(4*opt.nclass,2*opt.nclass,4,4,2,2,1,1)(p5L5ups2_relu)
p5L5ups3_bn=nn.SpatialBatchNormalization(2*opt.nclass)(p5L5ups3)
p5L5ups3_relu=cudnn.ReLU(true)(p5L5ups3_bn)
p5L5up=nn.SpatialFullConvolution(2*opt.nclass,opt.nclass,4,4,2,2,1,1)(p5L5ups3_relu)
p5L5up_bn=nn.SpatialBatchNormalization(opt.nclass)(p5L5up)
p5L5up_relu=cudnn.ReLU(true)(p5L5up_bn)
p5L6ups1=nn.SpatialFullConvolution(32*nc,16*opt.nclass,4,4,2,2,1,1)(p5L6res1_drop)
p5L6ups1_bn=nn.SpatialBatchNormalization(16*opt.nclass)(p5L6ups1)
p5L6ups1_relu=cudnn.ReLU(true)(p5L6ups1_bn)
p5L6ups2=nn.SpatialFullConvolution(16*opt.nclass,8*opt.nclass,4,4,2,2,1,1)(p5L6ups1_relu)
p5L6ups2_bn=nn.SpatialBatchNormalization(8*opt.nclass)(p5L6ups2)
p5L6ups2_relu=cudnn.ReLU(true)(p5L6ups2_bn)
p5L6ups3=nn.SpatialFullConvolution(8*opt.nclass,4*opt.nclass,4,4,2,2,1,1)(p5L6ups2_relu)
p5L6ups3_bn=nn.SpatialBatchNormalization(4*opt.nclass)(p5L6ups3)
p5L6ups3_relu=cudnn.ReLU(true)(p5L6ups3_bn)
p5L6ups4=nn.SpatialFullConvolution(4*opt.nclass,2*opt.nclass,4,4,2,2,1,1)(p5L6ups3_relu)
p5L6ups4_bn=nn.SpatialBatchNormalization(2*opt.nclass)(p5L6ups4)
p5L6ups4_relu=cudnn.ReLU(true)(p5L6ups4_bn)
p5L6up=nn.SpatialFullConvolution(2*opt.nclass,opt.nclass,4,4,2,2,1,1)(p5L6ups4_relu)
p5L6up_bn=nn.SpatialBatchNormalization(opt.nclass)(p5L6up)
p5L6up_relu=cudnn.ReLU(true)(p5L6up_bn)
p5Lallup1 = nn.JoinTable(1,3)({p5L1up0_relu,p5L2up_relu,p5L3up_relu,p5L4up_relu,p5L5up_relu,p5L6up_relu})
p5Lallup2 = nn.SpatialConvolution(6*opt.nclass, opt.nclass, 3, 3, 1, 1, 1, 1)(p5Lallup1)
p5Lallup2_bn = nn.SpatialBatchNormalization(opt.nclass)(p5Lallup2)
p5Lallup3 = cudnn.ReLU(true)(p5Lallup2_bn)
p5Lallup4 = nn.SpatialConvolution(opt.nclass, opt.nclass, 1, 1, 1, 1, 0, 0)(p5Lallup3)
p5L1up20=nn.SpatialConvolution(nc, opt.nclass, 3, 3, 1, 1, 1, 1)(p5L1c_relu)
p5L1up20_bn=nn.SpatialBatchNormalization(opt.nclass)(p5L1up20)
p5L1up20_relu=cudnn.ReLU(true)(p5L1up20_bn)
p5L2up2=nn.SpatialFullConvolution(8*nc,opt.nclass,4,4,2,2,1,1)(p5L2res2_relu)
p5L2up2_bn=nn.SpatialBatchNormalization(opt.nclass)(p5L2up2)
p5L2up2_relu=cudnn.ReLU(true)(p5L2up2_bn)
p5L3up2s1=nn.SpatialFullConvolution(16*nc,2*opt.nclass,4,4,2,2,1,1)(p5L3res1_relu)
p5L3up2s1_bn=nn.SpatialBatchNormalization(2*opt.nclass)(p5L3up2s1)
p5L3up2s1_relu=cudnn.ReLU(true)(p5L3up2s1_bn)
p5L3up2=nn.SpatialFullConvolution(2*opt.nclass,opt.nclass,4,4,2,2,1,1)(p5L3up2s1_relu)
p5L3up2_bn=nn.SpatialBatchNormalization(opt.nclass)(p5L3up2)
p5L3up2_relu=cudnn.ReLU(true)(p5L3up2_bn)
p5L4up2s1=nn.SpatialFullConvolution(32*nc,4*opt.nclass,4,4,2,2,1,1)(p5L4res1_drop) 
p5L4up2s1_bn=nn.SpatialBatchNormalization(4*opt.nclass)(p5L4up2s1)
p5L4up2s1_relu=cudnn.ReLU(true)(p5L4up2s1_bn)
p5L4up2s2=nn.SpatialFullConvolution(4*opt.nclass,2*opt.nclass,4,4,2,2,1,1)(p5L4up2s1_relu)
p5L4up2s2_bn=nn.SpatialBatchNormalization(2*opt.nclass)(p5L4up2s2)
p5L4up2s2_relu=cudnn.ReLU(true)(p5L4up2s2_bn)
p5L4up2=nn.SpatialFullConvolution(2*opt.nclass,opt.nclass,4,4,2,2,1,1)(p5L4up2s2_relu)
p5L4up2_bn=nn.SpatialBatchNormalization(opt.nclass)(p5L4up2)
p5L4up2_relu=cudnn.ReLU(true)(p5L4up2_bn)
p5L5up2s1=nn.SpatialFullConvolution(32*nc,8*opt.nclass,4,4,2,2,1,1)(p5L5res1_drop)
p5L5up2s1_bn=nn.SpatialBatchNormalization(8*opt.nclass)(p5L5up2s1)
p5L5up2s1_relu=cudnn.ReLU(true)(p5L5up2s1_bn)
p5L5up2s2=nn.SpatialFullConvolution(8*opt.nclass,4*opt.nclass,4,4,2,2,1,1)(p5L5up2s1_relu)
p5L5up2s2_bn=nn.SpatialBatchNormalization(4*opt.nclass)(p5L5up2s2)
p5L5up2s2_relu=cudnn.ReLU(true)(p5L5up2s2_bn)
p5L5up2s3=nn.SpatialFullConvolution(4*opt.nclass,2*opt.nclass,4,4,2,2,1,1)(p5L5up2s2_relu)
p5L5up2s3_bn=nn.SpatialBatchNormalization(2*opt.nclass)(p5L5up2s3)
p5L5up2s3_relu=cudnn.ReLU(true)(p5L5up2s3_bn)
p5L5up2=nn.SpatialFullConvolution(2*opt.nclass,opt.nclass,4,4,2,2,1,1)(p5L5up2s3_relu)
p5L5up2_bn=nn.SpatialBatchNormalization(opt.nclass)(p5L5up2)
p5L5up2_relu=cudnn.ReLU(true)(p5L5up2_bn)
p5L6up2s1=nn.SpatialFullConvolution(32*nc,16*opt.nclass,4,4,2,2,1,1)(p5L6res1_drop)
p5L6up2s1_bn=nn.SpatialBatchNormalization(16*opt.nclass)(p5L6up2s1)
p5L6up2s1_relu=cudnn.ReLU(true)(p5L6up2s1_bn)
p5L6up2s2=nn.SpatialFullConvolution(16*opt.nclass,8*opt.nclass,4,4,2,2,1,1)(p5L6up2s1_relu)
p5L6up2s2_bn=nn.SpatialBatchNormalization(8*opt.nclass)(p5L6up2s2)
p5L6up2s2_relu=cudnn.ReLU(true)(p5L6up2s2_bn)
p5L6up2s3=nn.SpatialFullConvolution(8*opt.nclass,4*opt.nclass,4,4,2,2,1,1)(p5L6up2s2_relu)
p5L6up2s3_bn=nn.SpatialBatchNormalization(4*opt.nclass)(p5L6up2s3)
p5L6up2s3_relu=cudnn.ReLU(true)(p5L6up2s3_bn)
p5L6up2s4=nn.SpatialFullConvolution(4*opt.nclass,2*opt.nclass,4,4,2,2,1,1)(p5L6up2s3_relu)
p5L6up2s4_bn=nn.SpatialBatchNormalization(2*opt.nclass)(p5L6up2s4)
p5L6up2s4_relu=cudnn.ReLU(true)(p5L6up2s4_bn)
p5L6up2=nn.SpatialFullConvolution(2*opt.nclass,opt.nclass,4,4,2,2,1,1)(p5L6up2s4_relu)
p5L6up2_bn=nn.SpatialBatchNormalization(opt.nclass)(p5L6up2)
p5L6up2_relu=cudnn.ReLU(true)(p5L6up2_bn)
p5Lallup21 = nn.JoinTable(1,3)({p5L1up20_relu,p5L2up2_relu,p5L3up2_relu,p5L4up2_relu,p5L5up2_relu,p5L6up2_relu})
p5Lallup22 = nn.SpatialConvolution(6*opt.nclass, opt.nclass, 3, 3, 1, 1, 1, 1)(p5Lallup21)
p5Lallup22_bn = nn.SpatialBatchNormalization(opt.nclass)(p5Lallup22)
p5Lallup23 = cudnn.ReLU(true)(p5Lallup22_bn)
p5Lallup24 = nn.SpatialConvolution(opt.nclass, opt.nclass, 1, 1, 1, 1, 0, 0)(p5Lallup23)
p1Lr1=nn.SpatialSoftMax()(p1Lallup4)
p1Lr=nn.MulConstant(1/5,false)(p1Lr1)
p2Lr1=nn.SpatialSoftMax()(p2Lallup4)
p2Lr=nn.MulConstant(1/5,false)(p2Lr1)
p3Lr1=nn.SpatialSoftMax()(p3Lallup4)
p3Lr=nn.MulConstant(1/5,false)(p3Lr1)
p4Lr1=nn.SpatialSoftMax()(p4Lallup4)
p4Lr=nn.MulConstant(1/5,false)(p4Lr1)
p5Lr1=nn.SpatialSoftMax()(p5Lallup4)
p5Lr=nn.MulConstant(1/5,false)(p5Lr1)
pfLout1 = nn.CAddTable(false)({p1Lr,p2Lr,p3Lr,p4Lr,p5Lr})
p1Rr1=nn.SpatialSoftMax()(p1Lallup24)
p1Rr=nn.MulConstant(1/5,false)(p1Rr1)
p2Rr1=nn.SpatialSoftMax()(p2Lallup24)
p2Rr=nn.MulConstant(1/5,false)(p2Rr1)
p3Rr1=nn.SpatialSoftMax()(p3Lallup24)
p3Rr=nn.MulConstant(1/5,false)(p3Rr1)
p4Rr1=nn.SpatialSoftMax()(p4Lallup24)
p4Rr=nn.MulConstant(1/5,false)(p4Rr1)
p5Rr1=nn.SpatialSoftMax()(p5Lallup24)
p5Rr=nn.MulConstant(1/5,false)(p5Rr1)
pfRout1 = nn.CAddTable(false)({p1Rr,p2Rr,p3Rr,p4Rr,p5Rr})
f1L1=nn.Mean(2,3)(p1L5res1_relu)
f1L2=nn.Mean(2,2)(f1L1)

f2L1=nn.Mean(2,3)(p2L5res1_relu)
f2L2=nn.Mean(2,2)(f2L1)

f3L1=nn.Mean(2,3)(p3L5res1_relu)
f3L2=nn.Mean(2,2)(f3L1)

f4L1=nn.Mean(2,3)(p4L5res1_relu)
f4L2=nn.Mean(2,2)(f4L1)

f5L1=nn.Mean(2,3)(p5L5res1_relu)
f5L2=nn.Mean(2,2)(f5L1)

fall=nn.JoinTable(1,1)({f1L2,f2L2,f3L2,f4L2,f5L2})
# #--print('debug1')
local unet2 = nn.gModule({input1,input2,input3,input4,input5},{pfLout1,pfRout1,p1Lallup4,p2Lallup4,p3Lallup4,p4Lallup4,p5Lallup4,p1Lallup24,p2Lallup24,p3Lallup24,p4Lallup24,p5Lallup24,fall}):cuda()
# #--print('debug2')
local x, gradx = unet2:getParameters()
# #-- get bias index
x_quanValue = torch.Tensor(x:size(1), 1):fill(0):cuda()    ##-- record quantized value
# #--x_quanIndex = torch.Tensor(x:size(1), 1):fill(0)         #-- record bias (0) quantized weight (1), and non-quantized weight (0)
x_noQuanIndex = torch.Tensor(x:size(1), 1):fill(0):cuda()  ##-- record bias (0) quantized weight (0), and non-quantized weight (1)
x_noChangeIndex = torch.Tensor(x:size(1), 1):fill(1):cuda()##-- record bias (1) quantized weight (0), and non-quantized weight (1)
biasNumber = 0
print("Top k sort initilazing...")
for i=1, x:size(1) do
	if x[i]==0 then
		biasNumber = biasNumber + 1
		x_noQuanIndex[i] = 0
	else 
		x_noQuanIndex[i] = 1
	end
end
print('bias num='..biasNumber)
end

# #-- network loading
print('loading: '..opt.model)
n_classifer=5
unet = torch.load(opt.model):cuda()

unet:training()

local finput, fgradInput
unet:apply(function(m)  if torch.type(m) == 'nn.SpatialConvolution' or torch.type(m) == 'nn.SpatialFullConvolution' then 
                           finput = finput or m.finput
                           fgradInput = fgradInput or m.fgradInput

                           m.finput = finput
                           m.fgradInput = fgradInput
                        end
            end)

criterion1 = nn.ParallelCriterion(false)
criterion1:add(cudnn.SpatialCrossEntropyCriterion(),0)
criterion1:add(cudnn.SpatialCrossEntropyCriterion(),0)
criterion1:add(cudnn.SpatialCrossEntropyCriterion(),0.1)
criterion1:add(cudnn.SpatialCrossEntropyCriterion(),0.1)
criterion1:add(cudnn.SpatialCrossEntropyCriterion(),0.1)
criterion1:add(cudnn.SpatialCrossEntropyCriterion(),0.1)
criterion1:add(cudnn.SpatialCrossEntropyCriterion(),0.1)
criterion1:add(cudnn.SpatialCrossEntropyCriterion(),0.1)
criterion1:add(cudnn.SpatialCrossEntropyCriterion(),0.1)
criterion1:add(cudnn.SpatialCrossEntropyCriterion(),0.1)
criterion1:add(cudnn.SpatialCrossEntropyCriterion(),0.1)
criterion1:add(cudnn.SpatialCrossEntropyCriterion(),0.1)
criterion1=criterion1:cuda()


print("net batch number: "..n_classifer)

parameters,gradParameters = unet:getParameters()

print("parameter number:")
print(parameters:size())

if (opt.model=='') then
for k,module in ipairs(unet:listModules()) do
   if (torch.type(module) == 'nn.SpatialFullConvolution') then
     local stdv=math.sqrt(2/(module.nInputPlane*4))
     module.weight:normal(0, stdv)
     module.bias:normal(0, stdv)
   end
   if (torch.type(module) == 'cudnn.SpatialConvolution') then
     local stdv=math.sqrt(2/(module.nInputPlane*module.kH*module.kW))
     module.weight:normal(0, stdv)
     module.bias:normal(0, stdv)
   end
end
end

config = {learningRate=opt.learningRate}
 
local idx = 1
local image_seq = image.load(train_files[1],1,'double')
local target_seq = image.load(label_files[1],1,'double');
local image_batch = torch.DoubleTensor(n_classifer,opt.batch_size,opt.imageType, ous, ous)
local target_batch = torch.DoubleTensor(n_classifer,opt.batch_size, ins, ins)

local test_image=torch.DoubleTensor(opt.imageType,opt.imageType,ous,ous)
local errlen=100
local meanerr=torch.DoubleTensor(errlen):fill(1)
local current_p=1
local epoch = 0
local cost_seq=torch.DoubleTensor(opt.batch_size,ins,ins)
# #--cost_image=torch.DoubleTensor(opt.batch_size, ins, ins):fill(1)

local feval = function (x)
   if x ~= parameters then parameters:copy(x) end
   gradParameters:zero()
   
   local input_image = image_batch:clone()
   local label_image = target_batch:clone()
   local label_image1=torch.eq(label_image,interv):add(1)
   local label_image2=torch.eq(label_image,2*interv):add(1)

#    #--label_image=label_image:div(interv):add(1)
  
#    #--cost_image:fill(1)
#    #--image.save(string.format('%s/label_%d.png',opt.CheckPointDir,idx),label_image[1]);
#    #--image.save(string.format('%s/train_%d.png',opt.CheckPointDir,idx),input_image[1][1]);
            
   input_image = input_image:cuda()
   label_image = label_image:cuda()

   local output_image = unet:forward({input_image[1],input_image[2],input_image[3],input_image[4],input_image[5]})

   local err = criterion1:forward(output_image, {label_image1[1],label_image2[1],label_image1[1],label_image1[2],label_image1[3],label_image1[4],label_image1[5],label_image2[1],label_image2[2],label_image2[3],label_image2[4],label_image2[5]})

   local grad_df = criterion1:backward(output_image, {label_image1[1],label_image2[1],label_image1[1],label_image1[2],label_image1[3],label_image1[4],label_image1[5],label_image2[1],label_image2[2],label_image2[3],label_image2[4],label_image2[5]})
   meanerr[current_p]=err
   print('iteration '..iteration..': bs='..opt.batch_size..' Meanerr='..meanerr:mean()..' lr='..config.learningRate)
   current_p=current_p+1
   if (current_p>errlen) then
       current_p=1
   end
   unet:backward({input_image[1],input_image[2],input_image[3],input_image[4],input_image[5]},grad_df)

   return err, gradParameters
end

local function apply (model,tI,ous,ins)
       local rI=tI:clone()
       rI:fill(0);
       local testI = torch.DoubleTensor(opt.imageType,rI:size(2)+ous-ins,rI:size(3)+ous-ins):fill(0)
       local recI = testI:sub(1,opt.imageType,(ous-ins)/2+1,(ous-ins)/2+rI:size(2),(ous-ins)/2+1,(ous-ins)/2+rI:size(3))
       local feed_image=torch.DoubleTensor(1,opt.imageType,ous,ous)
       local small_in=torch.DoubleTensor(2,1,ins,ins)
       local small_ou=torch.DoubleTensor(opt.imageType,ous,ous)
       local temp_small=torch.DoubleTensor(1,ins,ins)
       local nclassrI=torch.DoubleTensor(2,1,rI:size(2),rI:size(3)):fill(0)
       local avI=torch.DoubleTensor(2,1,rI:size(2),rI:size(3)):fill(0)
       local wI=torch.DoubleTensor(1,ins,ins)

       for i=1,ins do
         for j=1,ins do
           local dx=math.min(i-1,ins-i)
           local dy=math.min(j-1,ins-j)
           local d=math.min(dx,dy)+1
           wI[1][i][j]=d;
         end
       end
       wI=wI/wI:max()
    #    #--image.save(string.format('%s/w.png',opt.CheckPointDir),wI)
    #    #--print(string.format('%s/w.png',opt.CheckPointDir))

       feed_image:fill(0)
       small_in:fill(0)
       small_ou:fill(0)
       temp_small:fill(0)
   
       recI:copy(tI)
       
       for j=1,(ous-ins)/2 do
          for c=1,opt.imageType do
             testI[{c,j,{}}]=testI[{c,ous-ins+1-j,{}}]
             testI[{c,(ous-ins)/2+rI:size(2)+j,{}}]=testI[{c,(ous-ins)/2+rI:size(2)+1-j,{}}]
            #  #--testI[{c,j,{}}]:fill(0)
            #  #--testI[{c,(ous-ins)/2+rI:size(2)+j,{}}]:fill(0)
          end
       end
       for j=1,(ous-ins)/2 do
          for c=1,opt.imageType do
             testI[{c,{},j}]=testI[{c,{},ous-ins+1-j}]
             testI[{c,{},(ous-ins)/2+rI:size(3)+j}]=testI[{c,{},(ous-ins)/2+rI:size(3)+1-j}]
            #  #--testI[{c,{},j}]:fill(0)
            #  #--testI[{c,{},(ous-ins)/2+rI:size(3)+j}]:fill(0)
          end
       end
 
       local insti,inedi,instj,inedj
       local ousti,ouedi,oustj,ouedj
       local avk=4;
       for i1=1,math.ceil(avk*(rI:size(2)-ins)/ins)+1 do
           for j1=1,math.ceil(avk*(rI:size(3)-ins)/ins)+1 do
               insti=(i1-1)*ins/avk+1
               instj=(j1-1)*ins/avk+1
               inedi=insti+ins-1
               inedj=instj+ins-1
               if inedi>rI:size(2) then
                   inedi=rI:size(2)
                   insti=inedi-ins+1
               end
               if inedj>rI:size(3) then
                   inedj=rI:size(3)
                   instj=inedj-ins+1
               end
               ousti=insti
               ouedi=inedi+(ous-ins)
               oustj=instj
               ouedj=inedj+(ous-ins)
            #    #--print(ousti,ouedi,oustj,ouedj)

               small_in:fill(0)
               
               for j=1,4 do 
                   small_ou:copy(testI:sub(1,opt.imageType,ousti,ouedi,oustj,ouedj))
                   small_ou=image.rotate(small_ou,pi/2*(j-1))
                   feed_image[1]:copy(small_ou)
                   feed_image=feed_image:cuda()
                   local all_out=unet:forward({feed_image,feed_image,feed_image,feed_image,feed_image})
    
                   local classP1=torch.zeros(1,2,ins,ins)
                   local classP2=torch.zeros(1,2,ins,ins)
                   
                   local mlp1=nn.SpatialSoftMax():cuda()
                   local Prob1=all_out[1]
                   local classP1=Prob1
                #    #--local classP1=mlp1:forward(Prob1)

                   local mlp2=nn.SpatialSoftMax():cuda()
                   local Prob2=all_out[2]
                   local classP2=Prob2
                #    #--local classP2=mlp2:forward(Prob2)

                   temp_small:copy(classP1[1][2])
                   temp_small=image.rotate(temp_small,-pi/2*(j-1))
                   small_in[1]:add(temp_small)

                   temp_small:copy(classP2[1][2])
                   temp_small=image.rotate(temp_small,-pi/2*(j-1))
                   small_in[2]:add(temp_small)

               end

               small_in:div(4)
               for k=1,2 do
                   local weighted_result=small_in[k];
                   weighted_result:cmul(wI)
                   nclassrI:sub(k,k,1,1,insti,inedi,instj,inedj):add(small_in[k])
                   avI:sub(k,k,1,1,insti,inedi,instj,inedj):add(wI)
               end
           end
       end
       nclassrI:cdiv(avI)
       return nclassrI
   end
quanData = function(data)  
	data_c = data:clone()
	local temp = torch.Tensor(data_c:size(1), 1):fill(2):cuda()
	temp = temp:log()
	local logValue = data_c:log()
    logValue:cdiv(temp)
	local upperValue = torch.ceil(logValue)
	upperValue = torch.pow(2,upperValue)
	local lowerValue = torch.floor(logValue)
	lowerValue= torch.pow(2,lowerValue)
	local absUpper = torch.abs(upperValue-data)
	local absLower = torch.abs(lowerValue-data)
	for i = 1, data_c:size(1) do
        #--print('data[i]='..data[i]..',u:'..upperValue[i]..',l='..lowerValue[i])
		if absUpper[i] > absLower[i] then
		    data[i] = lowerValue[i]
		else
			data[i] = upperValue[i]	
		end
		if data[i] > 4 then    ##-- constraint to 5 bits 2(-12) to 2(2)
			data[i] = 4
		end
		if data[i] < 0.000244140625 then ##--1/4096 (5bits)   #--0.0078125 #--1/128
		    data[i] = 0 ##--1/4096(5bits) #--1/128
		end
 		##--print('new data[i]='..data[i])
    end	
	##--return 
end
  
epoch = 1
#--local flr=opt.learningRate
print('Init: ',parameters[1]);

inq_step_size = {0.3, 0.3, 0.2, 0.2}
inq_step = 4
inq_max_iters = para_iter
quanNumber = 0
print('Incremental quantization with totally '..inq_step..' steps') 
#-- the main process of the task
for iterStep=1, inq_step do
	print('Incremental quantization: Step='..iterStep)
	#-- quantize parameters
	local x_abs = torch.abs(parameters):cuda()
	x_abs:cmul(x_noQuanIndex)  #-- makes the bias and quantized weights equal to zero when finding the top k
	local quanN_temp = 0
	if iterStep == 4  then
		quanN_temp = x_quanValue:size(1) - biasNumber - quanNumber 
	else
		quanN_temp = torch.ceil(inq_step_size[iterStep]*(x_quanValue:size(1)-biasNumber))
		quanNumber = quanNumber + quanN_temp
	end
	print('quanN_temp =    '..quanN_temp..', quanNumber='..quanNumber..'/'..parameters:size(1))
	local x_sort, x_ind = torch.topk(x_abs, quanN_temp , 1, true)
    quanData(x_sort)
	local signX = torch.sign(parameters)
	for tempI = 1, x_sort:size(1) do
		#--print('index='..x_ind[tempI]..', Value='..x[x_ind[tempI]])
		parameters[x_ind[tempI]] = x_sort[tempI]*signX[x_ind[tempI]]
		#--print('index='..x_ind[tempI]..', quanValue='..x[x_ind[tempI]])
		x_quanValue[x_ind[tempI]] = parameters[x_ind[tempI]]
		#--x_quanIndex[x_ind[tempI]] = 1
		x_noQuanIndex[x_ind[tempI]] = 0
		x_noChangeIndex[x_ind[tempI]] = 0
	end

	for iter=1, opt.maxIter_INQ do 
		  iteration = iter
		  for i=1,n_classifer do
			for j=1,opt.batch_size do
			  tr_sp,la_sp = get_image()
			  image_batch[i][j] = tr_sp
			  target_batch[i][j] = la_sp
			end
		  end
		  config.learningRate=0.00001
		  optim.adam(feval, parameters, config)
			#-- freeze quantized parameters in every step 
		  parameters:cmul(x_noChangeIndex)
		  parameters:add(x_quanValue)
			
		  if iter%20==0 then
			print(target_batch[1][1]:max())
			image.save(string.format('%s/t.png',INQ_train_path),image_batch[1][1])
			image.save(string.format('%s/l.png',INQ_train_path),target_batch[1][1]/255)
		  end
		  if opt.checkpoint>0 and iterStep == 4  and iter%opt.checkpoint ==0 then
			 #--unet:clearState(); 
			 #--filename=string.format('%s/unet_INQ_%d.bin',INQ_train_path,iter);
			 #--if iter%(opt.checkpoint) == 0 then
			 #--  torch.save(filename,unet);
			 #--end
			 unet:evaluate();
			 for iter_file=1,#vali_files do
				Iname=string.split(vali_files[iter_file],'/')
				Iname=Iname[#Iname]
				print('Processing: '..Iname)
				local testimage=image.load(vali_files[iter_file],opt.imageType,'double')
				local totdim = testimage:size():size()
				local testx = testimage:size(totdim-1)
				local testy = testimage:size(totdim)
				testimage = image.scale(testimage,'*1/2','simple');
				local testfullimage=torch.DoubleTensor(opt.imageType,testimage:size(totdim-1),testimage:size(totdim));
				testfullimage=testimage;
				local Pmap=apply(unet,testfullimage,ous,ins)
				for k=1,2 do
					image.save(string.format('%s/iter%d_c%d_%s',INQ_train_path,iter,k,Iname),Pmap[k])
				end
			 end 
			 for iter_file=1,#test_files do
				Iname=string.split(test_files[iter_file],'/')
				Iname=Iname[#Iname]
				print('Processing: '..Iname)
				local testimage=image.load(test_files[iter_file],opt.imageType,'double')
				local totdim = testimage:size():size()
				local testx = testimage:size(totdim-1)
				local testy = testimage:size(totdim)
				testimage = image.scale(testimage,'*1/2','simple');
				local testfullimage=torch.DoubleTensor(opt.imageType,testimage:size(totdim-1),testimage:size(totdim));
				testfullimage=testimage;
				local Pmap=apply(unet,testfullimage,ous,ins)
				for k=1,2 do
					image.save(string.format('%s/iter%d_c%d_%s',final_INQ_path,iter,k,Iname),Pmap[k])
				end
			 end 
			 unet:training()
		  end  
	end
end

local x_abs = torch.abs(parameters):cuda()
local x_quan = torch.Tensor(40, 1):fill(0):cuda() 
local x_interval = torch.Tensor(3, 1):fill(0):cuda() 
print(x_abs[1])
print(x_abs[2])

local temp = torch.Tensor(parameters:size(1), 1):fill(2):cuda()
temp = temp:log()
local data_c = x_abs:clone()
local logValue = data_c:log():cuda()
logValue:cdiv(temp)
local relaValue = torch.floor(logValue)
relaValue = torch.pow(2,relaValue)
relaValue:csub(x_abs)
for i=1, x_abs:size(1) do #--x_abs:size(1)
    if(i%1000==0) then print(i) end
	#--print("x_abs:"..x_abs[i]..",relaValue="..relaValue[i])
	if(x_abs[i]==0) then x_quan[1] = x_quan[1]+1 
	elseif relaValue[i] == 0  then
		#--print("x_abs:"..x_abs[i]..",logValue="..logValue[i])
		x_quan[30+logValue[i]] = x_quan[30+logValue[i]]+1
	end
end


