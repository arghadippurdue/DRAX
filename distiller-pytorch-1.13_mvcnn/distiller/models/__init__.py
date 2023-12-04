#
# Copyright (c) 2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

"""This package contains ImageNet and CIFAR image classification models for pytorch"""

import copy
from functools import partial
import torch
import torchvision.models as torch_models
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from torchvision.ops.misc import FrozenBatchNorm2d
import torch.nn as nn
from . import cifar10 as cifar10_models
from . import mnist as mnist_models
from . import imagenet as imagenet_extra_models
import pretrainedmodels

from distiller.utils import set_model_input_shape_attr, model_setattr
from distiller.modules import Mean, EltwiseAdd

import logging

#AD
import os
import glob
import torchvision.models as models
from torch.autograd import Variable
from collections import OrderedDict

msglogger = logging.getLogger()

SUPPORTED_DATASETS = ('imagenet', 'cifar10', 'mnist', 'modelnet') #AD

# ResNet special treatment: we have our own version of ResNet, so we need to over-ride
# TorchVision's version.
RESNET_SYMS = ('ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101', 'resnet152',
               'resnext50_32x4d', 'resnext101_32x8d', 'wide_resnet50_2', 'wide_resnet101_2')

TORCHVISION_MODEL_NAMES = sorted(
                            name for name in torch_models.__dict__
                            if name.islower() and not name.startswith("__")
                            and callable(torch_models.__dict__[name]))

IMAGENET_MODEL_NAMES = copy.deepcopy(TORCHVISION_MODEL_NAMES)
IMAGENET_MODEL_NAMES.extend(sorted(name for name in imagenet_extra_models.__dict__
                                   if name.islower() and not name.startswith("__")
                                   and callable(imagenet_extra_models.__dict__[name])))
IMAGENET_MODEL_NAMES.extend(pretrainedmodels.model_names)

CIFAR10_MODEL_NAMES = sorted(name for name in cifar10_models.__dict__
                             if name.islower() and not name.startswith("__")
                             and callable(cifar10_models.__dict__[name]))

MNIST_MODEL_NAMES = sorted(name for name in mnist_models.__dict__
                           if name.islower() and not name.startswith("__")
                           and callable(mnist_models.__dict__[name]))
#AD
MVCNN_MODEL_NAMES = ['alexnet_mvcnn', 'resnet34_mvcnn', 'vgg11_mvcnn']

ALL_MODEL_NAMES = sorted(map(lambda s: s.lower(),
                            set(IMAGENET_MODEL_NAMES + CIFAR10_MODEL_NAMES + MNIST_MODEL_NAMES + MVCNN_MODEL_NAMES)))



def patch_torchvision_mobilenet_v2(model):
    """
    Patches TorchVision's MobileNetV2:
    * To allow quantization, this adds modules for tensor operations (mean, element-wise addition) to the
      model instance and patches the forward functions accordingly
    * Fixes a bug in the torchvision implementation that prevents export to ONNX (and creation of SummaryGraph)
    """
    if not isinstance(model, torch_models.MobileNetV2):
        raise TypeError("Only MobileNetV2 is acceptable.")

    def patched_forward_mobilenet_v2(self, x):
        x = self.features(x)
        # x = x.mean([2, 3]) # this was a bug: https://github.com/pytorch/pytorch/issues/20516
        x = self.mean32(x)
        x = self.classifier(x)
        return x
    model.mean32 = nn.Sequential(
        Mean(3), Mean(2)
    )
    model.__class__.forward = patched_forward_mobilenet_v2

    def is_inverted_residual(module):
        return isinstance(module, nn.Module) and module.__class__.__name__ == 'InvertedResidual'

    def patched_forward_invertedresidual(self, x):
        if self.use_res_connect:
            return self.residual_eltwiseadd(self.conv(x), x)
        else:
            return self.conv(x)

    for n, m in model.named_modules():
        if is_inverted_residual(m):
            if m.use_res_connect:
                m.residual_eltwiseadd = EltwiseAdd()
            m.__class__.forward = patched_forward_invertedresidual


_model_extensions = {}


def create_model(pretrained, dataset, arch, parallel=True, device_ids=None):
    """Create a pytorch model based on the model architecture and dataset

    Args:
        pretrained [boolean]: True is you wish to load a pretrained model.
            Some models do not have a pretrained version.
        dataset: dataset name (only 'imagenet' and 'cifar10' are supported), added 'modelnet'
        arch: architecture name
        parallel [boolean]: if set, use torch.nn.DataParallel
        device_ids: Devices on which model should be created -
            None - GPU if available, otherwise CPU
            -1 - CPU
            >=0 - GPU device IDs
    """
    dataset = dataset.lower()
    if dataset not in SUPPORTED_DATASETS:
        raise ValueError('Dataset {} is not supported'.format(dataset))

    model = None
    cadene = False
    try:
        if dataset == 'imagenet':
            model, cadene = _create_imagenet_model(arch, pretrained)
        elif dataset == 'cifar10':
            model = _create_cifar10_model(arch, pretrained)
        elif dataset == 'mnist':
            model = _create_mnist_model(arch, pretrained)
        elif dataset == 'modelnet':
            model = _create_modelnet_model(arch, pretrained)
    except ValueError:
        if _is_registered_extension(arch, dataset, pretrained):
            model = _create_extension_model(arch, dataset)
        else:
            raise ValueError('Could not recognize dataset {} and arch {} pair'.format(dataset, arch))

    msglogger.info("=> created a %s%s model with the %s dataset" % ('pretrained ' if pretrained else '',
                                                                     arch, dataset))
    if torch.cuda.is_available() and device_ids != -1:
        device = 'cuda'
        if parallel and (dataset != 'modelnet'):    # Check if 'modelnet' works with dataparallel or not
            if arch.startswith('alexnet') or arch.startswith('vgg'):
                model.features = torch.nn.DataParallel(model.features, device_ids=device_ids)
            else:
                model = torch.nn.DataParallel(model, device_ids=device_ids)
        model.is_parallel = parallel
    else:
        device = 'cpu'
        model.is_parallel = False

    # Cache some attributes which describe the model
    _set_model_input_shape_attr(model, arch, dataset, pretrained, cadene)
    model.arch = arch
    model.dataset = dataset
    return model.to(device)


def is_inception(arch):
    return arch in [ # Torchvision architectures
                    'inception_v3', 'googlenet',
                    # Cadene architectures
                    'inceptionv3', 'inceptionv4', 'inceptionresnetv2']


def _create_imagenet_model(arch, pretrained):
    dataset = "imagenet"
    cadene = False
    model = None
    if arch in RESNET_SYMS:
        model = imagenet_extra_models.__dict__[arch](pretrained=pretrained)
    elif arch in TORCHVISION_MODEL_NAMES:
        try:
            if is_inception(arch):
                model = getattr(torch_models, arch)(pretrained=pretrained, transform_input=False)
            else:
                model = getattr(torch_models, arch)(pretrained=pretrained)
            if arch == "mobilenet_v2":
                patch_torchvision_mobilenet_v2(model)

        except NotImplementedError:
            # In torchvision 0.3, trying to download a model that has no
            # pretrained image available will raise NotImplementedError
            if not pretrained:
                raise
    if model is None and (arch in imagenet_extra_models.__dict__) and not pretrained:
        model = imagenet_extra_models.__dict__[arch]()
    if model is None and (arch in pretrainedmodels.model_names):
        cadene = True
        model = pretrainedmodels.__dict__[arch](
            num_classes=1000,
            pretrained=(dataset if pretrained else None))
    if model is None:
        error_message = ''
        if arch not in IMAGENET_MODEL_NAMES:
            error_message = "Model {} is not supported for dataset ImageNet".format(arch)
        elif pretrained:
            error_message = "Model {} (ImageNet) does not have a pretrained model".format(arch)
        raise ValueError(error_message or 'Failed to find model {}'.format(arch))
    return model, cadene


def _create_cifar10_model(arch, pretrained):
    if pretrained:
        raise ValueError("Model {} (CIFAR10) does not have a pretrained model".format(arch))
    try:
        model = cifar10_models.__dict__[arch]()
    except KeyError:
        raise ValueError("Model {} is not supported for dataset CIFAR10".format(arch))
    return model


def _create_mnist_model(arch, pretrained):
    if pretrained:
        raise ValueError("Model {} (MNIST) does not have a pretrained model".format(arch))
    try:
        model = mnist_models.__dict__[arch]()
    except KeyError:
        raise ValueError("Model {} is not supported for dataset MNIST".format(arch))
    return model

def _create_modelnet_model(arch, pretrained):
    # pretrained is dummy    
    try:
        # cnn_name = arch #"vgg11" #"alexnet" # "reset34"
        num_views = 12    # <FIXME> hardcoded
        # path = '/content/drive/MyDrive/Arghadip/MVCNN/jongchyisu/Trained_Models/Sensor_Subsampling_Baselines/w_gen_pool/epoch_15/pkl' # <FIXME> hardcoded
        path = '/content/acc_models'
        if arch == 'alexnet_mvcnn':
          cnn_name = 'alexnet'
          part_point = 5
          modelfile_mvcnn = 'Alexnet_MVCNN_PP_5_pkl.pt'
        elif arch == 'resnet34_mvcnn':
          cnn_name = 'resnet34'
          part_point = 5
          modelfile_mvcnn = 'Resnet34_MVCNN_PP_5_pkl.pt'
        elif arch == 'vgg11_mvcnn':
          cnn_name = 'vgg11'
          part_point = 10
          modelfile_mvcnn = 'Vgg11_MVCNN_PP_10_pkl.pt'

        name = cnn_name
        cnet = SVCNN(name, nclasses=40, pretraining=False, cnn_name=cnn_name, part_point=part_point)
        model = MVCNN(name, cnet, nclasses=40, cnn_name=cnn_name, num_views=num_views, part_point=part_point)
        model.load(path, modelfile_mvcnn) # Load model parameters
        # cnet_2.eval()
        del cnet
    except KeyError:
        raise ValueError("Model {} is not supported for dataset ModelNet".format(arch))
    return model

def _set_model_input_shape_attr(model, arch, dataset, pretrained, cadene):
    if cadene and pretrained:
        # When using pre-trained weights, Cadene models already have an input size attribute
        # We add the batch dimension to it
        input_size = model.module.input_size if isinstance(model, torch.nn.DataParallel) else model.input_size
        shape = tuple([1] + input_size)
        set_model_input_shape_attr(model, input_shape=shape)
    elif arch == 'inception_v3':
        set_model_input_shape_attr(model, input_shape=(1, 3, 299, 299))
    else:
        set_model_input_shape_attr(model, dataset=dataset)


def register_user_model(arch, dataset, model):
    """A simple mechanism to support models that are not part of distiller.models"""
    _model_extensions[(arch, dataset)] = model


def _is_registered_extension(arch, dataset, pretrained):
    try:
        return _model_extensions[(arch, dataset)] is not None
    except KeyError:
        return False


def _create_extension_model(arch, dataset):
    return _model_extensions[(arch, dataset)]()

#AD
# Inherit Model from "torch.nn.Module"
class Model(nn.Module):

    def __init__(self, name):
        super(Model, self).__init__()
        self.name = name
        train_on_gpu = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if train_on_gpu else "cpu")


    # Function to save model
    def save(self, path, epoch=0):
        # Complete path to save the Model
        complete_path = os.path.join(path, self.name)
        if not os.path.exists(complete_path):
            os.makedirs(complete_path)
        # Uses PyTorch's torch.save to save the model every epoch
        torch.save(self.state_dict(), 
                os.path.join(complete_path, 
                    "model-{}.pth".format(str(epoch).zfill(5))))


    # Function to save results ??
    # What is this function?
    def save_results(self, path, data):
        raise NotImplementedError("Model subclass must implement this method.")
        

    # Function to load Model
    def load(self, path, modelfile=None):
        # Get the path from where the saved model to be loaded
        # complete_path = os.path.join(path, self.name)
        complete_path = path
        # If model does not exist, raise error
        if not os.path.exists(complete_path):
            raise IOError("{} directory does not exist in {}".format(self.name, path))

        # If no modelfile name is given
        if modelfile is None:
            # Grab the latest model file
            model_files = glob.glob(complete_path+"/*")
            mf = max(model_files)
        else:
            # Else grab the specific model file
            mf = os.path.join(complete_path, modelfile)

        # load that model file
        # print(mf)
        if self.device.type == 'cpu':
          self.load_state_dict(torch.load(mf, map_location=torch.device('cpu')))
        else:
          # print('I am loading GPU')
          # self.load_state_dict(torch.jit.load(mf))
          self.load_state_dict(torch.load(mf))
        # self.load_state_dict(torch.load(mf, map_location=torch.device('cpu')))
        # self.load_state_dict(torch.load(mf), strict=False)


# Import "Model" class from  Model.py
# from .Model import Model
class SVCNN(Model):

    def __init__(self, name, nclasses=40, pretraining=True, cnn_name='vgg11', part_point=5):
        super(SVCNN, self).__init__(name)
        # "name" is a dummy parameter

        # All the classes in ModelNet40
        self.classnames=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
                         'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
                         'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
                         'person','piano','plant','radio','range_hood','sink','sofa','stairs',
                         'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']

        self.nclasses = nclasses
        self.pretraining = pretraining
        self.cnn_name = cnn_name
        self.part_point = part_point
        
        # use_resnet = True if "cnn_name" starts with "resnet"
        self.use_resnet = cnn_name.startswith('resnet')

        # HARDCODING of the inherent division of the CNN (feature extractor and classifier)
        # Check values and update manually
        if not self.use_resnet:
          if self.cnn_name == 'alexnet':
            self.feature_extractors = 12
          elif self.cnn_name == 'vgg11':
            self.feature_extractors = 20
          elif self.cnn_name == 'vgg16':
            self.feature_extractors = 30
        else:
          self.feature_extractors = 8

        train_on_gpu = torch.cuda.is_available()
        device = torch.device("cuda:0" if train_on_gpu else "cpu")

        # Mean and standard deviation of the dataset used (NOT used anywhere)
        self.mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]), requires_grad=False).to(device)
        self.std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]), requires_grad=False).to(device)

        # If "resnet"
        if self.use_resnet:
          if self.cnn_name == 'resnet18':
              # Load the model from PyTorch repo
              # Set the "pretrained" value based on the passed argument
              self.net = models.resnet18(pretrained=self.pretraining)
              # Change the input and output dimensions of the final classifier (fc) layer
              # 40 classes in ModelNet40
              self.net.fc = nn.Linear(512,40)
          elif self.cnn_name == 'resnet34':
              self.net = models.resnet34(pretrained=self.pretraining)
              self.net.fc = nn.Linear(512,40)
          elif self.cnn_name == 'resnet50':
              self.net = models.resnet50(pretrained=self.pretraining)
              self.net.fc = nn.Linear(2048,40)

          self.resnet_modules = [
              ('conv1',self.net.conv1),
              ('bn1', self.net.bn1),
              ('relu', self.net.relu),
              ('maxpool', self.net.maxpool),
              ('layer1', self.net.layer1),
              ('layer2', self.net.layer2),
              ('layer3', self.net.layer3),
              ('layer4', self.net.layer4),
              ('avgpool', self.net.avgpool)
          ]
          # Partitioning
          if (self.part_point < self.feature_extractors):
            self.net_1 = nn.Sequential(OrderedDict(self.resnet_modules[0:self.part_point+1]))
            self.net_2 = nn.Sequential(OrderedDict(self.resnet_modules[self.part_point+1:]))
          else:
            self.net_1 = nn.Sequential(OrderedDict(self.resnet_modules[0:self.feature_extractors+1]))
            self.net_2 = nn.Sequential(OrderedDict(self.resnet_modules[self.feature_extractors:self.feature_extractors]))
          self.net_3 = self.net.fc
        else:
          if (self.part_point < self.feature_extractors):
            if self.cnn_name == 'alexnet':
                self.net_1 = models.alexnet(pretrained=self.pretraining).features[0:self.part_point+1]
                self.net_2 = models.alexnet(pretrained=self.pretraining).features[self.part_point+1:self.feature_extractors+1]
                self.net_3 = models.alexnet(pretrained=self.pretraining).classifier
            elif self.cnn_name == 'vgg11':
                self.net_1 = models.vgg11(pretrained=self.pretraining).features[0:self.part_point+1]
                self.net_2 = models.vgg11(pretrained=self.pretraining).features[self.part_point+1:self.feature_extractors+1]
                self.net_3 = models.vgg11(pretrained=self.pretraining).classifier
            elif self.cnn_name == 'vgg16':
                self.net_1 = models.vgg16(pretrained=self.pretraining).features[0:self.part_point+1]
                self.net_2 = models.vgg16(pretrained=self.pretraining).features[self.part_point+1:self.feature_extractors+1]
                self.net_3 = models.vgg16(pretrained=self.pretraining).classifier
          else:
            if self.cnn_name == 'alexnet':
                self.net_1 = models.alexnet(pretrained=self.pretraining).features
                self.net_2 = models.alexnet(pretrained=self.pretraining).classifier[0:min((self.part_point-self.feature_extractors),6)]
                self.net_3 = models.alexnet(pretrained=self.pretraining).classifier[min((self.part_point-self.feature_extractors),6):]
            elif self.cnn_name == 'vgg11':
                self.net_1 = models.vgg11(pretrained=self.pretraining).features
                self.net_2 = models.vgg11(pretrained=self.pretraining).classifier[0:min((self.part_point-self.feature_extractors),6)]
                self.net_3 = models.vgg11(pretrained=self.pretraining).classifier[min((self.part_point-self.feature_extractors),6):]
            elif self.cnn_name == 'vgg16':
                self.net_1 = models.vgg16(pretrained=self.pretraining).features
                self.net_2 = models.vgg16(pretrained=self.pretraining).classifier[0:min((self.part_point-self.feature_extractors),6)]
                self.net_3 = models.vgg16(pretrained=self.pretraining).classifier[min((self.part_point-self.feature_extractors),6):]
          # Outside if-else
          self.net_3._modules['6'] = nn.Linear(4096,40)

            

    # Forward function for the inputs
    def forward(self, x):
        if (self.use_resnet) or (self.part_point < self.feature_extractors):
          # Pass through net_1
          y1 = self.net_1(x)
          # Pass through net_2
          y2 = self.net_2(y1)
          # Retain the first dimension of y i.e num_classes (=40)
          # (-1) in torch.view() automatically get the 2nd dimension by merging the other dimensions
          # For example, if original shape of y is [40 256, 6, 6], then after applying this, the shape
          # will be [40, 9216]
          return self.net_3(y2.view(y2.shape[0],-1))
        else:
          # Pass through net_1
          y1 = self.net_1(x)
          # Flatten y1 and Pass through net_2
          y2 = self.net_2(y1.view(y1.shape[0],-1))
          # Pass through net_3
          return self.net_3(y2)


class MVCNN(Model):

    def __init__(self, name, model, nclasses=40, cnn_name='vgg11', num_views=12, part_point=5):
        super(MVCNN, self).__init__(name)

        self.classnames=['airplane','bathtub','bed','bench','bookshelf','bottle','bowl','car','chair',
                         'cone','cup','curtain','desk','door','dresser','flower_pot','glass_box',
                         'guitar','keyboard','lamp','laptop','mantel','monitor','night_stand',
                         'person','piano','plant','radio','range_hood','sink','sofa','stairs',
                         'stool','table','tent','toilet','tv_stand','vase','wardrobe','xbox']

        self.cnn_name = model.cnn_name
        self.nclasses = nclasses
        self.num_views = num_views
        self.part_point = part_point
        self.feature_extractors = model.feature_extractors

        train_on_gpu = torch.cuda.is_available()
        device = torch.device("cuda:0" if train_on_gpu else "cpu")
        # Mean and standard deviation of the dataset used (NOT used anywhere)
        self.mean = Variable(torch.FloatTensor([0.485, 0.456, 0.406]), requires_grad=False).to(device)
        self.std = Variable(torch.FloatTensor([0.229, 0.224, 0.225]), requires_grad=False).to(device)

        # use_resnet = True if "cnn_name" starts with "resnet"
        self.use_resnet = cnn_name.startswith('resnet')
        
        self.net_1 = model.net_1
        self.net_2 = model.net_2
        self.net_3 = model.net_3

    # Forward function defines how the inputs will pass through the network
    def forward(self, x):
      if (self.use_resnet) or (self.part_point < self.feature_extractors):
        y1 = self.net_1(x)
        y1 = y1.view((int(x.shape[0]/self.num_views),self.num_views,y1.shape[-3],y1.shape[-2],y1.shape[-1]))
        y2 = self.net_2(torch.max(y1,1)[0])
        return self.net_3(y2.view(y2.shape[0],-1))
      else:
        y1 = self.net_1(x)
        y2 = self.net_2(y1.view(y1.shape[0],-1))
        y2 = y2.view((int(x.shape[0]/self.num_views),self.num_views,y2.shape[-1]))
        return self.net_3(torch.max(y2,1)[0])