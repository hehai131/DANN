# dann models
from __future__ import division
import torch
import torch.nn as nn
from torch.autograd import Function

import utils
import network


class GradReverse(Function):

    @staticmethod
    def forward(ctx, x, lamda):
        ctx.lamda = lamda

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = (grad_output.neg() * ctx.lamda)

        return output, None



class DANN(nn.Module):
    def __init__(self, opt):
        super(DANN, self).__init__()
        print('===============> DANN task! <===============')
        self.opt = opt
        #########################################
        #
        # Feature Extrator
        #
        #########################################
        self.feature_extrator = nn.Sequential()
        self.feature_extrator.add_module('f_conv1', nn.Conv2d(3, opt.nf, kernel_size=5))
        self.feature_extrator.add_module('f_bn1', nn.BatchNorm2d(opt.nf))
        self.feature_extrator.add_module('f_pool1', nn.MaxPool2d(2))
        self.feature_extrator.add_module('f_relu1', nn.ReLU(True))

        self.feature_extrator.add_module('f_conv2', nn.Conv2d(opt.nf, opt.nf, kernel_size=5))
        self.feature_extrator.add_module('f_bn2', nn.BatchNorm2d(opt.nf))
        self.feature_extrator.add_module('f_pool2', nn.MaxPool2d(2))
        self.feature_extrator.add_module('f_relu3', nn.ReLU(True))

        self.feature_extrator.add_module('f_con3', nn.Conv2d(opt.nf, opt.nf*2, kernel_size=5))
        self.feature_extrator.add_module('f_bn3', nn.BatchNorm2d(opt.nf*2))
        # self.feature_extrator.add_module('f_dp3', nn.Dropout2d())
        # self.feature_extrator.add_module('f_pool3', nn.MaxPool2d(2))
        self.feature_extrator.add_module('f_relu3', nn.ReLU(True))
        #########################################
        #
        # Label Classifier
        #
        #########################################
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(opt.nf*2, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm2d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_dp1', nn.Dropout2d())

        self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm2d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))

        self.class_classifier.add_module('c_fc3', nn.Linear(100, opt.num_classes))
        # self.class_classifier.add_module('c_lsm', nn.LogSoftmax())
        #########################################
        #
        # Domain Classifier
        #
        #########################################
        self.domain_classifier = nn.Sequential()

        self.domain_classifier.add_module('d_fc1', nn.Linear(opt.nf*2, 100))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm2d(100))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))

        self.domain_classifier.add_module('d_fc2', nn.Linear(100, 1))
        self.domain_classifier.add_module('d_sm', nn.Sigmoid())

    def forward(self, input, alpha=-1, sln='dann'):
        input = input.expand(input.data.shape[0], 3, self.opt.image_size, self.opt.image_size)
        feature = self.feature_extrator(input)
        feature = feature.view(feature.size(0), -1)
        if sln == 'dann':
            class_out = self.class_classifier(feature)
            reverse_feature = GradReverse.apply(feature, alpha)
            domian_out = self.domain_classifier(reverse_feature)
            return class_out, domian_out
        else:
            class_out = self.class_classifier(feature)
            return class_out


class Classifier(nn.Module):
    def __init__(self, num_classes=10):
        super(Classifier, self).__init__()
        print('===============> classifier task! <===============')
        #########################################
        #
        # Feature Extrator
        #
        #########################################
        self.feature_extrator = nn.Sequential()
        self.feature_extrator.add_module('f_conv1', nn.Conv2d(3, 64, kernel_size=5))
        self.feature_extrator.add_module('f_bn1', nn.BatchNorm2d(64))
        self.feature_extrator.add_module('f_pool1', nn.MaxPool2d(2))
        self.feature_extrator.add_module('f_relu1', nn.ReLU(True))

        self.feature_extrator.add_module('f_con2', nn.Conv2d(64, 50, kernel_size=5))
        self.feature_extrator.add_module('f_bn2', nn.BatchNorm2d(50))
        self.feature_extrator.add_module('f_dp2', nn.Dropout2d())
        self.feature_extrator.add_module('f_pool2', nn.MaxPool2d(2))
        self.feature_extrator.add_module('f_relu2', nn.ReLU(True))
        #########################################
        #
        # Label Classifier
        #
        #########################################
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc1', nn.Linear(50*4*4, 100))
        self.class_classifier.add_module('c_bn1', nn.BatchNorm2d(100))
        self.class_classifier.add_module('c_relu1', nn.ReLU(True))
        self.class_classifier.add_module('c_dp1', nn.Dropout2d())

        self.class_classifier.add_module('c_fc2', nn.Linear(100, 100))
        self.class_classifier.add_module('c_bn2', nn.BatchNorm2d(100))
        self.class_classifier.add_module('c_relu2', nn.ReLU(True))

        self.class_classifier.add_module('c_fc3', nn.Linear(100, num_classes))
        self.class_classifier.add_module('c_lsm', nn.LogSoftmax())

    def forward(self, input):
        input = input.expand(input.data.shape[0], 3, 28, 28)
        feature = self.feature_extrator(input)
        feature = feature.view(-1, 50*4*4)
        class_out = self.class_classifier(feature)
        return class_out


class DANN_resnet50(nn.Module):
    def __init__(self, opt):
        super(DANN_resnet50, self).__init__()
        print('===============> DANN_resnet50 task! <===============')
        self.opt = opt
        #########################################
        #
        # Feature Extrator
        #
        #########################################
        pretrained_dict = torch.load('/media/b3-542/196AE2835A1F87B0/HeHai/Models/resnet50-19c8e357.pth')
        self.feature_extrator = network.resnet50()
        model_dict = self.feature_extrator.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict)
        self.feature_extrator.load_state_dict(model_dict)
        #########################################
        #
        # Label Classifier
        #
        #########################################
        self.class_classifier = nn.Sequential()
        self.class_classifier.add_module('c_fc', nn.Linear(2048 * 4, 31))#opt.num_classes))
        self.class_classifier.apply(utils.weights_init)
        #########################################
        #
        # Domain Classifier
        #
        #########################################
        self.domain_classifier = nn.Sequential()

        self.domain_classifier.add_module('d_fc1', nn.Linear(2048*4, 1024))
        self.domain_classifier.add_module('d_bn1', nn.BatchNorm2d(1024))
        self.domain_classifier.add_module('d_relu1', nn.ReLU(True))

        self.domain_classifier.add_module('d_fc2', nn.Linear(1024, 256))
        self.domain_classifier.add_module('d_bn2', nn.BatchNorm2d(256))
        self.domain_classifier.add_module('d_relu2', nn.ReLU(True))

        self.domain_classifier.add_module('d_fc3', nn.Linear(256, 1))
        self.domain_classifier.add_module('d_sm', nn.Sigmoid())
        self.domain_classifier.apply(utils.weights_init)

    def forward(self, input, alpha=-1, sln='dann'):
        input = input.expand(input.data.shape[0], 3, self.opt.image_size, self.opt.image_size)
        feature = self.feature_extrator(input)
        feature = feature.view(feature.size(0), -1)
        if sln == 'dann':
            class_out = self.class_classifier(feature)
            reverse_feature = GradReverse.apply(feature, alpha)
            domian_out = self.domain_classifier(reverse_feature)
            return class_out, domian_out
        else:
            class_out = self.class_classifier(feature)
            return class_out

