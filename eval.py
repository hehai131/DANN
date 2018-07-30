from __future__ import division
import os
from numpy import random
import argparse

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable

import utils
import trainer
import models
import datetime
# from logger import Logger

# Set the logger
# logger = Logger('./logs')


def eval():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='emm')
    parser.add_argument('--dataset', type=str, default='mnist_svhn')
    parser.add_argument('--data_root', type = str, default='/media/b3-542/196AE2835A1F87B0/HeHai/Dataset/Mnist')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--image_size', type=int, default=28)
    parser.add_argument('--nf', type=int, default=64)
    parser.add_argument('--nepochs', type=int, default=100)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.002)
    parser.add_argument('--beta', type=float, default=0.8)
    parser.add_argument('--lr_patience', type=float, default=8)

    opt = parser.parse_args()

    # target_dataset_name = 'mnist_m'
    # target_image_root = os.path.join(opt.data_root, target_dataset_name)
    target_dataset_name = 'svhn'
    opt.data_root = '/media/b3-542/196AE2835A1F87B0/HeHai/Dataset/Mnist/digits'
    target_train_root = os.path.join(opt.data_root, target_dataset_name, 'trainset')

    cudnn.benchmark = True

    # load data
    # source data
    img_transform = transforms.Compose([
        transforms.Resize(opt.image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
    ])

    # target data
    # train_list = os.path.join(target_image_root, 'mnist_m_train_labels.txt')
    # dataset_target = utils.GetLoader(
    #     data_root=os.path.join(target_image_root, 'mnist_m_train'),
    #     data_list=train_list,
    #     transform=img_transform
    # )

    # target data
    dataset_target = datasets.ImageFolder(
        root=target_train_root,
        transform=img_transform,
    )

    loader = torch.utils.data.DataLoader(
        dataset=dataset_target,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=8)

    if opt.method == 'dann':
        model = models.DANN(opt).cuda()
        model_path = './models/'+opt.dataset+'/best_dann.pth'
    elif opt.method == 'sourceonly':
        model = models.Classifier().cuda()
        model_path = './models/'+opt.dataset+'/best_cls.pth'
    if opt.method == 'emm':
        # model = models.CNNModel().cuda()
        model_path = './models/'+opt.dataset+'/dann.pth'
        model = torch.load(model_path)
    print(model_path)

    # model.load_state_dict(torch.load(model_path))
    model.eval()
    total = 0
    correct = 0

    for i, src_data in enumerate(loader):
        src_image, src_label = src_data
        src_image = src_image.cuda()
        src_image = Variable(src_image, volatile=True)
        if opt.method == 'dann':
            class_out, _ = model(src_image, -1)
        else:#if opt.method == 'sourceonly':
            class_out, _ = model(src_image, 0)
        _, predicted = torch.max(class_out.data, 1)
        total += src_label.size(0)
        correct += ((predicted == src_label.cuda()).sum())

    val_acc = 100 * float(correct) / total
    print('%s| Test Accuracy: %f %%' % (datetime.datetime.now(), val_acc))

    return val_acc




if __name__ == '__main__':
    with torch.cuda.device(0):
        eval()