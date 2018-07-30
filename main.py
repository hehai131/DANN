from __future__ import division
import os
from numpy import random
import argparse
import numpy as np

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.datasets as datasets

import utils
import trainer
# from logger import Logger

# Set the logger
# logger = Logger('./logs')
cuda_device = 1


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='dann')
    parser.add_argument('--dataset', type=str, default='mnist_svhn')
    parser.add_argument('--data_root', type = str, default='/media/b3-542/196AE2835A1F87B0/HeHai/Dataset/Mnist')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--image_size', type=int, default=32)
    parser.add_argument('--nf', type=int, default=64)
    parser.add_argument('--nepochs', type=int, default=30)
    parser.add_argument('--num_classes', type=int, default=10)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--beta', type=float, default=0.8)
    parser.add_argument('--lr_patience', type=float, default=8)

    opt = parser.parse_args()
    print(opt)

    manual_seed = random.randint(1, 10000)
    random.seed(manual_seed)
    torch.manual_seed(manual_seed)

    cudnn.benchmark = True
    model_path = os.path.join('./models', opt.dataset)
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    if opt.dataset == 'mnist_mnistm':
        opt.lr = 0.002
        source_dataset_name = 'mnist'
        target_dataset_name = 'mnist_m'
        opt.data_root = '/media/b3-542/196AE2835A1F87B0/HeHai/Dataset/Mnist'
        source_train_root = os.path.join(opt.data_root, source_dataset_name)
        target_train_root = os.path.join(opt.data_root, target_dataset_name)

        # load data
        # source data
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.5, 0.5, 0.5])
        img_transform = transforms.Compose([
            transforms.Resize(opt.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        train_dataset_source = datasets.MNIST(
            root=source_train_root,
            train=True,
            transform=img_transform,
            download=False,
        )

        val_dataset_source = datasets.MNIST(
            root=source_train_root,
            train=False,
            transform=img_transform,
            download=False,
        )

        # target data
        train_list = os.path.join(target_train_root, 'mnist_m_train_labels.txt')
        dataset_target = utils.GetLoader(
            data_root=os.path.join(target_train_root, 'mnist_m_train'),
            data_list=train_list,
            transform=img_transform
        )

    elif opt.dataset == 'mnist_svhn':
        opt.lr = 0.002
        source_dataset_name = 'svhn'
        target_dataset_name = 'mnist'
        opt.data_root = '/media/b3-542/196AE2835A1F87B0/HeHai/Dataset/Mnist/digits'
        source_train_root = os.path.join(opt.data_root, source_dataset_name, 'trainset')
        source_val_root = os.path.join(opt.data_root, source_dataset_name, 'testset')
        target_train_root = os.path.join(opt.data_root, target_dataset_name, 'trainset')

        # load data
        # source data
        # mean = np.array([0.44, 0.44, 0.44])
        # std = np.array([0.19, 0.19, 0.19])
        mean = np.array([0.5, 0.5, 0.5])
        std = np.array([0.5, 0.5, 0.5])
        img_transform = transforms.Compose([
            transforms.Resize(opt.image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

        train_dataset_source = datasets.ImageFolder(
            root=source_train_root,
            transform=img_transform,
        )

        val_dataset_source = datasets.ImageFolder(
            root=source_val_root,
            transform=img_transform,
        )

        # target data
        dataset_target = datasets.ImageFolder(
            root=target_train_root,
            transform=img_transform,
        )

    elif opt.dataset == 'amazon_webcam':
        opt.lr = 0.001
        source_dataset_name = 'amazon'
        target_dataset_name = 'webcam'
        opt.data_root = '/media/b3-542/196AE2835A1F87B0/HeHai/Dataset/office/domain_adaptation_images'
        source_train_root = os.path.join(opt.data_root, source_dataset_name, 'images')
        # source_val_root = os.path.join(opt.data_root, source_dataset_name, 'images')
        target_train_root = os.path.join(opt.data_root, target_dataset_name, 'images')
        img_transform = transforms.Compose([
            transforms.Resize(227),
            # transforms.RandomCrop(args['image_size']),
            # transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
        ])
        train_dataset_source = datasets.ImageFolder(
            root=source_train_root,
            transform=img_transform,
        )

        # target data
        dataset_target = datasets.ImageFolder(
            root=target_train_root,
            transform=img_transform,
        )

        val_dataset_source = dataset_target

    # print(dataset_target.classes)
    source_train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset_source,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )

    source_val_loader = torch.utils.data.DataLoader(
        dataset=val_dataset_source,
        batch_size=opt.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
    )

    target_train_loader = torch.utils.data.DataLoader(
        dataset=dataset_target,
        batch_size=opt.batch_size,
        shuffle=True,
        num_workers=8,
        pin_memory=True,
        drop_last=True,
    )

    DA_trainer = trainer.DA(source_train_loader, source_val_loader, target_train_loader, opt)
    if opt.method == 'dann':
        # DA_trainer.train()
        DA_trainer.test(opt.nepochs)
    elif opt.method == 'cls':
        DA_trainer.train_cls()
        DA_trainer.test(opt.nepochs)
    else:
        raise ValueError('method argument should be dann or cls')





if __name__ == '__main__':
    with torch.cuda.device(cuda_device):
        main()