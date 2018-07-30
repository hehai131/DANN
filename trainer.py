# train
import torch
import torch.nn as nn
from torch.autograd import Variable, Function
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch.optim as optim
import torchvision.utils as vutils
import itertools
import numpy as np
import cv2
import os
import datetime


from models import DANN, Classifier, DANN_resnet50
import utils
import eval


class DA(object):
    def __init__(self, source_trainloader, source_valloader, target_train_loader, opt):
        self.source_trainloader = source_trainloader
        self.source_valloader = source_valloader
        self.target_train_loader = target_train_loader
        self.best_val = 0
        self.opt = opt

        self.num_classes = self.opt.num_classes
        if opt.dataset == 'amazon_webcam':
            self.model = DANN_resnet50(opt).cuda()
            self.optimizer = torch.optim.SGD([{'params': self.model.feature_extrator.parameters(), 'lr': 0.1 * opt.lr},
                                             {'params': self.model.class_classifier.parameters(), 'lr': opt.lr},
                                             {'params': self.model.domain_classifier.parameters(), 'lr': opt.lr}],
                                            momentum=0.9,
                                            weight_decay=1e-4)
        else:
            self.model = DANN(opt).cuda()
            # self.model.apply(utils.weights_init)
            self.optimizer = optim.Adam(self.model.parameters(), lr=opt.lr, betas=(opt.beta, 0.999))
            # self.optimizer = optim.SGD(self.model.parameters(), lr=opt.lr, momentum=0.9)

        self.criterion_c = nn.CrossEntropyLoss().cuda()
        self.criterion_s = nn.BCELoss().cuda()

        # self.scheduler = ReduceLROnPlateau(self.optimizer, 'min', patience=opt.lr_patience, min_lr=1e-10, verbose=True)

        self.real_label = 1
        self.fake_label = 0

        for p in self.model.parameters():
            p.requires_grad = True

    def validate(self, epoch):
        if epoch == self.opt.nepochs:
            self.model.load_state_dict(torch.load('./models/'+self.opt.dataset+'/best_cls.pth'))
        self.model.eval()
        total = 0
        correct = 0
        val_loss = 0

        for i, src_data in enumerate(self.source_valloader):
            src_image, src_label = src_data
            src_image = src_image.cuda()
            src_image = Variable(src_image, volatile=True)

            class_out = self.model(input=src_image, sln=self.opt.method)
            # loss = self.criterion_c(class_out, Variable(src_label.cuda(), volatile=True))
            _, predicted = torch.max(class_out.data, 1)
            total += src_label.size(0)
            correct += ((predicted == src_label.cuda()).sum())
            # val_loss += loss.data[0]

        val_acc = 100*float(correct)/total
        # val_loss = float(val_loss)/total
        print('%s| Epoch: %d, Val Accuracy: %f %%' % (datetime.datetime.now(), epoch, val_acc))

        # self.scheduler.step(val_loss)
        # Saving checkpoints
        # torch.save(self.model.state_dict(), './models/'+self.opt.dataset+'/cls.pth')

        return val_acc

    def test(self, epoch, sln='dann'):
        if epoch == self.opt.nepochs and sln == 'dann':
            self.model.load_state_dict(torch.load('./models/'+self.opt.dataset+'/best_dann.pth'))
        elif epoch == self.opt.nepochs and sln == 'cls':
            self.model.load_state_dict(torch.load('./models/'+self.opt.dataset+'/best_cls.pth'))

        self.model.eval()
        total = 0
        correct = 0

        for i, tgt_data in enumerate(self.target_train_loader):
            tgt_image, tgt_label = tgt_data
            tgt_image = tgt_image.cuda()
            tgt_image = Variable(tgt_image, volatile=True)

            if sln == 'dann':
                class_out, _ = self.model(tgt_image)
            else:
                class_out = self.model(tgt_image, sln='sln')
            _, predicted = torch.max(class_out.data, 1)
            total += tgt_label.size(0)
            correct += ((predicted == tgt_label.cuda()).sum())

        test_acc = 100*float(correct)/total
        print('%s| Epoch: %d, Test Accuracy: %f %%' % (datetime.datetime.now(), epoch, test_acc))

        return test_acc


    def train(self):
        real_label = torch.FloatTensor(self.opt.batch_size).fill_(self.real_label).cuda()
        fake_label = torch.FloatTensor(self.opt.batch_size).fill_(self.fake_label).cuda()
        real_label = Variable(real_label)
        fake_label = Variable(fake_label)

        min_len = min(len(self.source_trainloader), len(self.target_train_loader))
        print_frequency = min_len//20

        for epoch in range(self.opt.nepochs):
            self.model.train()

            for i, (source_data, target_data) in enumerate(itertools.izip(self.source_trainloader, self.target_train_loader)):
                alpha = utils.adjust_alpha(i, epoch, min_len, self.opt.nepochs)

                src_image, src_label = source_data
                tgt_image, _ = target_data

                # if src_image.size(0) != tgt_image.size(0):
                #     break

                src_image, src_label = src_image.cuda(), src_label.cuda()
                tgt_image = tgt_image.cuda()
                src_image, src_label = Variable(src_image), Variable(src_label)
                tgt_image = Variable(tgt_image)


                # print(src_label[0].data)
                # cv2.imshow('src', np.transpose(src_image[0].squeeze().data.cpu().numpy(), (1, 2, 0)))
                # cv2.imshow('tgt', np.transpose(tgt_image[0].squeeze().data.cpu().numpy(), (1, 2, 0)))
                # if (cv2.waitKey(0) == 27):
                #     cv2.destroyAllWindows()

                self.model.zero_grad()

                class_out, domian_out = self.model(src_image, alpha, sln=self.opt.method)


                class_err = self.criterion_c(class_out, src_label)
                src_domain_err = self.criterion_s(domian_out, fake_label)

                _, domian_out = self.model(tgt_image, alpha, sln=self.opt.method)
                tgt_domain_err = self.criterion_s(domian_out, real_label)

                total_loss = class_err + src_domain_err + tgt_domain_err

                total_loss.backward()
                self.optimizer.step()

                if i % print_frequency == 0:
                    print('epoch: {0}| {1}/{2}  class_err: {c:.4f}  src_err: {s:.4f}  tgt_err: {t:.4f}'.format(
                        epoch, i, min_len, c=class_err.data[0], s=src_domain_err.data[0], t=tgt_domain_err.data[0]
                    ))

            torch.save(self.model.state_dict(), './models/' + self.opt.dataset + '/dann.pth')
            test_acc = self.test(epoch+1)
            # self.validate(epoch+1)

            if test_acc > self.best_val:
                self.best_val = test_acc
                torch.save(self.model.state_dict(), './models/' + self.opt.dataset + '/best_dann.pth')

    def train_cls(self):
        min_len = len(self.source_trainloader)
        print_frequency = min_len//20

        for epoch in range(self.opt.nepochs):
            self.model.train()

            for i, source_data in enumerate(self.source_trainloader):
                src_image, src_label = source_data

                src_image, src_label = src_image.cuda(), src_label.cuda()
                src_image, src_label = Variable(src_image), Variable(src_label)


                self.model.zero_grad()

                class_out = self.model(input=src_image, sln=self.opt.method)
                class_err = self.criterion_c(class_out, src_label)

                total_loss = class_err

                total_loss.backward()
                self.optimizer.step()

                if i % print_frequency == 0:
                    print('epoch: {0}| {1}/{2}  class_err: {c:.4f}'.format(
                        epoch, i, min_len, c=class_err.data[0]
                    ))

            torch.save(self.model.state_dict(), './models/' + self.opt.dataset + '/cls.pth')
            # test_acc = self.validate(epoch+1)
            test_acc = self.test(epoch+1, sln='cls')

            if test_acc > self.best_val:
                self.best_val = test_acc
                torch.save(self.model.state_dict(), './models/' + self.opt.dataset + '/best_cls.pth')

