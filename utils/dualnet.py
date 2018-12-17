from __future__ import print_function
import argparse
import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.nn.functional as F
import glob
import time

class DualNet(object):
    def __init__(self, batchsize=32, num_points=2500, num_epoch=25, outf='cls', model='', num_classes=2, alpha=0.05, beta=0.05, ptype=''):
        self.batchsize=batchsize
        self.num_points=num_points	
        self.num_epoch=num_epoch
        self.outf=outf
        self.model=model
        self.num_classes=num_classes
        self.alpha=alpha
        self.beta=beta

        global DualNetCls
        if ptype == '':
            from utils.torchnet.dualnet_small import DualNetCls
        elif ptype == 'small':
            from utils.torchnet.pointnet_small import DualNetCls, PointNetDenseCls
        elif ptype == 'dropout':
            from utils.torchnet.pointnet_dropout import DualNetCls, PointNetDenseCls
        elif ptype == 'normless':
            from utils.torchnet.pointnet_normless import DualNetCls, PointNetDenseCls
        elif ptype == 'small+dropout':
            from utils.torchnet.pointnet_small_dropout import DualNetCls, PointNetDenseCls



    def train(self, dataset, test_dataset):

        def randomAugment(points, alpha, beta):
            # taken from https://en.wikipedia.org/wiki/Rotation_matrix
            x1, x2, x3 = np.random.rand(3)*alpha
            d1, d2, d3 = np.random.rand(3)*beta

            Rz = np.matrix([[np.cos(2 * np.pi * x1), -np.sin(2 * np.pi * x1), 0],
                        [np.sin(2 * np.pi * x1), np.cos(2 * np.pi * x1), 0],
                        [0, 0, 1]])
            Ry = np.matrix([[np.cos(2 * np.pi * x2), 0,-np.sin(2 * np.pi * x2)],
                        [0,1,0],
                        [-np.sin(2 * np.pi * x2), 0,np.cos(2 * np.pi * x2)]])
            Rx = np.matrix([[1, 0, 0],
                        [0, np.cos(2 * np.pi * x3),-np.sin(2 * np.pi * x3)],
                        [0, np.sin(2 * np.pi * x3), np.cos(2 * np.pi * x3)]])
            rand_rot = np.dot(np.dot(Rx,Ry),Rz)
            disp = np.array([[d1],[d2],[d3]])
            for i in range(len(points)):
                points[i] = Variable(torch.from_numpy(np.dot(rand_rot, points[i]) + disp))

        blue = lambda x:'\033[94m' + x + '\033[0m'

        # initialise dataloader as single thread else socket errors
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.batchsize,
                                                shuffle=True)#, num_workers=int(self.workers))

        testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batchsize,
                                                shuffle=True)#, num_workers=int(self.workers))

        print("size of train: ", len(dataset))
        print("size of test: ", len(test_dataset))

        print('classes: ', self.num_classes)

        try:
            os.makedirs(self.outf)
        except OSError:
            pass


        classifier = DualNetCls(k = self.num_classes)


        if self.model != '':
            classifier.load_state_dict(torch.load(self.model))


        optimizer = optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)
        classifier.cuda()

        num_batch = len(dataset)/self.batchsize
        test_acc = []
        for epoch in range(self.num_epoch):
            for i, data in enumerate(dataloader, 0):
                
                # Ungodly indexing magic
                points, target = data[:,:,0:self.num_points].type(torch.FloatTensor), data[:,:,self.num_points:self.num_points+1][:,0].type(torch.LongTensor)

                # add translation/rotation augmentation
                # randomAugment(points, self.alpha, self.beta)

                points, target = Variable(points), Variable(target[:,0])
                points, target = points.cuda(), target.cuda()
                optimizer.zero_grad()
                classifier = classifier.train()
                pred, _ = classifier(points, points)
                # print(pred)
                loss = F.nll_loss(pred, target)
                loss.backward()
                optimizer.step()
                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(target.data).cpu().sum()
                # print('[%d: %d/%d] train loss: %f accuracy: %f' %(epoch, i, num_batch, loss.item(),correct.item() / float(self.batchsize)))

                if i % 10 == 0:
                    j, data = next(enumerate(testdataloader, 0))
                    points, target = data[:,:,0:self.num_points].type(torch.FloatTensor), data[:,:,self.num_points:self.num_points+1][:,0].type(torch.LongTensor)
                    points, target = Variable(points), Variable(target[:,0])
                    # points = points.transpose(2,1)
                    points, target = points.cuda(), target.cuda()
                    classifier = classifier.eval()
                    pred, _ = classifier(points, points)
                    # print(pred)
                    loss = F.nll_loss(pred, target)
                    pred_choice = pred.data.max(1)[1]
                    correct = pred_choice.eq(target.data).cpu().sum()
                    print('[%d: %d/%d] %s loss: %f accuracy: %f' %(epoch, i, num_batch, blue('test'), loss.item(), correct.item()/float(self.batchsize)))

            # out, _ = classifier.sim_data
            # print(out[0:10])
            torch.save(classifier.state_dict(),"model")# '%s/cls_model_%d.pth' % (self.outf, epoch))

        acc = 0
        confusion_matrix = np.zeros((self.num_classes, self.num_classes))
        # empty matrix for recording instance accuracies
        accuracy_matrix = np.zeros((self.num_classes, int(len(test_dataset)/self.num_classes)))

        classifier.eval()
        with torch.no_grad():
            # self.batchsize = 1
            for i in range(len(test_dataset)//self.batchsize):
                data = test_dataset[i*self.batchsize:i*self.batchsize+self.batchsize]   
                points, target = data[:,:,0:self.num_points].type(torch.FloatTensor), data[:,:,self.num_points:self.num_points+1][:,0].type(torch.LongTensor)
                identifier =  data[:,:,self.num_points+1:self.num_points+2][:,0]

                points, target = Variable(points), Variable(target[:,0])
                points, target = points.cuda(), target.cuda()
                start = time.time()
                pred, _ = classifier(points)
                # print("time: ", time.time()-start)
                # print(target, pred)
                loss = F.nll_loss(pred, target)
                pred_choice = pred.data.max(1)[1]
                correct = pred_choice.eq(target.data).cpu().sum()
                # print(correct.item())
                acc += correct.item()

                # update confusion matrix and accuracy
                for i, t in enumerate(target):
                    confusion_matrix[int(t), int(pred_choice[i])] += 1
                    if int(t) == int(pred_choice[i]): # if correct set to 1
                        accuracy_matrix[t,int(identifier[i])] = 1 
            
        print("final acc: ", acc/len(test_dataset))
        return acc/len(test_dataset), confusion_matrix, accuracy_matrix
