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
from datasets import PartDataset
from pointnet import PointNetCls
import torch.nn.functional as F
import glob


parser = argparse.ArgumentParser()
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--num_points', type=int, default=2500, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument('--nepoch', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='cls',  help='output folder')
parser.add_argument('--model', type=str, default = '',  help='model path')
parser.add_argument('--classes', type=int, default = 2,  help='num classes')

opt = parser.parse_args()
print (opt)

blue = lambda x:'\033[94m' + x + '\033[0m'

opt.manualSeed = random.randint(1, 10000) # fix seed
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)

# Literal nonsense:

# load arrays from file_path and label with class_num
def loadData(file_path, class_num,  num_points=2500, file_type="*.npy"):
    #helper function for subsampling
    def subsampleColumns(arr, num_points):
        subsampling = np.random.choice([i for i in range(len(arr[0]))], num_points)
        return arr[:,subsampling]
    
    filenames = glob.glob(file_path + file_type)
    datum = np.load(filenames[0])
    datum = subsampleColumns(datum, num_points)
    
    # add class label
    datum = np.concatenate((datum,np.array([[class_num],[class_num],[class_num]])), axis=1)
    datum = np.expand_dims(datum, axis=0)
    print(datum.shape)

    for name in filenames[1:]:
        arr = np.load(name)
        arr = subsampleColumns(arr, num_points)
        arr = np.concatenate((arr,np.array([[class_num],[class_num],[class_num]])), axis=1)
        arr = np.expand_dims(arr, axis=0)
        datum = np.concatenate((datum,arr),axis=0)
    print(datum.shape)

    return datum
    
data = np.concatenate((loadData("../train/open1/npys/", 0), (loadData("../train/thumbdown1/npys/", 1))), axis=0)
dataset = Variable(torch.from_numpy(data))

test = np.concatenate((loadData("../train/open2/npys/", 0), (loadData("../train/thumbdown2/npys/", 1))), axis=0)
test_dataset = Variable(torch.from_numpy(test))



# dataset = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0', classification = True, npoints = opt.num_points)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                          shuffle=True, num_workers=int(opt.workers))

# test_dataset = PartDataset(root = 'shapenetcore_partanno_segmentation_benchmark_v0', classification = True, train = False, npoints = opt.num_points)
testdataloader = torch.utils.data.DataLoader(test_dataset, batch_size=opt.batchSize,
                                          shuffle=True, num_workers=int(opt.workers))

print(len(dataset), len(test_dataset))
num_classes = opt.classes
print('classes', num_classes)

try:
    os.makedirs(opt.outf)
except OSError:
    pass


classifier = PointNetCls(k = num_classes)


if opt.model != '':
    classifier.load_state_dict(torch.load(opt.model))


optimizer = optim.SGD(classifier.parameters(), lr=0.01, momentum=0.9)
classifier.cuda()

num_batch = len(dataset)/opt.batchSize

for epoch in range(opt.nepoch):
    for i, data in enumerate(dataloader, 0):
        
        # Ungodly indexing magic
        points, target = data[:,:,0:opt.num_points].type(torch.FloatTensor), data[:,:,opt.num_points:opt.num_points+1][:,0].type(torch.LongTensor)
        # print(points)
        print(target)

        points, target = Variable(points), Variable(target[:,0])
        # points = points.transpose(2,1)
        # print(points.shape)
        points, target = points.cuda(), target.cuda()
        # print(target)
        # raise Exception("no")
        optimizer.zero_grad()
        classifier = classifier.train()
        pred, _ = classifier(points)
        print(pred)
        loss = F.nll_loss(pred, target)
        loss.backward()
        optimizer.step()
        pred_choice = pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        print('[%d: %d/%d] train loss: %f accuracy: %f' %(epoch, i, num_batch, loss.item(),correct.item() / float(opt.batchSize)))

        if i % 10 == 0:
            j, data = next(enumerate(testdataloader, 0))
            points, target = data[:,:,0:opt.num_points].type(torch.FloatTensor), data[:,:,opt.num_points:opt.num_points+1][:,0].type(torch.LongTensor)
            points, target = Variable(points), Variable(target[:,0])
            # points = points.transpose(2,1)
            points, target = points.cuda(), target.cuda()
            classifier = classifier.eval()
            pred, _ = classifier(points)
            loss = F.nll_loss(pred, target)
            pred_choice = pred.data.max(1)[1]
            correct = pred_choice.eq(target.data).cpu().sum()
            print('[%d: %d/%d] %s loss: %f accuracy: %f' %(epoch, i, num_batch, blue('test'), loss.item(), correct.item()/float(opt.batchSize)))


    # out, _ = classifier.sim_data
    # print(out[0:10])
    torch.save(classifier.state_dict(),"model")# '%s/cls_model_%d.pth' % (opt.outf, epoch))
