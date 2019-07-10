from utils.dualnet import DualNet
from utils.datasets import DuoDataset
import argparse
import numpy as np
## load arguments

parser = argparse.ArgumentParser()
parser.add_argument('--batchsize', type=int, default=32, help='input batch size')
parser.add_argument('--num_points', type=int, default=320, help='input batch size')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=1)
parser.add_argument('--num_epoch', type=int, default=1, help='number of epochs to train for')
parser.add_argument('--outf', type=str, default='cls',  help='output folder')
parser.add_argument('--model', type=str, default = '',  help='model path')
parser.add_argument('--num_classes', type=int, default = 10,  help='num classes')

opt = parser.parse_args()
print (opt)

train_paths = ["train/zero/", "train/one/", "train/two/", "train/three/", "train/four/", "train/five/", "train/thumbup/", "train/thumbdown/", "train/frame/", "train/bird/"]
test_paths = ["test/zero/", "test/one/", "test/two/", "test/three/", "test/four/", "test/five/", "test/thumbup/", "test/thumbdown/", "test/frame/", "test/bird/"]

dataset = DuoDataset(num_points=opt.num_points, file_paths=train_paths)
test_dataset = DuoDataset(num_points=opt.num_points, file_paths=test_paths)

pnt = DualNet(batchsize=opt.batchsize, num_points=opt.num_points, num_epoch=opt.num_epoch, outf=opt.outf, model=opt.model, num_classes=opt.num_classes, ptype='modified')
print(pnt.train(dataset, test_dataset))
