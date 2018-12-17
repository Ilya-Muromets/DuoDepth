from utils.pointnet import PointNet
from utils.loader import Loader
import argparse
import numpy as np
import datetime

current_time = datetime.datetime.now()
prefix = current_time.strftime("%Y-%m-%d:%H:%M")


test = []
trial = []
ldr = Loader(left=False, right=True, num_points=320)
train_paths = ["train/open/", "train/thumbup/", "train/thumbdown/", "train/twofinger/", "train/bird/", "train/frame/"]
test_paths =  ["test/open/", "test/thumbup/", "test/thumbdown/","test/twofinger/","test/bird/","test/frame/"]
dataset = ldr.loadFolders(train_paths)
test_dataset = ldr.loadFolders(test_paths)

trial_num = 25

for j in range(trial_num):
    print("small test number 1: ", j)
    pnt = PointNet(num_points=320, num_classes=6, num_epoch=64, batchsize=20, ptype='small', alpha=0, beta=0)
    trial.append(pnt.train(dataset, test_dataset)[0])
test.append(trial)
trial = []
np.save("test_results/" + prefix, np.array(test))

for j in range(trial_num):
    print("small test number 2: ", j)
    pnt = PointNet(num_points=320, num_classes=6, num_epoch=64, batchsize=20, ptype='small', alpha=0.01, beta=0.01)
    trial.append(pnt.train(dataset, test_dataset)[0])
test.append(trial)
trial = []
np.save("test_results/" + prefix, np.array(test))

for j in range(trial_num):
    print("small test number 3: ", j)
    pnt = PointNet(num_points=320, num_classes=6, num_epoch=64, batchsize=20, ptype='small', alpha=0.02, beta=0.02)
    trial.append(pnt.train(dataset, test_dataset)[0])
test.append(trial)
trial = []
np.save("test_results/" + prefix, np.array(test))

for j in range(trial_num):
    print("small test number 4: ", j)
    pnt = PointNet(num_points=320, num_classes=6, num_epoch=64, batchsize=20, ptype='small', alpha=0.03, beta=0.03)
    trial.append(pnt.train(dataset, test_dataset)[0])
test.append(trial)
trial = []
np.save("test_results/" + prefix, np.array(test))

for j in range(trial_num):
    print("small test number 5: ", j)
    pnt = PointNet(num_points=320, num_classes=6, num_epoch=64, batchsize=20, ptype='small', alpha=0.04, beta=0.04)
    trial.append(pnt.train(dataset, test_dataset)[0])
test.append(trial)
trial = []
np.save("test_results/" + prefix, np.array(test))


for j in range(trial_num):
    print("small test number 5: ", j)
    pnt = PointNet(num_points=320, num_classes=6, num_epoch=64, batchsize=20, ptype='small', alpha=0.05, beta=0.05)
    trial.append(pnt.train(dataset, test_dataset)[0])
test.append(trial)
trial = []
np.save("test_results/" + prefix, np.array(test))