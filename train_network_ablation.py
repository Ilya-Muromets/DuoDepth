
from utils.pointnet import PointNet
from utils.dualnet import DualNet
from utils.datasets import DuoDataset, MonoDataset
import argparse
import numpy as np
import datetime

current_time = datetime.datetime.now()
prefix = current_time.strftime("%Y-%m-%d:%H:%M")

test_acc = []
test_mat = []
trial_acc = []
trial_mat = np.zeros((10,10))

train_paths = ["train/zero/", "train/one/", "train/two/", "train/three/", "train/four/", "train/five/", "train/thumbdown/", "train/thumbup/", "train/frame/", "train/bird/"]
test_paths = ["test/zero/", "test/one/", "test/two/", "test/three/", "test/four/", "test/five/", "test/thumbdown/", "test/thumbup/", "test/frame/", "test/bird/"]

dataset = MonoDataset(left=True, right=False, num_points=320, file_paths=train_paths)
test_dataset = MonoDataset(left=True, right=False, num_points=320, file_paths=test_paths)

# single test nominal:
num_trials = 10

for j in range(trial_num):
    print("nominal: ", j)
    pnt = PointNet(num_points=320, num_classes=10, num_epoch=30, batchsize=32, ptype='', alpha=0, beta=0)
    res = pnt.train(dataset, test_dataset)
    trial_acc.append(res[0])
    trial_mat += res[1]
    print(trial_mat)

test_acc.append(trial_acc)
test_mat.append(trial_mat)
trial_acc = []
trial_mat = np.zeros((10,10))

np.save("test_results/acc" + prefix, np.array(test_acc))
np.save("test_results/mat" + prefix, np.array(test_mat))

# single test nominal+dropout:

for j in range(trial_num):
    print("dropout: ", j)
    pnt = PointNet(num_points=320, num_classes=10, num_epoch=30, batchsize=32, ptype='dropout', alpha=0, beta=0)
    res = pnt.train(dataset, test_dataset)
    trial_acc.append(res[0])
    trial_mat += res[1]
    print(trial_mat)

test_acc.append(trial_acc)
test_mat.append(trial_mat)
trial_acc = []
trial_mat = np.zeros((10,10))

np.save("test_results/acc" + prefix, np.array(test_acc))
np.save("test_results/mat" + prefix, np.array(test_mat))

# single test small:

for j in range(trial_num):
    print("smallnet: ", j)
    pnt = PointNet(num_points=320, num_classes=10, num_epoch=30, batchsize=32, ptype='small', alpha=0, beta=0)
    res = pnt.train(dataset, test_dataset)
    trial_acc.append(res[0])
    trial_mat += res[1]
    print(trial_mat)

test_acc.append(trial_acc)
test_mat.append(trial_mat)
trial_acc = []
trial_mat = np.zeros((10,10))

# single test small+dropout:

np.save("test_results/acc" + prefix, np.array(test_acc))
np.save("test_results/mat" + prefix, np.array(test_mat))

for j in range(trial_num):
    print("smallnet+dropout: ", j)
    pnt = PointNet(num_points=320, num_classes=10, num_epoch=30, batchsize=32, ptype='small+dropout', alpha=0, beta=0)
    res = pnt.train(dataset, test_dataset)
    trial_acc.append(res[0])
    trial_mat += res[1]
    print(trial_mat)

test_acc.append(trial_acc)
test_mat.append(trial_mat)
trial_acc = []
trial_mat = np.zeros((10,10))


np.save("test_results/acc" + prefix, np.array(test_acc))
np.save("test_results/mat" + prefix, np.array(test_mat))

dataset = DuoDataset(num_points=320, file_paths=train_paths)
test_dataset = DuoDataset(num_points=320, file_paths=test_paths)

for j in range(trial_num):
    print("dualnet: ", j)
    pnt = DualNet(num_points=320, num_classes=10, num_epoch=30, batchsize=32, ptype='', alpha=0, beta=0)
    res = pnt.train(dataset, test_dataset)
    trial_acc.append(res[0])
    trial_mat += res[1]
    print(trial_mat)

test_acc.append(trial_acc)
test_mat.append(trial_mat)
trial_acc = []
trial_mat = np.zeros((10,10))


np.save("test_results/acc" + prefix, np.array(test_acc))
np.save("test_results/mat" + prefix, np.array(test_mat))

for j in range(trial_num):
    print("dualnet modified: ", j)
    pnt = DualNet(num_points=320, num_classes=10, num_epoch=30, batchsize=32, ptype='modified', alpha=0, beta=0)
    res = pnt.train(dataset, test_dataset)
    trial_acc.append(res[0])
    trial_mat += res[1]
    print(trial_mat)

test_acc.append(trial_acc)
test_mat.append(trial_mat)
trial_acc = []
trial_mat = np.zeros((10,10))


np.save("test_results/acc" + prefix, np.array(test_acc))
np.save("test_results/mat" + prefix, np.array(test_mat))


