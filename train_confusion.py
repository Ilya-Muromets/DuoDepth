
from utils.pointnet import PointNet
from utils.loader import Loader
import argparse
import numpy as np
import datetime

current_time = datetime.datetime.now()
prefix = current_time.strftime("%Y-%m-%d:%H:%M")

test_acc = []
test_mat = []
trial_acc = []
trial_mat = np.zeros((6,6))
ldr_left = Loader(left=True, right=False, num_points=320)
ldr_right = Loader(left=False, right=True, num_points=320)
ldr_fused = Loader(left=True, right=True, num_points=320)

train_paths = ["train/open/", "train/thumbup/", "train/thumbdown/", "train/twofinger/", "train/bird/", "train/frame/"]
test_paths =  ["test/open/", "test/thumbup/", "test/thumbdown/","test/twofinger/","test/bird/","test/frame/"]


# single test LL:

dataset = ldr_left.loadFolders(train_paths)
test_dataset = ldr_left.loadFolders(test_paths)

trial_num = 100

for j in range(trial_num):
    print("FF: ", j)
    pnt = PointNet(num_points=320, num_classes=6, num_epoch=32, batchsize=20, ptype='small', alpha=0, beta=0)
    res = pnt.train(dataset, test_dataset)
    trial_acc.append(res[0])
    trial_mat += res[1]
    print(trial_mat)

test_acc.append(trial_acc)
test_mat.append(trial_mat)
trial_acc = []
trial_mat = np.zeros((6,6))


np.save("test_results/acc" + prefix, np.array(test_acc))
np.save("test_results/mat" + prefix, np.array(test_mat))

# single test RR:

dataset = ldr_right.loadFolders(train_paths)
test_dataset = ldr_right.loadFolders(test_paths)

trial_num = 100

for j in range(trial_num):
    print("FF: ", j)
    pnt = PointNet(num_points=320, num_classes=6, num_epoch=32, batchsize=20, ptype='small', alpha=0, beta=0)
    res = pnt.train(dataset, test_dataset)
    trial_acc.append(res[0])
    trial_mat += res[1]
    print(trial_mat)

test_acc.append(trial_acc)
test_mat.append(trial_mat)
trial_acc = []
trial_mat = np.zeros((6,6))



np.save("test_results/acc" + prefix, np.array(test_acc))
np.save("test_results/mat" + prefix, np.array(test_mat))


# single test FF:

dataset = ldr_fused.loadFolders(train_paths)
test_dataset = ldr_fused.loadFolders(test_paths)

trial_num = 100

for j in range(trial_num):
    print("FF: ", j)
    pnt = PointNet(num_points=320, num_classes=6, num_epoch=32, batchsize=20, ptype='small', alpha=0, beta=0)
    res = pnt.train(dataset, test_dataset)
    trial_acc.append(res[0])
    trial_mat += res[1]
    print(trial_mat)

test_acc.append(trial_acc)
test_mat.append(trial_mat)
trial_acc = []
trial_mat = np.zeros((6,6))

np.save("test_results/acc" + prefix, np.array(test_acc))
np.save("test_results/mat" + prefix, np.array(test_mat))

