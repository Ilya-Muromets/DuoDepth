from utils.pointnet import PointNet
from utils.datasets import Mono
import argparse
import numpy as np
import datetime

current_time = datetime.datetime.now()
prefix = current_time.strftime("%m-%d:%H:%M") + "aug"


train_paths = ["train/zero/", "train/one/", "train/two/", "train/three/", "train/four/", "train/five/", "train/thumbup/", "train/thumbdown/", "train/frame/", "train/bird/"]
test_paths = ["test/zero/", "test/one/", "test/two/", "test/three/", "test/four/", "test/five/", "test/thumbup/", "test/thumbdown/", "test/frame/", "test/bird/"]

dataset = Mono(left=True, right=True, num_points=320, file_paths=train_paths)
test_dataset = Mono(left=True, right=True, num_points=320, file_paths=test_paths)

num_trials = 25
trial = []
test = []

for j in range(num_trials):
    print("small test number 1: ", j)
    pnt = PointNet(num_points=320, num_classes=10, num_epoch=64, batchsize=32, ptype='small', alpha=0, beta=0)
    trial.append(pnt.train(dataset, test_dataset)[0])
test.append(trial)
trial = []
np.save("test_results/" + prefix, np.array(test))

for j in range(num_trials):
    print("small test number 2: ", j)
    pnt = PointNet(num_points=320, num_classes=10, num_epoch=64, batchsize=32, ptype='small', alpha=0, beta=0.01)
    trial.append(pnt.train(dataset, test_dataset)[0])
test.append(trial)
trial = []
np.save("test_results/" + prefix, np.array(test))

for j in range(num_trials):
    print("small test number 3: ", j)
    pnt = PointNet(num_points=320, num_classes=10, num_epoch=64, batchsize=32, ptype='small', alpha=0, beta=0.02)
    trial.append(pnt.train(dataset, test_dataset)[0])
test.append(trial)
trial = []
np.save("test_results/" + prefix, np.array(test))

for j in range(num_trials):
    print("small test number 4: ", j)
    pnt = PointNet(num_points=320, num_classes=10, num_epoch=64, batchsize=32, ptype='small', alpha=0, beta=0.03)
    trial.append(pnt.train(dataset, test_dataset)[0])
test.append(trial)
trial = []
np.save("test_results/" + prefix, np.array(test))

for j in range(num_trials):
    print("small test number 5: ", j)
    pnt = PointNet(num_points=320, num_classes=10, num_epoch=64, batchsize=32, ptype='small', alpha=0, beta=0.04)
    trial.append(pnt.train(dataset, test_dataset)[0])
test.append(trial)
trial = []
np.save("test_results/" + prefix, np.array(test))


for j in range(num_trials):
    print("small test number 5: ", j)
    pnt = PointNet(num_points=320, num_classes=10, num_epoch=64, batchsize=32, ptype='small', alpha=0, beta=0.08)
    trial.append(pnt.train(dataset, test_dataset)[0])
test.append(trial)
trial = []
np.save("test_results/" + prefix, np.array(test))