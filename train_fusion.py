
from utils.pointnet import PointNet
from utils.dualnet import DualNet
from utils.datasets import Mono, Siamese
import argparse
import numpy as np
import datetime

current_time = datetime.datetime.now()
prefix = current_time.strftime("%m-%d:%H:%M") + "fusion"

# initialize final variables for return
test_acc = []
test_conf = []
test_energy = []

train_paths = ["train/zero/", "train/one/", "train/two/", "train/three/", "train/four/", "train/five/", "train/thumbup/", "train/thumbdown/", "train/frame/", "train/bird/"]
test_paths = ["test/zero/", "test/one/", "test/two/", "test/three/", "test/four/", "test/five/", "test/thumbup/", "test/thumbdown/", "test/frame/", "test/bird/"]

# initialize running variables for collection
trial_acc = []
trial_conf = np.zeros((len(test_paths),len(test_paths)))
trial_energy = np.zeros(len(test_paths)*201)

# IMPORTANT
num_trials = 10
epochs = 64
bs = 32
num_points = 320

# Single test LL

dataset = Mono(left=True, right=False, num_points=num_points, file_paths=train_paths)
test_dataset = Mono(left=True, right=False, num_points=num_points, file_paths=test_paths)

for j in range(num_trials):
    print("LL: ", j)
    pnt = PointNet(num_points=320, num_classes=len(test_paths), num_epoch=epochs, batchsize=bs, ptype='small', alpha=0, beta=0)
    res = pnt.train(dataset, test_dataset)
    trial_acc.append(res[0])
    trial_conf += res[1]
    trial_energy += res[2]

test_acc.append(trial_acc)
test_conf.append(trial_conf)
test_energy.append(trial_energy)

# reinitialize running variables for collection
trial_acc = []
trial_conf = np.zeros((len(test_paths),len(test_paths)))
trial_energy = np.zeros(len(test_paths)*201)

np.save("test_results/acc" + prefix, np.array(test_acc))
np.save("test_results/conf" + prefix, np.array(test_conf))
np.save("test_results/energy" + prefix, np.array(test_energy))

# Single test RR

dataset = Mono(left=False, right=True, num_points=num_points, file_paths=train_paths)
test_dataset = Mono(left=False, right=True, num_points=num_points, file_paths=test_paths)

for j in range(num_trials):
    print("RR: ", j)
    pnt = PointNet(num_points=320, num_classes=len(test_paths), num_epoch=epochs, batchsize=bs, ptype='small', alpha=0, beta=0)
    res = pnt.train(dataset, test_dataset)
    trial_acc.append(res[0])
    trial_conf += res[1]
    trial_energy += res[2]

test_acc.append(trial_acc)
test_conf.append(trial_conf)
test_energy.append(trial_energy)

# reinitialize running variables for collection
trial_acc = []
trial_conf = np.zeros((len(test_paths),len(test_paths)))
trial_energy = np.zeros(len(test_paths)*201)

np.save("test_results/acc" + prefix, np.array(test_acc))
np.save("test_results/conf" + prefix, np.array(test_conf))
np.save("test_results/energy" + prefix, np.array(test_energy))

# Single test F

dataset = Mono(left=True, right=True, num_points=num_points, file_paths=train_paths)
test_dataset = Mono(left=True, right=True, num_points=num_points, file_paths=test_paths)

for j in range(num_trials):
    print("F: ", j)
    pnt = PointNet(num_points=320, num_classes=len(test_paths), num_epoch=epochs, batchsize=bs, ptype='small', alpha=0, beta=0)
    res = pnt.train(dataset, test_dataset)
    trial_acc.append(res[0])
    trial_conf += res[1]
    trial_energy += res[2]

test_acc.append(trial_acc)
test_conf.append(trial_conf)
test_energy.append(trial_energy)

# reinitialize running variables for collection
trial_acc = []
trial_conf = np.zeros((len(test_paths),len(test_paths)))
trial_energy = np.zeros(len(test_paths)*201)

np.save("test_results/acc" + prefix, np.array(test_acc))
np.save("test_results/conf" + prefix, np.array(test_conf))
np.save("test_results/energy" + prefix, np.array(test_energy))

# Dual test

dataset = Siamese(num_points=320, file_paths=train_paths)
test_dataset = Siamese(num_points=320, file_paths=test_paths)

for j in range(num_trials):
    print("Dual: ", j)
    pnt = PointNet(num_points=320, num_classes=len(test_paths), num_epoch=epochs, batchsize=bs, ptype='', alpha=0, beta=0)
    res = pnt.train(dataset, test_dataset)
    trial_acc.append(res[0])
    trial_conf += res[1]
    trial_energy += res[2]

test_acc.append(trial_acc)
test_conf.append(trial_conf)
test_energy.append(trial_energy)

# reinitialize running variables for collection
trial_acc = []
trial_conf = np.zeros((len(test_paths),len(test_paths)))
trial_energy = np.zeros(len(test_paths)*201)

np.save("test_results/acc" + prefix, np.array(test_acc))
np.save("test_results/conf" + prefix, np.array(test_conf))
np.save("test_results/energy" + prefix, np.array(test_energy))