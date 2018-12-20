import numpy as np
import glob
import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset

class Mono(Dataset):
    def __init__(self, transform=None, left=True, right=True, num_points=320, file_paths=''):
        self.identifiers = []
        self.targets = []
        self.data = []

        def subsampleColumns(arr, num_points):
            # no camera data
            if arr.shape[1] == 0:
                return np.zeros((3,num_points))

            subsampling = np.linspace(0,len(arr[0]) - 1,num_points)
            subsampling = np.asarray(subsampling, dtype=np.int)
            return arr[:,subsampling]
        identity = 0
        for target, path in enumerate(file_paths):
            # define what files to load
            if left and right:
                file_type="*/npys/*fused.npy"
            elif left:
                file_type="*/npys/*leftreduced.npy"
            else:
                file_type="*/npys/*rightreduced.npy"

            filenames = glob.glob(path + file_type)
            print(path + file_type, len(filenames))
            for name in filenames:
                arr = np.load(name)
                arr = subsampleColumns(arr, num_points)
                # attach point cloud tensor
                self.data.append(Variable(torch.from_numpy(arr)))
                # attach label to each sample for error energy calculation
                self.identifiers.append(identity)
                identity += 1
                # attach class target
                self.targets.append(target)

            print("loaded: ", path)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        #get images and labels here 
        #returned images must be tensor
        #labels should be int
        return self.data[idx], self.targets[idx], self.identifiers[idx]

class Siamese(Dataset):
    def __init__(self, transform=None, num_points=320, file_paths=''):
        self.identifiers = []
        self.targets = []
        self.data_left = []
        self.data_right = []

        def subsampleColumns(arr, num_points):
            # no camera data
            if arr.shape[1] == 0:
                return np.zeros((3,num_points))
            subsampling = np.linspace(0,len(arr[0]) - 1,num_points)
            subsampling = np.asarray(subsampling, dtype=np.int)
            return arr[:,subsampling]
        identity = 0
        for target, path in enumerate(file_paths):
            # define what files to load
            filenames_left = glob.glob(path + "*/npys/*leftreduced.npy")
            filenames_right = glob.glob(path + "*/npys/*rightreduced.npy")

            for i in range(len(filenames_left)):
                arr_left = np.load(filenames_left[i])
                arr_right = np.load(filenames_right[i])

                arr_left = subsampleColumns(arr_left, num_points)
                arr_right = subsampleColumns(arr_right, num_points)

                # attach point cloud tensor
                self.data_left.append(Variable(torch.from_numpy(arr_left)))
                self.data_right.append(Variable(torch.from_numpy(arr_right)))

                # attach label to each sample for error energy calculation
                self.identifiers.append(identity)
                identity += 1
                # attach class target
                self.targets.append(target)

            print("loaded: ", path)

    def __len__(self):
        return len(self.data_left)

    def __getitem__(self, idx):
        #get images and labels here 
        #returned images must be tensor
        #labels should be int
        return self.data_left[idx], self.data_right[idx], self.targets[idx], self.identifiers[idx]