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
    def __init__(self, transform=None):
        return 1
    #init data here

    def __len__(self):
        return 1  #length of the data

    def __getitem__(self, idx):
        #get images and labels here 
        #returned images must be tensor
        #labels should be int 
        return img1, img2 , label1, label2 