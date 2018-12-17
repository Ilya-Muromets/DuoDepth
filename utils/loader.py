import numpy as np
import glob
import torch
from torch.autograd import Variable

class Loader(object):
    def __init__(self, left=True, right=True, num_points=2500):
        self.left = left
        self.right = right
        self.num_points = num_points

    # load one folder
    def loadData(self, file_path, class_num):
    #helper function for subsampling
        def subsampleColumns(arr, num_points):
            # no camera data
            if arr.shape[1] == 0:
                return np.zeros((3,num_points))

            subsampling = np.linspace(0,len(arr[0]) - 1,num_points)
            subsampling = np.asarray(subsampling, dtype=np.int)
            return arr[:,subsampling]
    
        # define what files to load
        if self.left and self.right:
            file_type="*/npys/*fused.npy"
        elif self.left:
            file_type="*/npys/*leftreduced.npy"
        else:
            file_type="*/npys/*rightreduced.npy"

        filenames = glob.glob(file_path + file_type)
        datum = np.load(filenames[0])
        datum = subsampleColumns(datum, self.num_points)
    
        # add class label
        datum = np.concatenate((datum,np.array([[class_num],[class_num],[class_num]])), axis=1)
        # add identifier (which sample from the class it is)
        datum = np.concatenate((datum,np.array([[0],[0],[0]])), axis=1)
        datum = np.expand_dims(datum, axis=0)

        for i, name in enumerate(filenames[1:]):
            arr = np.load(name)
            arr = subsampleColumns(arr, self.num_points)

            # add class label
            arr = np.concatenate((arr,np.array([[class_num],[class_num],[class_num]])), axis=1)
            # add identifier (which sample from the class it is)
            arr = np.concatenate((arr,np.array([[i+1],[i+1],[i+1]])), axis=1)
            arr = np.expand_dims(arr, axis=0)
            datum = np.concatenate((datum,arr),axis=0)

        print("loaded: ", file_path)
        return datum

    # load all folders contents as specified in file_paths
    def loadFolders(self, file_paths):
        data = []

        for class_num, path in enumerate(file_paths):
            data.append(self.loadData(path, class_num))

        data_arr = np.concatenate(data)
        np.random.shuffle(data_arr)
        return Variable(torch.from_numpy(data_arr))

    class Siamese(Dataset):
        def __init__(self, transform=None):

        #init data here

        def __len__(self):
            return   #length of the data

        def __getitem__(self, idx):
            #get images and labels here 
            #returned images must be tensor
            #labels should be int 
            return img1, img2 , label1, label2 
