import numpy as np
import glob
import open3d as od
import re
import os
import threading

class Processor(object):
    # initialise with limits for point cloud cropping, as well as file path
    def __init__(self, file_path, xlim=10, ylim=10, zlim=0.1, crop=True, overwrite=False):
        self.xlim = xlim
        self.ylim = ylim
        self.zlim = zlim
        self.file_path = file_path
        self.crop = crop
        self.overwrite = overwrite

    def compact(self):
        def atoi(text):
            return int(text) if text.isdigit() else text.lower()
            
        def natural_keys(text):
            return [atoi(c) for c in re.split('(\d+)', text)]

        def croppedValues(arr):
            crop = []
            for i, entry in enumerate(arr):
                if abs(entry[0]) <= self.xlim and abs(entry[1]) <= self.ylim and abs(entry[2]) <= self.zlim:
                    crop.append(i)
            return crop

        npy_list = glob.glob(self.file_path + "*.npy")
        npy_list.sort(key=natural_keys)
        npy_lists = np.array_split(npy_list, 6)
        threads = []
        for i in range(len(npy_lists)):
            # print(npy_list)
            t = threading.Thread(target=self.compact_worker, args=(npy_lists[i],))
            threads.append(t)
            t.start()
       
    def compact_worker(self, npy_list):
            for npy in npy_list:
                points = np.load(npy)
                print("started: ", npy)
                # if this is an unprocessed file, process
                if type(points[0]) is type(np.void(0)):
                    points = points[np.nonzero(points)]
                    points_new = np.empty((3,len(points)),dtype=np.ndarray)

                    # convert np.void (what realsense returns) back into float32 array
                    for i in range(len(points)):
                        points_new[0][i] = points[i][0]
                        points_new[1][i] = points[i][1]
                        points_new[2][i] = points[i][2]
                    points_new = np.asarray(points_new, dtype=np.float32)
                    # if crop, crop
                    if self.crop:
                        points_new = points_new[croppedValues(points_new)]
                    
                    print("finished: ", npy)
                    if self.overwrite:
                        np.save(npy, points_new)
                    else:
                        new_name = self.file_path + "processed" + os.path.basename(npy)
                        np.save(new_name, points_new)

                else:
                    print("finished: ", npy)
                    print("skipped")

    def compactSingleThread(self):
        def atoi(text):
            return int(text) if text.isdigit() else text.lower()
            
        def natural_keys(text):
            return [atoi(c) for c in re.split('(\d+)', text)]

        def croppedValues(arr):
            crop = []
            for i, entry in enumerate(arr):
                if abs(entry[0]) <= self.xlim and abs(entry[1]) <= self.ylim and abs(entry[2]) <= self.zlim:
                    crop.append(i)
            return crop

        npy_list = glob.glob(self.file_path + "npys/*.npy")
        npy_list.sort(key=natural_keys)
        for npy in npy_list:
                points = np.load(npy)
                print("started: ", npy)
                # if this is an unprocessed file, process
                if type(points[0]) is type(np.void(0)):
                    points = points[np.nonzero(points)]
                    points_new = np.array([[points[0][0]],[points[0][1]],[points[0][2]]])

                    # convert np.void (what realsense returns) back into float32 array
                    for i in range(1, len(points)):
                        arr = np.array([[points[i][0]],[points[i][1]],[points[i][2]]])
                        points_new = np.hstack((points_new, arr))
                    
                    # if crop, crop
                    if self.crop:
                        points_new = points_new[croppedValues(points_new)]

                    self.plot(points_new)
                    
                    print("finished: ", npy)
                    if self.overwrite:
                        np.save(npy, points_new)
                    else:
                        new_name = self.file_path + "npys/processed" + os.path.basename(npy)
                        np.save(new_name, points_new)

                else:
                    print("finished: ", npy)
                    print("skipped")

    def plot(self, arr): 
        arr = np.transpose(arr)
        for i in range(len(arr)):
            arr[i] = (arr[i][0],arr[i][1],arr[i][2])
        pcd = od.PointCloud()
        pcd.points = od.Vector3dVector(arr)
        od.draw_geometries([pcd])

    def fuse(self):
        def atoi(text):
            return int(text) if text.isdigit() else text.lower()
            
        def natural_keys(text):
            return [atoi(c) for c in re.split('(\d+)', text)]

        transform = np.load("base_transform_new.npy")

        npy_list_left = glob.glob(self.file_path + "*left.npy")
        npy_list_left.sort(key=natural_keys)

        npy_list_right = glob.glob(self.file_path + "*right.npy")
        npy_list_right.sort(key=natural_keys)

        if len(npy_list_left) != len(npy_list_right):
            raise Exception("Mismatch in file numbers.")

        npy_lists_left = np.array_split(npy_list_left, 6)
        npy_lists_right = np.array_split(npy_list_right, 6)
        
        threads = []
        for i in range(len(npy_lists_left)):
            t = threading.Thread(target=self.fuse_worker, args=(npy_lists_left[i],npy_lists_right[i],transform,))
            threads.append(t)
            t.start()

    def fuse_worker(self, npy_list_left, npy_list_right, transform):
        def croppedValues(arr, min_dist):
            crop = []
            if min_dist > 0.05:
                for i, entry in enumerate(arr):
                    if abs(entry[0]) <= self.xlim and abs(entry[1]) <= self.ylim and abs(entry[2]) <= (min_dist + self.zlim):
                        crop.append(i)
            else:
                for i, entry in enumerate(arr):
                    if abs(entry[0]) <= self.xlim and abs(entry[1]) <= self.ylim and abs(entry[2]) <= 0.65:
                        crop.append(i)
            return crop

        for i in range(len(npy_list_left)):
            source_name = npy_list_right[i]
            target_name = npy_list_left[i]
            source = np.load(source_name)
            target = np.load(target_name)
            print("loaded: ", source_name," and ", target_name)

            source = np.transpose(source)
            target = np.transpose(target)

            source = source[croppedValues(source, np.min(source.T[2]))]
            target = target[croppedValues(target, np.min(target.T[2]))]

            source_pcd = od.PointCloud()
            source_pcd.points = od.Vector3dVector(source)

            target_pcd = od.PointCloud()
            target_pcd.points = od.Vector3dVector(target)
            
            # voxel downsampling
            source_pcd = od.voxel_down_sample(source_pcd, voxel_size = 0.0035)
            target_pcd = od.voxel_down_sample(target_pcd, voxel_size = 0.0035)

            source_pcd.transform(transform)

            # od.draw_geometries([source_pcd])
            # od.draw_geometries([target_pcd])

            source = np.asarray(source_pcd.points)
            target = np.asarray(target_pcd.points)

            source = np.transpose(source)
            target = np.transpose(target)
            np.save(source_name[:-4] + "reduced", source)
            np.save(target_name[:-4] + "reduced", target)
            np.save(target_name[:-8] + "fused", np.concatenate((source, target), axis=1))
            print("reduced: ", source_name," and ", target_name)
    
    # def fuseSingleThread(self):
    #     def atoi(text):
    #         return int(text) if text.isdigit() else text.lower()
            
    #     def natural_keys(text):
    #         return [atoi(c) for c in re.split('(\d+)', text)]

    #     transform = np.load("test2.npy")

    #     npy_list_left = glob.glob(self.file_path + "*left.npy")
    #     npy_list_left.sort(key=natural_keys)

    #     npy_list_right = glob.glob(self.file_path + "*right.npy")
    #     npy_list_right.sort(key=natural_keys)

    #     if len(npy_list_left) != len(npy_list_right):
    #         raise Exception("Mismatch in file numbers.")

    #     def croppedValues(arr):
    #         crop = []
    #         for i, entry in enumerate(arr):
    #             if abs(entry[0]) <= self.xlim and abs(entry[1]) <= self.ylim and abs(entry[2]) <= self.zlim:
    #                 crop.append(i)
    #         return crop

    #     for i in range(len(npy_list_left)):
    #         source_name = npy_list_right[i]
    #         target_name = npy_list_left[i]
    #         source = np.load(source_name)
    #         target = np.load(target_name)
    #         # print("loaded: ", source_name," and ", target_name)

    #         source = np.transpose(source)
    #         target = np.transpose(target)

    #         source = source[croppedValues(source)]
    #         target = target[croppedValues(target)]

    #         source_pcd = od.PointCloud()
    #         source_pcd.points = od.Vector3dVector(source)

    #         target_pcd = od.PointCloud()
    #         target_pcd.points = od.Vector3dVector(target)
            
    #         # voxel downsampling
    #         source_pcd = od.voxel_down_sample(source_pcd, voxel_size = 0.0035)
    #         target_pcd = od.voxel_down_sample(target_pcd, voxel_size = 0.0035)

    #         source_pcd.transform(transform)

    #         # od.draw_geometries([source_pcd])
    #         # od.draw_geometries([target_pcd])

    #         source = np.asarray(source_pcd.points)
    #         target = np.asarray(target_pcd.points)

    #         source = np.transpose(source)
    #         target = np.transpose(target)
    #         np.save(source_name[:-4] + "reduced", source)
    #         np.save(target_name[:-4] + "reduced", target)
    #         np.save(target_name[:-8] + "fused", np.concatenate((source, target), axis=1))
    #         # print("reduced: ", source_name," and ", target_name)

    ## convert old npys to new npys
    # def convert(self):
    #     def atoi(text):
    #         return int(text) if text.isdigit() else text.lower()
            
    #     def natural_keys(text):
    #         return [atoi(c) for c in re.split('(\d+)', text)]

    #     npy_list = glob.glob(self.file_path + "npys/*.npy")
    #     npy_list.sort(key=natural_keys)
    #     for npy in npy_list:
    #         points = np.load(npy)
    #         print("started: ", npy)
    #         points_new = np.array([[points[0][0]],[points[0][1]],[points[0][2]]])

    #         # convert np.void (what realsense returns) back into float32 array
    #         for i in range(1, len(points)):
    #             arr = np.array([[points[i][0]],[points[i][1]],[points[i][2]]])
    #             points_new = np.hstack((points_new, arr))
    #         # self.plot(points_new)
    #         print("finished: ", npy)
    #         if self.overwrite:
    #             np.save(npy, points_new)
    #         else:
    #             new_name = self.file_path + "npys/processed" + os.path.basename(npy)
    #             np.save(new_name, points_new)

    # register npys based on single transform, and also downsample
