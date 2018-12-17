import numpy as np
import open3d as od
import copy
import argparse
import numpy as np
import glob

class Fuser(object):
    def __init__(self, file_path):
        self.file_path = file_path
        self.base_transform = np.linalg.inv(np.load("base_transform_new.npy"))
        self.voxel_radius = [0.01, 0.01, 0.01]
        self.max_iter = [50, 50, 50]
        self.max_nn = 3
        self.relative_fitness = 1e-6
        self.relative_rmse = 1e-6

    def draw_registration_result_original_color(self, source, target, transformation):
        source_temp = copy.deepcopy(source)
        source_temp.transform(transformation)
        od.draw_geometries([source_temp, target])

    def estimateAverageTransform(self):
        targets = glob.glob(self.file_path+"*left.ply")
        sources = glob.glob(self.file_path+"*right.ply")
        assert len(sources) == len(targets)
        transform_list = []
        for i in range(len(sources)):
            tran = (self.estimateTransform(sources[i], targets[i]))
            transform_list.append(tran)
            if np.linalg.norm(self.base_transform - tran) < 0.1:
                self.base_transform =  tran
        
        # estimate good average transform
        # avg_transformation = self.base_transform

        avg_transformation =  np.mean(transform_list, axis=0)
        print(avg_transformation)
        # print(transform_list)

        # display final result
        for i in range(len(transform_list) - 1):
            source = od.read_point_cloud(sources[i])
            target = od.read_point_cloud(targets[i])
            self.draw_registration_result_original_color(    
        source, target, avg_transformation)
        np.save(self.file_path + "transform", avg_transformation)
        print(avg_transformation)

    # use color icp to estimate transform between two point clouds
    def estimateTransform(self, source, target):
        source = od.read_point_cloud(source)
        target = od.read_point_cloud(target)
        current_transformation = self.base_transform

        for scale in range(len(self.voxel_radius)):
            iterations = self.max_iter[scale]
            radius = self.voxel_radius[scale]

            source_down = od.voxel_down_sample(source, radius)
            target_down = od.voxel_down_sample(target, radius)

            od.estimate_normals(source_down, od.KDTreeSearchParamHybrid(
                    radius = 2*radius, max_nn = self.max_nn))
            od.estimate_normals(target_down, od.KDTreeSearchParamHybrid(
                    radius = 2*radius, max_nn = self.max_nn))

            result_icp = od.registration_colored_icp(source_down, target_down,
                    radius, current_transformation,
                    od.ICPConvergenceCriteria(relative_fitness = self.relative_fitness,
                    relative_rmse = self.relative_rmse, max_iteration = iterations))

            current_transformation = result_icp.transformation
            # self.draw_registration_result_original_color(
            # source_down, target_down, current_transformation)
        
        return current_transformation