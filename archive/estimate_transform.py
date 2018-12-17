import numpy as np
import open3d as od
import copy
import argparse
import numpy as np

def draw_registration_result_original_color(source, target, transformation):
    source_temp = copy.deepcopy(source)
    source_temp.transform(transformation)
    od.draw_geometries([source_temp, target])

if __name__ == "__main__":
    # Create object for parsing command-line options
    parser = argparse.ArgumentParser(description="Stitch two ply files into one.")

    parser.add_argument("-i1", "--input1", type=str, help="Path to the left ply file")
    parser.add_argument("-i2", "--input2", type=str, help="Path to the right ply file")
    # Parse the command line arguments to an object
    args = parser.parse_args()
    # Safety if no parameter have been given
    if not args.input1 or not args.input2:
        print("No input paramater have been given.")
        print("For help type --help")
        exit()
    
    source = od.read_point_cloud(args.input1)
    target = od.read_point_cloud(args.input2)
    # draw initial alignment
    current_transformation = np.load("base_transform.npy")
    # current_transformation = np.array([[-0.035209, 0.720650, 0.692404, -0.435650], [-0.689849, 0.483771, -0.538585, 0.248754], [-0.723097, -0.496617, 0.480106, 0.236933], [0.000000, 0.000000, 0.000000, 1.000000]])

    draw_registration_result_original_color(
            source, target, current_transformation)


    # colored pointcloud registration
    # This is implementation of following paper
    # J. Park, Q.-Y. Zhou, V. Koltun,
    # Colored Point Cloud Registration Revisited, ICCV 2017
    # colored pointcloud registration



    voxel_radius = [0.02, 0.02, 0.02, 0.01, 0.005];
    max_iter = [ 50, 50, 50, 500, 500];

    # current_transformation = np.identity(4)

    for scale in range(len(voxel_radius)):
        iter = max_iter[scale]
        radius = voxel_radius[scale]
        print([iter,radius,scale])

        print("3-1. Downsample with a voxel size %.2f" % radius)
        # source_down = source
        # target_down = target
        source_down = od.voxel_down_sample(source, radius)
        target_down = od.voxel_down_sample(target, radius)

        print("3-2. Estimate normal.")
        od.estimate_normals(source_down, od.KDTreeSearchParamHybrid(
                radius = 2*radius, max_nn = 5))
        od.estimate_normals(target_down, od.KDTreeSearchParamHybrid(
                radius = 2*radius, max_nn = 5))

        print("3-3. Applying colored point cloud registration")
        result_icp = od.registration_colored_icp(source_down, target_down,
                radius, current_transformation,
                od.ICPConvergenceCriteria(relative_fitness = 1e-6,
                relative_rmse = 1e-6, max_iteration = iter))
        current_transformation = result_icp.transformation
        print("Current Transform: ", current_transformation)
        draw_registration_result_original_color(
            source_down, target_down, current_transformation)

    draw_registration_result_original_color(    
            source, target, current_transformation)
    np.save("transform", current_transformation)
    