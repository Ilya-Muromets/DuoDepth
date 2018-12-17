import open3d as od
import numpy as np
import argparse

parser = argparse.ArgumentParser(description="Stitch two npy files into one.")

parser.add_argument("-i1", "--input1", type=str, help="Path to the left npy file")
parser.add_argument("-i2", "--input2", type=str, help="Path to the right npy file")
# Parse the command line arguments to an object
args = parser.parse_args()
# Safety if no parameter have been given
if not args.input1 or not args.input2:
    print("No input paramater have been given.")
    print("For help type --help")
    exit()

source = np.load(args.input1)
target = np.load(args.input2)


transform = np.load("test2.npy")

def applyTransform(points, transform):
    for i in range(len(points[0])):
        pt = np.array([[points[0][i]], [points[1][i]], [points[2][i]], [1]])
        # print(pt)
        pt_new = np.dot(transform, pt) 
        points[0][i] = pt_new[0,0]
        points[1][i] = pt_new[1,0]
        points[2][i] = pt_new[2,0]

def croppedValues(arr, xlim, ylim, zlim):
    crop = []
    for i, entry in enumerate(arr):
        if abs(entry[0]) <= xlim and abs(entry[1]) <= ylim and abs(entry[2]) <= zlim:
            crop.append(i)
    return crop

def subsampleColumns(arr, num_points):
    subsampling = np.random.choice([i for i in range(len(arr[0]))], num_points)
    subsampling = np.linspace(0,len(arr[0]) - 1,num_points)
    subsampling = np.asarray(subsampling, dtype=np.int)
    return arr[:,subsampling]

# combined = np.concatenate((source,target), axis=1)
# print(combined.shape)
# np.save("result", combined)

applyTransform(target, transform)

source_sub = subsampleColumns(source, 1500)


source_sub = np.transpose(source_sub)
source = np.transpose(source)
target = np.transpose(target)

source = source[croppedValues(source, 10, 10, 0.63)]
source_sub = source_sub[croppedValues(source_sub, 10, 10, 0.63)]
target = target[croppedValues(target, 10, 10, 0.63)]

source_sub_pcd = od.PointCloud()
source_sub_pcd.points = od.Vector3dVector(source_sub)


source_pcd = od.PointCloud()
source_pcd.points = od.Vector3dVector(source)

target_pcd = od.PointCloud()
target_pcd.points = od.Vector3dVector(target)

# od.draw_geometries([target_pcd])
od.draw_geometries([source_pcd])
# od.draw_geometries([source_pcd, target_pcd])

downpcd = od.voxel_down_sample(source_pcd, voxel_size = 0.0035)
od.draw_geometries([downpcd])
print(len(downpcd.points))

# od.draw_geometries([target_pcd])
# od.draw_geometries([source_pcd])
# od.draw_geometries([source_pcd, target_pcd])