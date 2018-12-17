import open3d as od
import numpy as np
import argparse

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
transform = np.load("test.npy")

target.transform(transform)
od.draw_geometries([source, target])
