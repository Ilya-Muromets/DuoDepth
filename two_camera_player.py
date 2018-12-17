import pyrealsense2 as rs
import numpy as np
import cv2
import open3d as od
from utils.player import Player
from utils.processor import Processor
import datetime
import os
import time

def main():
    record = True
    plr = Player(colour=True, record=False)

    # initialize placeholder point clouds
    pc_left = rs.pointcloud()
    pc_right = rs.pointcloud()

    # create prefix and directories for naming files
    current_time = datetime.datetime.now()
    prefix = current_time.strftime("%Y-%m-%d:%H:%M")

    npy_dir = "data/" + prefix + "/npys/"
    ply_dir = "data/" + prefix + "/plys/"

    os.makedirs(npy_dir)
    os.makedirs(ply_dir)


    # initialise counters for naming files
    ply_counter = npy_counter = 0

    try:
        while True:
            [frame_left, frame_right] = plr.getFrames()
            if not frame_left or not frame_right:
                continue
            depth_frame_left = frame_left.get_depth_frame()
            depth_frame_right = frame_right.get_depth_frame()
            
            depth_color_frame_left = rs.colorizer().colorize(depth_frame_left)
            depth_color_frame_right = rs.colorizer().colorize(depth_frame_right)

            # Convert depth_frame to numpy array to render image
            depth_color_image_left = np.asanyarray(depth_color_frame_left.get_data())
            depth_color_image_right = np.asanyarray(depth_color_frame_right.get_data())

            depth_color_image = np.hstack((depth_color_image_left,depth_color_image_right))

            # resize to fit screen, change as desired
            image = cv2.resize(depth_color_image, (1440, 720))

            # Render image in opencv window
            cv2.imshow("Depth Stream", image)
            key = cv2.waitKey(1)

            # if s is pressed save .npy
            if key == ord('s') and not record:

                points_left = pc_left.calculate(depth_frame_left)
                points_right = pc_right.calculate(depth_frame_right)
                
                np.save(npy_dir + str(npy_counter) + "left", np.array(points_left.get_vertices()))
                print("File saved to " + npy_dir + str(npy_counter) + "left.npy")

                np.save(npy_dir + str(npy_counter) + "right", np.array(points_right.get_vertices()))
                print("File saved to " + npy_dir + str(npy_counter) + "right.npy")

                npy_counter += 1
            if record:
                # convert and save left file
                points_left = pc_left.calculate(depth_frame_left)
                points_left = np.array(points_left.get_vertices())
                # print(points_left.shape)
                points_left = points_left[np.nonzero(points_left)]
                np.save(npy_dir + str(npy_counter) + "left", points_left)
                print("File saved to " + npy_dir + str(npy_counter) + "left.npy")

                # convert and save right file
                points_right = pc_right.calculate(depth_frame_right)
                points_right = np.array(points_right.get_vertices())
                points_right = points_right[np.nonzero(points_right)]
                np.save(npy_dir + str(npy_counter) + "right", points_right)
                print("File saved to " + npy_dir + str(npy_counter) + "right.npy")

                npy_counter += 1
                if npy_counter > 200:
                    raise Exception("finished recording")
                # time.sleep(0.1)

            
            # if a is pressed save .ply
            if key == ord('a'):

                color_frame_left = frame_left.get_color_frame()
                color_frame_right = frame_right.get_color_frame()

                # ply's require a colour mapping
                pc_left.map_to(color_frame_left)
                pc_right.map_to(color_frame_right)

                points_left = pc_left.calculate(depth_frame_left)
                points_right = pc_right.calculate(depth_frame_right)

                points_left.export_to_ply(ply_dir + str(ply_counter) + "left.ply", color_frame_left)
                print("File saved to " + ply_dir + str(ply_counter) + "left.ply")

                points_right.export_to_ply(ply_dir + str(ply_counter) + "right.ply", color_frame_right)
                print("File saved to " + ply_dir + str(ply_counter) + "right.ply")

                ply_counter += 1

            # if pressed escape exit program 
            if key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        print("Stopping pipelines.")
        plr.stop()
        print("Compacting files.")
        # pcr = Processor(npy_dir,crop=False, overwrite=True)
        # pcr.compact()
        

if __name__ == "__main__":
    main()
