import pyrealsense2 as rs
import numpy as np
import cv2
import argparse
import os.path
import time

class Player(object):
    def __init__(self, colour=False, infrared=False, record=False, hw_sync=False):
        # Create object for parsing command-line options
        parser = argparse.ArgumentParser(description="Read recorded bag file and display depth stream in jet colormap.\
                                        Remember to change the stream resolution, fps and format to match the recorded.")
        # Add argument which takes path to a bag file as an input
        parser.add_argument("-i1", "--input1", type=str, help="Path to the left bag file")
        parser.add_argument("-i2", "--input2", type=str, help="Path to the right bag file")
        # Parse the command line arguments to an object
        args = parser.parse_args()
        # Safety if no parameter have been given
        if not args.input1 or not args.input2:
            print("No input paramater have been given.")
            print("For help type --help")
            exit()
        # Check if the given file have bag extension
        if os.path.splitext(args.input1)[1] != ".bag" or os.path.splitext(args.input2)[1] != ".bag":
            print("The given file is not of correct file format.")
            print("Only .bag files are accepted")
            exit()


        self.pipeline_left = rs.pipeline()
        self.config_left = rs.config()
        rs.config.enable_device_from_file(self.config_left, args.input1)
        self.config_left.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30) 

        if infrared:    
            self.config_left.enable_stream(rs.stream.infrared, 1, 1280, 720, rs.format.y8, 30)
        if colour:
            self.config_left.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)

        self.pipeline_right = rs.pipeline()
        self.config_right = rs.config()
        rs.config.enable_device_from_file(self.config_right, args.input2)
        self.config_right.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        if infrared:
            self.config_right.enable_stream(rs.stream.infrared, 1, 1280, 720, rs.format.y8, 30)
        if colour:
            self.config_right.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)

        # Start streaming from file and pause the streams
        self.start_left = self.pipeline_left.start(self.config_left)
        self.playback_left = self.start_left.get_device().as_playback()
        time.sleep(0.2)
        # time.sleep(0.219)
        self.start_right = self.pipeline_right.start(self.config_right)
        self.playback_right =  self.start_right.get_device().as_playback()

        # Create opencv window to render image in
        cv2.namedWindow("Depth Stream", cv2.WINDOW_AUTOSIZE)

        self.profile_left = self.start_left.get_stream(rs.stream.depth)
        self.intrinsics_left = self.profile_left.as_video_stream_profile().get_intrinsics()
        print("left intrinsics: ", self.intrinsics_left)

        self.profile_right = self.start_right.get_stream(rs.stream.depth)
        self.intrinsics_right = self.profile_right.as_video_stream_profile().get_intrinsics() 
        print("right intrinsics: ", self.intrinsics_right)


        
    # return current camera frames  (depth and colour)
    def getFrames(self): 
        # frame_right = self.pipeline_right.wait_for_frames() 
        # frame_left = self.pipeline_left.wait_for_frames()
        time.sleep(0.04)
        frame_left = rs.composite_frame(rs.frame())
        frame_right = rs.composite_frame(rs.frame())

        self.playback_right.resume()
        s_left = self.pipeline_right.poll_for_frames(frame_left)
        self.playback_right.pause()

        self.playback_left.resume()
        s_right = self.pipeline_left.poll_for_frames(frame_right)
        self.playback_left.pause()
        if not s_right or not s_left:
            [frame_left, frame_right] = self.getFrames()

        return [frame_left, frame_right]
        
            
            
    def stop(self):
        self.pipeline_right.stop()
        self.pipeline_left.stop()



