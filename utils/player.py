import pyrealsense2 as rs
import numpy as np
import cv2
import datetime
import time
import json
import os
import shutil


class Player(object):
    def __init__(self, colour=False, infrared=False, record=True, hw_sync=True):
        self.colour = colour
        self.ctx = rs.context()
        self.devices = self.ctx.query_devices()
        print("Detected device serial numbers: ")
        try:
            self.device_left = self.devices[0].get_info(rs.camera_info.serial_number)
            self.device_right = self.devices[1].get_info(rs.camera_info.serial_number)
        except Exception:
            raise(Exception("Two devices not found."))

        print(self.device_left)
        print(self.device_right)

        # load settings from json

        json_data = open("hand.json").read()
        adv_string = json.loads(json_data)
        json_string = str(adv_string).replace("'", '\"')
        advnc_mode_left = rs.rs400_advanced_mode(self.devices[0])
        advnc_mode_right = rs.rs400_advanced_mode(self.devices[1])
        advnc_mode_left.load_json(json_string)
        advnc_mode_right.load_json(json_string)

        # create save folder
        current_time = datetime.datetime.now()
        prefix = current_time.strftime("%Y-%m-%d:%H:%M")

        path = "data/" + prefix + "/"
        if not os.path.exists(path):
            os.makedirs(path)
        else:
            shutil.rmtree(path)
            os.makedirs(path)


        self.pipeline_left = rs.pipeline()
        self.config_left = rs.config()
        self.config_left.enable_device(self.device_left)
        self.config_left.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30) 


        if infrared:    
            self.config_left.enable_stream(rs.stream.infrared, 1, 1280, 720, rs.format.y8, 30)
        if colour:
            self.config_left.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)
        if record:
            self.config_left.enable_record_to_file("data/" + prefix + "/left.bag")


        


        self.pipeline_right = rs.pipeline()
        self.config_right = rs.config()
        self.config_right.enable_device(self.device_right)
        self.config_right.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        if infrared:
            self.config_right.enable_stream(rs.stream.infrared, 1, 1280, 720, rs.format.y8, 30)
        if colour:
            self.config_right.enable_stream(rs.stream.color, 1280, 720, rs.format.rgb8, 30)
        if record:
            self.config_right.enable_record_to_file("data/" + prefix + "/right.bag")


        #print("Starting Recording in:")
        # print("3")
        # time.sleep(1)
        # print("2")
        # time.sleep(1)
        # print("1")
        # time.sleep(1)
        # print("now")

        # Start streaming
        
        self.playback_left = self.pipeline_left.start(self.config_left)
        start = time.time()
        self.playback_right = self.pipeline_right.start(self.config_right)
        print("Stream desync: ", time.time() - start)

        self.profile_left = self.playback_left.get_stream(rs.stream.depth)
        self.intrinsics_left = self.profile_left.as_video_stream_profile().get_intrinsics()
        print("left intrinsics: ", self.intrinsics_left)

        self.profile_right = self.playback_right.get_stream(rs.stream.depth)
        self.intrinsics_right = self.profile_right.as_video_stream_profile().get_intrinsics() 
        print("right intrinsics: ", self.intrinsics_right)
        if hw_sync:
            self.ds_left = self.devices[0].query_sensors()[0]
            self.ds_left.set_option(rs.option.inter_cam_sync_mode, 1)
            print("left option: ", ["null", "master", "slave"][int(self.ds_left.get_option(rs.option.inter_cam_sync_mode))])

            self.ds_right = self.devices[1].query_sensors()[0]
            self.ds_right.set_option(rs.option.inter_cam_sync_mode, 2)
            print("right option: ", ["null", "master", "slave"][int(self.ds_right.get_option(rs.option.inter_cam_sync_mode))])
        

        
    # return current camera frames  (depth and colour)
    def getFrames(self):
        #frameset_left = rs.composite_frame(rs.frame())
        #frameset_right = rs.composite_frame(rs.frame())
        #frame_right = self.pipeline_right.wait_for_frames()
        #frame_left = self.pipeline_left.poll_for_frames(frameset_left)

        #self.playback_left.resume()    
        frame_left = self.pipeline_left.wait_for_frames()
        #self.playback_left.pause()

        #self.playback_right.resume()
        frame_right = self.pipeline_right.wait_for_frames()
        #self.playback_right.pause()

        ret_frames = [None, None]

        if frame_left:
            ret_frames[0] = frame_left
        # else:
        #     time.sleep(0.015)

        if frame_right:
            ret_frames[1] = frame_right
        # else:
        #     time.sleep(0.015) 
        #print(ret_frames[0].timestamp - ret_frames[1].timestamp)
        return ret_frames
        
            
            
    def stop(self):
        self.pipeline_right.stop()
        self.pipeline_left.stop()



