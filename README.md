This repository contains tools I used to record and process data from two Intel RealSense D415 cameras.  
It might not be the most efficient implementation of these tools, but I hope it to be functional and relatively human readable.  

### Prerequisites
You'll need the **opencv-python, numpy, pyrealsense2, and open3d** libraries for full code functionality. I highly recommended making a Conda environment for this.  
Code was tested on machines running **Python 3.6.5 and 3.7.0** on **Ubuntu 18.04 and 16.04**.  

## The Basics
    python3 two_camera_player.py
This will start recording from the two cameras, saving the recorded data in data/the_current_time/, pressing the **a** button will save 2 color .ply files from the left and right cameras into data/the_current_time/plys/, similarly pressing **s** will save 2 numpy arrays of points (without colour) from the left and right cameras into data/the_current_time/npys/.  

If this code does not run properly, try unplugging and re-plugging the cameras and running again (sometimes flipping the wires works even though they’re type-c and theoretically reversible). I highly recommend always checking that the hardware is working with Intel's RealSense Viewer before attempting to run this code.
At the end of recording the player will "compact" the numpy arrays, removing zero distance entries and converting it into a 3xnum_nonzero_points array, which can then be directly loaded into PyTorch.  
Parameters such as turning on and off the color and infrared streams, hardware sync, and .bag recording can be found within the player.py utility and toggled accordingly.

## Other Files and Folders
    archive/
Old, and possibly unused, code for transform estimation and stitching; may be useful for experimenting.
    
    test/
Put individual folders full of test data here.

    train/
Put individual folders full of train data here.

    utils/
The heart of this project, contains utilities for fusing, cropping, processing, recording, and playing back recordings of point clouds.
    
    hand.json
Json file containing camera properties for recording such as max distance, color correction, and point density.

    two_camera_reader.py
Carbon copy of two_camera_player.py except it takes two inputs "-i1 left.bag -i2 right.bag" and will play from file instead of from device.

    train_*.py
Various files for training with various parameters and datasets, used for ablation testing primarily.

    compact.py
Re-processing for point clouds that didn't come out in the right data format.

    graphs.ipynb
Graphs.

## The ML Stuff
All architecture is written using **PyTorch** and is ran on **PyTorch 0.4.0**.  
The train_\*.py files are the most top-level files and execute loading the datasets, training on them, and saving/printing accuracy results. utils/pointnet.py and utils/dualnet.py implement training and optimization. utils/torchnet/pointnet/ contains all the various architectures to train on.  

This is not production ready code, but most research isn’t; I hope you can find it useful.  

A big thanks fxia22 as much of this code would not be possible without https://github.com/fxia22/pointnet.pytorch.  

## Dataset
Dataset can be downloaded from:
https://drive.google.com/open?id=1-nHLWO1fVMprPBvNDVsLZKLB8zvrqUWk

Good luck.  
-	Ilya
