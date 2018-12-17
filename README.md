This repository contains tools I used to record and process data simultaneously from two Intel RealSense D415 cameras.
It might not be the most efficient implementation of these tools, but it is relatively human readable (I hope.).

### Prerequisites
You'll need the **opencv-python, numpy, pyrealsense2, and open3d** libraries in order to properly run everything.
Code was tested for machines running **Python 3.6.5 and 3.7.0** on **Ubuntu 18.04 and 16.04**.  

### The Basics
    python3 two_camera_player.py
This will start recording from the two cameras, saving the recorded data in data/the_current_time/, pressing the **a** button will save 2 color .ply files from the left and right cameras into data/the_current_time/plys/, similarly pressing **s** will save 2 numpy arrays of points (without colour) from the left and rigth cameras into data/the_current_time/npys/.  

If this code does not start, try unplugging and replugging the cameras and running again. You can always check that the hardware is working with Intel's RealSense Viewer before attemping to run this code.

At the end of recording the player will "compact" the numpy arrays, removing zero distance entries and converting it into a 3xnum_nonzero_points array, which can then be directly loaded into PyTorch.

Parameters such as turning on and off the color and infrared streams, hardware sync, and .bag recording can be found within the player.py utility and toggled accordingly.

### Other Files
    fuser.py
A utility for attempting to register left and right point clouds using color icp. Works sometimes, deemed unreliable for this application.

    hand.json
Json file containing parameters for recording such as max distance, color correction, point density, etc.

    two_camera_reader.py
Carbon copy of two_camera_player.py except it takes two inputs "-i1 left.bag -i2 right.bag" and will play from file instead of from device.

Good luck.
-Ilya
