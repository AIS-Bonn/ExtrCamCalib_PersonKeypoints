*Online Marker-free Extrinsic Camera Calibration using Person Keypoint Detections*

https://user-images.githubusercontent.com/109513136/188647343-c289a007-dbff-44d4-a7c9-b5b39c992892.mp4

This ROS-package provides extrinsic calibration for a static [camera network](https://github.com/AIS-Bonn/SmartEdgeSensor3DHumanPose) providing person keypoint detections.<br>
We assume the intrinsic calibration and a rough estimate of the extrinsic calibration to be available.

## Installation

### Dependencies

The package was tested with ROS melodic and Ubuntu 18.04, as well as ROS noetic and Ubuntu 20.04. 

The former requires the [geometry2](https://github.com/ros/geometry2) and [cv_bridge](https://github.com/ros-perception/vision_opencv) packages to be placed in the `catkin_ws/src` folder.<br>
Both packages must be built with Python3 support, e.g. using 
```bash
catkin_make -DPYTHON_EXECUTABLE:FILEPATH=/usr/bin/python3 -DPYTHON_INCLUDE_DIR=/usr/include/python3.6m -DPYTHON_LIBRARY=/usr/lib/x86_64-linux-gnu/libpython3.6m.so
```

The received person keypoint detections must be encoded using the [person_msgs](https://github.com/AIS-Bonn/SmartEdgeSensor3DScenePerception/tree/master/person_msgs/msg) package with the default joint order being defined [here](keypoint_camera_calibration/examples/keypoint_correspondences.txt).
Place the `person_msgs` package it in your `catkin_ws/src` folder and build it via `catkin_make` or `catkin_build person_msgs`.

The factor graph optimization is implemented using the [GTSAM library](https://github.com/borglab/gtsam),<br>
which can be installed by `pip install gtsam==4.1.1`.

For additional standard-dependencies, see [calibration.py](keypoint_camera_calibration/scripts/calibration.py).

### Build

Place the `keypoint_camera_calibration` package in your `catkin_ws/src` folder.<br>
Navigate to your `catkin_ws` and run `catkin_make` or `catkin_build keypoint_camera_calibration`,<br> 
depending on your build system.

### Demo

The [examples](keypoint_camera_calibration/examples) folder contains calibration files for the presented camera network.<br>
The initial estimate of the extrinsic calibration gets generated automatically by retracting the reference calibration.<br>
The corresponding calibration and evaluation bagfiles can be found [here](https://cloud.vi.cs.uni-bonn.de/index.php/s/F8DqX7sFCHaodBN).<br>
All parameters are preset for this scenario.<br>

Simply start the calibration pipeline by 
```bash
rosrun keypoint_camera_calibration calibration.py
```
Play one of the provided bagfiles to start calibration
```bash
rosbag play $(rospack find keypoint_camera_calibration)/examples/bagfiles/2022-05-26_calib_2persons_3min.bag
```

Results will be placed in the `logs` folder.<br>
Visualization presets for `rqt` and `rviz` are [provided](keypoint_camera_calibration/examples/presets).

### General Usage

Provide `.yaml` files following the syntax established in the [example files](keypoint_camera_calibration/examples):
* Intrinsic calibration
* Estimated extrinsic calibration 
* Reference extrinsic calibration (optional)

Edit the required parameters in the `__init__` function of [calibration.py](keypoint_camera_calibration/scripts/calibration.py) to match your scenario:
* File locations
* Message properties
* Method parameters

Start calibration by
```bash
rosrun keypoint_camera_calibration calibration.py
```
Provide `person_msgs` by playing a bagfile or accessing a sensor network. 

## Citation

Bastian Pätzold, Simon Bultmann, and Sven Behnke:<br>
*Online Marker-free Extrinsic Camera Calibration using Person Keypoint Detections*.<br>
DAGM German Conference on Pattern Recognition (GCPR), Konstanz, September 2022.

## License

This package is licensed under BSD-3.
