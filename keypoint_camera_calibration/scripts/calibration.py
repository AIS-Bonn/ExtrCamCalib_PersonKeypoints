#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from person_msgs.msg import Person2DOcclusionList as Person2DList # New bagfiles
#from person_msgs.msg import Person2DList as Person2DList # Old bagfiles

import rospy
import rospkg 
import tf2_ros
import message_filters
from std_msgs.msg import ColorRGBA
from sensor_msgs.msg import CameraInfo
from sensor_msgs.msg import Image as ROS_Image
from visualization_msgs.msg import MarkerArray,Marker
from geometry_msgs.msg import TransformStamped,Pose,Point
from keypoint_camera_calibration.msg import Person2DWithID

import gtsam
import quaternion
import numpy as np
import cv2
from cv_bridge import CvBridge
from scipy.spatial.distance import pdist
from scipy.optimize import minimize_scalar
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib import cm as cmx

import os
import sys
import copy
import time
import yaml
import shutil
import datetime
import threading

class calibration:
    
    def __init__(self):

        # -- Method Parameters -- #

        # Synchronizer
        self.sync_min_gap = 0.0                         # time in seconds in which incomming frame-sets are discarded after the last accepted frame-set 
        self.sync_queue = 10                            # parametrizes the ROS approximate time synchronizer
        self.sync_slop = 10                             # parametrizes the ROS approximate time synchronizer

        # Filter
        self.filter_stamps_skip = False                 # ignores all _stamps_ constraints
        self.filter_stamps_span_max = 0.05              # frame-sets with larger span between the earliest and last detection in seconds are removed entirely
        self.filter_stamps_std_max = 0.025              # frame-sets with larger std. of all timestamps in seconds are removed entirely
        self.filter_scores_skip = False
        self.filter_scores_min = 0.6                    # removes joint detections with lower scores/confidence values
        self.filter_bbox_minimum_distance = 100         # removes person detections within one camera, where the shortest distance in pixels between their bounding boxes is shorter 
        
        # Depth Estimation
        self.person_size_min = 1.70                     # minimum person height to be expeced during calibration
        self.person_size_max = 2.00                     # maximum person height to be expeced during calibration
        self.person_size_ratio_hips      = 0.1526136692 # assumed ratio: hip distance / person height
        self.person_size_ratio_shoulders = 0.1752612993 # assumed ratio: shoulder distance / person height
        self.person_size_ratio_torso     = 0.2924273380 # assumed ratio: torso distance / person height

        # Data Association
        self.data_association_views_min = 2             # specifies the number of required perspectives on each joint within a person hypothesis
        self.data_association_threshold = 1.0           # required distance in meters for assigning a person detection to a person hypothesis
        
        # Hypothesis Selection
        self.hyp_sel_random = False                     # disables the selection algorithm (parameters below) and selects at random instead
        self.hyp_sel_spacing = 0.2                      # minimum distance that is enforced between all selected hypohese (if possible)
        self.hyp_sel_timing = 0.05                      # weight for prefering the selection of newer over older hypotheses: 0-> uniform selection ... 1-> linear decay

        # Optimization
        self.optimization_flag = True                   # enables/disables optimization
        self.optimization_interval = 0.50               # optimization interval in seconds
        self.optimization_scope_size = 50               # number of person hypotheses included in one optimization
        self.optimization_wait_mult = 2.0               # blocks optimization until the queue contains at least _wait_mult*_scope_size hypotheses
        self.optimization_delta = 10.0                  # parametrizes the GTSAM dogleg optimizer 
        
        # Temporal Smoothing
        self.kal_init_translation = 0.025                                   # initial covariance for all prior factors (std. in meters)
        self.kal_init_rotation = 1.0                                        # initial covariance for all prior factors (std. in degrees) 
        self.kal_process_translation = self.kal_init_translation * 0.05     # process noise added to the measured covariance (std. in meters)
        self.kal_process_rotation = self.kal_init_rotation * 0.05           # process noise added to the measured covariance (std. in degrees)

        # Scaling
        self.smoothing_enforce_scale = True             # enforces the scale of the initial estimate by applying Umeyama's method
        
        # -- Inputs -- #

        self.cam_ID_range = np.arange(0,100)            # scan all input .yaml input-files for the scpecified range of IDs in the form "cam_ID"

        # Intrinsics
        self.intrinsics_file = "examples/camera_intrinsics.yaml"
        
        # Initial estimate extrinsics
        self.extrinsics_estimate_readfromfile = False
        self.extrinsics_estimate_file = "examples/camera_extrinsics_estimate.yaml"
        
        # Reference extrinsics
        self.extrinsics_reference_readfromfile = True
        self.extrinsics_reference_file = "examples/camera_extrinsics_reference.yaml"
        
        # Generate new initial estimate extrinsics by retracting the reference calibration, which will be written to self.extrinsics_estimate_file
        self.extrinsics_estimate_generate = True
        self.extrinsics_estimate_generate_noise_translation = 0.25              # std. in meters
        self.extrinsics_estimate_generate_noise_translation_matchnoise = True   # True-> retraction will match- False-> normal distribution around -self.extrinsics_estimate_generate_noise_translation
        self.extrinsics_estimate_generate_noise_rotation = 10.0                 # std. in degrees
        self.extrinsics_estimate_generate_noise_rotation_matchnoise = True      # True-> retraction will match- False-> normal distribution around -self.extrinsics_estimate_generate_noise_rotation

        # Properties of the recieved person.msg 
        self.person_msg_keys_num = 17                       # number of included keypoints, specifying the available keypoint IDs in [0 .. person_msg_keys_num-1]
        self.person_msg_keys_ids = []                       # leave empty to include all contained keypoints, otherwise list the IDs from 0 to person_msg_keys_num - 1 to be used
        self.person_msg_keys_hips_shoulders = (5,6,11,12)   # IDs for shoulders and hips, which are required during data association
        self.person_msg_coords_undistort = True             # undistorts keypoint coordinates w.r.t. the distortion coefficients provided in the intrinsics file
        self.person_msg_scores_normalize = True             # scores between scores_min and scores_max will be normalized to [0,1]. Otheriwse scores will be clipped between 0 and 1
        self.person_msg_scores_min = 0.0                    # scores are expected to be larger or equal before normalization. This adaptively changes when finding smaller scores
        self.person_msg_scores_max = 1.0                    # scores are expected to be smaller or equal before normalization. This adaptively changes when finding larger scores
        self.person_msg_persons_max = 10                    # all detections within a camera with more person detections are rejected

        # -- Outputs -- #

        # Topics #

        self.publish_3D_flag = True                         # master control for all other publish_3D flags
        self.publish_3D_interval = 0.1                      # refresh interval in seconds

        self.publish_3D_landmarks_triangulation = True      # publishes MarkerArrays for the skeletons of all person hypotheses included in the latest optimization before optimization
        self.publish_3D_landmarks_optimization = True       # publishes MarkerArrays for the skeletons of all person hypotheses included in the latest optimization after optimization
        self.publish_3D_depth_estimation = True             # publishes MarkerArrays for the line-segments computed during depth estimation 
        self.publish_3D_marker_topic = '/keypoint_projections'

        self.publish_3D_camera_poses_result = True          # publishes the resulting/current set of camera poses
        self.publish_3D_camera_poses_reference = True       # publishes the reference set of camera poses
        self.publish_3D_camera_poses_initialization = True  # publishes the initial set of camera poses before calibration
        self.publish_3D_camera_poses_result_label = ""            
        self.publish_3D_camera_poses_initialization_label = "_ini"
        self.publish_3D_camera_poses_reference = "_ref"

        self.publish_3D_camera_poses_base_to_cam0 = [-3.44546785,  0.73389514,  2.61657272, 0.79924155, 0.37410778, 0.19631249, 0.42745495] # transformation from some base to cam_0
        self.publish_3D_base_to_map = [13.57584285736084, 13.884191513061523, 0.0, 0.0, 0.0, 0.999999972294683, -0.0002353946332988968]     # transformation from some base to the map
        self.publish_3D_base_name = "base"                  # name of the base frame
        self.publish_3D_map_name = "/map"                   # name of the map frame

        self.publish_2D = False                             # visualizes the received frame-sets and filtering constraints (only use for debugging, this drastically reduces throughput)
        self.publish_2D_interval = 1.0                      # refresh interval in seconds
        self.publish_2D_topic = '/filter_and_data_association'

        # Files #

        self.log_dir = "logs/"+str(datetime.datetime.now())[2:-7]   # location relative to the package path for storing files specified below

        self.log_extrinsics_result_flag = True                      # saves the resulting calibration
        self.log_extrinsics_result_file = "extrinsics_result.yaml"

        self.log_history_flag = True                                # logs all intermediate camera poses into a file
        self.log_history_file = "extrinsics_history.yaml"

        self.log_extrinsics_reference_file = True                   # copies reference file into the log folder
        self.log_extrinsics_estimate_file = True                    # copies estimate file into the log folder
        self.log_intrinsics_file = True                             # copies intirnsics file into the log folder

        self.log_terminal_flag = True                               # writes the entire terminal output into a file
        self.log_terminal_file = "log.txt"

        # -- Program Behavior -- # 

        # Terminal #

        self.print_calibration = False      # prints all intrinsic and extrinsic camera parameters before and after optimization
        self.print_error = True             # prints errors towards reference calibration before and after optimization
        self.print_error_delta = True       # prints the difference in error after optimization
        self.print_verbose = False          # various notifications and warnings
        
        self.print_status = True            # prints real-time status in the last row
        self.print_status_interval = 0.1    # update interval in seconds
        
        # Exit #

        self.autoexit_duration = 250.0      # automatically ends calibration after the specified time in seconds. Set 0.0 to deactive 
        
        ##############
        self.startup()

    def startup(self):

        print()
        self.check_parameters()

        np.random.seed(2)
        np.set_printoptions(suppress=True)

        # Data structures / Variables

        self.path = rospkg.RosPack().get_path('keypoint_camera_calibration')

        self.first_msg_received = False     # blocks optimization, termination, status_prints, etc, until first frame is recieved
        self.processing = False             # blocks shutdown while inside joints_callback or graph_callback
        self.shutdown = False               # blocks all callbacks once shutdown is initiated
        
        # Read files

        ids = []
        if self.extrinsics_reference_readfromfile:
            self.file_reader_extrinsics_reference()
            ids.append(self.found_IDs_reference)
        if self.extrinsics_estimate_readfromfile:
            self.file_reader_extrinsics_estimate()
            ids.append(self.found_IDs_estimate)
        else:
            self.file_writer_extrinsics_estimate(self.extrinsics_estimate_generate_noise_translation, self.extrinsics_estimate_generate_noise_rotation)  
        self.file_reader_intrinsics()
        ids.append(self.found_IDs_intrinsics)
        for i in range(len(ids)-1):
            if ids[i] != ids[i+1]:
                print("Files do not match!")
                os._exit(1)
        self.cams = ids[0]
        self.cams_num = len(self.cams)
        assert self.cams_num > 1 , "A minimum of 2 cameras is required for calibration!"

        # Publisher

        if self.publish_3D_depth_estimation:
            self.publisher_3D_depth_estimation_prepare()
        if self.publish_3D_landmarks_triangulation or self.publish_3D_landmarks_optimization or self.publish_3D_depth_estimation:
            if self.publish_3D_landmarks_triangulation or self.publish_3D_landmarks_optimization:
                self.publisher_3D_landmarks_prepare()
            self.marker_array_pub = rospy.Publisher(self.publish_3D_marker_topic,MarkerArray,queue_size=10)
        if self.publish_3D_flag and (self.publish_3D_camera_poses_result or self.publish_3D_camera_poses_reference or self.publish_3D_camera_poses_initialization):
            self.tf_broadcaster = tf2_ros.TransformBroadcaster()
            self.publisher_3D_update = None
            rospy.Timer(rospy.Duration(self.publish_3D_interval), self.publisher_3D_callback)

        if self.publish_2D:
            self.publisher_2D_prepare()  
            self.img_pub = rospy.Publisher(self.publish_2D_topic, ROS_Image, queue_size=1)
        
        # Subscriber
    
        self.subs_joints = []
        for i in range(self.cams_num):
            self.subs_joints.append(message_filters.Subscriber(self.extrinsics_estimate[i]["topic"], Person2DList, queue_size=1))
        self.ts = message_filters.ApproximateTimeSynchronizer(self.subs_joints, queue_size=self.sync_queue, slop=self.sync_slop) 
        self.ts.registerCallback(self.joints_callback)

        # Prepare and start optimization

        self.graph_initialize()

        if self.print_status:
            self.status_length_max = 0
            rospy.Timer(rospy.Duration(self.print_status_interval), self.printer_status)

        self.print("\nWaiting for incoming detections...",stamp=True,flush=True)
        
        if self.optimization_flag:
            rospy.Timer(rospy.Duration(self.optimization_interval), self.graph_callback)
        
    def check_parameters(self):

        assert self.sync_min_gap >= 0.0
        assert self.filter_stamps_skip or (not self.filter_stamps_skip and self.filter_stamps_span_max >= 0.0)
        assert self.filter_stamps_skip or (not self.filter_stamps_skip and self.filter_stamps_std_max >= 0.0)
        assert self.filter_stamps_skip or (not self.filter_stamps_skip and self.filter_scores_min >= 0.0)
        assert self.filter_stamps_skip or (not self.filter_stamps_skip and self.filter_bbox_minimum_distance >= 0)
        assert self.person_size_min > 0.0
        assert self.person_size_max >= self.person_size_min
        assert self.person_size_ratio_hips >= 0.0
        assert self.person_size_ratio_shoulders >= 0.0
        assert self.person_size_ratio_torso >= 0.0
        assert self.data_association_views_min >= 2 , "Cannot triangulate from less than 2 perspectives on a single joint!"
        assert self.data_association_threshold >= 0.0
        assert self.hyp_sel_random or (not self.hyp_sel_random and self.hyp_sel_spacing >= 0.0)
        assert self.hyp_sel_random or (not self.hyp_sel_random and self.hyp_sel_timing >= 0.0 and self.hyp_sel_timing <= 1.0)
        assert self.optimization_interval >= 0.0
        assert self.optimization_scope_size > 0
        assert self.optimization_wait_mult >= 1.0
        assert self.optimization_delta > 0.0
        assert self.kal_init_translation >= 0.0
        assert self.kal_init_rotation >= 0.0
        assert self.kal_process_translation >= 0.0
        assert self.kal_process_rotation >= 0.0

        assert self.person_msg_keys_num > 0
        assert len(self.person_msg_keys_hips_shoulders) == 4
        if len(self.person_msg_keys_ids) == 0:
            self.person_msg_keys_ids = np.arange(self.person_msg_keys_num,dtype=np.uint8)
        self.person_msg_keys_ids = np.sort(np.unique(np.append(self.person_msg_keys_ids,self.person_msg_keys_hips_shoulders).astype(np.uint8)))        
        assert np.min(self.person_msg_keys_ids) >= 0 and np.max(self.person_msg_keys_ids) < self.person_msg_keys_num , "Specified keypoint IDs must be in [0.."+str(self.person_msg_keys_num)+"]"
        assert self.person_msg_scores_min <= self.person_msg_scores_max
        assert self.person_msg_persons_max > 0
        assert len(self.person_msg_keys_hips_shoulders) == 4
        
        assert self.intrinsics_file != ""
        assert not self.extrinsics_estimate_readfromfile or (self.extrinsics_estimate_readfromfile and self.extrinsics_estimate_file != "")
        assert not self.extrinsics_reference_readfromfile or (self.extrinsics_reference_readfromfile and self.extrinsics_reference_file != "")
        assert self.extrinsics_estimate_readfromfile or self.extrinsics_reference_readfromfile , "Choose at least one!"
        assert (not self.extrinsics_estimate_readfromfile and self.extrinsics_estimate_generate) or (self.extrinsics_estimate_readfromfile and not self.extrinsics_estimate_generate) 

        assert not self.extrinsics_estimate_generate or (self.extrinsics_estimate_generate and self.extrinsics_reference_readfromfile)
        assert not self.extrinsics_estimate_generate or (self.extrinsics_estimate_generate and self.extrinsics_estimate_generate_noise_translation >= 0.0)
        assert not self.extrinsics_estimate_generate or (self.extrinsics_estimate_generate and self.extrinsics_estimate_generate_noise_rotation >= 0.0)

        assert len(self.publish_3D_camera_poses_base_to_cam0) == 7
        assert np.isclose(np.linalg.norm(self.publish_3D_camera_poses_base_to_cam0[3:]), 1.0) , "Rotation component should be a unit quternion" 
        assert len(self.publish_3D_base_to_map) == 7
        assert np.isclose(np.linalg.norm(self.publish_3D_base_to_map[3:]),1.0) , "Rotation component should be a unit quternion" 

        assert not self.publish_2D or (self.publish_2D and self.publish_2D_interval >= 0.0)

        assert not self.log_extrinsics_result_flag or (self.log_extrinsics_result_flag and self.log_extrinsics_result_file != "")
        assert not self.log_history_flag or (self.log_history_flag and self.log_history_file != "")
        assert not self.log_terminal_flag or (self.log_terminal_flag and self.log_terminal_file != "")
        assert not self.log_extrinsics_reference_file or (self.log_extrinsics_reference_file and self.extrinsics_reference_readfromfile)

        assert not self.print_status or (self.print_status and self.print_status_interval >= 0.0)
        assert self.autoexit_duration >= 0.0

    # Graph
    def graph_initialize(self):
        
        self.print("Initializing graph...",stamp=True,flush=True)

        # Data structures & variables

        self.extrinsics_history = []
        
        self.next_landmark_id = 0
        self.frame_success = []
        self.cams_constrained = []
        
        self.covariance_measurements = []
        self.covariance_predictions = []

        prediction_row = []
        for c in range(self.cams_num):
            prediction_row.append(np.eye(6))
        self.covariance_predictions.append(prediction_row)

        self.means_measurements = []
        self.means_predictions = []

        prediction_row = []
        for c in range(self.cams_num):
            prediction_row.append(np.concatenate((R.from_quat(self.extrinsics_estimate[c]["rotation"]).as_euler("xyz",degrees=False),self.extrinsics_estimate[c]["translation"])))
        self.means_predictions.append(prediction_row)

        self.noise = []

        # initialize noise
        noise_row = []

        noise_zero = np.zeros(shape=(6,6)) 
        noise_row.append(noise_zero)

        noise_default = np.zeros(shape=(6,6)) 
        noise_default[range(0,3), range(0,3)] = np.deg2rad(self.kal_init_rotation) ** 2
        noise_default[range(3,6), range(3,6)] = self.kal_init_translation ** 2
        for i in range(1,self.cams_num):
            noise_row.append(noise_default)    

        self.noise.append(noise_row)

        self.uncertainty_process = np.zeros((6,6)) # process noise
        self.uncertainty_process[range(0,3), range(0,3)] = np.deg2rad(self.kal_process_rotation) ** 2
        self.uncertainty_process[range(3,6), range(3,6)] = self.kal_process_translation ** 2
        
        self.frame_detections = []
        self.projections = []
        self.landmarks_projected = []
        self.landmark_history = []                    
        self.times_optimization = []
        self.analyzed_frames = []
        self.cam_counters = np.zeros(self.cams_num,dtype=int)
        self.cam_counters_history = []

        self.graph_compute_initposes()
        self.L = gtsam.symbol_shorthand.L
        self.X = gtsam.symbol_shorthand.X   
        self.graph = gtsam.NonlinearFactorGraph()
        self.estimate = gtsam.Values()
        self.graph_estimate_cameras()
        self.extrinsics_update(self.estimate)
        self.print("done!")
        if self.print_calibration:
            self.printer_extrinsics_raw()
        if self.print_error and self.extrinsics_reference_readfromfile:
            self.printer_extrinsics_error()

    def graph_compute_initposes(self):
       
        self.graph_init_poses = [gtsam.Pose3()]
        for i in range(1,self.cams_num):
            rotation = gtsam.Rot3(R.from_quat(self.extrinsics_estimate[i]["rotation"]).as_matrix())
            pose = gtsam.Pose3(rotation,self.extrinsics_estimate[i]["translation"])
            self.graph_init_poses.append(pose)

    def graph_add_factors_cameras(self):
        
        #Fix cam_0 to zero-pose
        factor = gtsam.NonlinearEqualityPose3(self.X(0), gtsam.Pose3())
        self.graph.push_back(factor)

        # Add prior factors to graph
        for i in range(1,self.cams_num):            
            noise = gtsam.noiseModel.Gaussian.Covariance(self.noise[-1][i])
            rotation = gtsam.Rot3(R.from_euler("xyz",self.means_predictions[-1][i][0:3],degrees=False).as_matrix())
            pose = gtsam.Pose3(rotation,self.means_predictions[-1][i][3:6])
            factor = gtsam.PriorFactorPose3(self.X(i), pose, noise)
            self.graph.push_back(factor)
        
    def graph_add_factors_projections(self):

        cams_constrained = []
        cam_counters = np.zeros(self.cams_num,dtype=int)

        start = 0
        if len(self.projections) > self.optimization_scope_size:
            start = len(self.projections) - self.optimization_scope_size

        for n in range(start,len(self.projections)):
            for i in range(0,len(self.projections[n])):

                if self.projections[n][i][3] in self.landmarks_projected[n-start]["landmark_ids"]:
                    factor = gtsam.GenericProjectionFactorCal3_S2(
                        self.projections[n][i][0], self.projections[n][i][1], self.X(self.projections[n][i][2]), self.L(self.projections[n][i][3]), self.intrinsics_cal3s2[self.projections[n][i][2]])
                    self.graph.push_back(factor)
                    cam_counters[self.projections[n][i][2]] += 1
                    cams_constrained.append(self.projections[n][i][2])
        
        cams_constrained = np.unique(np.asarray(cams_constrained))
        self.cams_constrained.append(cams_constrained)
        self.cam_counters_history.append(cam_counters)
        self.cam_counters += cam_counters

    def graph_add_factors_landmarks(self):
        return

    def graph_estimate_cameras(self):

        for i in range(self.cams_num):
            if len(self.frame_success) == 0:
                pose = self.graph_init_poses[i]
            else:
                rotation = gtsam.Rot3(R.from_euler("xyz",self.means_predictions[-1][i][0:3],degrees=False).as_matrix())
                pose = gtsam.Pose3(rotation,self.means_predictions[-1][i][3:6])
            self.estimate.insert(self.X(i), pose)

    def graph_estimate_landmarks(self):

        landmarks = self.landmarks_projected

        start = 0
        if len(self.projections) > self.optimization_scope_size:
            start = len(self.projections)-self.optimization_scope_size

        for n in range(start,len(self.projections)):
            for p in range(0,len(self.frame_detections[n])):
                for k in range(0,len(self.frame_detections[n][p])):
                    if self.frame_detections[n][p][k][1] in landmarks[n-start]["landmark_ids"]:

                        self.estimate.insert(self.L(self.frame_detections[n][p][k][1]),
                        gtsam.Point3(landmarks[n-start]["points"][self.frame_detections[n][p][k][0]][0], 
                            landmarks[n-start]["points"][self.frame_detections[n][p][k][0]][1], 
                            landmarks[n-start]["points"][self.frame_detections[n][p][k][0]][2]))

    def graph_optimize(self):

        params = gtsam.DoglegParams()
        params.setDeltaInitial(self.optimization_delta)
        #params.setVerbosityDL("VERBOSE")
        optimizer = gtsam.DoglegOptimizer(self.graph, self.estimate, params)
        
        timestamp = rospy.get_time()
        try:    
            result = optimizer.optimize()
            success = True
        except Exception as e:
            result = self.estimate    
            success = False
        self.times_optimization.append(rospy.get_time() - timestamp)
        
        self.frame_success.append(success)     

        if self.frame_success[-1]:
            means_row = []
            covariance_row = []
            try:
                i = -1
                marginals = gtsam.Marginals(self.graph,result)     
                for i in range(self.cams_num):
                    pose = gtsam.Pose3(result.atPose3(self.X(i)).rotation(),result.atPose3(self.X(i)).translation())
                    position = pose.translation()
                    orientation = R.from_matrix(pose.rotation().matrix()).as_euler('xyz',degrees=False)  
                    means_row.append(np.concatenate((orientation,position)))
                    covariance_row.append(marginals.marginalCovariance(self.X(i)))
            except Exception as e:
                if self.print_verbose:
                    self.print("\tError during marginalization! - i="+str(i)+" - "+str(e))
                self.frame_success[-1] = False
                
        if self.frame_success[-1]:
            self.means_measurements.append(means_row)
            self.covariance_measurements.append(covariance_row)
        
        return result
    
    def graph_noise_update_kalman(self,result):
        
        if self.frame_success[-1] and np.sum(self.frame_success)>1:

            prediction_covariance_row = []
            prediction_covariance_row.append(np.eye(6))

            prediction_means_row = []
            prediction_means_row.append(np.zeros(6))

            for c in range(1,self.cams_num):    
                if c in self.cams_constrained[-1]:

                    P = self.covariance_predictions[-1][c] + self.uncertainty_process
                    K = P @ np.linalg.inv( P + self.covariance_measurements[-1][c] )
                    
                    means_prev = self.means_predictions[-1][c]
                    
                    diff = self.means_measurements[-1][c] - means_prev
                    diff[0:3] = np.mod(diff[0:3]+np.pi,2*np.pi)-np.pi
                    
                    means = means_prev + K @ (diff)
                    means[0:3] = np.mod(means[0:3]+np.pi,2*np.pi)-np.pi
                    
                    cov = (np.eye(6) - K) @ P
                    
                    pose = gtsam.Pose3(gtsam.Rot3(R.from_euler('xyz',means[0:3],degrees=False).as_matrix()),means[3:])
                    result.update(self.X(c),pose)

                else:

                    cov = self.covariance_predictions[-1][c]
                    means = self.means_predictions[-1][c]

                prediction_covariance_row.append(cov)
                prediction_means_row.append(means)   
        else:

            prediction_covariance_row = self.covariance_predictions[-1]
            prediction_means_row = self.means_predictions[-1]

        self.covariance_predictions.append(prediction_covariance_row)
        self.means_predictions.append(prediction_means_row)

        return result

    def graph_triangulate(self,proj_matricies,points,confidences=None):

        n_views = len(proj_matricies)

        if confidences is None:
            confidences = np.ones(n_views, dtype=float)

        A = np.zeros((2 * n_views, 4))
        for j in range(len(proj_matricies)):
            A[j * 2 + 0] = points[j][0] * proj_matricies[j][2, :] - proj_matricies[j][0, :]
            A[j * 2 + 0] = A[j * 2 + 0] / np.linalg.norm(A[j * 2 + 0]) # for numerical stability
            A[j * 2 + 0] = A[j * 2 + 0] * confidences[j]
            
            A[j * 2 + 1] = points[j][1] * proj_matricies[j][2, :] - proj_matricies[j][1, :]
            A[j * 2 + 1] = A[j * 2 + 1] / np.linalg.norm(A[j * 2 + 1]) # for numerical stability
            A[j * 2 + 1] = A[j * 2 + 1] * confidences[j]

        u, s, vh =  np.linalg.svd(A, full_matrices=False)
        point_3d_homo = vh[3, :]

        point_3d = (point_3d_homo.T[:-1] / point_3d_homo.T[-1]).T

        return point_3d

    def graph_compute_landmarks(self):

        if len(self.frame_detections) >= self.optimization_scope_size:
            l = self.optimization_scope_size
        else:
            l = len(self.frame_detections)

        landmarks_projected = []
        projection_matricies = [None] * self.cams_num

        for n in range(0,l):

            # find corresponding detections that are observed from at least n different cameras
            persons = 0
            shared_detections = []
            for p in range(0,len(self.frame_detections[-l+n])):
                for d in range(0,len(self.frame_detections[-l+n][p])):
                    if len(self.frame_detections[-l+n][p][d][2]) >= self.data_association_views_min:
                        shared_detections.append((self.frame_detections[-l+n][p][d][2],p,self.frame_detections[-l+n][p][d][0],self.frame_detections[-l+n][p][d][1],self.frame_detections[-l+n][p][d][3]))
                        if persons <= p: persons += 1
                    else:
                        print("This should not happen")
            if not len(shared_detections) > 0:
                print("No shared detections, shouldnt happen")

            # find all cameras that have at least one shared detections
            involved_cams = []
            for i in range(len(shared_detections)):
                for j in range(len(shared_detections[i][0])):
                    if shared_detections[i][0][j] not in involved_cams:
                        involved_cams.append(shared_detections[i][0][j])
                        if len(involved_cams) == self.cams_num:
                            break
                if len(involved_cams) == self.cams_num:
                    break

            # compute necessary 3x4 projection matricies w.r.t. current extrinsics
            for i in range(0,len(involved_cams)):
                if projection_matricies[involved_cams[i]] is None:
                    if len(self.extrinsics_history) > 0:
                        RT_inv = np.column_stack((R.from_euler("xyz",self.means_predictions[-1][involved_cams[i]][0:3],degrees=False).as_matrix(),self.means_predictions[-1][involved_cams[i]][3:6]))
                        RT_inv = np.row_stack((RT_inv,np.array((0.0,0.0,0.0,1.0))))
                        RT_inv = np.linalg.inv(RT_inv)
                        RT_inv = np.delete(RT_inv,3,0)
                    else:
                        RT_inv = np.delete(np.linalg.inv(self.graph_init_poses[involved_cams[i]].matrix()),3,0)
                    P = self.intrinsics_cal3s2[involved_cams[i]].K() @ RT_inv
                    projection_matricies[involved_cams[i]] = P

            # pack all shared_detection projections/points and triangulate corresponding 3D points
            point3s = []
            for i in range(0,len(shared_detections)):
                keypoint = shared_detections[i][2]
                proj_matricies = []
                point2s = []
                confidences = []
                for c in range(0,len(shared_detections[i][0])):

                    # find corresponding projection
                    projection = None
                    for j in range(0,len(self.projections[-l+n])):
                        if self.projections[-l+n][j][2] == shared_detections[i][0][c] and self.projections[-l+n][j][5] == shared_detections[i][1] and self.projections[-l+n][j][4] == shared_detections[i][2]:  #cam, person, keypoint
                            projection = self.projections[-l+n][j]
                            break
                    assert projection != None

                    proj_matricies.append(projection_matricies[shared_detections[i][0][c]])
                    point2s.append(projection[0])
                    confidences.append(shared_detections[i][4][c])
                    
                proj_matricies = np.array(proj_matricies)
                point2s = np.array(point2s)
                point3 = self.graph_triangulate(proj_matricies,point2s,confidences)
                point3s.append((shared_detections[i][1],shared_detections[i][2],point3,shared_detections[i][3])) # person, keypoint_id, 3D-point, landmark_id 
            
            # format for output
            stamps = [None] * len(self.person_msg_keys_ids) * persons
            points = np.full((len(self.person_msg_keys_ids) * persons,3),np.inf)
            landmarks_ids = [None] * len(self.person_msg_keys_ids) * persons
            for i in range(len(point3s)):
                stamps[point3s[i][1]*(point3s[i][0]+1)] = rospy.Time.now()
                points[point3s[i][1]*(point3s[i][0]+1)] = point3s[i][2]
                landmarks_ids[point3s[i][1]*(point3s[i][0]+1)] = point3s[i][3]
            result = {"timestamps":stamps, "persons":persons, "points":points, "landmark_ids":landmarks_ids}
            landmarks_projected.append(result)

        self.landmarks_projected = landmarks_projected

    def graph_select_hypotheses(self):
        
        # Select a set of frames
        
        if self.hyp_sel_random:
            return np.random.choice(len(self.queue_hyps), self.optimization_scope_size, replace = False)
        else:
            
            filter_queue_len = len(self.queue_hyps)
            p_linear = np.arange(1,filter_queue_len+1,1)
            p_linear = p_linear / np.sum(p_linear)
            p_uniform = np.full(filter_queue_len,1.0/filter_queue_len)
            p = self.hyp_sel_timing * p_linear + (1-self.hyp_sel_timing) * p_uniform
            rng = np.random.default_rng()
            ranking = rng.choice(filter_queue_len, filter_queue_len, replace = False, p=p)

            props = np.asarray(self.queue_props)
            d0  = props[:,0] # valid detec
            d1  = props[:,1] # center of mass x
            d2  = props[:,2] # center of mass y 
            d3  = props[:,3] # center of mass z
            d4  = props[:,4] # mean distance between center of mass and camera poses
            
            zeta = np.stack((d1,d2,d3),axis=1)
            selection = [ranking[0]]
            
            i = 1
            while len(selection) < self.optimization_scope_size and i < filter_queue_len:
                tooclose = False
                for j in range(len(selection)):
                    if np.linalg.norm(zeta[ranking[i]]-zeta[selection[j]]) < self.hyp_sel_spacing:
                        tooclose = True
                        break
                if not tooclose:
                    selection.append(ranking[i])
                i += 1
            
            i = 1
            while len(selection) < self.optimization_scope_size:
                if i == 1 and self.print_verbose:
                    self.print("\tSpacing constraint violated during hypothesis selection after having selected "+str(len(selection))+"/"+str(self.optimization_scope_size)+" hypotheses.")
                if not ranking[i] in selection:
                    selection.append(ranking[i])
                i += 1
                
            selection = np.asarray(selection)
            
            return selection

    def graph_update_hypotheses_properties(self,selection):
        self.lock.acquire()
        for i in range(len(selection)):
            self.queue_props[selection[i]][12] += 1
            if not self.frame_success[-1]:
                self.queue_props[selection[i]][13] += 1  
        self.lock.release()

    def graph_analyze_hypotheses(self,selection):
        
        # frame_detections: k l_id [c] [scores]
        # projections: point noise c landmark_id k_index person_id_pos

        # hyp  0 1 2     3 4 5   6   7   8     9 
        #      c k score x y cov cov cov stamp depth

        for h in selection:
            frame_detections = []
            projections = []
            frame_detections.append([])
            appended_k = []
            
            for i in range(len(self.queue_hyps[h])):
                
                c = int(self.queue_hyps[h][i][0])
                k = int(self.queue_hyps[h][i][1])
                s = self.queue_hyps[h][i][2]
                
                if k in appended_k:
                    
                    landmark_id = -1
                    for m in range(0,len(frame_detections[0])):
                        if frame_detections[0][m][0] == k:
                            landmark_id = frame_detections[0][m][1]
                
                else:

                    all_k = np.where(self.queue_hyps[h][:,1]==self.queue_hyps[h][i][1])[0].astype(int)
                    cams = self.queue_hyps[h][all_k,0].astype(int)
                    scores = self.queue_hyps[h][all_k,2]
                    landmark_id = self.next_landmark_id

                    frame_detections[0].append([k, landmark_id, cams, scores])
                    appended_k.append(k)
                    self.next_landmark_id += 1

                noise = gtsam.noiseModel.Gaussian.Covariance([
                    [self.queue_hyps[h][i][5], self.queue_hyps[h][i][6]], 
                    [self.queue_hyps[h][i][6], self.queue_hyps[h][i][7]]])            
                point = (self.queue_hyps[h][i][3],self.queue_hyps[h][i][4])
                projections.append((point,noise,c,landmark_id,k,0))

            assert len(self.frame_detections) == len(self.projections)
            self.analyzed_frames.append((h,len(self.frame_detections)))
            self.frame_detections.append(frame_detections)
            self.projections.append(projections)

    # Joints
    def joints_check_reference(self):

        # Check if intrinsics exist
        if len(self.intrinsics_cal3s2) < len(self.msg):
            if self.print_verbose:
                self.print("\tSkipping image frame! - Waiting for camera intrinsics...")
            self.joints_finisher(reason="Waiting for intrinsics")
            return False

        # Check if reference extrinsics exists
        if self.extrinsics_reference_readfromfile and len(self.extrinsics_reference) < len(self.msg):
            if self.print_verbose:
                self.print("\tSkipping image frame! - Waiting for reference extrinsics...")
            self.joints_finisher(reason="Waiting for reference extrinsics")
            return False

        return True

    def joints_filter_detections(self):

        # remove cams without- or more than expected detections
        self.filterstage = 0
        detections = np.zeros(self.cams_num,dtype=np.uint8)
        for c in self.cams_used:
            detections[c] = len(self.msg[c].persons)
        if np.sum(detections) == 0:
            self.joints_finisher(reason = "No detections")
            return False
        cams_no_detections = np.where(detections == 0)[0]
        if self.remove_cams(cams_no_detections, reason = "No detections") == -1:
            return False
        cams_too_many_detections = np.where(np.delete(detections,cams_no_detections) > self.person_msg_persons_max)[0]
        if cams_too_many_detections.shape[0] > 0:
            if self.print_verbose:
                self.print("\tWarning! - Exceeding 'max_number_of_persons_per_cam' in cam "+str(np.take(self.cams,np.take(self.cams_used,cams_too_many_detections)))
                    +". Should be <="+str(self.person_msg_persons_max)+" but is "+str(np.take(detections,np.take(self.cams_used,cams_too_many_detections)))+". Cam was removed from frame.")
            if self.remove_cams(cams_too_many_detections, reason = "Too many detections (>"+str(self.person_msg_persons_max)+")") == -1:
                return False

        return True

    def joints_filter_bbox(self):

        self.filterstage = 4

        for c in copy.deepcopy(self.cams_used):

            removal = True
            while removal == True:
                removal = False

                persons = len(self.msg[c].persons)

                i = 0
                j = 1

                while j < persons:

                    r1 = self.msg[c].persons[i].bbox
                    r2 = self.msg[c].persons[j].bbox
                    d, a1, a2 = self.distance_rec2rec(r1,r2)
               
                    if d <= self.filter_bbox_minimum_distance:

                        if self.remove_person(c,j) == -1:
                            return False
                        if self.remove_person(c,i) == -1:
                            return False
                        
                        removal = True
                        j = persons

                    else:

                        if j+1 < persons:
                            j = j+1
                        else:
                            i = i+1
                            j = i+1

        return True

    def joints_filter_timing(self):

        # remove cams with insufficient timing
        if self.filter_stamps_skip == False:
            self.filterstage = 1
            stamps_secs = []
            stamps_nsecs = []
            for c in self.cams_used:
                stamps_secs.append(self.msg[c].header.stamp.secs)
                stamps_nsecs.append(self.msg[c].header.stamp.nsecs)
            stamps = np.asarray(stamps_secs) * 1000000000 + np.asarray(stamps_nsecs) 
            if np.amax(stamps) - np.amin(stamps) > self.filter_stamps_span_max:
                self.joints_finisher(reason = "filter_stamps_span_max")
                return False
            stamps_std = np.std(stamps)
            if stamps_std > self.filter_stamps_std_max:
                self.joints_finisher(reason = "filter_stamps_std_max")
                return False

        return True

    def joints_filter(self):

        # remove individual keypoints
        self.filterstage = 2
        for c in copy.deepcopy(self.cams_used):
            
            p_offset = 0
            for p in range(0,len(self.msg[c].persons)):
                
                p = p + p_offset # compensate indexing after persons were removed from list

                # remove unused keypoints
                for k in self.person_msg_keys_ids_unused:
                    self.msg[c].persons[p].keypoints[k].score = 0
                    self.msg[c].persons[p].keypoints[k].x = 0.0
                    self.msg[c].persons[p].keypoints[k].y = 0.0
                    self.msg[c].persons[p].keypoints[k].cov = (0.0,0.0,0.0)
                
                # read data
                data = np.asarray([np.array([self.msg[c].persons[p].keypoints[k].x, self.msg[c].persons[p].keypoints[k].y, self.msg[c].persons[p].keypoints[k].score]) for k in self.person_msg_keys_ids])
                
                scores = data[:,2]
                xs = data[:,0]
                ys = data[:,1]

                # remove keypoints outside of the image
                xs_out_of_image = np.where(np.logical_or(xs < 0, xs >= self.extrinsics_estimate[c]["resolution"][0]))[0]
                if xs_out_of_image.shape[0] > 0 and self.print_verbose:
                    self.print("\tEncoutered detection"+("s" if xs_out_of_image.shape[0] > 1 else " ")
                        +" with x-coordinate"+("s" if xs_out_of_image.shape[0] > 1 else " ")+" exceeding the image width  of "
                        +str(self.extrinsics_estimate[c]["resolution"][0])+": "+np.array2string(np.take(xs,xs_out_of_image)).replace('\n', '')
                        + " ; Detection"+("s are" if xs_out_of_image.shape[0] > 1 else " is")+" rejected!", stamp=True)
                
                ys_out_of_image = np.where(np.logical_or(ys < 0, ys >= self.extrinsics_estimate[c]["resolution"][1]))[0]
                if ys_out_of_image.shape[0] > 0 and self.print_verbose:
                    self.print("\tEncoutered detection"+("s" if ys_out_of_image.shape[0] > 1 else " ")
                        +" with y-coordinate"+("s" if ys_out_of_image.shape[0] > 1 else " ")+" exceeding the image height of "
                        +str(self.extrinsics_estimate[c]["resolution"][1])+": "+np.array2string(np.take(ys,ys_out_of_image)).replace('\n', '')
                        + " ; Detection"+("s are" if ys_out_of_image.shape[0] > 1 else " is")+" rejected!", stamp=True)
                outliers = np.unique(np.concatenate((xs_out_of_image,ys_out_of_image)))
                if np.any(np.isin(outliers,self.filter_scores_musthahaveids_relative,assume_unique=True)):
                    if self.remove_person(c,p) == -1: # will report in 2D pub as "rejecting while checking scores"
                        return False
                    p_offset -= 1
                    continue

                # normalize or clip scores
                normalized = False
                if self.person_msg_scores_normalize:

                    value = np.max(scores)
                    if value > self.person_msg_scores_max:
                        if self.print_verbose:
                            self.print("\tEncoutered higher score than expected: "
                                +self.format_num(value,1,3)+" > "+self.format_num(self.person_msg_scores_max,1,3)
                                +" ; Scores are normalized accordingly from now on!", stamp=True) 
                        self.person_msg_scores_max = value # old frames will be ranked too high in frame selection
                        
                    value = np.min(scores)
                    if value < self.person_msg_scores_min:
                        if self.print_verbose:
                            self.self.format_num("\tEncoutered smaller score than expected: "
                                +self.format_num(value,1,3)+" < "+self.format_num(self.person_msg_scores_min,1,3)
                                +" ; Scores are normalized accordingly from now on!", stamp=True) 
                        self.person_msg_scores_min = value # old frames will be ranked too low in frame selection
                        
                    if self.person_msg_scores_min != 0.0 or self.person_msg_scores_max != 1.0:
                        normalized = True
                        scores = self.person_msg_scores_min + scores / (self.person_msg_scores_max-self.person_msg_scores_min)
                        for k in range(len(self.msg[c].persons[p].keypoints)):
                            self.msg[c].persons[p].keypoints[k].score = self.person_msg_scores_min + self.msg[c].persons[p].keypoints[k].score / (self.person_msg_scores_max-self.person_msg_scores_min)
                else:

                    clip = False
                    value = np.max(scores)
                    if value > 1.0:
                        if self.print_verbose:
                            self.print("\tEncoutered higher score than expected, "
                                +self.format_num(value,1,3)+" > 1.0"
                                +" : Scores are clipped at 1.0!", stamp=True)
                        clip = True

                    value = np.min(scores)
                    if value < 0.0:
                        if self.print_verbose:
                            self.self.format_num("\tEncoutered smaller score than expected, "
                                +self.format_num(value,1,3)+" < 0.0"
                                +" : Scores are clipped at 0.0!", stamp=True)
                        clip = True

                    if clip:
                        normalized = True
                        scores = np.clip(scores,0.0,1.0)
                        for k in range(len(self.msg[c].persons[p].keypoints)):
                            self.msg[c].persons[p].keypoints[k].score = np.clip(self.msg[c].persons[p].keypoints[k].score,0.0,1.0)

                # remove keypoints with low scores
                scores_to_low = np.where(scores < self.filter_scores_min)[0]
                
                outliers = np.unique(np.concatenate((outliers,scores_to_low)))
                if np.any(np.isin(outliers,self.filter_scores_musthahaveids_relative,assume_unique=True)):
                    if self.remove_person(c,p) == -1:
                        return False
                    p_offset -= 1
                    continue

                # remove person with high standard deviation
                scores_filter_scope = np.delete(scores,outliers)
                scores_std = np.std(scores_filter_scope)
                
                # remove all outliers from msg
                for k in outliers:
                    self.msg[c].persons[p].keypoints[k].score = 0
                    self.msg[c].persons[p].keypoints[k].x = 0.0
                    self.msg[c].persons[p].keypoints[k].y = 0.0
                    self.msg[c].persons[p].keypoints[k].cov = (0.0,0.0,0.0)
                    
                # update person info in msg
                if normalized or outliers.shape[0] > 0 or self.person_msg_keys_ids_unused.shape[0] > 0:

                    # average score 
                    scores_mean = np.mean(scores_filter_scope)
                    self.msg[c].persons[p].score = scores_mean

                    # bounding box
                    if outliers.shape[0] > 0 or self.person_msg_keys_ids_unused.shape[0] > 0:
                        xs_filter_scope = np.delete(xs,outliers)
                        x0 = np.min(xs_filter_scope)
                        x1 = np.max(xs_filter_scope)
                        ys_filter_scope = np.delete(ys,outliers)
                        y0 = np.min(ys_filter_scope)
                        y1 = np.max(ys_filter_scope)
                        self.msg[c].persons[p].bbox = (x0,y0,x1,y1)

        return True   

    def joints_undistort(self):

        for c in self.cams_used:
            if len(self.intrinsics_caminfomsg[c].D) > 0:
                K = np.asarray(self.intrinsics_caminfomsg[c].K).reshape(3,3)
                for p in range(0,len(self.msg[c].persons)):
                    for k in self.person_msg_keys_ids:
                        if self.msg[c].persons[p].keypoints[k].score != 0.0:
                            
                            P = np.array([[[self.msg[c].persons[p].keypoints[k].x,self.msg[c].persons[p].keypoints[k].y]]])
                            P_undist = cv2.undistortPoints(src=P, cameraMatrix=K, distCoeffs=np.asarray(self.intrinsics_caminfomsg[c].D),P=K)[0][0]
                            
                            self.msg[c].persons[p].keypoints[k].x = P_undist[0]
                            self.msg[c].persons[p].keypoints[k].y = P_undist[1]

    def joints_depth_estimation(self):

        depths = []
        rays = [[[None for k in range(self.person_msg_keys_num)] for p in range(len(self.msg[c].persons))] for c in range(self.cams_num)]
        line_segments = [[[None for k in range(self.person_msg_keys_num)] for p in range(len(self.msg[c].persons))] for c in range(self.cams_num)] 
        
        for c in self.cams_used:

            K = np.asarray(self.intrinsics_caminfomsg[c].K).reshape(3,3)
            cx = K[0][2]
            cy = K[1][2]
            fx = K[0][0]
            fy = K[1][1]
            F = (fx+fy)/2

            for p in range(len(self.msg[c].persons)):

                depth = []
                for size in (self.person_size_min,self.person_size_max):

                    dist_hips_px = np.linalg.norm(np.array([self.msg[c].persons[p].keypoints[11].x - self.msg[c].persons[p].keypoints[12].x,
                                                            self.msg[c].persons[p].keypoints[11].y - self.msg[c].persons[p].keypoints[12].y]))

                    dist_shoulders_px = np.linalg.norm(np.array([self.msg[c].persons[p].keypoints[5].x - self.msg[c].persons[p].keypoints[6].x,
                                                                 self.msg[c].persons[p].keypoints[5].y - self.msg[c].persons[p].keypoints[6].y]))

                    hips_mid = np.array([(self.msg[c].persons[p].keypoints[11].x + self.msg[c].persons[p].keypoints[12].x) / 2,
                                         (self.msg[c].persons[p].keypoints[11].y + self.msg[c].persons[p].keypoints[12].y) / 2])

                    shoulders_mid = np.array([(self.msg[c].persons[p].keypoints[5].x + self.msg[c].persons[p].keypoints[6].x) / 2,
                                              (self.msg[c].persons[p].keypoints[5].y + self.msg[c].persons[p].keypoints[6].y) / 2])

                    dist_torso_px = np.linalg.norm(shoulders_mid-hips_mid)

                    dist_hips_m = size * self.person_size_ratio_hips
                    dist_shoulders_m = size * self.person_size_ratio_shoulders
                    dist_torso_m = size * self.person_size_ratio_torso

                    if size == self.person_size_min:

                        if dist_hips_px > 0.0:
                            depth_hips = F * dist_hips_m / dist_hips_px
                        else:
                            depth_hips = np.inf

                        if dist_shoulders_px > 0.0:
                            depth_shoulders = F * dist_shoulders_m / dist_shoulders_px
                        else:
                            depth_shoulders = np.inf

                        if dist_torso_px > 0.0:
                                depth_torso = F * dist_torso_m / dist_torso_px
                        else:
                            depth_torso = np.inf

                    else:

                        if dist_hips_px > 0.0:
                            depth_hips = F * dist_hips_m / (dist_hips_px*self.sinus_pi_45_degrees)
                        else:
                            depth_hips = np.inf

                        if dist_shoulders_px > 0.0:
                            depth_shoulders = F * dist_shoulders_m / (dist_shoulders_px*self.sinus_pi_45_degrees)
                        else:
                            depth_shoulders = np.inf

                        if dist_torso_px > 0.0:
                            depth_torso = F * dist_torso_m / (dist_torso_px*self.sinus_pi_45_degrees)
                        else:
                            depth_torso = np.inf

                    smallest = np.min(np.array([depth_hips,depth_shoulders,depth_torso]))
                    
                    depth.append(smallest)

                if not depth[1] > depth[0]:
                    depth[0] = 0.0
                    depth[1] = 50.0

                depths.append((depth[0],depth[1],(depth[0]+depth[1])/2,c,p))

                self.msg[c].persons[p].depth = depths[-1][2]

                for k in self.person_msg_keys_hips_shoulders:
                    if self.msg[c].persons[p].keypoints[k].score > 0.0:

                        u = self.msg[c].persons[p].keypoints[k].x
                        v = self.msg[c].persons[p].keypoints[k].y

                        z = 1.0
                        x = (u-cx)*z/fx
                        y = (v-cy)*z/fy
                        
                        P = np.array([x,y,z])
                        P = P / np.linalg.norm(P)

                        r = R.from_euler("xyz",self.means_predictions[-1][c][0:3],degrees=False).as_matrix()
                        t = self.means_predictions[-1][c][3:6]
                        P = r @ P + t

                        a = t
                        b = P-t

                        rays[c][p][k] = (a,b)

                        a2 = a + depth[0] * b
                        b2 = a + depth[1] * b

                        line_segments[c][p][k] = (a2,b2)

        depths = np.asarray(depths)

        if self.publish_3D_depth_estimation:
            self.publisher_3D_depth_estimation(rays, line_segments)
            self.publsihed_depth_information_this_frame = True
            
        return rays, line_segments, depths

    def joints_data_association(self,line_segments,depths):

        # find order, from near to far
        order = np.argsort(depths[:,2])
        order = np.stack((np.take(depths[:,3],order),np.take(depths[:,4],order)),axis=1).astype(int) 
   
        hyps = []
        for n in range(len(order)):
            c = order[n][0]
            p = order[n][1]
           
            # find matching hypothesis
            distances = []
            assigned = False
            for h in range(len(hyps)):
                
                # check if a detetion in c is already assigned to h and skip
                skip = False
                for i in range(len(hyps[h])):
                    if hyps[h][i][0] == c:
                        skip = True
                        break
                if skip:
                    distances.append(self.data_association_threshold)
                    continue

                # calc distance
                i = 0
                distance_linesegments = 0
                distance_rays = 0
                for k in self.person_msg_keys_hips_shoulders:
                    if self.msg[c].persons[p].keypoints[k].score > 0.0:
                        for hyp_cp in range(len(hyps[h])):
                            if self.msg[hyps[h][hyp_cp][0]].persons[hyps[h][hyp_cp][1]].keypoints[k].score > 0.0:                                    
                                distance_linesegments += self.distance_segment2segment(line_segments[c][p][k][0],line_segments[c][p][k][1], 
                                                                                     line_segments[hyps[h][hyp_cp][0]][hyps[h][hyp_cp][1]][k][0], 
                                                                                     line_segments[hyps[h][hyp_cp][0]][hyps[h][hyp_cp][1]][k][1])[2]
                                i += 1
                distances.append(distance_linesegments/i)
            
            # assign to hypothesis with smallest distance if this distance is below the treshold
            if len(distances) > 0:
                distances = np.asarray(distances)
                i_min = np.argmin(distances)
                if distances[i_min] < self.data_association_threshold:
                    self.msg[c].persons[p].id = i_min
                    hyps[i_min].append((c,p))
                    assigned = True  

            # create new hypothesis
            if not assigned:
                self.msg[c].persons[p].id = len(hyps)
                hyps.append([(c,p)])
        
        return len(hyps)

    def joints_pruning(self,hyps):
        
        self.filterstage = 3

        do_again = True
        while(do_again):
            do_again = False

            # counting person keypoint detections
            detections = np.zeros((hyps,self.person_msg_keys_num),dtype=np.uint8)
            for c in self.cams_used:    
                for p in range(len(self.msg[c].persons)):
                    for k in range(len(self.msg[c].persons[p].keypoints)):
                        if self.msg[c].persons[p].keypoints[k].score > 0.0:
                            detections[self.msg[c].persons[p].id][k] += 1

            # don't do any pruning if not necessary
            if np.where(np.logical_and(detections > 0, detections < self.data_association_views_min))[0].shape[0] == 0:
                break

            # pruning person keypoint detections that are observed from less than self.data_association_views_min cameras
            for c in copy.deepcopy(self.cams_used):    
                
                for p in range(0,len(self.msg[c].persons)):
                
                    person_id = self.msg[c].persons[p].id

                    # skip if this person_id doesn't need pruning
                    if np.where(np.logical_and( detections[person_id] > 0, detections[person_id] < self.data_association_views_min))[0].shape[0] == 0:
                        continue

                    #Read data
                    data = np.asarray([np.array([k.x, k.y, k.score]) for k in self.msg[c].persons[p].keypoints]) 
                    scores = data[:,2]
                    xs = data[:,0]
                    ys = data[:,1]

                    remove = np.where(np.logical_and(scores > 0.0, detections[person_id] < self.data_association_views_min))[0]
                    if np.any(np.isin(remove,self.person_msg_keys_hips_shoulders,assume_unique=True)):
                        if self.remove_person(c,p) == -1: # this can affect other detections, so pruning must be repeated
                            return False
                        do_again = True
                        break
                    zeros = np.where(scores == 0.0)[0]
                    
                    # remove all outliers from msg
                    for k in remove:

                        self.msg[c].persons[p].keypoints[k].score = 0
                        self.msg[c].persons[p].keypoints[k].x = 0.0
                        self.msg[c].persons[p].keypoints[k].y = 0.0
                        self.msg[c].persons[p].keypoints[k].cov = (0.0,0.0,0.0)
                        detections[person_id][k] -= 1

                    # update person info in msg
                    if remove.shape[0] > 0:

                        outliers = np.concatenate((remove,zeros))

                        # average score
                        scores_mean_scope = np.delete(scores,outliers)
                        scores_mean = np.mean(scores_mean_scope)
                        self.msg[c].persons[p].score = scores_mean

                        # bounding box
                        xs_filter_scope = np.delete(xs,outliers)
                        x0 = np.min(xs_filter_scope)
                        x1 = np.max(xs_filter_scope)
                        ys_filter_scope = np.delete(ys,outliers)
                        y0 = np.min(ys_filter_scope)
                        y1 = np.max(ys_filter_scope)
                        self.msg[c].persons[p].bbox = (x0,y0,x1,y1)
                
                if do_again:
                    break

        return True

    def joints_extract_hyp(self):

        hyps = [None]
        hyp_ids = []

        for c in self.cams_used:

            stamp_secs = self.msg[c].header.stamp.secs
            stamp_nsecs =self.msg[c].header.stamp.nsecs
            stamp = np.asarray(stamp_secs) * 1000000000 + np.asarray(stamp_nsecs) 

            for p in range(len(self.msg[c].persons)):

                i = self.msg[c].persons[p].id
                if i not in hyp_ids: 
                    hyp_ids.append(i)
                    if len(hyp_ids)>len(hyps):
                        hyps.append(None)
                i = hyp_ids.index(i)
                
                for k in range(len(self.msg[c].persons[p].keypoints)):

                    d2=self.msg[c].persons[p].keypoints[k].score
                    if d2 > 0:
                        d3=self.msg[c].persons[p].keypoints[k].x
                        d4=self.msg[c].persons[p].keypoints[k].y
                        d5=self.msg[c].persons[p].keypoints[k].cov[0]
                        d6=self.msg[c].persons[p].keypoints[k].cov[1]
                        d7=self.msg[c].persons[p].keypoints[k].cov[2]
                        d8=stamp
                        d9 = self.msg[c].persons[p].depth

                        row = np.array([c,k,d2,d3,d4,d5,d6,d7,d8,d9],dtype=np.float64)    

                        if hyps[i] is None:
                            hyps[i] = [row]
                        else:
                            hyps[i].append(row)

        for i in range(len(hyps)):
            hyps[i] = np.asarray(hyps[i])

        return hyps

    def joints_analyze_hyp(self,hyps):

        p_num = 14+self.cams_num
        properties = []
        centers = []
        for h in range(len(hyps)):
            p = np.zeros(p_num,dtype=np.float64)
            
            p[0] = hyps[h].shape[0] # number of valid detections
    
            point = np.zeros(3,dtype=np.float64)
            counter = 0
            for i in range(hyps[h].shape[0]):

                if int(hyps[h][i][1]) in self.person_msg_keys_hips_shoulders:

                    c = int(hyps[h][i][0])
                    K = np.asarray(self.intrinsics_caminfomsg[c].K).reshape(3,3)
                    cx = K[0][2]
                    cy = K[1][2]
                    fx = K[0][0]
                    fy = K[1][1]
                    F = (fx+fy)/2

                    u = hyps[h][i][3]
                    v = hyps[h][i][4]

                    z = 1.0
                    x = (u-cx)*z/fx
                    y = (v-cy)*z/fy
                    
                    P = np.array([x,y,z])
                    P = P / np.linalg.norm(P)

                    r = R.from_euler("xyz",self.means_predictions[-1][c][0:3],degrees=False).as_matrix()
                    t = self.means_predictions[-1][c][3:6]
                    P = r @ P + t

                    a = t
                    b = P-t

                    point += (a + hyps[h][i][9] * b)
                    counter += 1

            point /= counter
            point = np.asarray(point)

            centers.append(point)

            p[1] = point[0] # center of mass x
            p[2] = point[1] # center of mass y
            p[3] = point[2] # center of mass z
            
            distances = []
            cam_counters = np.zeros(self.cams_num,dtype=int)
            for i in range(hyps[h].shape[0]):
                distances.append(np.linalg.norm(point-self.means_predictions[-1][int(hyps[h][i][0])][3:6]))
                cam_counters[int(hyps[h][i][0])] += 1
            distances = np.asarray(distances)

            p[4] = np.mean(distances) # mean distance between center of mass and camera poses
            p[5] = np.std(distances) # std distance between center of mass and camera poses

            keys = np.unique(hyps[h][:,1])            
            occurences = []
            for i in range(len(keys)):
                occurences.append(np.where(hyps[h][:,1]==keys[i])[0].shape[0])
            occurences = np.asarray(occurences)

            p[6] = np.mean(occurences) # average detections per joint
            p[7] = np.std(occurences) # std detections per joint

            p[8] = np.mean(hyps[h][:,2]) # mean confidence
            p[9] = np.std(hyps[h][:,2]) # std confidence

            p[10] = np.mean(hyps[h][:,8]) # mean timestamp
            p[11] = np.std(hyps[h][:,8]) # std timestamp

            p[12] = 0 # opt. uses
            p[13] = 0 # opt. fails

            p[13:-1] = cam_counters # number of detections of each cam

            properties.append(p)

        if self.publish_3D_depth_estimation:
            self.publisher_3D_live_centers_of_mass(centers)

        return properties

    def remove_cams(self,cams_to_be_removed,reason=""):
            
        # returns -1 if cam count drops below minimum_shared_detections, returns 0 otherwise

        if isinstance(cams_to_be_removed,np.ndarray):
            pass
        elif isinstance(cams_to_be_removed,(int,np.integer)):
            cams_to_be_removed = np.array([cams_to_be_removed],dtype=np.uint8)
        elif isinstance(cams_to_be_removed,list):
            cams_to_be_removed = np.asarray(cams_to_be_removed,dtype=np.uint8)
        else:
            raise TypeError("Got "+str(cams_to_be_removed)+" of type "+str(type(cams_to_be_removed))+", but only integers, lists and numpy arrays are allowed") 

        for c in cams_to_be_removed:
            self.msg[self.cams_used[c]].persons = []

            if self.publish_2D:
                if self.publisher_2D_do:
                    if self.reasons_per_cam[self.cams_used[c]] != "":
                        self.reasons_per_cam[self.cams_used[c]] += "\n"
                    self.reasons_per_cam[self.cams_used[c]] += reason

        self.cams_used = np.delete(self.cams_used,cams_to_be_removed)
        self.persons_used = np.delete(self.persons_used,cams_to_be_removed)

        if self.cams_used.shape[0] < self.data_association_views_min:
            self.joints_finisher(reason = self.filterstage_reasons[self.filterstage])
            return -1
        return 0

    def remove_person(self, cam, person):
        
        # returns -1 if person count of cam drops to zero, returns 0 otherwise

        self.msg[cam].persons.pop(person)
        cam_index = np.where(self.cams_used==cam)[0]
        self.persons_used[cam_index] -= 1

        if self.persons_used[cam_index] == 0:
            if self.remove_cams(cam_index, reason = self.filterstage_reasons[self.filterstage]) == -1:
                return -1
        return 0

    def joints_finisher(self,used=False,reason=""):

        self.frame_counter_total += 1
        self.timestamp_lastframe = rospy.get_time()

        if used:
            self.hypotheses_num += len(self.hypotheses)
            self.lock.acquire()
            for i in range(len(self.hypotheses)):
                self.queue_hyps.append(self.hypotheses[i])
                self.queue_props.append(self.properties[i])  
            self.lock.release()

            self.frame_counter_eligble += 1
            if self.sync_min_gap > 0.0:
                self.timestamp_lastframe_used = self.timestamp_lastframe
        
        self.timestamp_processing_end = self.timestamp_lastframe
        
        if self.publish_3D_depth_estimation:
            if not self.publsihed_depth_information_this_frame:
                self.publisher_3D_depth_estimation()
            
        if self.publish_2D:
            if self.publisher_2D_do:
                self.timestamp_last_2Dpublisher = self.timestamp_lastframe
                self.publisher_2D_do = False
                for i in range(self.cams_num):
                    if self.reasons_per_cam[i] == "" and not used:
                        self.reasons_per_cam[i] = "Rejected before further reasoning"
                self.publisher_2D(self.original_msg, self.msg, used, reasons_per_cam=self.reasons_per_cam, reason_frame=reason)

    # Main callbacks
    def graph_callback(self,timer):
        
        if self.shutdown or not self.first_msg_received or len(self.queue_hyps) < self.optimization_scope_size * self.optimization_wait_mult:
            return
        
        # Prevent shutdown
        self.processing = True
        
        self.lock.acquire()
        selection = self.graph_select_hypotheses()
        self.graph_analyze_hypotheses(selection)
        self.lock.release()
        
        # build new graph
        self.graph = gtsam.NonlinearFactorGraph()
        self.graph_add_factors_cameras()
        self.graph_compute_landmarks()
        self.graph_add_factors_projections()
        
        # insert values
        self.estimate = gtsam.Values()
        self.graph_estimate_cameras()
        self.graph_estimate_landmarks()

        # optimize
        result = self.graph_optimize()

        # refine
        result = self.graph_noise_update_kalman(result)
        
        # update datastructures, trigger error calculation and publisher
        self.graph_update_hypotheses_properties(selection)
        self.extrinsics_update(result)
        self.landmarks_update(result)
        self.publisher_3D_update = True

        # Allow shutdown
        self.timestamp_processing_end = rospy.get_time()
        self.processing = False

    def joints_callback(self,*msg):
        
        if self.shutdown:
            return

        # Initialize callback
        if not self.first_msg_received:

            self.filterstage_reasons = ["Rejected while cheking detection counts","Rejected while checking timings","Rejected while checking scores","Rejected while pruning unique detections","Rejected while checking bounding boxes"]
            
            self.lock=threading.Lock()

            self.filter_stamps_span_max *= 1000000000
            self.filter_stamps_std_max *= 1000000000
            self.sinus_pi_45_degrees = np.sin(45*np.pi/180)

            self.queue_hyps = []
            self.queue_props = []
            self.hypotheses_num = 0
            self.frame_counter_eligble = 0
            self.frame_counter_total = 0

            self.person_msg_keys_ids = np.asarray(self.person_msg_keys_ids)
            self.person_msg_keys_ids_unused = np.delete(np.arange(self.person_msg_keys_num),self.person_msg_keys_ids)
            
            self.person_msg_keys_hips_shoulders = np.asarray(self.person_msg_keys_hips_shoulders)
            self.filter_scores_musthahaveids_relative = []
            for i in self.person_msg_keys_hips_shoulders:
                self.filter_scores_musthahaveids_relative.append(np.where(self.person_msg_keys_ids == i)[0][0])
            self.filter_scores_musthahaveids_relative = np.asarray(self.filter_scores_musthahaveids_relative)

            self.timestamp_processing_begin = rospy.get_time()
            if self.sync_min_gap > 0.0:
                self.timestamp_lastframe_used = self.timestamp_processing_begin - rospy.Duration.from_sec(self.sync_min_gap).to_sec()
            if self.publish_2D:
                self.timestamp_last_2Dpublisher = self.timestamp_processing_begin - rospy.Duration.from_sec(self.publish_2D_interval).to_sec()

            self.first_msg_received = True
            
            self.print("first message received at "+"["+str(datetime.datetime.now())[2:-3]+"].")

            if self.autoexit_duration > 0.0:
                rospy.Timer(rospy.Duration(self.autoexit_duration), self.autoexit_callback, oneshot=True)
                self.print("Calibration will automatically shutdown in "+str(self.autoexit_duration)+" seconds.")

            self.print("\nOptimizing camara extrinsics...")

        # Reject frame-set if previously used frame-set was less than sync_min_gap seconds ago
        if self.sync_min_gap > 0.0:
            if rospy.get_rostime() - self.timestamp_lastframe_used > rospy.Duration.from_sec(self.sync_min_gap).to_sec():
                self.joints_finisher(reason = "sync_min_gap")
                return

        self.msg = msg  
        self.cams_used = np.arange(0,self.cams_num,1)
        self.persons_used = np.zeros(self.cams_num,dtype=np.uint8)
        
        if self.publish_3D_depth_estimation:
            self.publsihed_depth_information_this_frame = False
        
        # Publish filtering and data association for this callback
        if self.publish_2D:
            if rospy.get_time() - self.timestamp_last_2Dpublisher >= rospy.Duration.from_sec(self.publish_2D_interval).to_sec():
                self.reasons_per_cam = ["" for _ in range(self.cams_num)]
                self.publisher_2D_do = True
                self.original_msg = copy.deepcopy(self.msg)

        if not self.joints_filter_detections():
            return

        for c in range(len(self.cams_used)):
            self.persons_used[c] = len(self.msg[self.cams_used[c]].persons)

        if not self.joints_filter_bbox():
            return

        if not self.joints_filter_timing():
            return


        if not self.joints_filter():
            return
        
        if self.person_msg_coords_undistort:
            self.joints_undistort()

        # copy data to a new message type with a person ID
        msg_new = [Person2DList() for _ in self.msg]
        for c in range(self.cams_num):
            msg_new[c].header = self.msg[c].header
            msg_new[c].fb_delay = self.msg[c].fb_delay
            for p in range(len(self.msg[c].persons)):
                msg_new[c].persons.append(Person2DWithID())
                msg_new[c].persons[p].id = -1
                msg_new[c].persons[p].depth = -1
                msg_new[c].persons[p].score = self.msg[c].persons[p].score
                msg_new[c].persons[p].bbox = self.msg[c].persons[p].bbox
                msg_new[c].persons[p].keypoints = copy.deepcopy(self.msg[c].persons[p].keypoints)        
        self.msg = msg_new

        rays, line_segments, depths = self.joints_depth_estimation()

        hyps = self.joints_data_association(line_segments,depths)
    
        if not self.joints_pruning(hyps):
            return
           
        self.hypotheses = self.joints_extract_hyp()
        self.properties = self.joints_analyze_hyp(self.hypotheses)

        self.joints_finisher(used = True)

    # Updaters & Publishers
    def landmarks_update(self,values):

        landmarks = copy.deepcopy(self.landmarks_projected)

        for n in range(0,len(self.landmarks_projected)):
            for i in range(0,len(landmarks[n]["points"])):
                if landmarks[n]["landmark_ids"][i] is not None:
                    landmarks[n]["points"][i] = values.atPoint3(self.L(landmarks[n]["landmark_ids"][i]))
            landmarks[n].pop("timestamps")

        self.landmark_history.append(landmarks)

    def extrinsics_update(self,values):

        extrinsics = []
        
        for i in range(self.cams_num):

            topic = self.extrinsics_estimate[i]["topic"] 
            origin = self.extrinsics_estimate[i]["origin"]  # TODO does not work without reference
            stamp = rospy.get_rostime()  
            translation = values.atPose3(self.X(i)).translation()
            rotation = R.as_quat(R.from_matrix(values.atPose3(self.X(i)).rotation().matrix()))

            extrinsics.append({"topic":topic, "origin":origin, "stamp":stamp, "translation":translation, "rotation":rotation})

        if self.smoothing_enforce_scale:
            extrinsics = self.extrinsics_enforce_scale(extrinsics)

        self.extrinsics_history.append(extrinsics)
        
        if self.extrinsics_reference_readfromfile:
            self.extrinsics_calc_error()

    def extrinsics_enforce_scale(self,extrinsics):

        A = []
        B = []
        for c in range(0,self.cams_num):
            A.append(self.extrinsics_estimate[c]["translation"])
            B.append(extrinsics[c]["translation"])
        A = np.asarray(A)
        B = np.asarray(B)

        r, scale, t = self.kabsch_umeyama(A,B, False)
        
        for c in range(1,self.cams_num):
            extrinsics[c]["translation"] *= scale

        return extrinsics

    def extrinsics_calc_error(self):

        # initialize data structures
        if len(self.frame_success) == 0:    
            self.extrinsics_error = []
            self.global_error_translation = []
            self.global_error_translation_distance = []
            self.global_error_rotation = []
            self.global_error_rotation_distance = []
            self.average_distibuted_pos_errors = []

        # calc average distributed error
        A = []
        B = []
        for i in range(0,self.cams_num):
            A.append(self.extrinsics_reference[i]["translation"])
            B.append(self.extrinsics_history[-1][i]["translation"])
        A = np.asarray(A)
        B = np.asarray(B)

        r, c, t = self.kabsch_umeyama(A,B, False)
        aligned_pos = np.array([t + c * r @ b for b in B])

        distances = []
        for i in range(self.cams_num):
            translation = self.extrinsics_reference[i]["translation"] - aligned_pos[i]    
            distances.append(np.linalg.norm(translation))
        self.average_distibuted_pos_errors.append(np.mean(distances))

        # reset errors
        error = []
        self.global_error_translation.append([0.0, 0.0, 0.0])
        self.global_error_translation_distance.append(0.0)
        self.global_error_rotation.append([0.0, 0.0, 0.0])
        self.global_error_rotation_distance.append(0.0)

        # update errors
        for i in range(0,self.cams_num):

                translation = self.extrinsics_reference[i]["translation"] - self.extrinsics_history[-1][i]["translation"]
                    
                translation_distance = np.linalg.norm(translation)

                rotation = R.from_quat(self.extrinsics_reference[i]["rotation"]).as_euler('xyz', degrees=True) - R.from_quat(self.extrinsics_history[-1][i]["rotation"]).as_euler('xyz', degrees=True)
                for j in range(0,3):
                    while rotation[j] > 180:
                        rotation[j] -= 360
                    while rotation[j] < -180:
                        rotation[j] += 360

                rotation_distance = self.distance_rotation(np.asarray(self.extrinsics_reference[i]["rotation"]),np.asarray(self.extrinsics_history[-1][i]["rotation"]),is_unit=False,degrees=True)

                error.append({"translation":translation, "translation_distance":translation_distance, "rotation":rotation, "rotation_distance":rotation_distance})
                self.global_error_translation[-1] += np.abs(translation)
                self.global_error_translation_distance[-1] += translation_distance
                self.global_error_rotation[-1] += np.abs(rotation)
                self.global_error_rotation_distance[-1] += rotation_distance

        self.extrinsics_error.append(error)

    def publisher_3D_live_centers_of_mass(self,centers=[]):

        msg = MarkerArray()

        marker = Marker()    
        marker.header.frame_id = self.extrinsics_estimate[0]["name"]
        marker.id = 77
        marker.type = 7
        marker.ns = "Center of mass"
        marker.action = 0
        marker.scale.x = 0.3
        marker.scale.y = 0.3
        marker.scale.z = 0.3
        marker.pose.orientation.x = 0
        marker.pose.orientation.y = 0
        marker.pose.orientation.z = 0
        marker.pose.orientation.w = 1
    
        points = []
        colors = []

        color = ColorRGBA(0, 0, 0, 1)
        for i in range(len(centers)):
        
            point = centers[i]
            point = Point(point[0],point[1],point[2])

            points.append(point)
            colors.append(color)

        marker.points=points
        marker.colors=colors

        if len(points) == 0:                 
            marker.action = 2
                
        msg.markers.append(marker)
        self.marker_array_pub.publish(msg)

    def publisher_3D_depth_estimation_prepare(self):

        self.segment_namespace = "Depth Estimation"
        self.segment_thickness = 0.04 * 0.75
        self.segment_color = ColorRGBA(0, 0.8, 0, 1)
        
        self.ray_namespace = "Depth Estimation" 
        self.ray_thickness = 0.03 * 0.75
        self.ray_color = ColorRGBA(0, 0, 0, 1)
        self.ray_max_range = 100.0 # meters

    def publisher_3D_depth_estimation(self,rays=[],line_segments=[]):

        msg = MarkerArray()

        marker = Marker()     
        marker.header.frame_id = self.extrinsics_estimate[0]["name"]
        marker.id = 69
        marker.type = 5 
        marker.ns = self.ray_namespace
        marker.action = 0
        marker.scale.x = self.ray_thickness
        marker.scale.y = self.ray_thickness
        marker.scale.z = self.ray_thickness
        marker.pose.orientation.x = 0
        marker.pose.orientation.y = 0
        marker.pose.orientation.z = 0
        marker.pose.orientation.w = 1
    
        ray_points = []
        ray_colors = []

        segment_points = []
        segment_colors = []

        for c in range(len(rays)):
            for p in range(len(rays[c])):
                for k in range(len(rays[c][p])):
                    if not rays[c][p][k] is None:
                        
                        # cam to minimum depth
                        point = rays[c][p][k][0]
                        point1 = Point(point[0],point[1],point[2])
                        ray_points.append(point1)
                        
                        point = line_segments[c][p][k][0]
                        point2 = Point(point[0],point[1],point[2])
                        ray_points.append(point2)

                        # minimum to maximum depth
                        segment_points.append(point2)
                        
                        point = line_segments[c][p][k][1]
                        point3 = Point(point[0],point[1],point[2])
                        segment_points.append(point3)
                        
                        # maximum depth to max_range
                        ray_points.append(point3)

                        point = point + rays[c][p][k][1] * self.ray_max_range
                        point4 = Point(point[0],point[1],point[2])
                        ray_points.append(point4)
        
        num_points = len(ray_points)
        if num_points == 0:                 
            marker.action = 2
        marker.points= ray_points
        marker.colors= [self.ray_color] * num_points             
        msg.markers.append(marker)

        marker = Marker()     
        marker.header.frame_id = self.extrinsics_estimate[0]["name"]
        marker.id = 42
        marker.type = 5 
        marker.ns = self.segment_namespace
        marker.action = 0
        marker.scale.x = self.segment_thickness
        marker.scale.y = self.segment_thickness
        marker.scale.z = self.segment_thickness
        marker.pose.orientation.x = 0
        marker.pose.orientation.y = 0
        marker.pose.orientation.z = 0
        marker.pose.orientation.w = 1

        num_points = len(segment_points)
        if num_points == 0:                 
            marker.action = 2
        marker.points=segment_points
        marker.colors= [self.segment_color] * num_points 
        msg.markers.append(marker)

        self.marker_array_pub.publish(msg)

    def publisher_3D_landmarks_prepare(self):
    
        color_table = [ColorRGBA(1.0, 0.0, 0.0, 1.0),   #0 nose
            ColorRGBA(1.0, 0.4, 0.0, 1.0),              #1 left eye
            ColorRGBA(1.0, 0.6, 0.0, 1.0),              #2 right eye
            ColorRGBA(1.0, 1.0, 0.0, 1.0),              #3 left ear
            ColorRGBA(0.5, 1.0, 0.0, 1.0),              #4 right ear
            ColorRGBA(0.0, 1.0, 0.2, 1.0),              #5 left shoulder
            ColorRGBA(0.0, 1.0, 0.0, 1.0),              #6 right shoulder
            ColorRGBA(0.0, 1.0, 0.5, 1.0),              #7 left elbow
            ColorRGBA(0.0, 0.7, 0.7, 1.0),              #8 right elbow
            ColorRGBA(0.0, 1.0, 1.0, 1.0),              #9 left wrist
            ColorRGBA(0.0, 0.7, 1.0, 1.0),              #10 right wrist
            ColorRGBA(0.0, 0.0, 1.0, 1.0),              #11 left hip
            ColorRGBA(0.0, 0.0, 1.0, 1.0),              #12 right hip
            ColorRGBA(0.0, 0.0, 1.0, 1.0),              #13 left knee
            ColorRGBA(0.2, 0.0, 0.5, 1.0),              #14 right knee
            ColorRGBA(0.5, 0.0, 0.7, 1.0),              #15 left ankle
            ColorRGBA(1.0, 0.0, 1.0, 1.0)]              #16 right ankle

        black_table = [ColorRGBA(0.0, 0.0, 0.0, 1.0)] * 17

        self.publish_3D_landmarks_connections = [(0,1),(0,2),(1,3),(2,4),(5,6),(5,7),(6,8),(7,9),(8,10),(11,12),(11,13),(12,14),(13,15),(14,16)]
        self.publish_3D_landmarks_between = [(5,6),(11,12)]

        self.publish_3D_landmarks_mtype = [None] * 6
        self.publish_3D_landmarks_mname = [None] * 6
        self.publish_3D_landmarks_mscale = [None] * 6
        self.publish_3D_landmarks_mcolor = [None] * 6
        self.publish_3D_landmarks_types = []

        if self.publish_3D_landmarks_triangulation:
            self.publish_3D_landmarks_types.append(0)
            self.publish_3D_landmarks_mtype[0] = 7
            self.publish_3D_landmarks_mname[0] = "Triangulation"
            self.publish_3D_landmarks_mscale[0] = 0.06
            self.publish_3D_landmarks_mcolor[0] = black_table
            self.publish_3D_landmarks_mtype[1] = 5
            self.publish_3D_landmarks_mname[1] = "Triangulation"
            self.publish_3D_landmarks_mscale[1] = 0.02
            self.publish_3D_landmarks_mcolor[1] = black_table
        if self.publish_3D_landmarks_optimization:
            self.publish_3D_landmarks_types.append(1)
            self.publish_3D_landmarks_mtype[2] = 7
            self.publish_3D_landmarks_mname[2] = "Optimization"
            self.publish_3D_landmarks_mscale[2] = 0.07
            self.publish_3D_landmarks_mcolor[2] = color_table
            self.publish_3D_landmarks_mtype[3] = 5
            self.publish_3D_landmarks_mname[3] = "Optimization"
            self.publish_3D_landmarks_mscale[3] = 0.03
            self.publish_3D_landmarks_mcolor[3] = color_table
            
            dimmer = 0.75
            for i in range(len(color_table)):
                color = self.publish_3D_landmarks_mcolor[3][i]
                color.r = color.r * dimmer
                color.g = color.g * dimmer
                color.b = color.b * dimmer
                color.a = color.a * dimmer
                self.publish_3D_landmarks_mcolor[3][i] = color
                
    def publisher_3D_landmarks(self):

        def index(array,value):
            i = next((idx for idx, val in np.ndenumerate(array) if val==value),np.nan)
            if np.isfinite(i): i = i[0]
            return i

        try:
            self.frame_success
        except:
            return
        if len(self.frame_success) == 0:
            return

        msg = MarkerArray()

        for t in self.publish_3D_landmarks_types:
            
            if t == 0 and len(self.landmarks_projected) > 0:
                data = self.landmarks_projected
            elif t == 1 and len(self.landmark_history) > 0:
                data = self.landmark_history[-1]

            for sl in range(2):

                i = t*2+sl
                
                marker = Marker() 
                
                marker.header.frame_id = self.extrinsics_estimate[0]["name"]
                marker.id = i
                marker.type = self.publish_3D_landmarks_mtype[i]     
                marker.ns = self.publish_3D_landmarks_mname[i]

                if not self.frame_success[-1] and i == 1:
                    marker.action = 2 # remove
                else:
                    marker.action = 0 # add

                marker.scale.x = self.publish_3D_landmarks_mscale[i]
                marker.scale.y = self.publish_3D_landmarks_mscale[i]
                marker.scale.z = self.publish_3D_landmarks_mscale[i]

                marker.pose.orientation.x = 0
                marker.pose.orientation.y = 0
                marker.pose.orientation.z = 0
                marker.pose.orientation.w = 1

                points = []
                colors = []

                if marker.action == 0:

                    # Points
                    if sl == 0:
                        for n in range(0,len(data)):

                            # find existing keypoints
                            keypoints = np.arange(self.person_msg_keys_num*data[n]["persons"]) % self.person_msg_keys_num
                            keypoints_existing = np.full(self.person_msg_keys_num*data[n]["persons"],None)

                            # all 3D points correpsonding to 2D keypoints
                            for j in range(0,len(data[n]["points"])):
                                if not (data[n]["points"][j] == np.inf).any():

                                    keypoints_existing[j] = keypoints[j]

                                    point = Point(data[n]["points"][j][0], data[n]["points"][j][1], data[n]["points"][j][2])
                                    color = self.publish_3D_landmarks_mcolor[i][self.person_msg_keys_ids[j%self.person_msg_keys_num]]

                                    points.append(point)
                                    colors.append(color)

                            # extra points between hips and shoulders
                            for j in range(0,len(self.publish_3D_landmarks_between)):

                                i_from = index(keypoints_existing,self.publish_3D_landmarks_between[j][0])
                                i_to =index(keypoints_existing,self.publish_3D_landmarks_between[j][1])

                                if i_from is not np.nan and i_to is not np.nan:
                                    
                                    point = ( data[n]["points"][i_from] + data[n]["points"][i_to] ) / 2
                                    point = Point(point[0],point[1],point[2])
                                    points.append(point)
                                    
                                    color = self.publish_3D_landmarks_mcolor[i][self.person_msg_keys_ids[i_from%self.person_msg_keys_num]]
                                    colors.append(color)

                    # Lines
                    elif sl == 1:
                        for n in range(0,len(data)):

                            # find existing keypoints
                            keypoints = np.arange(self.person_msg_keys_num*data[n]["persons"]) % self.person_msg_keys_num
                            keypoints_existing = np.full(self.person_msg_keys_num*data[n]["persons"],None)
                            for j in range(len(data[n]["points"])):
                                if not (data[n]["points"][j] == np.inf).any():
                                   keypoints_existing[j] = keypoints[j]

                            for p in range(data[n]["persons"]):
                                
                                # all self.publish_3D_landmarks_connections between keypoints that correspond to 2d keypoints
                                for j in range(len(self.publish_3D_landmarks_connections)):

                                    i_from = index(keypoints_existing[ self.person_msg_keys_num * p : self.person_msg_keys_num + p * self.person_msg_keys_num ], self.publish_3D_landmarks_connections[j][0]) + p * self.person_msg_keys_num
                                    i_to = index(keypoints_existing[ self.person_msg_keys_num * p : self.person_msg_keys_num + p * self.person_msg_keys_num ], self.publish_3D_landmarks_connections[j][1]) + p * self.person_msg_keys_num
                                    
                                    if not (np.isnan(i_from) or np.isnan(i_to)):

                                        point = data[n]["points"][i_from]
                                        point = Point(point[0],point[1],point[2])
                                        points.append(point)

                                        color = self.publish_3D_landmarks_mcolor[i][self.person_msg_keys_ids[i_from%self.person_msg_keys_num]]
                                        colors.append(color)

                                        point = data[n]["points"][i_to]
                                        point = Point(point[0],point[1],point[2])
                                        points.append(point)

                                        color = self.publish_3D_landmarks_mcolor[i][self.person_msg_keys_ids[i_to%self.person_msg_keys_num]]
                                        colors.append(color)

                                # connection self.publish_3D_landmarks_between center of hips and center of shoulders
                                i_hip_left = index(keypoints_existing[ self.person_msg_keys_num * p : self.person_msg_keys_num + p * self.person_msg_keys_num ], self.publish_3D_landmarks_connections[4][0]) + p * self.person_msg_keys_num
                                i_hip_right = index(keypoints_existing[ self.person_msg_keys_num * p : self.person_msg_keys_num + p * self.person_msg_keys_num ], self.publish_3D_landmarks_connections[4][1]) + p * self.person_msg_keys_num
                                i_shoulder_left = index(keypoints_existing[ self.person_msg_keys_num * p : self.person_msg_keys_num + p * self.person_msg_keys_num ], self.publish_3D_landmarks_connections[9][0]) + p * self.person_msg_keys_num
                                i_shoulder_right = index(keypoints_existing[ self.person_msg_keys_num * p : self.person_msg_keys_num + p * self.person_msg_keys_num ], self.publish_3D_landmarks_connections[9][1]) + p * self.person_msg_keys_num
                                
                                if not (np.isnan(i_hip_left) or np.isnan(i_hip_right) or np.isnan(i_shoulder_left) or np.isnan(i_shoulder_right)):

                                    point = ( data[n]["points"][i_hip_left] + data[n]["points"][i_hip_right] ) / 2
                                    point = Point(point[0],point[1],point[2])
                                    points.append(point)

                                    color = self.publish_3D_landmarks_mcolor[i][self.person_msg_keys_ids[i_hip_left%self.person_msg_keys_num]]
                                    colors.append(color)

                                    point = ( data[n]["points"][i_shoulder_left] + data[n]["points"][i_shoulder_right] ) / 2
                                    point = Point(point[0],point[1],point[2])
                                    points.append(point)

                                    color = self.publish_3D_landmarks_mcolor[i][self.person_msg_keys_ids[i_shoulder_left%self.person_msg_keys_num]]
                                    colors.append(color)

                                # connection between center of hips and center of shoulders
                                i_nose = index(keypoints_existing[ self.person_msg_keys_num * p : self.person_msg_keys_num + p * self.person_msg_keys_num ], 0) + p * self.person_msg_keys_num
                                i_hip_left = index(keypoints_existing[ self.person_msg_keys_num * p : self.person_msg_keys_num + p * self.person_msg_keys_num ], self.publish_3D_landmarks_connections[4][0]) + p * self.person_msg_keys_num
                                i_hip_right = index(keypoints_existing[ self.person_msg_keys_num * p : self.person_msg_keys_num + p * self.person_msg_keys_num ], self.publish_3D_landmarks_connections[4][1]) + p * self.person_msg_keys_num
                                
                                if not (np.isnan(i_nose) or np.isnan(i_hip_left) or np.isnan(i_hip_right)):

                                    point = data[n]["points"][i_nose]
                                    point = Point(point[0],point[1],point[2])
                                    points.append(point)

                                    color = self.publish_3D_landmarks_mcolor[i][self.person_msg_keys_ids[i_nose%self.person_msg_keys_num]]
                                    colors.append(color)

                                    point = ( data[n]["points"][i_hip_left] + data[n]["points"][i_hip_right] ) / 2
                                    point = Point(point[0],point[1],point[2])
                                    points.append(point)

                                    color = self.publish_3D_landmarks_mcolor[i][self.person_msg_keys_ids[i_hip_left%self.person_msg_keys_num]]
                                    colors.append(color)

                    marker.points=points
                    marker.colors=colors
            
                    if len(points) == 0:                 
                        marker.action = 2
                
                msg.markers.append(marker)

        self.marker_array_pub.publish(msg)

    def publisher_3D_cameras(self):

        stamp = rospy.Time.now()

        msg = TransformStamped()

        msg.header.stamp = stamp
        msg.header.frame_id = self.publish_3D_base_name
        msg.child_frame_id = self.publish_3D_map_name

        msg.transform.translation.x = self.publish_3D_base_to_map[0]
        msg.transform.translation.y = self.publish_3D_base_to_map[1]
        msg.transform.translation.z = self.publish_3D_base_to_map[2]
        msg.transform.rotation.x = self.publish_3D_base_to_map[3]
        msg.transform.rotation.y = self.publish_3D_base_to_map[4]
        msg.transform.rotation.z = self.publish_3D_base_to_map[5]
        msg.transform.rotation.w = self.publish_3D_base_to_map[6]

        self.tf_broadcaster.sendTransform(msg) # base to map
        
        if self.publish_3D_camera_poses_result:

            msg = TransformStamped()

            msg.header.stamp = stamp
            msg.header.frame_id = self.publish_3D_base_name
            msg.child_frame_id = (self.extrinsics_estimate[0]["name"]+self.publish_3D_camera_poses_result_label)

            msg.transform.translation.x = self.publish_3D_camera_poses_base_to_cam0[0]
            msg.transform.translation.y = self.publish_3D_camera_poses_base_to_cam0[1]
            msg.transform.translation.z = self.publish_3D_camera_poses_base_to_cam0[2]
            msg.transform.rotation.x = self.publish_3D_camera_poses_base_to_cam0[3]
            msg.transform.rotation.y = self.publish_3D_camera_poses_base_to_cam0[4]
            msg.transform.rotation.z = self.publish_3D_camera_poses_base_to_cam0[5]
            msg.transform.rotation.w = self.publish_3D_camera_poses_base_to_cam0[6]

            self.tf_broadcaster.sendTransform(msg) # base to cam_0

            for i in range(1,self.cams_num):
                
                msg = TransformStamped()

                msg.header.stamp = stamp
                msg.header.frame_id = (self.extrinsics_estimate[0]["name"]+self.publish_3D_camera_poses_result_label)
                msg.child_frame_id = (self.extrinsics_estimate[i]["name"]+self.publish_3D_camera_poses_result_label)

                msg.transform.translation.x = self.extrinsics_history[-1][i]["translation"][0]
                msg.transform.translation.y = self.extrinsics_history[-1][i]["translation"][1]
                msg.transform.translation.z = self.extrinsics_history[-1][i]["translation"][2]
                msg.transform.rotation.x = self.extrinsics_history[-1][i]["rotation"][0]
                msg.transform.rotation.y = self.extrinsics_history[-1][i]["rotation"][1]
                msg.transform.rotation.z = self.extrinsics_history[-1][i]["rotation"][2]
                msg.transform.rotation.w = self.extrinsics_history[-1][i]["rotation"][3]

                self.tf_broadcaster.sendTransform(msg) # cam_i to cam_0

        if self.publish_3D_camera_poses_initialization:
            
            msg = TransformStamped()

            msg.header.stamp = stamp
            msg.header.frame_id = self.publish_3D_base_name
            msg.child_frame_id = (self.extrinsics_estimate[0]["name"]+self.publish_3D_camera_poses_initialization_label)

            msg.transform.translation.x = self.publish_3D_camera_poses_base_to_cam0[0]
            msg.transform.translation.y = self.publish_3D_camera_poses_base_to_cam0[1]
            msg.transform.translation.z = self.publish_3D_camera_poses_base_to_cam0[2]
            msg.transform.rotation.x = self.publish_3D_camera_poses_base_to_cam0[3]
            msg.transform.rotation.y = self.publish_3D_camera_poses_base_to_cam0[4]
            msg.transform.rotation.z = self.publish_3D_camera_poses_base_to_cam0[5]
            msg.transform.rotation.w = self.publish_3D_camera_poses_base_to_cam0[6]

            self.tf_broadcaster.sendTransform(msg) # base to cam_0

            for i in range(1,self.cams_num):
                
                msg = TransformStamped()

                msg.header.stamp = stamp
                msg.header.frame_id = (self.extrinsics_estimate[0]["name"]+self.publish_3D_camera_poses_initialization_label)
                msg.child_frame_id = (self.extrinsics_estimate[i]["name"]+self.publish_3D_camera_poses_initialization_label)

                msg.transform.translation.x = self.extrinsics_estimate[i]["translation"][0]
                msg.transform.translation.y = self.extrinsics_estimate[i]["translation"][1]
                msg.transform.translation.z = self.extrinsics_estimate[i]["translation"][2]
                msg.transform.rotation.x = self.extrinsics_estimate[i]["rotation"][0]
                msg.transform.rotation.y = self.extrinsics_estimate[i]["rotation"][1]
                msg.transform.rotation.z = self.extrinsics_estimate[i]["rotation"][2]
                msg.transform.rotation.w = self.extrinsics_estimate[i]["rotation"][3]

                self.tf_broadcaster.sendTransform(msg) # cam_i to cam_0

        if self.publish_3D_camera_poses_reference and self.extrinsics_reference_readfromfile:     
       
            msg = TransformStamped()

            msg.header.stamp = stamp
            msg.header.frame_id = self.publish_3D_base_name
            msg.child_frame_id = (self.extrinsics_estimate[0]["name"]+self.publish_3D_camera_poses_reference)

            msg.transform.translation.x = self.publish_3D_camera_poses_base_to_cam0[0]
            msg.transform.translation.y = self.publish_3D_camera_poses_base_to_cam0[1]
            msg.transform.translation.z = self.publish_3D_camera_poses_base_to_cam0[2]
            msg.transform.rotation.x = self.publish_3D_camera_poses_base_to_cam0[3]
            msg.transform.rotation.y = self.publish_3D_camera_poses_base_to_cam0[4]
            msg.transform.rotation.z = self.publish_3D_camera_poses_base_to_cam0[5]
            msg.transform.rotation.w = self.publish_3D_camera_poses_base_to_cam0[6]

            self.tf_broadcaster.sendTransform(msg) # base to cam_0

            for i in range(1,self.cams_num):
                
                msg = TransformStamped()

                msg.header.stamp = stamp
                msg.header.frame_id = (self.extrinsics_estimate[0]["name"]+self.publish_3D_camera_poses_reference)
                msg.child_frame_id = (self.extrinsics_estimate[i]["name"]+self.publish_3D_camera_poses_reference)

                msg.transform.translation.x = self.extrinsics_reference[i]["translation"][0]
                msg.transform.translation.y = self.extrinsics_reference[i]["translation"][1]
                msg.transform.translation.z = self.extrinsics_reference[i]["translation"][2]
                msg.transform.rotation.x = self.extrinsics_reference[i]["rotation"][0]
                msg.transform.rotation.y = self.extrinsics_reference[i]["rotation"][1]
                msg.transform.rotation.z = self.extrinsics_reference[i]["rotation"][2]
                msg.transform.rotation.w = self.extrinsics_reference[i]["rotation"][3]

                self.tf_broadcaster.sendTransform(msg) # cam_i to cam_0

    def publisher_3D_callback(self,timer):

        if self.publisher_3D_update is None:
            rospy.sleep(0.2) # publishing for the first time doesn't seem to work without this for some reason
        elif not self.publisher_3D_update:
            return

        self.publisher_3D_cameras()
        self.publisher_3D_landmarks()
        self.publisher_3D_update = False

    def publisher_2D_prepare(self):

        self.publisher_2D_cvbridge = CvBridge()

        self.publisher_2D_pairs = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 8), (7, 9), (8, 10),
                 (5, 11), (6, 12), (11, 13), (12, 14), (13, 15), (14, 16)]
                
        self.publisher_2D_colors_persons = [(57, 167, 169), (102, 40, 149), (155, 0, 0), (255, 222, 0), (197, 255, 68), (211, 206, 171), (255, 121, 0), (169, 94, 132), 
                         (0, 81, 0), (255, 164, 136), (0, 48, 65), (0, 0, 173), (67, 51, 36), (132, 196, 144), (132, 90, 90), (133, 0, 59)]

        self.publisher_2D_color_black = (0,0,0) # (144, 144, 133)
        
        self.publisher_2D_colors_heatmap_levels = 500 # set to one for black keypoints
        assert self.publisher_2D_colors_heatmap_levels > 0

        if self.publisher_2D_colors_heatmap_levels < 2:
            self.publisher_2D_colors_heatmap = [self.publisher_2D_color_black] 
            return

        minimum = 0.0
        
        cMap = plt.get_cmap('jet') # Color map from blue over green to red
        
        cNorm  = colors.Normalize(vmin=minimum, vmax=1.0)
        publisher_2D_scalarMap = cmx.ScalarMappable(norm=cNorm, cmap=cMap)

        step = (((1.0 - minimum) / (self.publisher_2D_colors_heatmap_levels - 1)))

        publisher_2D_colors_heatmap = []
        for i in range(self.publisher_2D_colors_heatmap_levels):
            color = np.multiply(publisher_2D_scalarMap.to_rgba((i * step)/self.person_msg_scores_max)[:-1],255)
            publisher_2D_colors_heatmap.append(color)
        publisher_2D_colors_heatmap = np.asarray(publisher_2D_colors_heatmap)

        self.publisher_2D_colors_heatmap = publisher_2D_colors_heatmap  

        self.publisher_2D_color_good = (0, 155, 0)
        self.publisher_2D_color_bad = self.publisher_2D_color_black
        
        self.publisher_2D_size_gap = 40 # between cams
        self.publisher_2D_size_border = 8 # around cams
        self.publisher_2D_size_radius = 12 # keypoints
        self.publisher_2D_size_line = 4 # between keypoints
        self.publisher_2D_size_text = 2
        self.publisher_2D_size_info = 180 #info area below cams

        assert (self.publisher_2D_size_gap/4) % 1 == 0 and (self.publisher_2D_size_border/2) % 1 == 0 and (self.publisher_2D_size_radius/4) % 1 == 0\
        , str(self.publisher_2D_size_gap/2) + " " + str(self.publisher_2D_size_border/2) + " " + str(self.publisher_2D_size_radius/4) 

        self.publisher_2D_size_quarter_gap = int(self.publisher_2D_size_gap/4)
        self.publisher_2D_size_half_gap = int(self.publisher_2D_size_gap/2)
        self.publisher_2D_size_half_border = int(self.publisher_2D_size_border/2)
        self.publisher_2D_size_half_radius = int(self.publisher_2D_size_radius/2)

        # Create background image of appropriate size
        if self.cams_num <= 4:
            horizontal = self.cams_num
            vertical = 1
        else:
            i = 0
            while(not np.sqrt(self.cams_num+i).is_integer()):
                i += 1
            square = np.sqrt(self.cams_num+i).astype(int)
            horizontal = square 
            vertical = horizontal
            del i, square
            while(horizontal * (vertical-1) >= self.cams_num):
                vertical -= 1
        
        self.publisher_2D_grid_horizontal = horizontal
        self.publisher_2D_grid_vertical = vertical
        
        resolutions = []
        for i in range(self.cams_num):
                resolutions.append(self.extrinsics_reference[i]["resolution"])
        resolutions = np.asarray(resolutions)
        
        self.resolution_max_width = np.max(resolutions[:,0])
        self.resolution_max_height = np.max(resolutions[:,1])
        
        img = 255 * np.ones((self.resolution_max_height*self.publisher_2D_grid_vertical+self.publisher_2D_size_gap*(self.publisher_2D_grid_vertical+3)+self.publisher_2D_size_border*2+self.publisher_2D_size_info,
            self.resolution_max_width*self.publisher_2D_grid_horizontal+self.publisher_2D_size_gap*(self.publisher_2D_grid_horizontal+3)+self.publisher_2D_size_border*2,3), dtype=np.uint8)

        #Legend

        #Frame/Cam rejected
        xy = (self.publisher_2D_size_gap*2+self.publisher_2D_size_border, img.shape[0]-1-self.publisher_2D_size_gap*3-self.publisher_2D_size_half_gap*2)
        
        cv2.putText(img, "Frame/Cam rejected", (xy[0]+self.publisher_2D_size_gap+self.publisher_2D_size_half_gap, xy[1]),
            cv2.FONT_HERSHEY_PLAIN, 1.5, self.publisher_2D_color_black, self.publisher_2D_size_text)
        cv2.rectangle(img, (xy[0], xy[1]-self.publisher_2D_size_half_gap-self.publisher_2D_size_border), 
            (xy[0]+self.publisher_2D_size_gap, xy[1]+self.publisher_2D_size_half_gap-self.publisher_2D_size_border),
            color = self.publisher_2D_color_bad, thickness = self.publisher_2D_size_border, lineType=cv2.LINE_AA)

        #Frame/Cam accepted
        xy = (xy[0], img.shape[0]-1-self.publisher_2D_size_gap*2-self.publisher_2D_size_half_gap)
        
        cv2.putText(img, "Frame/Cam accepted", (xy[0]+self.publisher_2D_size_gap+self.publisher_2D_size_half_gap, xy[1]),
            cv2.FONT_HERSHEY_PLAIN, 1.5, self.publisher_2D_color_black, self.publisher_2D_size_text)
        cv2.rectangle(img, (xy[0], xy[1]-self.publisher_2D_size_half_gap-self.publisher_2D_size_border), 
            (xy[0]+self.publisher_2D_size_gap, xy[1]+self.publisher_2D_size_half_gap-self.publisher_2D_size_border),
            color = self.publisher_2D_color_good, thickness = self.publisher_2D_size_border, lineType=cv2.LINE_AA)

        #Keypoint used
        xy = (xy[0], img.shape[0]-1-self.publisher_2D_size_gap)
        
        cv2.putText(img, "Keypoint accepted", (xy[0]+self.publisher_2D_size_gap+self.publisher_2D_size_half_gap, xy[1]),
            cv2.FONT_HERSHEY_PLAIN, 1.5, self.publisher_2D_color_black, self.publisher_2D_size_text)
        img = cv2.circle(img, (xy[0]+self.publisher_2D_size_half_gap, xy[1]-self.publisher_2D_size_quarter_gap), self.publisher_2D_size_half_gap, self.publisher_2D_color_black, thickness=-1, lineType=cv2.LINE_AA)
        if self.publisher_2D_colors_heatmap_levels > 1:
            img = cv2.circle(img, (xy[0]+self.publisher_2D_size_half_gap, xy[1]-self.publisher_2D_size_quarter_gap), self.publisher_2D_size_quarter_gap, (255,255,255), thickness=-1, lineType=cv2.LINE_AA)
        
        #Heatmap
        if self.publisher_2D_colors_heatmap_levels > 1:
            self.publisher_2D_heatmap_width = self.resolution_max_width+self.publisher_2D_size_border*2 # self.publisher_2D_size_gap*10
            self.publisher_2D_heatmap_height = self.publisher_2D_size_gap*2
            
            self.publisher_2D_heatmap_xy = (self.resolution_max_width * (self.publisher_2D_grid_horizontal-1) + self.publisher_2D_size_gap * (self.publisher_2D_grid_horizontal+1),
                img.shape[0]-1-self.publisher_2D_size_gap*2-self.publisher_2D_heatmap_height+10)
            
            # Colors
            height_level = self.publisher_2D_heatmap_height - 2 * self.publisher_2D_size_border + 2
            width_level = int( ((self.publisher_2D_heatmap_width-2*self.publisher_2D_size_border)) / self.publisher_2D_colors_heatmap_levels)
            for i in range(self.publisher_2D_colors_heatmap_levels):
                xy_level = (self.publisher_2D_heatmap_xy[0]+1+self.publisher_2D_size_border+int(i*(self.publisher_2D_heatmap_width-2*self.publisher_2D_size_border)/self.publisher_2D_colors_heatmap_levels),self.publisher_2D_heatmap_xy[1]+1+self.publisher_2D_size_border)
                cv2.rectangle(img, xy_level, (xy_level[0]+width_level, xy_level[1]+height_level), color = self.color_scalar_to_heatmap(i/(self.publisher_2D_colors_heatmap_levels-1)), 
                    thickness = -1, lineType=cv2.LINE_AA)

            # Frame
            cv2.rectangle(img, (self.publisher_2D_heatmap_xy[0]+self.publisher_2D_size_half_border, self.publisher_2D_heatmap_xy[1]+self.publisher_2D_size_half_border), (self.publisher_2D_heatmap_xy[0]-self.publisher_2D_size_half_border+self.publisher_2D_heatmap_width, self.publisher_2D_heatmap_xy[1]+self.publisher_2D_heatmap_height), color = self.publisher_2D_color_black, 
                thickness = self.publisher_2D_size_border, lineType=cv2.LINE_AA)

            # Text
            cv2.putText(img, "Keypoint scores", (self.publisher_2D_heatmap_xy[0], self.publisher_2D_heatmap_xy[1]-15), cv2.FONT_HERSHEY_PLAIN, 1.5, self.publisher_2D_color_black, self.publisher_2D_size_text)
            
            if not self.person_msg_scores_normalize:
                cv2.putText(img, "0.00", (self.publisher_2D_heatmap_xy[0]+self.publisher_2D_size_border, self.publisher_2D_heatmap_xy[1]+self.publisher_2D_heatmap_height+self.publisher_2D_size_gap), cv2.FONT_HERSHEY_PLAIN, 1.5, self.publisher_2D_color_black, self.publisher_2D_size_text)
                cv2.putText(img, "1.00", (self.publisher_2D_heatmap_xy[0]+self.publisher_2D_heatmap_width-62, self.publisher_2D_heatmap_xy[1]+self.publisher_2D_heatmap_height+self.publisher_2D_size_gap), cv2.FONT_HERSHEY_PLAIN, 1.5, self.publisher_2D_color_black, self.publisher_2D_size_text)
        
        self.publisher_2D_bg_image = img
         
    def publisher_2D(self,msg_original,msg_filtered,frame_used,reasons_per_cam,reason_frame=""):
        
        def draw_humans(img, humans, origin, filtered=False, reason="", scale=-1):

            # print reason of removal
            if reason != "":
                cv2.putText(img, reason, (origin[0]+self.publisher_2D_size_border, origin[1]+self.publisher_2D_size_border*4), cv2.FONT_HERSHEY_PLAIN, 1.5, self.publisher_2D_color_black, self.publisher_2D_size_text)

            # draw cam border to indicate if cam is dropped or used
            if not frame_used:
                cv2.rectangle(img, (origin[0]-self.publisher_2D_size_border, origin[1]-self.publisher_2D_size_border), 
                    (origin[0]+self.resolution_max_width+self.publisher_2D_size_border, origin[1]+self.resolution_max_height+self.publisher_2D_size_border), 
                    color = self.publisher_2D_color_bad, thickness = self.publisher_2D_size_border, lineType=cv2.LINE_AA)
            elif filtered:
                if len(humans) > 0:                  
                    color = self.publisher_2D_color_good
                else:
                    color = self.publisher_2D_color_bad
                cv2.rectangle(img, (origin[0]-self.publisher_2D_size_border, origin[1]-self.publisher_2D_size_border), 
                    (origin[0]+self.resolution_max_width+self.publisher_2D_size_border, origin[1]+self.resolution_max_height+self.publisher_2D_size_border), 
                    color = color, thickness = self.publisher_2D_size_border, lineType=cv2.LINE_AA)
            
            for human in humans:

                centers = {}

                # scale things down for smaller bounding boxes
                x1, y1, x2, y2 = human['bbox']
                if scale == -1:
                    if (x2-x1) * (y2-y1) < self.resolution_max_height * self.resolution_max_height * 0.05:
                        radius_scaled = int(self.publisher_2D_size_radius / 2)
                        half_radius_scaled = int(self.publisher_2D_size_half_radius / 2)
                        line_scaled = int(self.publisher_2D_size_line / 2)
                        scale = 0
                    else:
                        radius_scaled = self.publisher_2D_size_radius
                        half_radius_scaled = self.publisher_2D_size_half_radius
                        line_scaled = self.publisher_2D_size_line
                        scale = 1
                elif scale == 0:    
                    radius_scaled = int(self.publisher_2D_size_radius / 2)
                    half_radius_scaled = int(self.publisher_2D_size_half_radius / 2)
                    line_scaled = int(self.publisher_2D_size_line / 2)
                elif scale == 1:
                    radius_scaled = self.publisher_2D_size_radius
                    half_radius_scaled = self.publisher_2D_size_half_radius
                    line_scaled = self.publisher_2D_size_line

                # draw bounding box and label
                if filtered:

                    #x1, y1, x2, y2 = human['bbox']
                    x1 = np.round(x1).astype(int) - self.publisher_2D_size_half_border + origin[0] - self.publisher_2D_size_half_radius
                    y1 = np.round(y1).astype(int) - self.publisher_2D_size_half_border + origin[1] - self.publisher_2D_size_half_radius
                    x2 = np.round(x2).astype(int) + self.publisher_2D_size_half_border + origin[0] + self.publisher_2D_size_half_radius
                    y2 = np.round(y2).astype(int) + self.publisher_2D_size_half_border + origin[1] + self.publisher_2D_size_half_radius
                    
                    color = self.color_scalar_to_heatmap(human['score'])

                    cv2.rectangle(img, (x1, y1), (x2, y2), color = self.publisher_2D_colors_persons[human['id'] % len(self.publisher_2D_colors_persons)], thickness = self.publisher_2D_size_border)

                    if y1-self.publisher_2D_size_radius*5 < origin[1]:
                        y = y2+self.publisher_2D_size_radius*3
                    else:
                        y = y1-self.publisher_2D_size_radius*3

                    if x1-self.publisher_2D_size_radius*2 < origin[0]:
                        x = x1+self.publisher_2D_size_radius*3
                    elif x1+self.publisher_2D_size_radius*10 > origin[0]+self.resolution_max_width:
                        x = x1-self.publisher_2D_size_radius*7
                    else:
                        x = x1

                    img = cv2.circle(img, (x+self.publisher_2D_size_half_radius, y), self.publisher_2D_size_radius, color = color, thickness=-1, lineType=cv2.LINE_AA)
                    #cv2.putText(img, 'ID: {} - {:.1f}%'.format(max(0,human['id']) , human['score'] * 100), (x1, y1 - 5), cv2.FONT_HERSHEY_PLAIN, 1.5, colors[human['id'] % len(colors)], self.publisher_2D_size_line)
                    cv2.putText(img, '{:.2f}%'.format(human['score'] * 100), (x+self.publisher_2D_size_radius*2+3, y+self.publisher_2D_size_half_radius+3), 
                        cv2.FONT_HERSHEY_PLAIN, 1.5, self.publisher_2D_color_black, self.publisher_2D_size_text)                    

                # analyse
                if filtered:
                    for i in range(self.person_msg_keys_num):
                        if isinstance(human['keypoints'], list):
                            if human['keypoints'][i] is None or human['keypoints'][i][2] == 0.0:
                                continue

                        centers[i] = (np.round(human['keypoints'][i][0]).astype(int)+origin[0], np.round(human['keypoints'][i][1]).astype(int)+origin[1])
                else:

                    
                    #for i in range(self.person_msg_keys_num):
                    for i in self.person_msg_keys_ids:
                        if isinstance(human['keypoints'], list):

                            # normalize or clip scores to match filtered scores
                            if self.person_msg_scores_normalize:

                                if human['keypoints'][i][2] > self.person_msg_scores_max:
                                    if self.print_verbose:
                                        self.print("\tWarning! - Found higher score than expected (in 2D publisher), "
                                            +self.format_num(human['keypoints'][i][2],1,3)+" > "+self.format_num(self.person_msg_scores_max,1,3)
                                            +" - Scores will be normalized accordingly from now!") 
                                    self.person_msg_scores_max = human['keypoints'][i][2] # old frames will be ranked too high in frame selection
                                
                                if human['keypoints'][i][2] < self.person_msg_scores_min:
                                    if self.print_verbose:
                                        self.self.format_num("\tWarning! - Found smaller score than expected (in 2D publisher), "
                                            +self.format_num(human['keypoints'][i][2],1,3)+" < "+self.format_num(self.person_msg_scores_min,1,3)
                                            +" - Scores will be normalized accordingly from now!") 
                                    self.person_msg_scores_min = human['keypoints'][i][2] # old frames will be ranked too low in frame selection

                                if self.person_msg_scores_min != 0.0 or self.person_msg_scores_max != 1.0:
                                    human['keypoints'][i][2] = self.person_msg_scores_min + human['keypoints'][i][2] / (self.person_msg_scores_max-self.person_msg_scores_min)
                            
                            else:

                                clip = False
                                if human['keypoints'][i][2] > 1.0:
                                    if self.print_verbose:
                                        self.print("\tWarning! - Found higher score than expected (in 2D publisher), "
                                            +self.format_num(human['keypoints'][i][2],1,3)+" > 1.0"
                                            +" - Score will be clipped at 1.0!")
                                    clip = True

                                if human['keypoints'][i][2] < 0.0:
                                    if self.print_verbose:
                                        self.self.format_num("\tWarning! - Found smaller score than expected (in 2D publisher), "
                                            +self.format_num(human['keypoints'][i][2],1,3)+" < 0.0"
                                            +" - Score will be clipped at 0.0!")
                                    clip = True

                                if clip:
                                    human['keypoints'][i][2] = np.clip(human['keypoints'][i][2],0.0,1.0) # this should be avoided. when scores exceed this interval too much, use normalizing
                                    
                            if human['keypoints'][i][2] == 0.0:
                                continue

                            centers[i] = (np.round(human['keypoints'][i][0]).astype(int)+origin[0], np.round(human['keypoints'][i][1]).astype(int)+origin[1])

                # draw lines
                if not filtered:
                    for pair_order, pair in enumerate(self.publisher_2D_pairs):
                        if pair[0] not in centers.keys() or pair[1] not in centers.keys():
                            continue
                        
                        color_a = self.color_scalar_to_heatmap(human['keypoints'][pair[0]][2])
                        color_b = self.color_scalar_to_heatmap(human['keypoints'][pair[1]][2])
                        color = np.add(color_a,color_b) * 0.5

                        img = cv2.line(img, centers[pair[0]], centers[pair[1]], color = color, thickness = line_scaled)

                # draw point
                for i in range(self.person_msg_keys_num):
                    if i not in centers.keys():
                        continue

                    if not filtered:
                        if self.publisher_2D_colors_heatmap_levels > 1:
                            color = self.color_scalar_to_heatmap(human['keypoints'][i][2])
                            img = cv2.circle(img, centers[i], radius_scaled, color, thickness=-1, lineType=cv2.LINE_AA)
                        else:
                            img = cv2.circle(img, centers[i], radius_scaled, self.publisher_2D_color_black, thickness=-1, lineType=cv2.LINE_AA)
                            img = cv2.circle(img, centers[i], half_radius_scaled, (255,255,255), thickness=-1, lineType=cv2.LINE_AA)
                    else:
                        if self.publisher_2D_colors_heatmap_levels > 1:
                            img = cv2.circle(img, centers[i], radius_scaled, self.publisher_2D_color_black, thickness=-1, lineType=cv2.LINE_AA)
                            color = self.color_scalar_to_heatmap(human['keypoints'][i][2])
                            img = cv2.circle(img, centers[i], half_radius_scaled, color, thickness=-1, lineType=cv2.LINE_AA)
                        else:
                            img = cv2.circle(img, centers[i], half_radius_scaled, self.publisher_2D_color_black, thickness=-1, lineType=cv2.LINE_AA)

            return img, scale

        bg_image = copy.deepcopy(self.publisher_2D_bg_image)

        for i in range(0,self.cams_num):
            humans_original = [{'id': -1, 'score': p.score, 'bbox': p.bbox, 'keypoints': [[kp.x, kp.y, kp.score] for kp in p.keypoints]} for p in msg_original[i].persons]
            if frame_used:
                humans_filtered = [{'id': p.id, 'score': p.score, 'bbox': p.bbox, 'keypoints': [[kp.x, kp.y, kp.score] for kp in p.keypoints]} for p in msg_filtered[i].persons]
            if i == 0:
                xy = (2*self.publisher_2D_size_gap+self.publisher_2D_size_border,2*self.publisher_2D_size_gap+self.publisher_2D_size_border)
                img, scale = draw_humans(bg_image, humans_original, xy, reason = reasons_per_cam[i])
                if frame_used:
                    img = draw_humans(bg_image, humans_filtered, xy, filtered=True, scale=scale)[0]
            else:
                xy = (self.resolution_max_width * (i % self.publisher_2D_grid_horizontal) + self.publisher_2D_size_gap * ((i % self.publisher_2D_grid_horizontal)+2) + self.publisher_2D_size_border, 
                    self.resolution_max_height * np.floor_divide(i,self.publisher_2D_grid_horizontal).astype(int) + self.publisher_2D_size_gap * (np.floor_divide(i,self.publisher_2D_grid_horizontal).astype(int)+2) + self.publisher_2D_size_border)
                img, scale = draw_humans(img, humans_original, xy, reason = reasons_per_cam[i])
                if frame_used:
                    img = draw_humans(img, humans_filtered, xy, filtered=True, scale=scale)[0]

        # Global border to indicate if frame is dropped or used
        if frame_used:
            color = self.publisher_2D_color_good
        else:
            color = self.publisher_2D_color_bad
        cv2.rectangle(img, (self.publisher_2D_size_gap+self.publisher_2D_size_half_border,self.publisher_2D_size_gap+self.publisher_2D_size_half_border), 
            (self.publisher_2D_bg_image.shape[1]-1-self.publisher_2D_size_half_border-self.publisher_2D_size_gap,
             self.publisher_2D_bg_image.shape[0]-1-self.publisher_2D_size_half_border-self.publisher_2D_size_gap-self.publisher_2D_size_info), 
            color = color, thickness = self.publisher_2D_size_border*2, lineType=cv2.LINE_AA)

        # Info
        if reason_frame != "":
            reason_frame = "> "+reason_frame
        cv2.putText(img, reason_frame, (self.publisher_2D_size_gap*2+self.publisher_2D_size_border+350, self.publisher_2D_bg_image.shape[0]-self.publisher_2D_size_gap*3-self.publisher_2D_size_half_gap*2-1), 
            cv2.FONT_HERSHEY_PLAIN, 1.5, self.publisher_2D_color_black, self.publisher_2D_size_text)

        if self.person_msg_scores_normalize:
            cv2.putText(img, self.format_num(self.person_msg_scores_min,1,2)[1:], (self.publisher_2D_heatmap_xy[0]+self.publisher_2D_size_border, self.publisher_2D_heatmap_xy[1]+self.publisher_2D_heatmap_height+self.publisher_2D_size_gap), cv2.FONT_HERSHEY_PLAIN, 1.5, self.publisher_2D_color_black, self.publisher_2D_size_text)
            cv2.putText(img, self.format_num(self.person_msg_scores_max,1,2)[1:], (self.publisher_2D_heatmap_xy[0]+self.publisher_2D_heatmap_width-62, self.publisher_2D_heatmap_xy[1]+self.publisher_2D_heatmap_height+self.publisher_2D_size_gap), cv2.FONT_HERSHEY_PLAIN, 1.5, self.publisher_2D_color_black, self.publisher_2D_size_text)
  
         # Convert to msg and publish
        img_msg = self.publisher_2D_cvbridge.cv2_to_imgmsg(img, "rgb8")
        img_msg.header = msg_original[0].header
        self.img_pub.publish(img_msg)

    # Filereaders
    def file_reader_intrinsics(self):
        
        path = self.get_path(self.intrinsics_file)
        self.print("Reading camera intrinsics from '"+path+"'...",stamp=True,flush=True)

        # Open file
        try:
            with open(path, 'r') as file:
                data = yaml.load(file, Loader=yaml.FullLoader)
        except:
            self.print("cannot open file!\n")
            os._exit(1)

        # Read file
        self.intrinsics_cal3s2 = []
        self.intrinsics_caminfomsg = []
        self.found_IDs_intrinsics = []
        warned = False
        for i in self.cam_ID_range:

            resolution_x, resolution_y, focal_x, focal_y, priciple_x, principle_y = None, None, None, None, None, None

            try:
                camera_info_msg = CameraInfo()
                         
                if self.extrinsics_estimate_generate:
                    target_resolution = self.extrinsics_reference[i]["resolution"]
                else:
                    target_resolution = self.extrinsics_estimate[i]["resolution"]
                
                camera_info_msg.width = target_resolution[0]
                camera_info_msg.height = target_resolution[1]
                
                resolution_x = data["cam_"+str(i)]["resolution"][0]
                resolution_y = data["cam_"+str(i)]["resolution"][1]
                resolution = (resolution_x,resolution_y)

                scale = ((target_resolution[0] / resolution[0]) + (target_resolution[1] / resolution[1])) / 2 

                if self.person_msg_coords_undistort:
                    try:
                        camera_info_msg.D = data["cam_"+str(i)]["distortion_coeffs"]
                    except:
                        if self.person_msg_coords_undistort:
                            if not warned:
                                self.print()
                                warned = True
                            self.print("\tFound 0 distortion coefficients for cam_"+str(i)+". Undistortion will not be applied for this camera.")
                        camera_info_msg.D = []

                    if not len(camera_info_msg.D) in [0,4,5,8,12,14]:
                        if not warned:
                            self.print()
                            warned = True
                        self.print("\tFound "+str(len(camera_info_msg.D))+" distortion coefficients for cam_"+str(i)+", which is not supported (4,5,8,12,14). Undistortion will not be applied for this camera.") 
                        camera_info_msg.D = []
                else:
                    camera_info_msg.D = []  

                focal_x = data["cam_"+str(i)]["intrinsics"][0]
                focal_y = data["cam_"+str(i)]["intrinsics"][1]

                priciple_x = data["cam_"+str(i)]["intrinsics"][2] 
                principle_y = data["cam_"+str(i)]["intrinsics"][3]

                camera_info_msg.K[0] = focal_x * scale
                camera_info_msg.K[1] = 0.0
                camera_info_msg.K[2] = priciple_x * scale
                camera_info_msg.K[3] = 0.0
                camera_info_msg.K[4] = focal_y * scale
                camera_info_msg.K[5] = principle_y * scale
                camera_info_msg.K[6] = 0.0
                camera_info_msg.K[7] = 0.0
                camera_info_msg.K[8] = 1.0
                
                self.intrinsics_caminfomsg.append(camera_info_msg)
                self.intrinsics_cal3s2.append(gtsam.Cal3_S2(camera_info_msg.K[0], camera_info_msg.K[4], 0.0, camera_info_msg.K[2], camera_info_msg.K[5])) # fx,fy,skew,ppx,ppy

                self.found_IDs_intrinsics.append(i)

            except Exception as e:

                partials = 0
                partials += 1 if not resolution_x is None else partials
                partials += 1 if not resolution_y is None else partials
                partials += 1 if not focal_x is None else partials
                partials += 1 if not focal_y is None else partials
                partials += 1 if not priciple_x is None else partials
                partials += 1 if not principle_y is None else partials

                if partials > 0:
                    self.print("file content is wrong at 'cam_"+str(i)+"'! - "+str(e))
                    os._exit(1)

        self.print("done! - Found "+str(len(self.found_IDs_intrinsics))+" entries: "+str(self.found_IDs_intrinsics))

    def file_reader_extrinsics_estimate(self):
        
        path = self.get_path(self.extrinsics_estimate_file)
        self.print("Reading estimate camera extrinsics from '"+path+"'...",stamp=True,flush=True)

        # Open file
        try:
            with open(path, 'r') as file:
                data = yaml.load(file, Loader=yaml.FullLoader)
        except:
            self.print("cannot open file!\n")
            os._exit(1)

        # Read file
        self.extrinsics_estimate = []
        self.found_IDs_estimate = []
        for i in self.cam_ID_range:

            topic, origin, name, resolution, stamp, translation, rotation = None, None, None, None, None, None, None

            try:
                topic = data["cam_"+str(i)]["topic"]
                origin = data["cam_"+str(i)]["origin"]
                name = data["cam_"+str(i)]["name"]
                resolution = data["cam_"+str(i)]["resolution"]
                stamp = rospy.Time.now()       
                translation = data["cam_"+str(i)]["translation"]
                rotation = data["cam_"+str(i)]["rotation"]
                self.extrinsics_estimate.append({"topic":topic, "origin":origin, "name":name, "resolution":resolution, "stamp":stamp, "translation":translation, "rotation":rotation})
                
                self.found_IDs_estimate.append(i)

            except Exception as e:

                partials = 0
                partials += 1 if not topic is None else partials
                partials += 1 if not origin is None else partials
                partials += 1 if not name is None else partials
                partials += 1 if not resolution is None else partials
                partials += 1 if not stamp is None else partials
                partials += 1 if not translation is None else partials
                partials += 1 if not rotation is None else partials

                if partials > 0:
                    self.print("file content is wrong at 'cam_"+str(i)+"'! - "+str(e))
                    os._exit(1)

        self.print("done! - Found "+str(len(self.found_IDs_estimate))+" entries: "+str(self.found_IDs_estimate))

    def file_reader_extrinsics_reference(self):
        
        path = self.get_path(self.extrinsics_reference_file)
        self.print("Reading reference camera extrinsics from '"+path+"'...",stamp=True,flush=True)

        # Open file
        try:
            with open(path, 'r') as file:
                data = yaml.load(file, Loader=yaml.FullLoader)
        except:
            self.print("cannot open file!\n")
            os._exit(1)

        # Read file
        self.extrinsics_reference = []
        self.found_IDs_reference = []
        for i in self.cam_ID_range:
            topic, origin, translation, rotation, name, resolution = None, None, None, None, None, None
            try:
                topic = data["cam_"+str(i)]["topic"]
                origin = data["cam_"+str(i)]["origin"]
                stamp = rospy.Time.now()       
                translation = np.asarray(data["cam_"+str(i)]["translation"])
                rotation = np.asarray(data["cam_"+str(i)]["rotation"])

                if self.extrinsics_estimate_generate:
                    name = data["cam_"+str(i)]["name"]
                    resolution = data["cam_"+str(i)]["resolution"]
                    self.extrinsics_reference.append({"topic":topic, "origin":origin, "name":name, "resolution":resolution, "stamp":stamp, "translation":translation, "rotation":rotation})
                else:
                    self.extrinsics_reference.append({"topic":topic, "origin":origin, "stamp":stamp, "translation":translation, "rotation":rotation})
                
                self.found_IDs_reference.append(i)

            except Exception as e:

                partials = 0
                partials += 1 if not topic is None else partials
                partials += 1 if not origin is None else partials
                partials += 1 if not translation is None else partials
                partials += 1 if not rotation is None else partials

                if self.extrinsics_estimate_generate:
                    partials += 1 if not name is None else partials
                    partials += 1 if not resolution is None else partials
                   
                if partials > 0:
                    self.print("file content is wrong at 'cam_"+str(i)+"'! - "+str(e))
                    os._exit(1)

        self.print("done! - Found "+str(len(self.found_IDs_reference))+" entries: "+str(self.found_IDs_reference))
    
    # Filewriters
    def file_writer_extrinsics_estimate(self,noise_translation,noise_rotation):

        # retract poses
        self.print("Retracting reference translation within +/- "+str(np.round(noise_translation,2))+"m "+("(exactly)" if self.extrinsics_estimate_generate_noise_translation_matchnoise else "(gaussian)")
            +" and rotation within +/- "+str(np.round(noise_rotation,2))+"° "+("(exactly)" if self.extrinsics_estimate_generate_noise_rotation_matchnoise else "(gaussian)")+" to use as estimate...",stamp=True,flush=True)
        extrinsics = [self.extrinsics_reference[0]]

        for i in range(1,len(self.extrinsics_reference)):
            topic = self.extrinsics_reference[i]["topic"]
            origin = self.extrinsics_reference[i]["origin"]
            name = self.extrinsics_reference[i]["name"]
            resolution = self.extrinsics_reference[i]["resolution"]
            pose_rotation = gtsam.Rot3(R.from_quat(self.extrinsics_reference[i]["rotation"]).as_matrix())
            pose_gt = gtsam.Pose3(pose_rotation,self.extrinsics_reference[i]["translation"])
            
            pose = pose_gt.retract(np.concatenate((np.deg2rad(noise_rotation)*np.random.randn(3,1)/3.0,noise_translation*np.random.randn(3,1)*0.577356)))
            
            if noise_translation == 0.0:
                translation = self.extrinsics_reference[i]["translation"]  
            elif self.extrinsics_estimate_generate_noise_translation_matchnoise:
                scale = noise_translation / np.linalg.norm(pose.translation()-self.extrinsics_reference[i]["translation"])
                translation = self.extrinsics_reference[i]["translation"] + (pose.translation()-self.extrinsics_reference[i]["translation"]) * scale
            else:
                translation = pose.translation()

            if noise_rotation == 0.0:
                rotation = self.extrinsics_reference[i]["rotation"]
            elif self.extrinsics_estimate_generate_noise_rotation_matchnoise:
                rotation = R.from_matrix(pose.rotation().matrix()).as_quat()
                n = 0
                n_max = 5           #
                accuracy = 0.0001   #
                assert accuracy < noise_rotation , "Choose smaller accuracy value for such small noise_rotation"
                while(np.abs(self.distance_rotation(rotation,self.extrinsics_reference[i]["rotation"])-noise_rotation) > accuracy and n < n_max):
                    if n == n_max-1:
                        n=0
                        #rotation = R.from_matrix(pose_gt.retract(np.concatenate((np.deg2rad(noise_rotation)*np.random.randn(3,1)/3.0,noise_translation*np.random.randn(3,1)*0.577356))).rotation().matrix()).as_quat()
                        pose = pose_gt.retract(np.concatenate((np.deg2rad(noise_rotation)*np.random.randn(3,1)/3.0,noise_translation*np.random.randn(3,1)*0.577356)))
                    def f(x):
                        return np.abs(self.distance_rotation(self.extrinsics_reference[i]["rotation"], R.from_euler("xyz",R.from_matrix(pose.rotation().matrix()).as_euler("xyz",degrees=True)*x,degrees=True).as_quat())-noise_rotation)
                    scale = minimize_scalar(f).x
                    rotation = R.from_euler("xyz",R.from_matrix(pose.rotation().matrix()).as_euler("xyz",degrees=True)*scale,degrees=True).as_quat()
                    n+=1
                #print(str(i)+" "+str(n)+" "+str(np.abs(self.distance_rotation(rotation,self.extrinsics_reference[i]["rotation"])-noise_rotation)))
            else:
                rotation = R.as_quat(R.from_matrix(pose.rotation().matrix()))

            extrinsics.append({"topic":topic, "origin":origin, "name":name, "resolution":resolution, "translation":translation, "rotation":rotation})
        self.print("done!")

        self.extrinsics_estimate = extrinsics

        # write estimate
        text = {}       
        for i in range(0,len(extrinsics)):
            text['cam_'+str(self.found_IDs_reference[i])] = {
                'topic': extrinsics[i]['topic'],
                'origin': extrinsics[i]['origin'],
                'resolution': extrinsics[i]['resolution'],
                'name': extrinsics[i]['name'],
                'translation': (np.round(extrinsics[i]['translation'],19).tolist() if i!=0 else [0.0, 0.0, 0.0]),
                'rotation': (np.round(extrinsics[i]['rotation'],19).tolist() if i!=0 else [0.0, 0.0, 0.0, 1.0])}
        path = self.get_path(self.extrinsics_estimate_file)
        self.print("Writing retracted camera extrinsics to '"+path+"'...",stamp=True,flush=True)
        try:
            with open(path, 'w') as file:
                yaml.dump(text, file, indent=2, sort_keys=False, default_flow_style=None, canonical=False)
        except Exception as e:
            self.print("failed!\n")
            self.print(e)
            os._exit(1)
        self.print("done!")

    def file_writer_extrinsics_result(self):

        path = self.get_path(self.log_extrinsics_result_file,folder=self.log_dir)
        self.print("Writing final camera extrinsics to '"+path+"'...",flush=True,stamp=True)

        text = {}       
        for i in range(0,len(self.extrinsics_history[-1])):
            text['cam_'+str(i+1)] = {
                'topic': self.extrinsics_estimate[i]['topic'],
                'origin': self.extrinsics_estimate[i]['origin'],
                'resolution': self.extrinsics_estimate[i]['resolution'],
                'name': self.extrinsics_estimate[i]['name'],
                'translation': (np.round(self.extrinsics_history[-1][i]['translation'],19).tolist() if i!=0 else [0.0, 0.0, 0.0]),
                'rotation': (np.round(self.extrinsics_history[-1][i]['rotation'],19).tolist() if i!=0 else [0.0, 0.0, 0.0, 1.0])}

        try:
            with open(path, 'w') as file:
                yaml.dump(text, file, indent=2, sort_keys=False, default_flow_style=None, canonical=False)
            self.print("done!")

        except Exception as e:
            self.print("failed!")
            if self.print_verbose:
                self.print(e)

    def file_writer_extrinsics_history(self):

        path = self.get_path(self.log_history_file,folder=self.log_dir)
        self.print("Writing history of camera extrinsics to '"+path+"'...",flush=True,stamp=True)

        text = {}
        for n in range(0,len(self.extrinsics_history)):

            text['iteration_'+str(n)] = {}

            for i in range(0,len(self.extrinsics_history[n])):

                text['iteration_'+str(n)]['cam_'+str(i+1)] = {
                    'timestamp': self.extrinsics_history[n][i]['stamp'].to_sec(),
                    'topic': self.extrinsics_history[n][i]['topic'],
                    'origin': self.extrinsics_history[n][i]['origin'],
                    'translation': (np.round(self.extrinsics_history[n][i]['translation'],19).tolist() if i!=0 else [0.0, 0.0, 0.0]),
                    'rotation': (np.round(self.extrinsics_history[n][i]['rotation'],19).tolist() if i!=0 else [0.0, 0.0, 0.0, 1.0])}

        try:
            with open(path, 'w') as file:
                yaml.dump(text, file, indent=2, sort_keys=False, default_flow_style=None, canonical=False)
            self.print("done!")

        except Exception as e:
            self.print("failed!")
            if self.print_verbose:
                self.print(e)

    def file_writer_log(self):
        
        path = self.get_path(self.log_terminal_file,folder=self.log_dir)
        self.print("Writing log to '"+path+"'...",flush=True,stamp=True)

        try:
            file = open(path,"w+")
            file.write(self.log)
            file.close()
            self.print("done!")

        except Exception as e:
            self.print("failed!")
            if self.print_verbose:
                self.print(e)
    
    # Printers
    def print(self,msg="",stamp=False,flush=False,logonly=False,status=False):

        msg = str(msg)

        if not logonly:
            
            try:
                self.print_last_was_status
            except:
                self.print_last_was_status = False

            if self.print_last_was_status:
                for i in range(0,self.status_length_max):
                    print("\b \b",end="")
                print("\r",end='')

            if status:
                print("\r"+msg, end="")
                self.print_last_was_status = True
            elif flush:
                print(msg, end="", flush=True)
                self.print_last_was_status = False
            else:
                print(msg)
                self.print_last_was_status = False

        if stamp:
            stamp = "["+str(datetime.datetime.now())[2:-3]+"] - "
            if msg[0] == "\t" or msg[0] == "\n":
                msg = msg[0] + stamp + msg[1:]
            else:
                msg = stamp + msg

        if self.log_terminal_flag and not status:
            try:
                self.log += msg + ("\n" if not flush else "")
            except Exception as e:
                self.log = msg + ("\n" if not flush else "")
    
    def printer_status(self,timer):

        if self.first_msg_received and not self.shutdown:

            successes = np.sum(self.frame_success)
            fails = len(self.frame_success)-successes
            if fails == 0:
                success_rate = 100
            else:
                success_rate = np.round( (successes / (successes+fails)) * 100,2)        
            
            processing_time = rospy.get_time() - self.timestamp_processing_begin
            
            if self.frame_counter_total > 0:
                fps_total = 1.0 / ( (self.timestamp_lastframe - self.timestamp_processing_begin) / self.frame_counter_total )
            else:
                fps_total = 0
            if self.frame_counter_eligble > 0:
                fps_eligble = 1.0 / ( (self.timestamp_lastframe - self.timestamp_processing_begin) / self.frame_counter_eligble )
            else:
                fps_eligble = 0
            if len(self.frame_success) == 0:
                fps_optimization = 0
            else:
                fps_optimization = 1.0 / ( processing_time / len(self.frame_success) )

            try:
                last_iput = "T-: "+self.format_num(self.timestamp_lastframe-(processing_time+self.timestamp_processing_begin),3,1)+"s"
            except:
                last_iput = "T-: -"
            time_processed = "T: "+self.format_num(processing_time,4,1)[1:]+"s"
            frames_total = "F: "+self.format_num(self.frame_counter_total,4,1)[1:-2]+" ("+self.format_num(fps_total,2,1)[1:]+"FPS)"
            frames_filtered = "F-: "+self.format_num(self.frame_counter_eligble,4,1)[1:-2]+" ("+self.format_num(fps_eligble,2,1)[1:]+"FPS)"
            successes_fails = "+/-: "+self.format_num(successes,4,1)[1:-2]+" / "+self.format_num(fails,3,1)[1:-2]+" ("+self.format_num(success_rate,3,2)[1:4]+"%)"
            fps_optimizations = "OPS: "+self.format_num(fps_optimization,1,1)[1:]

            if self.extrinsics_reference_readfromfile:
                distributed_error = self.format_num(self.average_distibuted_pos_errors[-1],1,3)[1:]+"m"
                avg_dist_err = self.format_num(self.global_error_translation_distance[-1] / (self.cams_num-1),1,3)[1:]+"m"
                avg_rot_err = self.format_num(np.sum(self.global_error_rotation_distance[-1]) / (self.cams_num-1),2,2)[1:]+"°"
                errors = "Err.: "+distributed_error+" (Umey.) / "+avg_dist_err+" (Raw) / "+avg_rot_err
            else:
                errors = ""

            covariance1 = "- / - (Meas.)"
            if len(self.covariance_measurements) > 0:
                covariance1 = np.sqrt(np.abs(np.diagonal(np.asarray(self.covariance_measurements[-1][1:]),0,2)))
                covariance_rotation = self.format_num(np.rad2deg(np.mean(covariance1[:,0:3])),2,3)[1:]+"°"
                covariance_translation = self.format_num(np.mean(covariance1[:,3:]),1,5)[1:]+"m"
                covariance1 = covariance_translation+" / "+covariance_rotation+" (Meas.)" # std

            covariance2 = "- / - (Pred.)"
            if len(self.covariance_predictions) > 0:
                covariance2 = np.sqrt(np.abs(np.diagonal(np.asarray(self.covariance_predictions[-1][1:]),0,2)))
                covariance_rotation = self.format_num(np.rad2deg(np.mean(covariance2[:,0:3])),2,3)[1:]+"°"
                covariance_translation = self.format_num(np.mean(covariance2[:,3:]),1,5)[1:]+"m"
                covariance2 = covariance_translation+" / "+covariance_rotation+" (Pred.)" # std
                
            gap1 = " \033[35m|\033[m " #purple
            gap2 = " \033[31m||\033[m " #red
            status = time_processed+gap1+last_iput+gap1+frames_total+gap1+frames_filtered+gap2+successes_fails+gap1+fps_optimizations+gap2+errors+(gap2 if self.extrinsics_reference_readfromfile else "")+covariance1+gap1+covariance2
            
            self.status_length_max = max(len(status),self.status_length_max)

            if not self.shutdown:
                self.print(status,status=True)
                self.status_printed = True

    def printer_report(self):

        self.print("")
        self.print("Report:",stamp=True)

        processing_time = self.timestamp_processing_end - self.timestamp_processing_begin
        if self.frame_counter_total > 0:
            average_processing_time = (self.timestamp_processing_end - self.timestamp_processing_begin)/self.frame_counter_total
        else:
            average_processing_time = 0
        self.print("Processing time: "+str(np.round(processing_time,3))+"s | Processed frame-sets: "+str(self.frame_counter_total)+" | Ratio: "+str(np.round(average_processing_time,3))+"s | Throughput "+(str(np.round(1.0/average_processing_time,2)) if average_processing_time > 0 else "-")+" FPS")

        if len(self.times_optimization) > 0:
            average_optimization_time = np.sum(self.times_optimization)/len(self.times_optimization)
        else:
            average_optimization_time = 0
        successes = np.sum(self.frame_success)
        self.print("Average optimization time: "+("-" if len(self.times_optimization) == 0 else str(np.round(average_optimization_time,3))+"s | "
            +(str(np.round(1.0 / ( (self.timestamp_processing_end - self.timestamp_processing_begin) / successes ),2)) if successes > 0 else "-" )+" OPS | "
            +(str(np.round((np.sum(self.times_optimization)/processing_time)*100,2)) if processing_time > 0 else "-")+"% of processing time"))    

        fails = len(self.frame_success)-successes
        if fails == 0:
            success_rate = 100
        else:
            success_rate = np.round( (successes / (successes+fails)) * 100,2)
        self.print("Optimizations/Failures: "+str(successes)+"/"+str(fails)+" | Success rate: "+str(success_rate)+"%")
        
        self.print("Projections per camera: "+str(list(self.cam_counters)))
        self.print("Found person hypotheses: "+str(self.hypotheses_num))
        if self.extrinsics_reference_readfromfile:
            self.print("Distributed position error: "+self.format_num(self.average_distibuted_pos_errors[-1],1,3)[1:]+"m | Raw Position error: "+str(np.round(self.global_error_translation_distance[-1]/(self.cams_num-1),3))+"m | Orientation error: "+str(np.round(self.global_error_rotation_distance[-1]/(self.cams_num-1),3))+"°")

    def printer_extrinsics_raw(self):
        
        if len(self.extrinsics_history[-1]) == self.cams_num:
            
            self.print("\n---------------------------------------<Calibration>-----------------------------------------")
            for i in range(0,self.cams_num):
                self.print(self.extrinsics_estimate[i]["name"]+":")
                self.print("\tTopic:       "+str(self.extrinsics_history[-1][i]["topic"]))
                self.print("\tOrigin:      "+str(self.extrinsics_history[-1][i]["origin"]))
                self.print("\tTimestamp:   "+str(self.extrinsics_history[-1][i]["stamp"].secs)+"."+str(self.extrinsics_history[-1][i]["stamp"].nsecs)+" (s.ns)")
                self.print("\tPosition:    "+str(self.extrinsics_history[-1][i]["translation"])+" (x,y,z in meters)")
                self.print("\tOrientation: "+str(self.extrinsics_history[-1][i]["rotation"])+" (x,y,z,w as quaternion)")
                self.print('\t          ": '+str(R.from_quat(self.extrinsics_history[-1][i]["rotation"]).as_euler('xyz', degrees=True))+" (x,y,z in degrees)")
                self.print("\tResolution:  "+str(self.intrinsics_caminfomsg[i].width)+" x "+str(self.intrinsics_caminfomsg[i].height))
                self.print("\tCal. Matrix: "+str(self.intrinsics_caminfomsg[i].K))
                self.print("\tDist. Coef.: "+str(self.intrinsics_caminfomsg[i].D))
            self.print("---------------------------------------</Calibration>----------------------------------------")

    def printer_extrinsics_error(self):
        
        label = "<Error>"
        ints = 3
        decs = 3
        gap = 5 * " "

        length = 34+7*len(gap)+4*ints+4*decs #29+7*gap+4*ints+4*decs

        if (length - len(label) ) % 2 == 1:
            pad_l = (int)((length-1-len(label))/2)
            pad_r = pad_l * "-" + "-"
            pad_l = pad_l * "-"
        else:
            pad_l = (int)((length-len(label))/2) * "-"
            pad_r = pad_l        

        if len(self.extrinsics_error[-1])==self.cams_num:

            self.print()
            self.print(""+pad_l+label+pad_r)
            
            for i in range(0,self.cams_num):
                self.print(self.extrinsics_estimate[i]["name"]+":")
                trans = self.extrinsics_error[-1][i]["translation"]
                self.print(gap+"Translation:"+gap+self.format_num(trans[0],ints,decs)+"m"+gap+self.format_num(trans[1],ints,decs)+"m"+gap+self.format_num(trans[2],ints,decs)+"m"
                    +gap+"|"+gap+"Distance:"+gap+self.format_num(self.extrinsics_error[-1][i]["translation_distance"],ints,decs)+"m")
                rot = self.extrinsics_error[-1][i]["rotation"]
                self.print(gap+"Rotation:   "+gap+self.format_num(rot[0],ints,decs)+"°"+gap+self.format_num(rot[1],ints,decs)+"°"+gap+self.format_num(rot[2],ints,decs)+"°"
                    +gap+"|"+gap+'       ":'+gap+self.format_num(self.extrinsics_error[-1][i]["rotation_distance"],ints,decs)+"°")
            
            self.print(length*"_")
            self.print("Accumulated:")
            trans = self.global_error_translation[-1]
            self.print(gap+"Translation:"+gap+self.format_num(trans[0],ints,decs)+"m"+gap+self.format_num(trans[1],ints,decs)+"m"+gap+self.format_num(trans[2],ints,decs)+"m"
                        +gap+"|"+gap+"Distance:"+gap+self.format_num(self.global_error_translation_distance[-1],ints,decs)+"m"
                        +gap+"Avg.: "+self.format_num(self.global_error_translation_distance[-1] / (self.cams_num-1),3,3)+"m")
            rot = self.global_error_rotation[-1]
            self.print(gap+"Rotation:   "+gap+self.format_num(rot[0],ints,decs)+"°"+gap+self.format_num(rot[1],ints,decs)+"°"+gap+self.format_num(rot[2],ints,decs)+"°"
                        +gap+"|"+gap+'       ":'+gap+self.format_num(self.global_error_rotation_distance[-1],ints,decs)+"°"
                        +gap+'   ": '+self.format_num(self.global_error_rotation_distance[-1] / (self.cams_num-1),3,3)+"°")
            
            self.print(length*"-")

    def printer_extrinsics_delta(self):
        
        label = "<Error-Delta>"
        ints = 3
        decs = 3
        gap = 5 * " "

        length = 34+7*len(gap)+4*ints+4*decs #29+7*gap+4*ints+4*decs

        if (length - len(label) ) % 2 == 1:
            pad_l = (int)((length-1-len(label))/2)
            pad_r = pad_l * "-" + "-"
            pad_l = pad_l * "-"
        else:
            pad_l = (int)((length-len(label))/2) * "-"
            pad_r = pad_l  

        if len(self.extrinsics_error[0]) == self.cams_num:

            self.print()
            self.print(""+pad_l+label+pad_r)                                                
            
            for i in range(0,self.cams_num):
                self.print(self.extrinsics_estimate[i]["name"]+":")
                trans = np.subtract(np.abs(self.extrinsics_error[-1][i]["translation"]),np.abs(self.extrinsics_error[0][i]["translation"])) # problem
                self.print(gap+"Translation:"+gap+self.format_num(trans[0],ints,decs)+"m"+gap+self.format_num(trans[1],ints,decs)+"m"+gap+self.format_num(trans[2],ints,decs)+"m"
                    +gap+"|"+gap+"Distance:"+gap
                    +self.format_num(np.subtract(self.extrinsics_error[-1][i]["translation_distance"],self.extrinsics_error[0][i]["translation_distance"]),ints,decs)+"m")
                rot = np.subtract(np.abs(self.extrinsics_error[-1][i]["rotation"]),np.abs(self.extrinsics_error[0][i]["rotation"])) # problem
                self.print(gap+"Rotation:   "+gap+self.format_num(rot[0],ints,decs)+"°"+gap+self.format_num(rot[1],ints,decs)+"°"+gap+self.format_num(rot[2],ints,decs)+"°"
                    +gap+"|"+gap+'       ":'+gap+self.format_num(np.subtract(self.extrinsics_error[-1][i]["rotation_distance"],self.extrinsics_error[0][i]["rotation_distance"]),ints,decs)+"°")
            
            trans = np.subtract(self.global_error_translation[-1],self.global_error_translation[0])
            trans_dist = np.subtract(self.global_error_translation_distance[-1],self.global_error_translation_distance[0])
            rot = np.subtract(self.global_error_rotation[-1],self.global_error_rotation[0])
            rot_dist = np.subtract(self.global_error_rotation_distance[-1],self.global_error_rotation_distance[0])

            self.print(length*"_")
            self.print("Accumulated:")
            self.print(gap+"Translation:"+gap+self.format_num(trans[0],ints,decs)+"m"+gap+self.format_num(trans[1],ints,decs)+"m"+gap+self.format_num(trans[2],ints,decs)+"m"
                        +gap+"|"+gap+"Distance:"+gap+self.format_num(trans_dist,ints,decs)+"m"+gap+"Avg.: "+self.format_num(trans_dist / (self.cams_num-1),3,3)+"m")
            self.print(gap+"Rotation:   "+gap+self.format_num(rot[0],ints,decs)+"°"+gap+self.format_num(rot[1],ints,decs)+"°"+gap+self.format_num(rot[2],ints,decs)+"°"
                        +gap+"|"+gap+'       ":'+gap+self.format_num(rot_dist,ints,decs)+"°"+gap+'   ": '+self.format_num(rot_dist / (self.cams_num-1),3,3)+"°")
            
            self.print(length*"-")
     
    # Tools
    def distance_rec2rec(self,r0,r1):

        assert r0[0] <= r0[2]
        assert r0[1] <= r0[3]
        assert r1[0] <= r1[2]
        assert r1[1] <= r1[3]

        r0 = np.asarray(r0)
        r1 = np.asarray(r1)

        w1 = np.abs(r0[0]-r0[2])
        w2 = np.abs(r1[0]-r1[2])

        if r0[0] >= r1[0] and r0[0] <= r1[2] or r0[2] >= r1[0] and r0[2] <= r1[2]:
            hd = 0
        else:
            hd = np.abs(np.min((r0[2]-r1[0],r1[2]-r0[0])))

        h1 = np.abs(r0[1]-r0[3])
        h2 = np.abs(r1[1]-r1[3])

        if r0[1] >= r1[1] and r0[1] <= r1[3] or r0[3] >= r1[1] and r0[3] <= r1[3]:
            vd = 0
        else:
            vd = np.abs(np.min((r0[3]-r1[1],r1[3]-r0[1])))

        d = np.sqrt(hd**2+vd**2)

        a1 = w1*h1
        a2 = w2*h2

        return d, a1, a2

    def distance_segment2segment(self,a0,a1,b0,b1):

        A = a1 - a0
        B = b1 - b0
        magA = np.linalg.norm(A)
        magB = np.linalg.norm(B)
        
        _A = A / magA
        _B = B / magB
        
        cross = np.cross(_A, _B);
        denom = np.linalg.norm(cross)**2
        
        if not denom:

            d0 = np.dot(_A,(b0-a0))
            d1 = np.dot(_A,(b1-a0))

            if d0 <= 0 >= d1:
                if np.absolute(d0) < np.absolute(d1):
                    return a0,b0,np.linalg.norm(a0-b0)
                return a0,b1,np.linalg.norm(a0-b1)
            elif d0 >= magA <= d1:
                if clampA1 and clampB0:
                    if np.absolute(d0) < np.absolute(d1):
                        return a1,b0,np.linalg.norm(a1-b0)
                    return a1,b1,np.linalg.norm(a1-b1)

            return None,None,np.linalg.norm(((d0*_A)+a0)-b0)

        t = (b0 - a0);
        detA = np.linalg.det([t, _B, cross])
        detB = np.linalg.det([t, _A, cross])

        t0 = detA/denom;
        t1 = detB/denom;

        pA = a0 + (_A * t0)
        pB = b0 + (_B * t1)

        if t0 < 0:
            pA = a0
        elif t0 > magA:
            pA = a1
        
        if t1 < 0:
            pB = b0
        elif t1 > magB:
            pB = b1
            
        if t0 < 0 or t0 > magA:
            dot = np.dot(_B,(pA-b0))
            if dot < 0:
                dot = 0
            elif dot > magB:
                dot = magB
            pB = b0 + (_B * dot)
    
        if t1 < 0 or t1 > magB:
            dot = np.dot(_A,(pB-a0))
            if dot < 0:
                dot = 0
            elif dot > magA:
                dot = magA
            pA = a0 + (_A * dot)

        return pA,pB,np.linalg.norm(pA-pB)

    def weighted_average_quaternions(self,Q,w):

        M = Q.shape[0]
        A = np.zeros(shape=(4,4))
        weightSum = 0

        for i in range(0,M):
            q = Q[i,:]
            A = w[i] * np.outer(q,q) + A
            weightSum += w[i]

        A = (1.0/weightSum) * A
        eigenValues, eigenVectors = np.linalg.eig(A)
        eigenVectors = eigenVectors[:,eigenValues.argsort()[::-1]]

        return np.real(eigenVectors[:,0].flatten())

    def weighted_avarage_distance(self,means,weights,angles=False):
       
        l = []
        w = []
        for i in range(len(means)):
            for j in range(len(means)):
                if j == i:
                    continue
                if angles:
                    d = means[i]-means[j]
                    if d < -np.pi: d += 2*np.pi
                    if d > +np.pi: d -= 2*np.pi
                    d = np.abs(d)
                    l.append(d)
                else:
                    l.append(np.abs(means[i]-means[j]))
                w.append(weights[i]*weights[j])
        l = np.asarray(l)
        w = np.asarray(w)
        w = w / np.sum(w)
        s = np.sum(l*w)
        
        return s

    def kabsch_umeyama(self,A,B,scaling=True):
        
        assert A.shape == B.shape 
        n, m = A.shape

        EA = np.mean(A, axis=0)
        EB = np.mean(B, axis=0)
        VarA = np.mean(np.linalg.norm(A - EA, axis=1) ** 2)

        H = ((A - EA).T @ (B - EB)) / n
        U, D, VT = np.linalg.svd(H)
        d = np.sign(np.linalg.det(U) * np.linalg.det(VT))
        S = np.diag([1] * (m - 1) + [d])

        R = U @ S @ VT
        c = VarA / np.trace(np.diag(D) @ S)
        if not scaling:
            c = 1.0
        t = EA - c * R @ EB
        
        return R, c, t

    def color_scalar_to_heatmap(self,scalar):
        
        assert scalar <= 1.0 and scalar >= 0.0 , str(scalar)
        color = self.publisher_2D_colors_heatmap[int(np.round( scalar * (self.publisher_2D_colors_heatmap_levels-1) ))]
        return color

    def distance_rotation(self,q0,q1,is_unit=False,degrees=True):

        q0 = np.asarray(q0)
        q1 = np.asarray(q1)

        if q0.shape != (4,):
            raise ValueError("Expected q0 to be 4-dimensional, got "+str(q0.shape)+" for shape")
        if np.any(np.invert(np.isfinite(q0))):
            raise ValueError("Expected q0 to be finite in all dimensions, got "+str(np.isfinite(q0))+" for isfinite()")
        if q1.shape != (4,):
            raise ValueError("Expected q1 to be 4-dimensional, got "+str(q1.shape)+" for shape")
        if np.any(np.invert(np.isfinite(q1))):
            raise ValueError("Expected q0 to be finite in all dimensions, got "+str(np.isfinite(q1)))  
        
        if np.allclose(q0,q1):
            return 0.0

        if not is_unit:
            q0_norm = np.linalg.norm(q0)
            if q0_norm != 0.0:
                q0 = q0 / q0_norm
            q1_norm = np.linalg.norm(q1)
            if q1_norm != 0.0:
                q1 = q1 / q1_norm

        if np.allclose(q0,q1):
            return 0.0

        inner = 2*np.inner(q0,q1)**2-1
        inner = np.clip(inner,0.0,1.0)
        theta_rad = np.arccos(inner)
        
        if not degrees:
            return theta_rad
        else:
            theta_deg = theta_rad * 180 / np.pi
            return theta_deg

    def get_path(self,file,folder=""):

        split = folder.split('/')
        try:
            while(True):
                split.remove("")
        except:
            pass

        path = self.path+'/'
        for i in range(0,len(split)):
            path += split[i]+"/"
            try:
                os.mkdir(path)
            except:
                pass # folder exists

        if file[0] == '/':
            path += file[1:]
        else:
            path += file

        return path

    def format_num(self,num,integers,decimals):

        # returns sign_integers_point_decimals of num with whitespace-padding for integers and zero-padding for decimals after rounding
        if decimals < 1:
            return "\033[31mX\033[m"
            #return "'Format error! - decimals must be > 0'"

        if integers < 1:
            return "\033[31mX\033[m"
            #return "'Format error! - integers must be > 0'"
        out = ""
        num = float(num)
        offset = integers - len(str(np.trunc(np.abs(num)))[:-2])
        if offset < 0:
            return " "+" "*integers+" "+" "*decimals
            #return "'Format error! - Not enough integer places - Should be <= "+str(integers)+" but is "+str(len(str(np.trunc(np.abs(num)))[:-2]))+" ("+str(num)+")'"
        
        if np.round(num,decimals) >= 0.0:
            out += "+"
        else:
            out += "-"
        for i in range(0, offset):
            #out += "0"
            out += " " 
        out += str(np.round(np.abs(num),decimals))
        while len(out) < 1+integers+1+decimals:
            out += "0"   
            #out += " "
        return out
    
    # Shutdown
    def autoexit_callback(self,timer):
        self.autoexit_trigger = True

        processing_time = self.timestamp_processing_end - self.timestamp_processing_begin
        if processing_time < self.autoexit_duration:
            rospy.sleep((self.autoexit_duration - processing_time))
        rospy.signal_shutdown("Autoexit after "+str(processing_time)+" seconds")

    def lastword(self):

        if not self.shutdown:
            self.shutdown = True
            if self.print_status:
                try:
                    if self.status_printed:
                        for i in range(0,self.status_length_max):
                            print("\b \b",end="")
                        print("\r",end='')
                except:
                    try:
                        self.ctrlc
                    except:
                        self.print()
            try:
                self.ctrlc
                self.print("\tInterupting with CTRL+C may cause post processing to be killed early. Use CTRL+D instead.",stamp=True)
            except:
                pass

            try:
                self.autoexit_trigger
                autoexit = True
            except:
                autoexit = False    
 
            self.print("...closing down at "+"["+str(datetime.datetime.now())[2:-3]+"], because "+("the autoexit duration of "+str(self.autoexit_duration)+" seconds has passed." if autoexit else "node was interupted by user."))

        if len(self.frame_success) > 0:
            self.printer_report()
            if self.print_calibration: 
                self.printer_extrinsics_raw()
            if self.print_error and self.extrinsics_reference_readfromfile:
                self.printer_extrinsics_error()
            if self.print_error_delta and self.extrinsics_reference_readfromfile:
               self.printer_extrinsics_delta()
            self.print()
            if self.log_extrinsics_result_flag:
                self.file_writer_extrinsics_result()
            if self.log_history_flag:
                self.file_writer_extrinsics_history()
            if self.log_extrinsics_reference_file and self.extrinsics_reference_readfromfile:
                path = self.get_path(self.extrinsics_reference_file[-1-self.extrinsics_reference_file[::-1].find("/"):],folder=self.log_dir)
                self.print("Copying reference camera extrinsics to '"+path+"'...",flush=True,stamp=True)
                try:
                    shutil.copy(self.get_path(self.extrinsics_reference_file),path)
                    self.print("done!")
                except Exception as e:
                    self.print("failed! - "+str(e))
            if self.log_extrinsics_estimate_file:
                path = self.get_path(self.extrinsics_estimate_file[-1-self.extrinsics_estimate_file[::-1].find("/"):],folder=self.log_dir)
                self.print("Copying estimate camera extrinsics to '"+path+"'...",flush=True,stamp=True)
                try:
                    shutil.copy(self.get_path(self.extrinsics_estimate_file),path)
                    self.print("done!")
                except Exception as e:
                    self.print("failed! - "+str(e))
            if self.log_intrinsics_file:
                path = self.get_path(self.intrinsics_file[-1-self.intrinsics_file[::-1].find("/"):],folder=self.log_dir)
                self.print("Copying camera intrinsics to '"+path+"'...",flush=True,stamp=True)
                try:
                    shutil.copy(self.get_path(self.intrinsics_file),path)
                    self.print("done!")
                except Exception as e:
                    self.print("failed! - "+str(e))
            if self.log_terminal_flag:
                self.file_writer_log()
            self.print("\nPost-processing finished!")
        else:
            self.print("\nNo results to be post-processed!")

        print()
        os._exit(0)

def main():

    rospy.init_node("calibration_node",disable_signals=True)
    
    calibrator = calibration()

    rospy.on_shutdown(calibrator.lastword)

    while not rospy.is_shutdown():
        try:
            line = sys.stdin.readline()    
            if not line:
                rospy.signal_shutdown("Keyboard-Interupt CTRL+D")
        except:
            calibrator.ctrlc = True
            rospy.signal_shutdown("Keyboard-Interupt CTRL+C")

if __name__ == "__main__":
    main()