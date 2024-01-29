#!/usr/bin/env python
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import argparse

import rospy
import tf
from sensor_msgs.msg import Image as ROS_Image
from sensor_msgs.msg import CameraInfo
from visualization_msgs.msg import MarkerArray, Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA, Header
from person_msgs.msg import Person2DOcclusionList

CocoColors = [(255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0), (170, 255, 0), (85, 255, 0), (0, 255, 0),
              (0, 255, 85), (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255), (0, 0, 255), (50, 0, 255), (100, 0, 255),
              (170, 0, 255), (255, 0, 255), (255, 150, 0), (85, 170, 0), (42, 128, 85), (0, 85, 170),
              (255, 0, 170), (255, 0, 85), (242, 165, 65)]
ADE20KIndoorColors = [(0,0,0), (217, 83, 25), (158, 158, 158), (0, 114, 189), (128, 64, 0), (255, 255, 64), (217, 83, 25), (162, 20, 47), (222,184,135), (126, 47, 142), (126, 47, 142), (222,184,135), (222,184,135), (77, 190, 238), (128, 128, 0), (230,230,250), (230,230,250), (196,64,128), (230,230,250), (127,255,0), (230,230,250), (230,230,250), (230,230,250), (230,230,250), (230,230,250), (0, 128, 128), (255,0,255), (255,0,255), (255,0,255), (230,230,250), (230,230,250), (230,230,250)]

CocoColors_inv = [(255 - color[0], 255 - color[1] , 255 - color[2]) for color in CocoColors]
CocoPairs = [(0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6), (5, 7), (6, 8), (7, 9), (8, 10),
               (5, 11), (6, 12), (11, 13), (12, 14), (13, 15), (14, 16)]

def draw_humans_3d(humans, stamp, frame_id, caminfo, pairs=CocoPairs):
    skel3d_msg = MarkerArray()
    
    fx = caminfo.K[0]
    fy = caminfo.K[4]
    cx = caminfo.K[2]
    cy = caminfo.K[5]
    color_idx = int(frame_id[4]) if frame_id[5] == '_' else int(frame_id[5])
    
    for h_idx, human in enumerate(humans):
        if human.score > 0.and len(human.depth_est) == len(human.keypoints):
            skel3d = Marker()
            skel3d.header.frame_id = frame_id
            skel3d.header.stamp = stamp
            skel3d.lifetime = rospy.Duration(0.5)
            skel3d.pose.orientation.w = 1.
            skel3d.type = Marker.LINE_LIST
            skel3d.scale.x = 0.05
            skel3d.ns = 'skeleton_local'
            skel3d.id = h_idx
            skel3d.color.a = 1.0
            
            skel3d_joints = Marker()
            skel3d_joints.header.frame_id = frame_id
            skel3d_joints.header.stamp = stamp
            skel3d_joints.lifetime = rospy.Duration(0.5)
            skel3d_joints.pose.orientation.w = 1.
            skel3d_joints.type = Marker.SPHERE_LIST
            skel3d_joints.scale.x = 0.07
            skel3d_joints.scale.y = 0.07
            skel3d_joints.scale.z = 0.07 # w.r.t depth sigma
            skel3d_joints.ns = 'joints_local'
            skel3d_joints.id = h_idx
            skel3d_joints.color.a = 1.0
            
            skel3d_depth = Marker()
            skel3d_depth.header.frame_id = frame_id
            skel3d_depth.header.stamp = stamp
            skel3d_depth.lifetime = rospy.Duration(0.5)
            skel3d_depth.pose.orientation.w = 1.
            skel3d_depth.type = Marker.LINE_LIST
            skel3d_depth.scale.x = 0.03
            skel3d_depth.ns = 'depth_local_interval'
            skel3d_depth.id = h_idx
            skel3d_depth.color.a = 1.0

            centers = []
            for i, kp in enumerate(human.keypoints):
                if kp.score <= 0. or (human.n_occluded > 0 and human.occluded[i]) or human.depth_est[i] == 0:
                    centers.append([])
                    continue
                
                c = ColorRGBA()
                c.a = 1.0
                c.r = CocoColors[color_idx][0] / 255.
                c.g = CocoColors[color_idx][1] / 255.
                c.b = CocoColors[color_idx][2] / 255.
                
                z = human.depth_est[i]
                x = (kp.x - cx) * z / fx
                y = (kp.y - cy) * z / fy
                
                z_min = max(0, z - human.depth_sigma[i])
                x_min = (kp.x - cx) * z_min / fx
                y_min = (kp.y - cy) * z_min / fy
                p_min = Point()
                p_min.x = x_min
                p_min.y = y_min
                p_min.z = z_min
                z_max = z + human.depth_sigma[i]
                x_max = (kp.x - cx) * z_max / fx
                y_max = (kp.y - cy) * z_max / fy
                p_max = Point()
                p_max.x = x_max
                p_max.y = y_max
                p_max.z = z_max
                skel3d_depth.points.append(p_min)
                skel3d_depth.points.append(p_max)
                skel3d_depth.colors.append(c)
                skel3d_depth.colors.append(c)
    
                center = [x, y, z]
                centers.append(center)
                
                p = Point()
                p.x = center[0]
                p.y = center[1]
                p.z = center[2]
                
                skel3d_joints.points.append(p)
                skel3d_joints.colors.append(c)
                
            for pair in pairs:
                if len(centers[pair[0]]) == 3 and len(centers[pair[1]]) == 3:
                    p1 = Point()
                    p1.x = centers[pair[0]][0]
                    p1.y = centers[pair[0]][1]
                    p1.z = centers[pair[0]][2]
                    
                    p2 = Point()
                    p2.x = centers[pair[1]][0]
                    p2.y = centers[pair[1]][1]
                    p2.z = centers[pair[1]][2]
                    
                    c1 = ColorRGBA()
                    c1.a = 1.0
                    c1.r = CocoColors[color_idx][0] / 255.
                    c1.g = CocoColors[color_idx][1] / 255.
                    c1.b = CocoColors[color_idx][2] / 255.
                    
                    c2 = ColorRGBA()
                    c2.a = 1.0
                    c2.r = CocoColors[color_idx][0] / 255.
                    c2.g = CocoColors[color_idx][1] / 255.
                    c2.b = CocoColors[color_idx][2] / 255.
                    
                    skel3d.points.append(p1)
                    skel3d.colors.append(c1)
                    skel3d.points.append(p2)
                    skel3d.colors.append(c2)

            skel3d_msg.markers.append(skel3d)
            skel3d_msg.markers.append(skel3d_joints)
            skel3d_msg.markers.append(skel3d_depth)

    return skel3d_msg

def draw_humans(img, humans):
        _CONF_THRESHOLD_DRAW = 0.25
        _OCCLUDED_COLOR = [255, 165, 0]
        
        image_w = img.shape[1]
        image_h = img.shape[0]
        
        num_joints = 17
        colors = CocoColors
        pairs = CocoPairs
            
        for human in humans:
            centers = {}
            body_parts = {}
            joints_occluded = {}
            occluded_idx = human['occluded_idx']
            n_occluded = human['n_occluded']
            n_valid = 0

            # draw point
            for i in range(num_joints):
                joint_occluded = ord(occluded_idx[i]) if n_occluded > 0 else 0
                if isinstance(human['keypoints'], dict):
                    if str(i) not in human['keypoints'].keys() or (human['keypoints'][str(i)][2] < _CONF_THRESHOLD_DRAW and joint_occluded == 0):
                        continue
                    body_part = human['keypoints'][str(i)]
                    
                elif isinstance(human['keypoints'], list):
                    if human['keypoints'][i] is None or (human['keypoints'][i][2] < _CONF_THRESHOLD_DRAW and joint_occluded == 0):
                        continue
                    body_part = human['keypoints'][i]
                    
                center = (int(body_part[0] + 0.5), int(body_part[1] + 0.5))

                centers[i] = center
                body_parts[i] = body_part
                joints_occluded[i] = joint_occluded
                n_valid += 1
                
                if joint_occluded > 0:
                    img = cv2.circle(img, center, max(1, int(img.shape[0] / 360)) * 5, _OCCLUDED_COLOR, thickness=-1, lineType=8, shift=0)
                else:
                    img = cv2.circle(img, center, max(1, int(img.shape[0] / 360)) * 5, colors[i], thickness=-1, lineType=8, shift=0)

            # draw line
            for pair_order, pair in enumerate(pairs):
                if pair[0] not in centers.keys() or pair[1] not in centers.keys():
                    continue
                
                if joints_occluded[pair[0]] and joints_occluded[pair[1]]:
                    img = cv2.line(img, centers[pair[0]], centers[pair[1]], _OCCLUDED_COLOR, max(1, int(img.shape[0] / 360)) * 4)
                else:
                    img = cv2.line(img, centers[pair[0]], centers[pair[1]], colors[pair[1]], max(1, int(img.shape[0] / 360)) * 4)
                
            # draw bounding box
            if n_valid > 0:
                x1, y1, x2, y2 = human['bbox']
                x1 = int(x1 + 0.5)-6
                y1 = int(y1 + 0.5)-6
                x2 = int(x2 + 0.5)+6
                y2 = int(y2 + 0.5)+6
                    
                if float(n_occluded) / n_valid > 0.8:
                    cv2.rectangle(img, (x1, y1), (x2, y2), color = _OCCLUDED_COLOR, thickness = max(1, int(img.shape[0] / 360)) * 2)
                else:
                    cv2.rectangle(img, (x1, y1), (x2, y2), color = colors[0], thickness = max(1, int(img.shape[0] / 360)) * 2)
                
                #cv2.putText(img, 'ID: {} - {:.1f}%'.format(max(0,human['id']) , human['score'] * 100), (x1, y1 - 5), cv2.FONT_HERSHEY_PLAIN, 1.5, colors[human['id'] % len(colors)], 2)
                #cv2.putText(img, '{:.1f}%'.format(human['score'] * 100), (x1, y1 - 5), cv2.FONT_HERSHEY_PLAIN, 1.5, colors[human['id'] % len(colors)], 2)

        return img

class pose_analyzer:
    
    def __init__(self, args):
        
        self.jetson = args.jetson
        self.last_obj_dets = None
        self.det2segm_class = [0, 14, 17, 16, 15, 9, 10, 8, 27, 26, 24, 24, 22, 20]
        self.flip = args.flip
        
        self.bridge = CvBridge()
        
        self.num_joints = 17
        
        self.caminfo = None
        self.last_obj_dets = None
        
        self.publisher_image_overlay = rospy.Publisher('image_overlay_from_json', ROS_Image, queue_size=1)
        if self.jetson:
            self.publisher_skeleton = rospy.Publisher('skel_3d_local', MarkerArray, queue_size=1)
                
        self.json_pose_sub = rospy.Subscriber('/human_joints', Person2DOcclusionList, self.callback_pose,  queue_size = 3)
        if self.jetson:
            from edgetpu_segmentation_msgs.msg import DetectionList
            #self.obj_labels = ["NONE", "person", "cycle", "vehicle", "animal", "chair", "couch", "table", "tv", "laptop", "microwave", "oven", "fridge", "book"]
            self.obj_labels = ["NONE", "person", "cycle", "other", "other", "chair", "chair", "table", "computer/tv", "computer/tv", "other", "other", "other", "other"]
            print('Object classes: {}'.format(self.obj_labels))
            self.obj_det_sub = rospy.Subscriber('/dets_obj', DetectionList, self.callback_obj_det, queue_size = 3)
            self.caminfo_sub = rospy.Subscriber('/camera_info', CameraInfo, self.callback_caminfo,  queue_size = 1)
                
    def callback_caminfo(self, msg):
        self.caminfo = msg
    def callback_obj_det(self, msg):
        self.last_obj_dets = msg.detections
    
    def callback_pose(self, msg):
        if self.jetson:
            bg_image = 255 * np.ones((480,848,3), dtype=np.uint8)
        else:
            bg_image = 255 * np.ones((480,640,3), dtype=np.uint8)
            
        if self.jetson and self.last_obj_dets is not None:
            for det in self.last_obj_dets:
                if det.label > 1: # don't display None and person (persons will be displayed as skeletons and bbox below..)
                    x0 = bg_image.shape[1] - det.bbox.xmax if self.flip else det.bbox.xmin
                    x1 = bg_image.shape[1] - det.bbox.xmin if self.flip else det.bbox.xmax
                    y0 = bg_image.shape[0] - det.bbox.ymax if self.flip else det.bbox.ymin
                    y1 = bg_image.shape[0] - det.bbox.ymin if self.flip else det.bbox.ymax
                    det_color = ADE20KIndoorColors[self.det2segm_class[det.label]]
                    cv2.rectangle(bg_image, (int(round(x0)), int(round(y0))), (int(round(x1)), int(round(y1))), color = det_color, thickness = max(1, int(bg_image.shape[0] / 360)) * 2)
                    
        if self.jetson and self.caminfo is not None:
            msg_skel = draw_humans_3d(msg.persons, msg.header.stamp, msg.header.frame_id, self.caminfo)
            if len(msg_skel.markers) > 0:
                if not rospy.is_shutdown():
                    self.publisher_skeleton.publish(msg_skel)
        
        #humans = [{'id': 0, 'score': p.score, 'bbox': p.bbox, 'keypoints': [[kp.x, kp.y, kp.score] for kp in p.keypoints]} for p in msg.persons]
        if self.jetson and self.flip:
            humans = [{'id': p.id, 'score': p.score, 'bbox': [bg_image.shape[1] - p.bbox[2], bg_image.shape[0] - p.bbox[3], bg_image.shape[1] - p.bbox[0], bg_image.shape[0] - p.bbox[1]],
                    'keypoints': [[bg_image.shape[1] - kp.x, bg_image.shape[0] - kp.y, kp.score] for kp in p.keypoints],
                    'debug_occ_kps_orig': [[bg_image.shape[1] - kp.x, bg_image.shape[0] - kp.y, kp.score] for kp in p.debug_occ_kps_orig],
                    'occluded_idx': p.occluded, 'n_occluded': p.n_occluded} for p in msg.persons]
        else:
            humans = [{'id': p.id, 'score': p.score, 'bbox': p.bbox, 'keypoints': [[kp.x, kp.y, kp.score] for kp in p.keypoints],
                        'debug_occ_kps_orig': [[kp.x, kp.y, kp.score] for kp in p.debug_occ_kps_orig],
                        'occluded_idx': p.occluded, 'n_occluded': p.n_occluded} for p in msg.persons]
        
        img = draw_humans(bg_image, humans)
        
        img_msg = self.bridge.cv2_to_imgmsg(img, "rgb8")
        img_msg.header = msg.header
            
        self.publisher_image_overlay.publish(img_msg)
 
def main():
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--jetson', default=False, action='store_true',
                        help='Pose and object detection running on jetson nx')
    parser.add_argument('--flip', default=False, action='store_true',
                        help='flip original image before drawing onto it')
    
    rospy.init_node('pose2D_plot_node')
    args = parser.parse_args(rospy.myargv()[1:])
    
    panalyzer = pose_analyzer(args)
    
    rospy.spin()

if __name__ == '__main__':
    main()
