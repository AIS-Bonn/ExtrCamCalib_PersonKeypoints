uint32 id # Person ID
float32 depth # distance towards camera
float32 score # average keypoint score
person_msgs/Keypoint2D[] keypoints # Array of keypoints
float32[4] bbox # Bounding box (x0, y0, x1, y1)