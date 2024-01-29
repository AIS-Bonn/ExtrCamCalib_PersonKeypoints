float32 score # avg score / confidence value from detector
uint32 id # ID

Keypoint2D[] keypoints # Array of keypoints
Keypoint2D[] debug_occ_kps_orig
float32[] depth_est
float32[] depth_sigma
uint8[] occluded
uint8 n_occluded
uint8 n_valid

float32[4] bbox # Bounding box (x0, y0, x1, y1)
float32[] reid_descr
