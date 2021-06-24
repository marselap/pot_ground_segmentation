#!/usr/bin/env python3

import rospy

import ros_numpy

from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, PointCloud2, PointField
from std_msgs.msg import Header 

from segmentation_class import Segmentor

from dynamic_reconfigure.server import Server
from pot_ground_segmentation.cfg import HsvMaskConfig

import cv2 as cv
import numpy as np
import math 

import pcl



def point_cloud(points, parent_frame):
    """ Creates a point cloud message.
    Args:
        points: Nx7 array of xyz positions (m) and rgba colors (0..1)
        parent_frame: frame in which the point cloud is defined
    Returns:
        sensor_msgs/PointCloud2 message
    """
    ros_dtype = PointField.FLOAT32
    dtype = np.float32
    itemsize = np.dtype(dtype).itemsize

    data = points.astype(dtype).tobytes()

    offsets = [0,4,8,16]

    fields = [PointField(
        name=n, offset=offsets[i], datatype=ros_dtype, count=1)
        for i, n in enumerate(['x','y','z','rgb'])]


    header = Header(frame_id=parent_frame, stamp=rospy.Time.now())

    return PointCloud2(
        header=header,
        height=1,
        width=points.shape[0],
        is_dense=False,
        is_bigendian=False,
        fields=fields,
        point_step=itemsize*4,
        row_step=32*points.shape[0],
        data=data
    )



def point_cloud_gray(points, parent_frame):
    """ Creates a point cloud message.
    Args:
        points: Nx3 array of xyz positions (m)
        parent_frame: frame in which the point cloud is defined
    Returns:
        sensor_msgs/PointCloud2 message
    """
    ros_dtype = PointField.FLOAT32
    dtype = np.float32
    itemsize = np.dtype(dtype).itemsize

    data = points.astype(dtype).tobytes()

    offsets = [0,4,8]

    fields = [PointField(
        name=n, offset=offsets[i], datatype=ros_dtype, count=1)
        for i, n in enumerate(['x','y','z'])]


    header = Header(frame_id=parent_frame, stamp=rospy.Time.now())

    return PointCloud2(
        header=header,
        height=1,
        width=points.shape[0],
        is_dense=False,
        is_bigendian=False,
        fields=fields,
        # point_step=itemsize*8,
        point_step=itemsize,
        # row_step=itemsize*4*points.shape[0],
        row_step=16*points.shape[0],
        data=data
    )



class GroundSegment():
    def __init__(self):
        self.bridge = CvBridge()
        self.image_pub = rospy.Publisher("/segmented", Image, queue_size=10)
        self.mask_pub = rospy.Publisher("/segmentation_mask", Image, queue_size=10)
        self.contours_pub = rospy.Publisher("/segmentation_contours", Image, queue_size=10)

        self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.img_callback)

        self.pc_sub = rospy.Subscriber("/camera/depth_registered/points", PointCloud2, self.pc_callback)

        self.pc_pub = rospy.Publisher("/segmented_cloud", PointCloud2, queue_size=10)
        self.pc_pub_plane = rospy.Publisher("/segmented_plane", PointCloud2, queue_size=10)

        self.segmentation = Segmentor()

        self.srv = Server(HsvMaskConfig, self.dynamic_recon_cb)

        self.pc_array = None    

    def dynamic_recon_cb(self, config, level):
        mask_lower = (config['hue_L'], config['sat_L'], config['val_L'])
        mask_upper = (config['hue_H'], config['sat_H'], config['val_H'])

        self.segmentation.min_contour_size = config['contour_size']
        self.segmentation.mask_lower = mask_lower
        self.segmentation.mask_upper = mask_upper
        self.segmentation.kernel = config['kernel_size']
        return config

    def img_callback(self, image_message):

        try:
            cv_image = self.bridge.imgmsg_to_cv2(image_message, 'rgb8')
        except CvBridgeError as e:
            print(e)
            return 
        
        self.segmentation.new_image(cv_image)



    def pc_callback(self, pc_message):

        self.pc_array = ros_numpy.point_cloud2.pointcloud2_to_array(pc_message)



def main():

    rospy.init_node("ground_segmentation")


    gs = GroundSegment()


    while not rospy.is_shutdown():

        if gs.segmentation.image is not None:
            masked, mask, contours = gs.segmentation.segment()
            gs.image_pub.publish(gs.bridge.cv2_to_imgmsg(masked, 'rgb8'))
            gs.mask_pub.publish(gs.bridge.cv2_to_imgmsg(mask, 'rgb8'))
            gs.contours_pub.publish(gs.bridge.cv2_to_imgmsg(contours, 'rgb8'))


            if gs.pc_array is not None:
                points = []
                temp_array = gs.pc_array
                gs.pc_array = None
                
                gray_mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)

                xmax, ymax = np.shape(gray_mask)
                for ix in range(xmax):
                    for iy in range(ymax):
                        if gray_mask[ix,iy] :
                            (x,y,z,rgb) = temp_array[ix, iy]
                            if not math.isnan(x):
                                points.append((x,y,z,rgb))

                if len(points)>1:
                # if len(np.shape(points))>1:
                    m, _ = np.shape(points)
                    points = np.array(points)
                    
                    pc = point_cloud(points, 'camera_color_optical_frame')
                    gs.pc_pub.publish(pc)

                    cloud = pcl.PointCloud()
                    cloud.from_array(points[:,0:3])
                    seg = cloud.make_segmenter_normals(ksearch=50)
                    seg.set_optimize_coefficients(True)
                    seg.set_model_type(pcl.SACMODEL_PLANE)
                    seg.set_normal_distance_weight(0.05)
                    seg.set_method_type(pcl.SAC_RANSAC)
                    seg.set_max_iterations(100)
                    seg.set_distance_threshold(0.005)
                    inliers, model = seg.segment()

                    print("model")
                    print(model)

                    if len(inliers)>1:

                        points2 = points[inliers, :]
                        points2 = np.array(points2)
                        pc = point_cloud(points2, 'camera_color_optical_frame')
                        gs.pc_pub_plane.publish(pc)


                    # points = np.expand_dims(points, axis=1)
                    
                    # points.dtype = {'names':['x','y','z','rgb'], 'formats':['<f4','<f4','<f4','<f4'], 'offsets':[0,4,8,16], 'itemsize':32}
                    # if m > 100:
                    #     np_pc = ros_numpy.point_cloud2.array_to_pointcloud2(points)
                    #     np_pc.header.frame_id = 'camera_color_optical_frame'
                    #     np_pc.header.stamp = rospy.Time.now()
                    #     gs.pc_pub.publish(np_pc)



    rospy.spin()


if __name__ == "__main__":
    main()