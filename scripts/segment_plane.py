#!/usr/bin/env python3

import rospy

import ros_numpy

from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header 

from segmentation_class import Segmentor

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



class GroundSegment():
    def __init__(self):

        self.pc_sub = rospy.Subscriber("/segmented_cloud", PointCloud2, self.pc_callback)

        self.pc_pub = rospy.Publisher("/segmented_plane", PointCloud2, queue_size=10)

        self.pc_array = None    


    def pc_callback(self, pc_message):

        print (pc_message.fields)
        print (pc_message.point_step)
        print (pc_message.row_step)
        print (pc_message.height)
        print (pc_message.width*pc_message.point_step)


        self.pc_array = ros_numpy.point_cloud2.pointcloud2_to_array(pc_message)



def main():

    rospy.init_node("ground_segmentation")


    gs = GroundSegment()


    while not rospy.is_shutdown():


        if gs.pc_array is not None:

            cloud = pcl.PointCloud()
            cloud.from_array(gs.pc_array)
            seg = cloud.make_segmenter_normals(ksearch=50)
            seg.set_optimize_coefficients(True)
            seg.set_model_type(pcl.SACMODEL_PLANE)
            seg.set_normal_distance_weight(0.05)
            seg.set_method_type(pcl.SAC_RANSAC)
            seg.set_max_iterations(100)
            seg.set_distance_threshold(0.005)
            inliers, model = seg.segment()

            if len(points)>1:
                # points = np.array(points)
                
                pc = point_cloud(inliers, 'camera_color_optical_frame')
                gs.pc_pub.publish(pc)




    rospy.spin()


if __name__ == "__main__":
    main()