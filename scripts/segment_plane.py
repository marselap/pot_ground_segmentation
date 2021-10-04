#!/usr/bin/env python3

import rospy

import ros_numpy

from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header 
from geometry_msgs.msg import PointStamped

from segmentation_class_2d import Segmentor
from pc_segmentation_class import PCSegmentor


import cv2 as cv
import numpy as np
import math 

import pcl

import tf2_ros
from tf2_sensor_msgs.tf2_sensor_msgs import do_transform_cloud
import time


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


        self.tfBuffer = tf2_ros.Buffer()
        listener = tf2_ros.TransformListener(self.tfBuffer)

        time.sleep(2.)

        # self.pc_sub = rospy.Subscriber("/segmented_cloud", PointCloud2, self.pc_callback)
        # self.pc_sub = rospy.Subscriber("/ground_pc", PointCloud2, self.pc_callback)
        self.pc_sub = rospy.Subscriber("/camera/depth_registered/points", PointCloud2, self.pc_callback)

        self.pc_pub = rospy.Publisher("/segmented_plane", PointCloud2, queue_size=10)

        self.ground_center_pub = rospy.Publisher("/ground_center", PointStamped, queue_size=10)

        self.pc_glob = rospy.Publisher("/full_cloud_glob", PointCloud2, queue_size=10)


        self.pc_array = None    

        self.segmentation = PCSegmentor(None)

        self.target_frame = 'panda_link0'
        self.source_frame = 'panda_camera'

        self.got_pc = False


    def pc_callback(self, pc_message):

        self.got_pc = True
        # print (pc_message.fields)
        # print (pc_message.point_step)
        # print (pc_message.row_step)
        # print (pc_message.height)
        # print (pc_message.width*pc_message.point_step)

        self.pc_msg = pc_message
        # self.pc_array = ros_numpy.point_cloud2.pointcloud2_to_array(pc_message)




def main():

    rospy.init_node("ground_segmentation")


    gs = GroundSegment()


    while not rospy.is_shutdown():

        # if gs.pc_array is not None:
        if gs.got_pc:


            try:
                trans = gs.tfBuffer.lookup_transform(gs.target_frame, gs.source_frame, rospy.Time())
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                print("Did not get transform")
                continue

            cloud_glob = do_transform_cloud(gs.pc_msg, trans)
            gs.pc_glob.publish(cloud_glob)

            gs.segmentation.set_pointcloud(cloud_glob)

            pc_pc2 = gs.segmentation.segment()


            gs.pc_array = ros_numpy.point_cloud2.pointcloud2_to_array(pc_pc2)

            # print (gs.pc_array.fields)
            # print (gs.pc_array.point_step)
            # print (gs.pc_array.row_step)
            # print (gs.pc_array.height)
            # print (gs.pc_array.width*gs.pc_array.point_step)

            points_g = []
            for row in gs.pc_array:
                
                points_g.append([row[0], row[1], row[2]])

            cloud = pcl.PointCloud()
            cloud.from_array(np.array(points_g))

            seg = cloud.make_segmenter_normals(ksearch=50)
            seg.set_optimize_coefficients(True)
            seg.set_model_type(pcl.SACMODEL_PLANE)
            seg.set_normal_distance_weight(0.05)
            seg.set_method_type(pcl.SAC_RANSAC)
            seg.set_max_iterations(100)
            seg.set_distance_threshold(0.005)
            inliers, model = seg.segment()

            # print("model") # ax + by + cz + d = 0 
            # print(model) 

            points = []
            for row in gs.pc_array:
                points.append([row[0], row[1], row[2], row[3]])
            points = np.array(points)


            if len(inliers)>1:

                points2 = points[inliers, :]
                points2 = np.array(points2)
                pc = point_cloud(points2, gs.target_frame)
                gs.pc_pub.publish(pc)

                not_nan = []
                for p in points2:
                    if not math.isnan(np.sum(p)) and not math.isinf(np.sum(p)):
                        not_nan.append(p[:3])
                not_nan = np.array(not_nan)
                not_nan=not_nan.transpose()

                try:
                    # print(np.min(not_nan[0][:]), np.mean(not_nan[0][:]), np.max(not_nan[0][:]), np.median(not_nan[0][:]))
                    # print("\n")
                    # print(np.min(not_nan[1][:]), np.mean(not_nan[1][:]), np.max(not_nan[1][:]), np.median(not_nan[1][:]))
                    # print("\n")
                    # print(np.min(not_nan[2][:]), np.mean(not_nan[2][:]), np.max(not_nan[2][:]), np.median(not_nan[2][:]))
                    # print("\n")
                    # print("r x: ", np.max(not_nan[0][:])-np.min(not_nan[0][:]))
                    # print("r y: ", np.max(not_nan[1][:])-np.min(not_nan[1][:]))

                    msg = PointStamped()
                    msg.header = Header(frame_id=gs.target_frame, stamp=rospy.Time.now())
                    msg.point.x = np.mean(not_nan[0][:])
                    msg.point.y = np.min(not_nan[1][:]+0.05) # 8cm od sredine za impedancija eksp s pikanjem
                    msg.point.z = np.mean(not_nan[2][:]+0.) # pomak od panda_link8 do vrha senzora

                    gs.ground_center_pub.publish(msg)

                    
                except IndexError:
                    print("ups")
                    pass
                
                gs.got_pc = False



    rospy.spin()


if __name__ == "__main__":
    main()