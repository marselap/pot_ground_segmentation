#!/usr/bin/env python3

import rospy

import ros_numpy

from sensor_msgs.msg import PointCloud2, PointField
from std_msgs.msg import Header 

import numpy as np



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
    offsets = [0,4,8,12]

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

