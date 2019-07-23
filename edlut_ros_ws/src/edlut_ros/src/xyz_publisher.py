#!/usr/bin/env python

##**************************************************************************
 #                           xyz_publisher.py      		                   *
 #                           -------------------                           *
 # copyright            : (C) 2019 by Ignacio Abadia                       *
 # email                : iabadia@ugr.es                        	       *
 #**************************************************************************/

##**************************************************************************
 #                                                                         *
 #   This program is free software; you can redistribute it and/or modify  *
 #   it under the terms of the GNU General Public License as published by  *
 #   the Free Software Foundation; either version 3 of the License, or     *
 #   (at your option) any later version.                                   *
 #                                                                         *
 #**************************************************************************/

# This node publishes the robot's endpoint position. 

import rospy
from std_msgs.msg import Time
from std_msgs.msg import Bool
import numpy as np
import baxter_interface
from baxter_interface import CHECK_VERSION
from baxter_core_msgs.msg import (
	JointCommand,
)
import edlut_ros.msg


class Endpoint_publisher(object):

	def __init__(self):
		self.limb_name = rospy.get_param("~limb")
		self.rate = rospy.get_param("~rate")
		self.output_topic = rospy.get_param("~output_topic")

		self.pub = rospy.Publisher(self.output_topic, edlut_ros.msg.AnalogCompact, queue_size=10)

		# Baxter limb to have access to the robot's position
		self.limb = baxter_interface.Limb(self.limb_name)

		self.xyz = []
		self.data = []
		self.time = 0.0
		self.msg = edlut_ros.msg.AnalogCompact()
		self.rate = rospy.Rate(500)

	def GetEndPoint(self):
		coordinates = self.limb.endpoint_pose()
		self.xyz.append(coordinates["position"][0])
		self.xyz.append(coordinates["position"][1])
		self.xyz.append(coordinates["position"][2])

		self.msg.header.stamp = rospy.get_rostime()
		self.msg.data = self.xyz
		self.pub.publish(self.msg)
		self.xyz = []
		self.rate.sleep()

def main():
	rospy.init_node('endpoint_publisher', anonymous=True, disable_signals = True)
	xyz = Endpoint_publisher()
	while not rospy.is_shutdown():
		xyz.GetEndPoint()
	rospy.signal_shutdown("enabling done")

if __name__ == '__main__':
	try:
		main()
	except rospy.ROSInterruptException:
		pass
