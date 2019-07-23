#!/usr/bin/env python

##**************************************************************************
 #                           TrajectoryPlot.py                             *
 #                           -------------------                           *
 # copyright            : (C) 2018 by Ignacio Abadia                       *
 # email                : iabadia@ugr.es           		                   *
 #**************************************************************************/

##**************************************************************************
 #                                                                         *
 #   This program is free software; you can redistribute it and/or modify  *
 #   it under the terms of the GNU General Public License as published by  *
 #   the Free Software Foundation; either version 3 of the License, or     *
 #   (at your option) any later version.                                   *
 #                                                                         *
 #**************************************************************************/


# This script creates an online 3D animation of the trajectory being performed
# by the end-effector.

import time
import argparse

import rospy

import sys

import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import threading

import matplotlib.lines
import matplotlib.animation as animation

import mpl_toolkits.mplot3d.axes3d as p3
from mpl_toolkits.mplot3d.art3d import Line3D

import baxter_interface
from baxter_interface import CHECK_VERSION

import edlut_ros.msg
from dynamic_reconfigure.server import Server
from edlut_ros.cfg import MaeDynParametersConfig

class TrajectoryAnimation(animation.TimedAnimation):

	def __init__(self):
		rospy.init_node('TrajectoryPlot', log_level=rospy.INFO)

		#Retrieve RosLaunch parameters
		self.limb_name = rospy.get_param("~limb")
		self.refreshRate = rospy.get_param("~refresh_frame")
		self.trajectory_period = rospy.get_param("~trajectory_period")
		self.trajectory_file_name = rospy.get_param("~trajectory_file")
		self.limb = baxter_interface.Limb(self.limb_name)
		self.trajectory_file = open(self.trajectory_file_name, "r")

		#Create parameters that can be modified dynamically to change the graph online
		self.display = False

		#Cartesian coordinates of the trajectory that the robot is actually doing
		self.robot_x = []
		self.robot_y = []
		self.robot_z = []

		self.rx = 0.0
		self.ry = 0.0
		self.rz = 0.0

		self.max_x = -1.0
		self.max_y = -1.0
		self.max_z = -1.0
		self.min_x = 1.0
		self.min_y = 1.0
		self.min_z = 1.0

		#Cartesian coordinates of the desired trajectory
		self.desired_x = []
		self.desired_y = []
		self.desired_z = []

		self.ReadFile()
		self.trajectory_file.close()

		self.samples = int(self.trajectory_period / (self.refreshRate/1000))
		self.data = np.empty((3,self.samples))

		self.deque_x = deque(maxlen=self.samples)
		self.deque_y = deque(maxlen=self.samples)
		self.deque_z = deque(maxlen=self.samples)


	def ReadFile(self):
	 	for line in self.trajectory_file:
	 		coordinates = line.split()
			x = float(coordinates[0])
			y = float(coordinates[1])
			z = float(coordinates[2])
	 		self.desired_x.append(x)
	 		self.desired_y.append(y)
	 		self.desired_z.append(z)
			if x > self.max_x:
				self.max_x = x
			elif x < self.min_x:
				self.min_x = x
			if y > self.max_y:
				self.max_y = y
			elif y < self.min_y:
				self.min_y = y
			if z > self.max_z:
				self.max_z = z
			elif z < self.min_z:
				self.min_z = z

	def GetEndPoint(self):
		coordinates = self.limb.endpoint_pose()
		self.robot_x.append(coordinates["position"][0])
		self.robot_y.append(coordinates["position"][1])
		self.robot_z.append(coordinates["position"][2])
		self.deque_x.append(coordinates["position"][0])
		self.deque_y.append(coordinates["position"][1])
		self.deque_z.append(coordinates["position"][2])


	#Callback to dynamically change parameters
	def param_callback(self, config, level):
		self.display = config["Display"]
		return config

	def update_lines(self, index):
		self.GetEndPoint()
		self.data[0, index] = self.robot_x
		self.data[1, index] = self.robot_y
		self.data[2, index] = self.robot_z


	def new_figure(self):
		# Create the figure and the axis
		self.figure = plt.figure()
		self.axis = self.figure.add_subplot(1,1,1, projection="3d")

		self.axis.set_title("Trajectory")
		self.axis.set_xlabel('x')
		self.axis.set_ylabel('y')
		self.axis.set_zlabel('z')

		self.line_ideal = Line3D([], [], [], color="blue", linestyle=':')
		self.line_real = Line3D([], [], [], color="red", linestyle='-')
		self.axis.add_line(self.line_ideal)
		self.axis.add_line(self.line_real)
		self.axis.set_xlim(self.min_x - 0.2, self.max_x + 0.2)
		self.axis.set_ylim(self.min_y - 0.2, self.max_y + 0.2)
		self.axis.set_zlim(self.min_z - 0.2, self.max_z + 0.2)

		self.line_ideal.set_data(self.desired_x, self.desired_y)
		self.line_ideal.set_3d_properties(self.desired_z)

		# Creating the Animation object
		animation.TimedAnimation.__init__(self, self.figure, interval=self.refreshRate, blit=True)


	def _draw_frame(self, framedata):
		if self.display:
			self.GetEndPoint()
			self.line_real.set_data(self.deque_x, self.deque_y)
			self.line_real.set_3d_properties(self.deque_z)
			self.line_ideal.set_data(self.desired_x, self.desired_y)
			self.line_ideal.set_3d_properties(self.desired_z)
			self._drawn_artists = [self.line_real, self.line_ideal]


			return self._drawn_artists

	def new_frame_seq(self):
		if self.display:
			return iter(xrange(sys.maxint))

	def _init_draw(self):
		if self.display:
			lines = [self.line_real]
			for l in lines:
				l.set_data([], [])

	def close_display(self):
		self.display = False

Animation = TrajectoryAnimation()
srv = Server(MaeDynParametersConfig, Animation.param_callback)
while not rospy.is_shutdown():
	if Animation.display:
		Animation.new_figure()
		plt.show()
		Animation.close_display()
	rospy.sleep(1)
