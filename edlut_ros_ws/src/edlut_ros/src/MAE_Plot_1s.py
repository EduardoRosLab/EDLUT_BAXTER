#!/usr/bin/env python

##**************************************************************************
 #                           MaePlot_1s.py                                 *
 #                           -------------------                           *
 # copyright            : (C) 2018 by Ignacio Abadia                       *
 # email                : iabadia@ugr.es                             	   *
 #**************************************************************************/

##**************************************************************************
 #                                                                         *
 #   This program is free software; you can redistribute it and/or modify  *
 #   it under the terms of the GNU General Public License as published by  *
 #   the Free Software Foundation; either version 3 of the License, or     *
 #   (at your option) any later version.                                   *
 #                                                                         *
 #**************************************************************************/


# This script is used to compute and plot the Mean Absolute Error (MAE)
# comparing two given topics (desired vs. current position) over a specified
# period of time (trial duration)

import time
import argparse

import rospy

import sys
import numpy
import matplotlib.pyplot

import threading

import matplotlib.lines
import matplotlib.animation

import edlut_ros.msg
from rosgraph_msgs.msg import Clock
from dynamic_reconfigure.server import Server
from edlut_ros.cfg import MaeDynParametersConfig


class MaeplotAnimation(matplotlib.animation.TimedAnimation):

	#Callback for current topic (position or velocity)
	def CurrentCallback(self, data):
		if self.first_current:
			for joint, value in self.current_values.items():
				for index, name in enumerate(data.names):
					if data.names[index] == joint:
						self.aux_current_index.append(index)
			self.first_current = False

		if self.init_error:
			self.start = rospy.get_time()
			self.init_error = False
		now = rospy.get_time()

		if now - self.start < self.duration_trial:
			counter2 = 0
			for joint, value in self.current_values.items():
				self.current_values[joint] = data.data[self.aux_current_index[counter2]]
				counter2 += 1

			current_error = self.GetError()
			self.total_calls += 1.0
			self.error = self.error + current_error

		else:
			if self.total_calls != 0.0:
				self.error = self.error / self.total_calls
			self.trial_error.append(self.error)
			if (self.error > self.error_max):
				self.error_max = self.error
			self.trial_number.append(self.total_trials)
			self.total_trials += 1

			# This piece of code is used for obtaining the statistics of the
			# position control mode. The first 100 trials are used.
			# if self.total_trials <= 100:
			# 	self.MAE_100.append(self.error)
			# if self.total_trials == 101:
			# 	self.mean_MAE_100 = numpy.mean(self.MAE_100)
			# 	self.std_100 = numpy.std(self.MAE_100)
			# 	print "MEAN MAE 100 = ", self.mean_MAE_100
			# 	print "STD MAE 100 = ", self.std_100

			self.total_calls = 0.0
			self.error = 0
			self.init_error = True

		if (now - self.start > self.duration_trial):
			self.init_error = True

	#Callback for desired topic (position or velocity)
	def DesiredCallback(self, data):
		if self.first_desired:
			for joint, value in self.desired_values.items():
				for index, name in enumerate(data.names):
					if data.names[index] == joint:
						self.aux_desired_index.append(index)
			self.first_desired = False

		counter2 = 0
		for joint, value in self.desired_values.items():
			self.desired_values[joint] = data.data[self.aux_desired_index[counter2]]
			counter2 += 1


	#Callback for simulation time
	def callback(self, clock):
		self.sim_time = clock.clock

	# Callback to dynamically change parameters
	def param_callback(self, config, level):
		self.display = config["Display"]
		return config

	def GetError(self):
		# num_joints = 0
		error = 0
		for x in range(0, self.num_joints):
			error = error + abs(self.desired_values[self.joint_list[x]] - self.current_values[self.joint_list[x]])
		error = error * self.inverse_num_joints

		return error

	def new_figure(self):
		# Create the figure, the axis and the animation
		self.figure = matplotlib.pyplot.figure()
		self.axis = self.figure.add_subplot(1, 1, 1)
		self.axis.set_title(self.figureName)

		self.axis.set_xlabel('Trials (n)')
		self.axis.set_ylabel('MAE')
		self.line = matplotlib.pyplot.Line2D([], [], color='red', marker='.', markersize=3,linestyle='solid', label = self.currentTopic)
		self.mean_line = matplotlib.pyplot.Line2D([], [], color='blue', marker='.', markersize=3, linestyle=':', label = self.currentTopic)

		self.axis.add_line(self.line)
		self.axis.add_line(self.mean_line)

		self.axis.set_ylim(0, self.error_max + 0.1)
		self.axis.set_xlim(1, self.total_trials + 1)

		matplotlib.animation.TimedAnimation.__init__(self, self.figure, interval=1./self.refreshRate*1000.0, repeat=False, blit=False)

	def __init__(self):
		rospy.init_node('MaePlot', log_level=rospy.INFO)

		#Retrieve RosLaunch parameters
		self.currentTopic = rospy.get_param("~current_topic")
		self.desiredTopic = rospy.get_param("~desired_topic")
		self.joint_list = rospy.get_param("~joint_list")
		self.figureName = rospy.get_param("~figure_name")
		self.refreshRate = rospy.get_param("~refresh_rate")
		self.duration_trial = rospy.get_param("~duration_trial")

		#Get global parameter use_sim_time and reference_time
		self.use_sim_time = rospy.get_param("use_sim_time")
		self.ref_time = rospy.get_param("reference_time")

		self.init_error = True
		self.start = 0.0

		self.total_calls = 0.0
		self.total_trials = 0

		self.trial_error = []
		self.trial_number = []

		self.desired_values = {}
		self.current_values = {}
		self.error = 0
		self.error_max = 0
		self.current_error = 0

		self.num_joints = len(self.joint_list)
		self.inverse_num_joints = 1.0 / self.num_joints

		self.first_desired = True
		self.first_current = True
		self.aux_desired_index = []
		self.aux_current_index = []


		for joint in self.joint_list:
			self.desired_values[joint] = 0
			self.current_values[joint] = 0

		# Define a lock for callback synchronization
		self.lock = threading.Lock()

		self.display = False

		#Simulation time in case use_sim_time True
		self.duration = rospy.Duration(0)
		self.checkpoint_time = rospy.Time(0)
		self.first = True
		self.sim_time = rospy.Time(0)

		#Subscribing to the topics
		rospy.Subscriber(self.currentTopic, edlut_ros.msg.AnalogCompact, self.CurrentCallback, queue_size=None)
		rospy.Subscriber(self.desiredTopic, edlut_ros.msg.AnalogCompact, self.DesiredCallback, queue_size=None)

		self.MAE_100 = []
		self.std_100 = 0.0
		self.mean_MAE_100 = 0.0


	def _draw_frame(self, framedata):
		self.lock.acquire()
		xdata, ydata = self.line.get_data()
		xdata = numpy.append(xdata, self.trial_number)
		ydata = numpy.append(ydata, self.trial_error)


		xmeandata, ymeandata = self.mean_line.get_data()
		index = 0
		while len(xmeandata)<len(xdata):
			counter = 0
			i = len(xdata)-1;
			mean = 0.0;
			while ( i >= 0 and counter < 50):
				mean+=ydata[i]
				i-=1
				counter+=1
			mean/=counter

			xmeandata = numpy.append(xmeandata, self.trial_number[index])
			ymeandata = numpy.append(ymeandata, mean)
			index+=1


		self.trial_number = []
		self.trial_error = []
		self.lock.release()

		self.line.set_data(xdata,ydata)
		self.mean_line.set_data(xmeandata,ymeandata)

		self._drawn_artists = [self.line]

		# self.axis.set_ylim(0.0, self.error_max + 0.1)
		self.axis.set_ylim(0.0, self.error_max)
		self.axis.set_xlim(1, self.total_trials + 1)

		return self._drawn_artists

	def new_frame_seq(self):
		return iter(xrange(sys.maxint))

	def _init_draw(self):
		lines = [self.line]
		for l in lines:
			l.set_data([], [])

	def close_display(self):
		self.display = False

Animation = MaeplotAnimation()
srv = Server(MaeDynParametersConfig, Animation.param_callback)
while not rospy.is_shutdown():
	if Animation.display:
		Animation.new_figure()
		matplotlib.pyplot.show()
		Animation.close_display()
	rospy.sleep(1)
