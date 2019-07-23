#!/usr/bin/env python

##**************************************************************************
 #                           RasterPlot_2Entries.py                        *
 #                           -------------------                           *
 # copyright            : (C) 2017 by Jesus Garrido                        *
 # email                : jesusgarrido@ugr.es                              *
 #**************************************************************************/

##**************************************************************************
 #                                                                         *
 #   This program is free software; you can redistribute it and/or modify  *
 #   it under the terms of the GNU General Public License as published by  *
 #   the Free Software Foundation; either version 3 of the License, or     *
 #   (at your option) any later version.                                   *
 #                                                                         *
 #**************************************************************************/

# This script generates an animated plot of the spiking activity of a cerebellar
# layer and an analogue signal.

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
from edlut_ros.cfg import DynParametersConfig

class RasterplotAnimation(matplotlib.animation.TimedAnimation):
	def SpikeCallback(self, data):
		# Get reference time from EDLUT once the simulator is running
		if self.first_spike and rospy.get_param("reference_time")!=0.0:
			self.first_spike = False
			self.checkpoint_time = rospy.get_param("reference_time")
		if self.display:
			# Append those spikes whose indexes are within the boundaries.
			for x in range(len(data.neuron_index)):
				if data.neuron_index[x] >= self.minNeuron_original and data.neuron_index[x] <= self.maxNeuron_original:
					self.lock.acquire()
					if self.use_sim_time:
						self.NeuronTime.append(data.time[x])
					else:
						self.NeuronTime.append(data.time[x] - self.checkpoint_time)
					self.NeuronIndex.append(data.neuron_index[x])
					self.lock.release()

			rospy.logdebug("Plot: %f. Spike received. Time %f and neuron %i", rospy.get_time(), data.time, data.neuron_index)

	def AnalogCallback(self, data):
		# Get reference time from EDLUT once the simulator is running
		if self.first_analog_signal and rospy.get_param("reference_time")!=0.0:
			self.first_analog_signal = False
			self.checkpoint_time = rospy.get_param("reference_time")
		if self.display:
			index_joint = -1
			for joint in self.joint_list:
				index_joint += 1
				index_name = -1
				for name in data.names:
					index_name += 1
					if joint == name:
						self.lock.acquire()
						self.analog_value_matrix[index_joint].append(data.data[index_name])
						if data.data[index_name] > self.maxAnalog:
							self.maxAnalog = data.data[index_name]
						elif data.data[index_name] < self.minAnalog:
							self.minAnalog = data.data[index_name]
						if self.use_sim_time:
							self.analog_time_matrix[index_joint].append(data.header.stamp)
						else:
							self.analog_time_matrix[index_joint].append(data.header.stamp.to_sec() - self.checkpoint_time)
						self.lock.release()



	def callback(self, clock):
		if not self.stop_plot:
			self.sim_time = clock.clock


	# Callback to dynamically change parameters
	def param_callback(self, config, level):
		if self.first_param:
			self.minNeuron = self.minNeuron_original
			self.maxNeuron = self.maxNeuron_original
		else:
			self.time_window = config["Time_Window"]
			first_neuron = config["Min_Neuron"]
			last_neuron = config["Max_Neuron"]
			self.stop_plot = config["Stop"]
			self.display = config["Display"]
			if (first_neuron >= self.minNeuron_original) and (first_neuron <= self.maxNeuron_original):
				self.minNeuron = first_neuron
			if (last_neuron <= self.maxNeuron_original) and (last_neuron >= self.minNeuron_original):
				self.maxNeuron = last_neuron
		self.first_param = False
		return config

	def new_figure(self):
		#Subscribing to the topics and creating the plot
		self.spike_subscriber = rospy.Subscriber(self.spike_inputTopic, edlut_ros.msg.Spike_group, self.SpikeCallback, queue_size=None)
		self.analog_subscriber = rospy.Subscriber(self.analog_inputTopic, edlut_ros.msg.AnalogCompact, self.AnalogCallback, queue_size=None)
		# Create the figure and the axis
		self.figure = matplotlib.pyplot.figure()
		self.axis1 = self.figure.add_subplot(2, 1, 1)
		self.axis1.set_title(self.figureName)

		self.axis1.set_xlabel('Time (s)')
		self.axis1.set_ylabel('Neuron Index')
		self.line_spike = matplotlib.pyplot.Line2D([], [], color='red', marker='.', markersize=3,linestyle='None', label = self.spike_inputTopic)
		self.axis1.add_line(self.line_spike)

		self.axis1.set_ylim(self.minNeuron -1, self.maxNeuron+1)


		if self.use_sim_time:
			self.axis1.set_xlim(self.sim_time.to_sec() - self.time_window , self.sim_time.to_sec())
		else:
			self.axis1.set_xlim(rospy.get_time()- self.time_window - self.checkpoint_time, rospy.get_time() - self.checkpoint_time)

		self.axis2 = self.figure.add_subplot(2, 1, 2)

		self.axis2.set_ylabel('Analog Value')

		self.analog_lines = []

		self.colours = ["blue", "red", "green", "yellow", "magenta", "black", "cyan"]

		for line in range(self.n_joints):
			self.analog_lines.append(matplotlib.pyplot.Line2D([], [], color=self.colours[line], linestyle='-', label = self.joint_list[line]))
			self.axis2.add_line(self.analog_lines[line])
		self.axis2.legend(loc="upper left")
		self.axis2.set_ylim(self.minAnalog, self.maxAnalog)

		if self.use_sim_time:
			self.axis2.set_xlim(self.sim_time.to_sec() - self.time_window , self.sim_time.to_sec())
		else:
			self.axis2.set_xlim(rospy.get_time()- self.time_window - self.checkpoint_time, rospy.get_time() - self.checkpoint_time)

		matplotlib.animation.TimedAnimation.__init__(self, self.figure, interval=1./self.refreshRate*1000.0, repeat=False, blit=False)

	# Kill subscriber
	def end_subscriber(self):
		self.spike_subscriber.unregister()
		self.analog_subscriber.unregister()


	def __init__(self):
		rospy.init_node('RasterPlot', log_level=rospy.INFO)

		#Retrieve RosLaunch parameters
		self.spike_inputTopic = rospy.get_param("~spike_input_topic")
		self.analog_inputTopic = rospy.get_param("~analog_input_topic")
		self.figureName = rospy.get_param("~figure_name")
		self.minNeuron_original = rospy.get_param("~min_neuron_index")
		self.maxNeuron_original = rospy.get_param("~max_neuron_index")
		self.joint_list = rospy.get_param("~joint_list")
		self.refreshRate = rospy.get_param("~refresh_rate")

		#Get global parameter use_sim_time and reference_time
		self.use_sim_time = rospy.get_param("use_sim_time")
		self.ref_time = rospy.get_param("reference_time")

		self.minNeuron = self.minNeuron_original
		self.maxNeuron = self.maxNeuron_original
		self.minAnalog = 0.0
		self.maxAnalog = 0.0
		self.first_param = True

		self.first_spike = True
		self.first_analog_signal = True

		self.NeuronTime = []
		self.NeuronIndex = []

		self.n_joints = len(self.joint_list)
		self.analog_value_matrix = [[]for _ in range(self.n_joints)]
		self.analog_time_matrix = [[]for _ in range(self.n_joints)]

		# Define a lock for callback synchronization
		self.lock = threading.Lock()

		#Create parameters that can be modified dynamically to change the graph online
		self.time_window = 1.0
		self.stop_plot = False
		self.display = False

		#Simulation time in case use_sim_time True
		self.sim_time = rospy.Time(0)

		#Start time to normalize x-axis when use_sim_time = False
		self.checkpoint_time = rospy.get_time()

		if self.use_sim_time:
 			self.clock_sub = rospy.Subscriber("/clock_sync", Clock, self.callback, queue_size=1000)

		# Initialize spike subscriber (mandatory so spike_subscriber exists) and unregister it
		# The subscriber will be registered again when the plotting occurs, meanwhile, for computation
		# economy the subscriber is closed so all the spikes are not processed while the plot is not
		# being showed.
		self.spike_subscriber = rospy.Subscriber(self.spike_inputTopic, edlut_ros.msg.Spike, self.SpikeCallback, queue_size=None)
		self.spike_subscriber.unregister()

		self.analog_subscriber = rospy.Subscriber(self.analog_inputTopic, edlut_ros.msg.AnalogCompact, self.AnalogCallback, queue_size=None)
		self.analog_subscriber.unregister()

	def _draw_frame(self, framedata):
		if self.display:
			if not self.stop_plot:
				self.lock.acquire()
				xdata1, ydata1 = self.line_spike.get_data()
				xdata1 = numpy.append(xdata1, self.NeuronTime)
				ydata1 = numpy.append(ydata1, self.NeuronIndex)
				self.line_spike.set_data(xdata1,ydata1)

				for i in range(self.n_joints):
					xdata2, ydata2 = self.analog_lines[i].get_data()
					xdata2 = numpy.append(xdata2, self.analog_time_matrix[i])
					ydata2 = numpy.append(ydata2, self.analog_value_matrix[i])
					self.analog_lines[i].set_data(xdata2,ydata2)

				self.NeuronTime = []
				self.NeuronIndex = []

				self.analog_value_matrix = [[]for _ in range(self.n_joints)]
				self.analog_time_matrix = [[]for _ in range(self.n_joints)]

				self.lock.release()


				if self.stop_plot:
					self._drawn_artists = []
				else:
					self._drawn_artists = [self.line_spike]
					for x in range(self.n_joints):
						self._drawn_artists.append(self.analog_lines[x])

				self.axis1.set_ylim(self.minNeuron - 1, self.maxNeuron+1)
				self.axis2.set_ylim(self.minAnalog, self.maxAnalog)

				if self.use_sim_time:
					self.axis1.set_xlim(self.sim_time.to_sec() - self.time_window , self.sim_time.to_sec())
					self.axis2.set_xlim(self.sim_time.to_sec() - self.time_window , self.sim_time.to_sec())
				else:
					self.axis1.set_xlim(rospy.get_time() - self.time_window - self.checkpoint_time, rospy.get_time() - self.checkpoint_time)
					self.axis2.set_xlim(rospy.get_time() - self.time_window - self.checkpoint_time, rospy.get_time() - self.checkpoint_time)

				return self._drawn_artists

	def new_frame_seq(self):
		if self.display:
			return iter(xrange(sys.maxint))

	def _init_draw(self):
		if self.display:
			lines = [self.line_spike]
			for x in range(self.n_joints):
				lines.append(self.analog_lines[x])
			for l in lines:
				l.set_data([], [])

	def close_display(self):
		self.display = False

Animation = RasterplotAnimation()
srv = Server(DynParametersConfig, Animation.param_callback)
while not rospy.is_shutdown():
	if Animation.display:
		Animation.new_figure()
		matplotlib.pyplot.show()
		Animation.close_display()
	Animation.end_subscriber()
 	rospy.sleep(1)
