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

# This script generates an animated plot of the spiking activity of two different
# cerebellar layers, specifying the min and max neuron of each layer.

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
from edlut_ros.cfg import DynParameters_2EntriesConfig

class RasterplotAnimation(matplotlib.animation.TimedAnimation):
	def FirstSpikeCallback(self, data):
		# Get reference time from EDLUT once the simulator is running
		if self.first_first_spike and rospy.get_param("reference_time")!=0.0:
			self.first_first_spike = False
			self.first_checkpoint_time = rospy.get_param("reference_time")
		if self.display:
			# Append those spikes whose indexes are within the boundaries.
			for x in range(len(data.neuron_index)):
				if data.neuron_index[x] >= self.first_minNeuron_original and data.neuron_index[x] <= self.first_maxNeuron_original:
					self.lock.acquire()
					if self.use_sim_time:
						self.first_NeuronTime.append(data.time[x])
					else:
						self.first_NeuronTime.append(data.time[x] - self.first_checkpoint_time)
					self.first_NeuronIndex.append(data.neuron_index[x])
					self.lock.release()

			rospy.logdebug("Plot: %f. Spike received. Time %f and neuron %i", rospy.get_time(), data.time, data.neuron_index)

	def SecondSpikeCallback(self, data):
		# Get reference time from EDLUT once the simulator is running
		if self.second_first_spike and rospy.get_param("reference_time")!=0.0:
			self.second_first_spike = False
			self.second_checkpoint_time = rospy.get_param("reference_time")
		if self.display:
			# Append those spikes whose indexes are within the boundaries.
			for x in range(len(data.neuron_index)):
				if data.neuron_index[x] >= self.second_minNeuron_original and data.neuron_index[x] <= self.second_maxNeuron_original:
					self.lock.acquire()
					if self.use_sim_time:
						self.second_NeuronTime.append(data.time[x])
					else:
						self.second_NeuronTime.append(data.time[x] - self.second_checkpoint_time)
					self.second_NeuronIndex.append(data.neuron_index[x])
					self.lock.release()

			rospy.logdebug("Plot: %f. Spike received. Time %f and neuron %i", rospy.get_time(), data.time, data.neuron_index)

	def callback(self, clock):
		if not self.stop_plot:
			self.sim_time = clock.clock


	# Callback to dynamically change parameters
	def param_callback(self, config, level):
		if self.first_param:
			self.first_minNeuron = self.first_minNeuron_original
			self.first_maxNeuron = self.first_maxNeuron_original
			self.second_minNeuron = self.second_minNeuron_original
			self.second_maxNeuron = self.second_maxNeuron_original
		else:
			self.time_window = config["Time_Window"]
			first_first_neuron = config["First_Min_Neuron"]
			first_last_neuron = config["First_Max_Neuron"]
			second_first_neuron = config["Second_Min_Neuron"]
			second_last_neuron = config["Second_Max_Neuron"]
			self.stop_plot = config["Stop"]
			self.display = config["Display"]
			if (first_first_neuron >= self.first_minNeuron_original) and (first_first_neuron <= self.first_maxNeuron_original):
				self.first_minNeuron = first_first_neuron
			if (first_last_neuron <= self.first_maxNeuron_original) and (first_last_neuron >= self.first_minNeuron_original):
				self.first_maxNeuron = first_last_neuron
			if (second_first_neuron >= self.second_minNeuron_original) and (second_first_neuron <= self.second_maxNeuron_original):
				self.second_minNeuron = second_first_neuron
			if (second_last_neuron <= self.second_maxNeuron_original) and (second_last_neuron >= self.second_minNeuron_original):
				self.second_maxNeuron = second_last_neuron
		self.first_param = False
		return config

	def new_figure(self):
		#Subscribing to the topics and creating the plot
		self.first_spike_subscriber = rospy.Subscriber(self.first_inputTopic, edlut_ros.msg.Spike_group, self.FirstSpikeCallback, queue_size=None)
		self.second_spike_subscriber = rospy.Subscriber(self.second_inputTopic, edlut_ros.msg.Spike_group, self.SecondSpikeCallback, queue_size=None)
		# Create the figure and the axis
		self.figure = matplotlib.pyplot.figure()
		self.axis1 = self.figure.add_subplot(1, 1, 1)
		self.axis1.set_title(self.figureName)

		self.axis1.set_xlabel('Time (s)')
		self.axis1.set_ylabel('First Neuron Index')
		self.line1 = matplotlib.pyplot.Line2D([], [], color='red', marker='.', markersize=3,linestyle='None', label = self.first_inputTopic)
		self.axis1.add_line(self.line1)

		self.axis1.set_ylim(self.first_minNeuron -1, self.first_maxNeuron+1)


		if self.use_sim_time:
			self.axis1.set_xlim(self.sim_time.to_sec() - self.time_window , self.sim_time.to_sec())
		else:
			self.axis1.set_xlim(rospy.get_time()- self.time_window - self.first_checkpoint_time, rospy.get_time() - self.first_checkpoint_time)

		self.axis2 = self.axis1.twinx()

		self.axis2.set_ylabel('Second Neuron Index')
		self.line2 = matplotlib.pyplot.Line2D([], [], color='blue', marker='.', markersize=3,linestyle='None', label = self.second_inputTopic)
		self.axis2.add_line(self.line2)

		self.axis2.set_ylim(self.second_minNeuron -1, self.second_maxNeuron+1)

		matplotlib.animation.TimedAnimation.__init__(self, self.figure, interval=1./self.refreshRate*1000.0, repeat=False, blit=False)

	# Kill subscriber
	def end_subscriber(self):
		self.first_spike_subscriber.unregister()
		self.second_spike_subscriber.unregister()


	def __init__(self):
		rospy.init_node('RasterPlot', log_level=rospy.INFO)

		#Retrieve RosLaunch parameters
		self.first_inputTopic = rospy.get_param("~first_input_topic")
		self.second_inputTopic = rospy.get_param("~second_input_topic")
		self.figureName = rospy.get_param("~figure_name")
		self.first_minNeuron_original = rospy.get_param("~first_min_neuron_index")
		self.first_maxNeuron_original = rospy.get_param("~first_max_neuron_index")
		self.second_minNeuron_original = rospy.get_param("~second_min_neuron_index")
		self.second_maxNeuron_original = rospy.get_param("~second_max_neuron_index")
		self.refreshRate = rospy.get_param("~refresh_rate")

		#Get global parameter use_sim_time and reference_time
		self.use_sim_time = rospy.get_param("use_sim_time")
		self.ref_time = rospy.get_param("reference_time")

		self.first_minNeuron = self.first_minNeuron_original
		self.first_maxNeuron = self.first_maxNeuron_original
		self.second_minNeuron = self.second_minNeuron_original
		self.second_maxNeuron = self.second_maxNeuron_original
		self.first_param = True

		self.first_first_spike = True
		self.second_first_spike = True


		self.first_NeuronTime = []
		self.first_NeuronIndex = []
		self.second_NeuronTime = []
		self.second_NeuronIndex = []

		# Define a lock for callback synchronization
		self.lock = threading.Lock()

		#Create parameters that can be modified dynamically to change the graph online
		self.time_window = 1.0
		self.stop_plot = False
		self.display = False

		#Simulation time in case use_sim_time True
		self.sim_time = rospy.Time(0)

		#Start time to normalize x-axis when use_sim_time = False
		self.first_checkpoint_time = rospy.get_time()
		self.second_checkpoint_time = rospy.get_time()


		if self.use_sim_time:
 			self.clock_sub = rospy.Subscriber("/clock_sync", Clock, self.callback, queue_size=1000)

		# Initialize spike subscriber (mandatory so spike_subscriber exists) and unregister it
		# The subscriber will be registered again when the plotting occurs, meanwhile, for computation
		# economy the subscriber is closed so all the spikes are not processed while the plot is not
		# being showed.
		self.first_spike_subscriber = rospy.Subscriber(self.first_inputTopic, edlut_ros.msg.Spike, self.FirstSpikeCallback, queue_size=None)
		self.first_spike_subscriber.unregister()

		self.second_spike_subscriber = rospy.Subscriber(self.second_inputTopic, edlut_ros.msg.Spike, self.SecondSpikeCallback, queue_size=None)
		self.second_spike_subscriber.unregister()

	def _draw_frame(self, framedata):
		if self.display:
			if not self.stop_plot:
				self.lock.acquire()
				xdata1, ydata1 = self.line1.get_data()
				xdata1 = numpy.append(xdata1, self.first_NeuronTime)
				ydata1 = numpy.append(ydata1, self.first_NeuronIndex)
				xdata2, ydata2 = self.line2.get_data()
				xdata2 = numpy.append(xdata2, self.second_NeuronTime)
				ydata2 = numpy.append(ydata2, self.second_NeuronIndex)

				self.first_NeuronTime = []
				self.first_NeuronIndex = []
				self.second_NeuronTime = []
				self.second_NeuronIndex = []
				self.lock.release()

				self.line1.set_data(xdata1,ydata1)
				self.line2.set_data(xdata2,ydata2)

				if self.stop_plot:
					self._drawn_artists = []
				else:
					self._drawn_artists = [self.line1, self.line2]

				self.axis1.set_ylim(self.first_minNeuron - 1, self.first_maxNeuron+1)
				self.axis2.set_ylim(self.second_minNeuron - 1, self.second_maxNeuron+1)

				if self.use_sim_time:
					self.axis1.set_xlim(self.sim_time.to_sec() - self.time_window , self.sim_time.to_sec())

				else:
					self.axis1.set_xlim(rospy.get_time() - self.time_window - self.first_checkpoint_time, rospy.get_time() - self.first_checkpoint_time)

				return self._drawn_artists

	def new_frame_seq(self):
		if self.display:
			return iter(xrange(sys.maxint))

	def _init_draw(self):
		if self.display:
			lines = [self.line1, self.line2]
			for l in lines:
				l.set_data([], [])

	def close_display(self):
		self.display = False

Animation = RasterplotAnimation()
srv = Server(DynParameters_2EntriesConfig, Animation.param_callback)
while not rospy.is_shutdown():
	if Animation.display:
		Animation.new_figure()
		matplotlib.pyplot.show()
		Animation.close_display()
	Animation.end_subscriber()
 	rospy.sleep(1)
