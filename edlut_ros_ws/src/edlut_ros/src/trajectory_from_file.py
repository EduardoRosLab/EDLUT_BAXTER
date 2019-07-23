#!/usr/bin/env python

##**************************************************************************
 #                           trajectory_from_file.py                         *
 #                           -------------------                           *
 # copyright            : (C) 2018 by Ignacio Abadia                       *
 # email                : iabadia@ugr.es                              *
 #**************************************************************************/

##**************************************************************************
 #                                                                         *
 #   This program is free software; you can redistribute it and/or modify  *
 #   it under the terms of the GNU General Public License as published by  *
 #   the Free Software Foundation; either version 3 of the License, or     *
 #   (at your option) any later version.                                   *
 #                                                                         *
 #**************************************************************************/

# This node reads a file with a series of joints angles and applies them to Baxter
# to follow the trajectory

import rospy
from std_msgs.msg import Time
from std_msgs.msg import Bool
import numpy as np
import baxter_interface
from baxter_interface import CHECK_VERSION
from baxter_core_msgs.msg import (
	JointCommand,
)

from mpl_toolkits import mplot3d
#%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class Trajectory_file(object):

	def __init__(self):
		# Publisher to put baxter's left arm in 0 position
		self.pub_position = rospy.Publisher('robot/limb/left/joint_command', JointCommand, queue_size = 10)

		# Baxter interface to have access to the state of the robot
		self.rs = baxter_interface.RobotEnable(CHECK_VERSION)

		# Baxter limb to have access to the robot's position
		self.limb = baxter_interface.Limb('left')

		self.file_read = open("/home/baxter/catkin_ws/src/BaxterCerebellum/src/edlut_ros/src/target_reaching/positions_target_reaching_1_5s.txt", "r")

		# Variable to store baxter state
		self.enabled = False

		#self.file_write = open("endpoint_pos.txt", "w")
		# Message components to put arm in 0 position (mode 1 = position control mode)
		self.mode = 1
		self.zero_position = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
		self.names = ['left_s0','left_s1', 'left_e0', 'left_e1', 'left_w0', 'left_w1', 'left_w2']
		self.names_dict = {'left_s0': 0.0, 'left_s1':0.0, 'left_e0':0.0, 'left_e1':0.0, 'left_w0':0.0, 'left_w1':0.0, 'left_w2':0.0}

		self.rate = rospy.Rate(500)

		self.path_x = []
		self.path_y = []
		self.path_z = []

	# Function to put baxter's left arm in 0 position
	def zero_position(self):
		self.pub_position.publish(self.mode, self.zero_position, self.names)

	# Function to enable the robot
	def robot_enabler(self):
		r = rospy.Rate(10) # 10hz
		self.rs.enable()

	def move_to_start(self):
		first_pos = self.file_read.readline()
		angles = first_pos.split()
		#angles[0] = angles[0].replace("(", "")
		## new -- try moving with limb object ###
		self.names_dict["left_s0"] = float(angles[0])
		self.names_dict["left_s1"] = float(angles[1])
		self.names_dict["left_e0"] = float(angles[2])
		self.names_dict["left_e1"] = float(angles[3])
		self.names_dict["left_w0"] = float(angles[4])
		self.names_dict["left_w1"] = float(angles[5])
		self.names_dict["left_w2"] = 0.0
		self.limb.move_to_joint_positions(self.names_dict, 1.0, 0.008)
		r=rospy.Rate(1)
		r.sleep()

	# Function to publish a series of joints angles read from a file.
	# A trajectory can be stored in a txt file and then sent to Baxter to follow it
	def traj_from_file(self):
		r=rospy.Rate(100)
		pos = []
		for line in self.file_read:
			angles = line.split()
			angles[0] = angles[0].replace("(", "")
			angles[6] = angles[6].replace(")", "")
			for x in range (0,7):
				if (x==6):
					pos.append(0.0)
				else:
					pos.append(float(angles[x]))
			self.pub_position.publish(self.mode, pos, self.names)
			print pos
			pos = []
			r.sleep()
			coordinates = self.limb.endpoint_pose()
			self.path_x.append(coordinates["position"][0])
			self.path_y.append(coordinates["position"][1])
			self.path_z.append(coordinates["position"][2])
			print coordinates

	def close_file(self):
		self.file_read.close()

	def draw_path(self):
		print "Drawing.."
		fig = plt.figure()
		ax = Axes3D(fig)
		ax.plot(self.path_x, self.path_y, '-b')
		plt.show()

	def write_file(self):
		file = open("position_trajectory.txt", "w")
		for i in range (len(self.path_x)):
			file.write(str(self.path_x[i])+" "+str(self.path_y[i])+" "+str(self.path_z[i])+"\n")
		file.close()

def main():
	rospy.init_node('robot_enabler', anonymous=True, disable_signals = True)
	path = Trajectory_file()
	path.robot_enabler()
	path.move_to_start()
	path.traj_from_file()
	path.close_file()
	path.write_file()
	path.draw_path()

if __name__ == '__main__':
	try:
		main()
	except rospy.ROSInterruptException:
		pass
