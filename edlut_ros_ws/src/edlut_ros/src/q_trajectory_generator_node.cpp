/***************************************************************************
 *                     q_trajectory_generator_node.cpp  	                 *
 *                           -------------------                           *
 * copyright            : (C) 2018 by Ignacio Abadia                       *
 * email                : iabadia@ugr.es         			                     *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

 // This is the trajectory generator node.
 // It generates trajectory position and velocity signals for every joint out of
 // .txt files in the format (one file for position, one for velocity):
 // J0 J1 J2 J3 J4 J5 J6
 // ...
 // J0 J1 J2 J3 J4 J5 J6

#include <edlut_ros/Analog.h>
#include <edlut_ros/AnalogCompact.h>
#include <baxter_core_msgs/JointCommand.h>
#include <sensor_msgs/JointState.h>
#include <edlut_ros/Q_TrajectoryGenerator.h>
#include <edlut_ros/Q_TrajectoryController.h>

#include "log4cxx/logger.h"
#include <ros/callback_queue.h>
#include <ros/ros.h>
#include <cstring>
#include <ctime>
#include <limits>
#include <signal.h>
#include <map>
#include "edlut_ros/ExternalClock.h"

/*--------------------------------------------------------------------
 * main()
 * Main function to  set up the delay node
 *------------------------------------------------------------------*/

static bool stop_node;

void rosShutdownHandler(int sig)
{
	stop_node = true;
}

int main(int argc, char **argv)
{
	// Set up ROS.
	ros::init(argc, argv, "sin_trajectory_generator", ros::init_options::NoSigintHandler);
	log4cxx::LoggerPtr my_logger = log4cxx::Logger::getLogger(ROSCONSOLE_DEFAULT_NAME);
	my_logger->setLevel(ros::console::g_level_lookup[ros::console::levels::Debug]);
	ros::NodeHandle nh;

	signal(SIGINT, rosShutdownHandler);

	// Declare variables that can be modified by launch file or command line.
	std::vector<std::string> joint_list;
	std::vector<double> phase, amplitude, frequency;
	std::string control_topic, joint_command_topic, joint_state_topic, desired_position_topic, desired_velocity_topic, positions_file_name, velocities_file_name;
	std::vector<double> max_pos_amplitude, min_pos_amplitude, max_vel_amplitude, min_vel_amplitude;
	double trial_length, update_frequency, trajectory_frequency;
	int number_of_trials, samples, columns;
	bool use_sim_time;

	stop_node = false;

	// Initialize node parameters from launch file or command line.
	// Use a private node handle so that multiple instances of the node can be run simultaneously
	// while using different parameters.
	ros::NodeHandle private_node_handle_("~");
	private_node_handle_.getParam("number_of_trials", number_of_trials);
	private_node_handle_.getParam("trial_length", trial_length);
	// private_node_handle_.getParam("joint_state_topic", joint_state_topic);
	private_node_handle_.getParam("joint_list", joint_list);
	private_node_handle_.getParam("max_pos_amplitude", max_pos_amplitude);
	private_node_handle_.getParam("min_pos_amplitude", min_pos_amplitude);
	private_node_handle_.getParam("max_vel_amplitude", max_vel_amplitude);
	private_node_handle_.getParam("min_vel_amplitude", min_vel_amplitude);
	private_node_handle_.getParam("update_frequency", update_frequency);
	private_node_handle_.getParam("samples", samples);
	private_node_handle_.getParam("trajectory_frequency", trajectory_frequency);
	private_node_handle_.getParam("positions_file_name", positions_file_name);
	private_node_handle_.getParam("velocities_file_name", velocities_file_name);


	// Output topics of the module
	private_node_handle_.getParam("control_topic", control_topic);
	private_node_handle_.getParam("desired_position_topic", desired_position_topic);
	private_node_handle_.getParam("desired_velocity_topic", desired_velocity_topic);
	private_node_handle_.getParam("joint_command_topic", joint_command_topic);

	nh.getParam("use_sim_time", use_sim_time);

	// Create the Sinusoidal Trajectory Generator
	Q_TrajectoryGenerator TrajGenerator (use_sim_time, max_pos_amplitude, min_pos_amplitude, max_vel_amplitude,
		min_vel_amplitude, samples, positions_file_name, velocities_file_name, trajectory_frequency);

	// Create the trajectory controller
	Q_TrajectoryController TrajController(joint_list,
			desired_position_topic,
			desired_velocity_topic,
			control_topic,
			joint_command_topic,
			//joint_state_topic,
			&TrajGenerator,
			number_of_trials,
			trial_length,
			update_frequency,
			use_sim_time);

	ROS_INFO("Baxter Sin Trajectory Generator node initialized");

	// Generate the sinusoidal trajectory until the specified total number of trials has been reached
	while (!stop_node && TrajController.NextControlStep(stop_node)){
	}

	ROS_INFO("Baxter Sin Trajectory Generator node ending");

	ros::shutdown();
	return 0;
} // end main()
