/***************************************************************************
 *                        p_supervisor_node.cpp                   				 *
 *                           -------------------                           *
 * copyright            : (C) 2018 by Francisco Naveros                    *
 * email                : fnaveros@ugr.es                                  *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

// This is the PD Supervisor node
// Its function is to supervise that the joints are kept within their working
// area. Only at the event of a joint going outside the working-area, the PD
// supervisor node adds a torque command to the cerebellar controller torques.


#include <edlut_ros/Analog.h>
#include <edlut_ros/PDSupervisor.h>
#include "log4cxx/logger.h"

#include <baxter_core_msgs/JointCommand.h>

#include <ros/callback_queue.h>
#include <ros/ros.h>
#include <cstring>
#include <ctime>
#include <limits>
#include <signal.h>
#include <map>

static bool stop_node;

void rosShutdownHandler(int sig)
{
	stop_node = true;
}


int main(int argc, char **argv)
{
	// Set up ROS.
	ros::init(argc, argv, "p_supervisor_node", ros::init_options::NoSigintHandler);
  log4cxx::LoggerPtr my_logger = log4cxx::Logger::getLogger(ROSCONSOLE_DEFAULT_NAME);
  my_logger->setLevel(ros::console::g_level_lookup[ros::console::levels::Info]);
  ros::NodeHandle nh;

	signal(SIGINT, rosShutdownHandler);

	// Declare variables that can be modified by launch file or command line.
	std::vector<std::string> joint_list;
	std::vector<double> min_position, max_position;
	std::string current_position_topic, current_velocity_topic, torque_topic;
  std::vector<double> kp;
	std::vector<double> kd;
	double update_frequency;
	bool use_sim_time;

	stop_node = false;

	// Initialize node parameters from launch file or command line.
	// Use a private node handle so that multiple instances of the node can be run simultaneously
	// while using different parameters.
	ros::NodeHandle private_node_handle_("~");
	private_node_handle_.getParam("current_position_topic", current_position_topic);
	private_node_handle_.getParam("current_velocity_topic", current_velocity_topic);
	private_node_handle_.getParam("output_topic", torque_topic);
	private_node_handle_.getParam("update_frequency", update_frequency);
  private_node_handle_.getParam("joint_list", joint_list);
	private_node_handle_.getParam("kp", kp);
	private_node_handle_.getParam("kd", kd);
	private_node_handle_.getParam("min_position_list", min_position);
	private_node_handle_.getParam("max_position_list", max_position);
	// Output topics of the module


	nh.getParam("use_sim_time", use_sim_time);

	// Create the trajectory controller
	PDSupervisor Supervisor(current_position_topic,
		current_velocity_topic,
		torque_topic,
		joint_list,
		kp,
		kd,
		min_position,
		max_position,
		use_sim_time);

  ros::Rate update(update_frequency);

	ROS_INFO("Baxter P Supervisor node initialized");

	while (!stop_node){
    Supervisor.NextSupervisorStep();
    update.sleep();
	}

	ROS_INFO("Baxter P Supervisor node ending");

	ros::shutdown();
	return 0;
} // end main()
