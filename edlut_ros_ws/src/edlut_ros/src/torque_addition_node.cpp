/***************************************************************************
 *                          torque_addition_node.cpp                       *
 *                           -------------------                           *
 * copyright            : (C) 2016 by Jesus Garrido                        *
 * email                : jesusgarrido@ugr.es                              *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

 // This node adds two torque signals.

#include <ros/ros.h>
#include <ros/callback_queue.h>
#include <edlut_ros/AnalogCompact.h>
#include <edlut_ros/ROSTorqueCalculatorAddition.h>
#include "log4cxx/logger.h"
#include "edlut_ros/ExternalClock.h"


#include <cstring>
#include <ctime>
#include <cmath>
#include <signal.h>


static bool stop_node;

void rosShutdownHandler(int sig)
{
	stop_node = true;
}



int main(int argc, char **argv)
{
	// Set up ROS.
	ros::init(argc, argv, "torque_estimator_node");
	log4cxx::LoggerPtr my_logger = log4cxx::Logger::getLogger(ROSCONSOLE_DEFAULT_NAME);
	my_logger->setLevel(ros::console::g_level_lookup[ros::console::levels::Info]);
	ros::NodeHandle nh;

	signal(SIGINT, rosShutdownHandler);

	// Declare variables that can be modified by launch file or command line.
	std::string input_topic_torque_1,  input_topic_torque_2, output_topic;
	std::vector<std::string> joint_list;
	double sampling_frequency;
	ros::Subscriber clock_subscriber;
	ExternalClock ext_clock;
	bool use_sim_time;

	stop_node = false;

	// Initialize node parameters from launch file or command line.
	// Use a private node handle so that multiple instances of the node can be run simultaneously
	// while using different parameters.
	ros::NodeHandle private_node_handle_("~");
	private_node_handle_.getParam("input_topic_torque_1", input_topic_torque_1);
	private_node_handle_.getParam("input_topic_torque_2", input_topic_torque_2);
	private_node_handle_.getParam("sampling_frequency", sampling_frequency);
	private_node_handle_.getParam("joint_list", joint_list);
	private_node_handle_.getParam("output_topic", output_topic);

	nh.getParam("use_sim_time", use_sim_time);

	// Create the subscriber
	ros::CallbackQueue CallbackQueue;
	nh.setCallbackQueue(&CallbackQueue);

	if (use_sim_time){
		ROS_DEBUG("RBF: Subscribing to topic /clock_sync");  //change from "clock" to "clock_ext"
		clock_subscriber = nh.subscribe("/clock_sync", 1000, &ExternalClock::ClockCallback, &ext_clock);
	}

	// Create the publisher
	ros::Publisher output_publisher = nh.advertise<edlut_ros::AnalogCompact>(output_topic, 1.0);
	ROSTorqueCalculatorAddition objTorqueCalculator(input_topic_torque_1,
			input_topic_torque_2,
			output_topic,
			joint_list);

	ROS_INFO("Torque estimator node initialized: writing to topic %s with frequency %f", output_topic.c_str(), sampling_frequency);

	ros::Rate rate(sampling_frequency);

	while (!stop_node){
		CallbackQueue.callAvailable(ros::WallDuration(0.001));
		ros::Time current_time;
		if (use_sim_time){
			current_time = ext_clock.GetLastConfirmedTime();
		}
		else{
			current_time = ros::Time::now();
		}
		ROS_DEBUG("Updating torque estimator at time %f", current_time.toSec());
		objTorqueCalculator.UpdateTorque(current_time);
		rate.sleep();
	}

	ROS_INFO("Ending torque estimator node");

	ros::shutdown();
	return 0;
} // end main()
