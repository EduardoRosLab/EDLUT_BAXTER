/***************************************************************************
 *                          error_estimator_node.cpp                       *
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

// This node computes the error signal comparing the desired and current state
// of the robot according to the specified gain assigned to each signal. 


#include <ros/ros.h>
#include <ros/callback_queue.h>
#include <edlut_ros/AnalogCompact.h>
#include <edlut_ros/ROSErrorCalculatorCompact.h>
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
	ros::init(argc, argv, "error_estimator_node");
	log4cxx::LoggerPtr my_logger = log4cxx::Logger::getLogger(ROSCONSOLE_DEFAULT_NAME);
	my_logger->setLevel(ros::console::g_level_lookup[ros::console::levels::Info]);
	ros::NodeHandle nh;

	signal(SIGINT, rosShutdownHandler);

	// Declare variables that can be modified by launch file or command line.
	std::string input_topic_pos_plus,  input_topic_vel_plus, input_topic_pos_minus, input_topic_vel_minus, output_topic;
	std::vector<std::string> joint_list;
	std::vector<double> error_pos_gain, error_vel_gain;
	double sampling_frequency;

	bool use_sim_time;

	ros::Subscriber clock_subscriber;
	ExternalClock ext_clock;

	ros::Time current_time(0.0);

	stop_node = false;

	// Initialize node parameters from launch file or command line.
	// Use a private node handle so that multiple instances of the node can be run simultaneously
	// while using different parameters.
	ros::NodeHandle private_node_handle_("~");
	private_node_handle_.getParam("input_topic_pos_plus", input_topic_pos_plus);
	private_node_handle_.getParam("input_topic_pos_minus", input_topic_pos_minus);
	private_node_handle_.getParam("input_topic_vel_plus", input_topic_vel_plus);
	private_node_handle_.getParam("input_topic_vel_minus", input_topic_vel_minus);
	private_node_handle_.getParam("sampling_frequency", sampling_frequency);
	private_node_handle_.getParam("joint_list", joint_list);
	private_node_handle_.getParam("error_position_gain", error_pos_gain);
	private_node_handle_.getParam("error_velocity_gain", error_vel_gain);
	private_node_handle_.getParam("output_topic", output_topic);


	nh.getParam("use_sim_time", use_sim_time);

	// Create the subscriber
	ros::CallbackQueue CallbackQueue;
	nh.setCallbackQueue(&CallbackQueue);

	if (use_sim_time){
		ROS_DEBUG("Error estimator node: Subscribing to topic /clock_sync");
		clock_subscriber = nh.subscribe("/clock_sync", 1000, &ExternalClock::ClockCallback, &ext_clock);
	}

	// Create the publisher
	ros::Publisher output_publisher = nh.advertise<edlut_ros::AnalogCompact>(output_topic, 1.0);
	ROSErrorCalculatorCompact objErrorCalculator(input_topic_pos_plus,
			input_topic_vel_plus,
			input_topic_pos_minus,
			input_topic_vel_minus,
			output_topic,
			joint_list,
			error_pos_gain,
			error_vel_gain);

	ROS_INFO("Error estimator node initialized: writing to topic %s with frequency %f", output_topic.c_str(), sampling_frequency);

	ros::Rate rate(sampling_frequency);

	while (!stop_node){
		CallbackQueue.callAvailable(ros::WallDuration(0.001));
		if (use_sim_time){
			ros::Time new_time = ext_clock.GetLastConfirmedTime();
			if (new_time > current_time){
				current_time = new_time;
				ROS_DEBUG("Updating error estimator at time %f", current_time.toSec());
				objErrorCalculator.UpdateError(current_time);
				rate.sleep();
				//ROS_INFO("Error node Current time: %f ", current_time.toSec());
			}
		}
		else{
			current_time = ros::Time::now();
			ROS_DEBUG("Updating error estimator at time %f", current_time.toSec());
			objErrorCalculator.UpdateError(current_time);
			rate.sleep();
		}
		//ROS_INFO("Error node Current time 2: %f ", current_time.toSec());
	}

	ROS_INFO("Ending error estimator node");

	ros::shutdown();
	return 0;
} // end main()
