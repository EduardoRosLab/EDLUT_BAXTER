/***************************************************************************
 *                           delay_analog_node.cpp                         *
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

// This node adds a delay to an analogue signal

#include <ros/ros.h>
#include <ros/callback_queue.h>
#include <edlut_ros/AnalogCompact.h>
#include <message_filters/subscriber.h>
#include <message_filters/time_sequencer.h>
#include "log4cxx/logger.h"

#include <cstring>
#include <ctime>
#include <limits>
#include <signal.h>


static bool stop_node;

/*
 * Create a class
 */
class Delayer {
private:
	double delay;

	ros::Publisher * Publisher;

public:
	Delayer (double delay, ros::Publisher * Publisher): delay(delay), Publisher(Publisher){};

	void AnalogCallback(const edlut_ros::AnalogCompact::ConstPtr& msg){
		edlut_ros::AnalogCompact analog;
		analog.header.stamp = msg->header.stamp + ros::Duration(this->delay);
		analog.data = msg->data;
		analog.names = msg->names;
		//ROS_INFO("Sending message with stamp %f: Original stamp: %f",analog.header.stamp.toSec(), msg->header.stamp.toSec());
		this->Publisher->publish(analog);
		return;
	};
};

void rosShutdownHandler(int sig)
{
	stop_node = true;
}

int main(int argc, char **argv)
{
	// Set up ROS.
	ros::init(argc, argv, "delay_analog", ros::init_options::NoSigintHandler);
	log4cxx::LoggerPtr my_logger = log4cxx::Logger::getLogger(ROSCONSOLE_DEFAULT_NAME);
	my_logger->setLevel(ros::console::g_level_lookup[ros::console::levels::Info]);
	ros::NodeHandle nh;

	signal(SIGINT, rosShutdownHandler);

	// Declare variables that can be modified by launch file or command line.
	std::string input_topic, output_topic;
	double delay;

	// Initialize node parameters from launch file or command line.
	// Use a private node handle so that multiple instances of the node can be run simultaneously
	// while using different parameters.
	ros::NodeHandle private_node_handle_("~");
	private_node_handle_.getParam("input_topic", input_topic);
	private_node_handle_.getParam("output_topic", output_topic);
	private_node_handle_.getParam("delay", delay);

	// Create the publisher
	ros::Publisher output_publisher = nh.advertise<edlut_ros::AnalogCompact>(output_topic, 1e9);
	Delayer objDelayer = Delayer(delay, &output_publisher);

	// Create the subscriber

	ros::CallbackQueue CallbackQueue;
	nh.setCallbackQueue(&CallbackQueue);

	message_filters::Subscriber<edlut_ros::AnalogCompact> input_subscriber (nh, input_topic, 1e9, ros::TransportHints(), &CallbackQueue);
	message_filters::TimeSequencer<edlut_ros::AnalogCompact> seq(input_subscriber, ros::Duration(delay), ros::Duration(0.001), 1e9, nh);
	seq.registerCallback(&Delayer::AnalogCallback, &objDelayer);

	ROS_INFO("Delay node initialized: reading from topic %s, writing to topic %s and using delay %f", input_topic.c_str(), output_topic.c_str(), delay);

	while (!stop_node){
		CallbackQueue.callAvailable(ros::WallDuration(0.010));
	}

	ROS_INFO("Ending delay node");

	ros::shutdown();
	return 0;
} // end main()
