/***************************************************************************
 *                          gazebo_control_node.cpp                        *
 *                           -------------------                           *
 * copyright            : (C) 2018 by Ignacio Abadia                       *
 * email                : iabadia@ugr.es                              		 *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

// This node is used when running the experimental setup in Gazebo with
// simulated time

#include "ros/ros.h"
#include "ros/console.h"
#include <ros/callback_queue.h>
#include "rosgraph_msgs/Clock.h"
#include <ros/subscribe_options.h>

#include <edlut_ros/ExternalClock.h>


#include <cstring>
#include <ctime>
#include <limits>
#include <signal.h>
#include <stdlib.h>

#include <gazebo/gazebo.hh>
#include <gazebo/gazebo_client.hh>
#include <gazebo/common/Plugin.hh>
#include <gazebo/msgs/msgs.hh>
#include <gazebo/transport/transport.hh>

#include "std_msgs/Bool.h"

static bool stop_simulation;

void rosShutdownHandler(int sig)
{
	stop_simulation = true;
}

//NODE to control GAZEBO simulation --- NOT USED in the actual setup
int main(int argc, char **argv)
{
	// Set up ROS.
	ros::init(argc, argv, "synchronizer", ros::init_options::NoSigintHandler);
	if( ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info) ) {
   		ros::console::notifyLoggerLevelsChanged();
	}

	ros::NodeHandle nh;

	//Create publisher fot step_cmd topic. The gazebo plugin is subscribed to this
	//topic and every time it receives a message on this topic it will advance
	//the simulation one step.
	ros::Publisher step_pub = nh.advertise<std_msgs::Bool>("/step_cmd", 10000);
	std_msgs::Bool msg_stepcmd;
	msg_stepcmd.data = true;

	signal(SIGINT, rosShutdownHandler);

	// Declare variables that can be modified by launch file or command line.

	bool use_sim_time = false;
	double checking_frequency;
	std::string clock_topic;
	ros::Publisher time_publisher;
	ros::Subscriber clock_subscriber, gazebo_clock_subscriber;
	ExternalClock ext_clock;
	ExternalClock gazebo_clock; //--

	struct timespec startt, endt;

	stop_simulation = false;

	// Initialize node parameters from launch file or command line.
	// Use a private node handle so that multiple instances of the node can be run simultaneously
	// while using different parameters.
	ros::NodeHandle private_node_handle_("~");
	private_node_handle_.getParam("checking_frequency", checking_frequency);
	private_node_handle_.getParam("clock_topic", clock_topic);

	nh.getParam("use_sim_time", use_sim_time);

	ros::CallbackQueue CallbackQueue;
	nh.setCallbackQueue(&CallbackQueue);

	if (use_sim_time){
		ROS_DEBUG("GZ control: Subscribing to topic /clock_sync");  //change from "clock" to "clock_ext"
		clock_subscriber = nh.subscribe("/clock_sync", 1000, &ExternalClock::ClockCallback, &ext_clock);  //change from "clock" to "clock_ext"
		gazebo_clock_subscriber = nh.subscribe("/clock", 1000, &ExternalClock::ClockCallback, &gazebo_clock); //--
		time_publisher  = nh.advertise<rosgraph_msgs::Clock>(clock_topic, 1000);
		//ROS_DEBUG("RBF: Publishing simulation time to topic %s",clock_topic.c_str());
	}


	if (!use_sim_time){
		ROS_WARN("GZ control: Simulated time is not enabled. Gazebo_control_node will shutdown. If you want to enable simulated time set use_sim_time variable to true.");
		ros::shutdown();
		return 0;
	}

	ROS_INFO("GZ control node initialized.");

	ros::Time time(0.0), last_sent_time(0.0);
	ros::WallRate sim_rate(checking_frequency);
	ros::WallRate init_rate(1.0);


	if (use_sim_time){
		// Wait until the first signal has been received from the synchronizer node
		while(!ext_clock.FirstReceived()){
			ROS_DEBUG("GZ control: Synchronizing");
			CallbackQueue.callAvailable(ros::WallDuration(0.001));
			init_rate.sleep();
		}

		time = ext_clock.GetLastConfirmedTime();
		ROS_DEBUG("GZ control: Node synchronized");
	}

	bool first = true;

	//When Gazebo time is 0.0 the clock is not published yet. So it is needed to
	//publish it manually in this first step so the synchronizer node gets time 0
	//from Gazebo.
	if (first){
		first=false;
		rosgraph_msgs::Clock current_time;
		current_time.clock = gazebo_clock.GetLastConfirmedTime();
		time_publisher.publish(current_time);
	}

	//Wait for the Gazebo plugin to be subscribed to step_cmd topic
	do{
		ROS_INFO("Waiting for connection");
	}while (step_pub.getNumSubscribers() == 0);

	while (!stop_simulation){
		CallbackQueue.callAvailable(ros::WallDuration(0.0001));
		ros::Time new_time = ext_clock.GetLastConfirmedTime();

		if (new_time>time){
			time = new_time;
			step_pub.publish(msg_stepcmd);
			last_sent_time = gazebo_clock.GetLastConfirmedTime();
			ROS_DEBUG("GZ control: Moving forward one step to %fs", last_sent_time.toSec());
		}
		sim_rate.sleep();
	}

	ROS_INFO("Ending GZ control node");
	ros::shutdown();

	return 0;
} // end main()
