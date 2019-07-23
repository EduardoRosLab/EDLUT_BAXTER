/***************************************************************************
 *                          synchronizer_node.cpp                          *
 *                           -------------------                           *
 * copyright            : (C) 2017 by Jesus Garrido                        *
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

// This is the synchronizer node. It controls that all nodes follow the same
// time evolution when running on simulation time.
// It waits for all nodes to have finished the current time step. Once they are
// all done, it tells them to execute the next time step, and so on.

#include "ros/ros.h"
#include "ros/console.h"
#include <ros/callback_queue.h>
#include "rosgraph_msgs/Clock.h"
#include <ros/subscribe_options.h>
#include "std_msgs/Bool.h"
#include <baxter_core_msgs/JointCommand.h>

#include <edlut_ros/ExternalClock.h>
#include <edlut_ros/EnableRobot.h>

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

#include <edlut_ros/SynchronizerDynParams.h>


static bool stop_simulation;


void rosShutdownHandler(int sig)
{
	stop_simulation = true;
}


int main(int argc, char **argv)
{
	// Set up ROS.
	ros::init(argc, argv, "synchronizer", ros::init_options::NoSigintHandler);
	if( ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info) ) {
   		ros::console::notifyLoggerLevelsChanged();
	}
	ros::NodeHandle nh;
	signal(SIGINT, rosShutdownHandler);

	// Declare variables that can be modified by launch file or command line.
	std::vector<std::string> synchronizing_topics;
	std::vector<ros::Subscriber> synchronizing_subscribers;
	std::vector<ExternalClock *> clock_objects;
	unsigned int num_topics;
	double checking_frequency, step_time;
	bool use_sim_time = false;
	ros::Publisher time_publisher;
	ros::Subscriber enable_sub;
	EnableRobot control_robot;
	ros::Publisher position_publisher;
	ros::Publisher unpause_publisher;
	ros::WallRate check_rate(1);

	struct timespec startt, endt;

	stop_simulation = false;

	// Initialize node parameters from launch file or command line.
	// Use a private node handle so that multiple instances of the node can be run simultaneously
	// while using different parameters.
	ros::NodeHandle private_node_handle_("~");
	private_node_handle_.getParam("clock_topics", synchronizing_topics);
	private_node_handle_.getParam("checking_frequency", checking_frequency);
	private_node_handle_.getParam("step_time", step_time);

	// Get global parameter use_sim_time
	nh.getParam("use_sim_time", use_sim_time);

	// Synchronizer node will shutdown when running in real time
	if (!use_sim_time){
		ROS_WARN("Synchronizer: Simulated time is not enabled. Synchronizer_node will shutdown. If you want to enable simulated time set use_sim_time variable to true.");
		ros::shutdown();
		return 0;
	}

	// Set callback queue
	ros::CallbackQueue CallbackQueue;
	nh.setCallbackQueue(&CallbackQueue);

	// Include the private node handle in the callback queue for the dynamic reconfigure server
	private_node_handle_.setCallbackQueue(&CallbackQueue);

	// Dynamic parameters to control synchronizer node behavior
	SynchronizerDynParams controller(&nh, &private_node_handle_);

	// // Wait until robot has been enabled.
	enable_sub = nh.subscribe("/robot_enabled", 10, &EnableRobot::EnabledCallback, &control_robot);
	while (!control_robot.GetEnabled()){
		CallbackQueue.callAvailable(ros::WallDuration(0.001));
		ROS_INFO("Waiting for robot");
		check_rate.sleep();
	}

	// Publisher to communicate with gazebo plugin to keep gazebo running
	unpause_publisher = nh.advertise<std_msgs::Bool>("/unpause_gazebo", 1);

	num_topics = synchronizing_topics.size();
	synchronizing_subscribers.resize(num_topics);
	clock_objects.resize(num_topics);

	// Create the subscribers and objects for every clock topic of the nodes that need to be synchronized
	for (unsigned int i=0; i<num_topics; ++i){
		clock_objects[i] = new ExternalClock();
		ROS_DEBUG("Synchronizer: Subscribing to clock topic %s",synchronizing_topics[i].c_str());
		synchronizing_subscribers[i] = nh.subscribe(synchronizing_topics[i], 1000, &ExternalClock::ClockCallback, clock_objects[i]);
	}

	// Publisher to advertise synchronizer node clock signal. This is the clock signal that all the nodes will follow
	time_publisher  = nh.advertise<rosgraph_msgs::Clock>("clock_sync", 1000);

	ROS_INFO("Synchronizer node initialized.");

	ros::WallRate rate(checking_frequency);
	ros::WallRate init_rate(1.0);

	ros::Time next_time (0.0), last_sent_time(0.0);
	bool all_first_received = false;

	// Message to advertise current synchronized time
	rosgraph_msgs::Clock current_time;

	// Publish start simulation time 0.0
	current_time.clock = ros::Time(0.0);
	ROS_DEBUG("Synchronizer: Publishing simulation time %f", current_time.clock.toSec());
	time_publisher.publish(current_time);

	// Wait until all nodes have published 0.0 time and the synchronizer has received it
	// The synchronizer node keeps publishing 0.0 time until all nodes are synchronized
	while(!all_first_received){
		CallbackQueue.callAvailable(ros::WallDuration(0.0001));
		unsigned int first_received_nodes=0;
		for (unsigned int i=0; i<num_topics; ++i){
			first_received_nodes += clock_objects[i]->FirstReceived();
		}
		all_first_received = (first_received_nodes==num_topics);

		if (!all_first_received){
			current_time.clock = ros::Time(0.0);
			ROS_DEBUG("Synchronizer: Publishing simulation time %f", current_time.clock.toSec());
			time_publisher.publish(current_time);
			init_rate.sleep();
		}
	}

	ROS_DEBUG("Synchronizer: Node synchronized");

	// Time stamp for statistical purposes
	ros::WallTime start = ros::WallTime::now();

	// Synchronizer loop
	while (!stop_simulation){
		CallbackQueue.callAvailable(ros::WallDuration(0.0001));

		// Dynamic parameters control. If paused button is true or the simulation has reached the time stamp
		// specified by the user ---> STOP THE SIMULATION
		while (controller.GetPaused() || (next_time.toSec()>=controller.GetTimeStamp() && controller.GetStopTS())){
			CallbackQueue.callAvailable(ros::WallDuration(0.0001));
		}

		// Check if all nodes have finished the current time step
		unsigned int finished_nodes = 0;
		for (unsigned int i=0; i<num_topics; ++i){
			if (clock_objects[i]->GetLastConfirmedTime()>=next_time){
				finished_nodes++;
			}
		}

		// When all nodes have finished the current time step, the synchronizer publishes the next time step
		if (finished_nodes==num_topics){
			next_time += ros::Duration(step_time);
			ROS_DEBUG("Synchronizer: All nodes finished. Sending new time stamp %f",next_time.toSec());
			current_time.clock = next_time;
			time_publisher.publish(current_time);
			ROS_DEBUG("Synchronizer: Publishing next time %f", next_time.toSec());
		}
	}

	ROS_INFO("Ending Synchronizer node");

	// Time statistics
	ros::Time sim_time = current_time.clock;
	ros::WallTime end = ros::WallTime::now();
	ros::WallDuration total = end - start;
	ROS_INFO("Elapsed time : %f", total.toSec());
	ROS_INFO("Simulated Time : %f", sim_time.toSec());
	ROS_INFO("Simulation ratio: %f", sim_time.toSec() / total.toSec());

	// Publish a message so the gazebo plugin keeps gazebo's clock running and puts the arm in 0 position
	std_msgs::Bool msg_bool;
	msg_bool.data = true;
	unpause_publisher.publish(msg_bool);

	// Shutdown node
	ros::shutdown();

	return 0;
} // end main()
