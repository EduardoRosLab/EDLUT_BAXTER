/***************************************************************************
 *                           baxter_torque_command_node.cpp                *
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

// This node transforms a given input torque command into an output
// robot/limb/left-right/joint_command that Baxter robot can actually process.


#include <edlut_ros/AnalogCompact.h>
#include <edlut_ros/LearningState.h>
#include <baxter_core_msgs/JointCommand.h>
#include <std_msgs/Empty.h>
#include "std_msgs/Bool.h"


#include "log4cxx/logger.h"
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


/*
 * Create a class 
 */
class Synchronizer {
private:
	ros::Publisher output_Publisher, gravity_Publisher;

	std::vector<std::string> joint_list;

	std::vector<std::string> selected_joint_list;

	std::vector<double> torque_values;

	bool in_learning, disable_gravity_compensation;

	int FindJointIndex(std::vector<std::string> strvector, std::string name){
		std::vector<std::string>::iterator first = strvector.begin();
		std::vector<std::string>::iterator last = strvector.end();
		unsigned int index = 0;
		bool found = false;

		while (first!=last && !found) {
			if (*first==name)
				found = true;
			else {
				++first;
				++index;
			}
		}

		if (found) {
			return index;
		} else {
			return -1;
		}
	};

public:
	Synchronizer (std::string output_topic,
		std::string limb,
		std::vector<std::string> selected_joint_list,
		bool disable_gravity_compensation,
		std::string gravity_topic):
		selected_joint_list(selected_joint_list),
		disable_gravity_compensation(disable_gravity_compensation)
	{
		ros::NodeHandle nh;

		this->joint_list.push_back(limb + "_s0");
		this->joint_list.push_back(limb + "_s1");
		this->joint_list.push_back(limb + "_e0");
		this->joint_list.push_back(limb + "_e1");
		this->joint_list.push_back(limb + "_w0");
		this->joint_list.push_back(limb + "_w1");
		this->joint_list.push_back(limb + "_w2");

		this->torque_values = std::vector<double> (this->joint_list.size(),0.0);

		this->in_learning = false;

		this->output_Publisher = nh.advertise<baxter_core_msgs::JointCommand>(output_topic, 1);
		ROS_INFO("Baxter Torque Command: Writing torque commands to topic %s", output_topic.c_str());

		if (disable_gravity_compensation){
			this->gravity_Publisher = nh.advertise<std_msgs::Empty>(gravity_topic, 1);
		}
		return;
	};

	void EventCallback(const edlut_ros::AnalogCompact::ConstPtr& trq){
		// ROS_DEBUG("Baxter Torque Command Node: Received event S0");

		for (unsigned int i=0; i<trq->names.size(); ++i){

			int index = this->FindJointIndex(this->joint_list, trq->names[i]);

			if (index!=-1) {
				this->torque_values[index] = trq->data[i];
			}
		}
		return;
	}

	void ControlEventCallback(const edlut_ros::LearningState::ConstPtr& lear){
		// ROS_DEBUG("Baxter Torque Command Node: Received control event");
		this->in_learning = lear->learning;
		return;
	}

	void PublishCommand(){
		if (this->in_learning){
			baxter_core_msgs::JointCommand newMsg;
			newMsg.mode = 3; // Torque control mode
			newMsg.names = this->joint_list;
			newMsg.command = this->torque_values;

			this->output_Publisher.publish(newMsg);

			if (this->disable_gravity_compensation){
				std_msgs::Empty msg;
				this->gravity_Publisher.publish(msg);
			}
		}
		return;
	};
};

int main(int argc, char **argv)
{
	// Set up ROS.
	ros::init(argc, argv, "baxter_torque_command", ros::init_options::NoSigintHandler);
	log4cxx::LoggerPtr my_logger = log4cxx::Logger::getLogger(ROSCONSOLE_DEFAULT_NAME);
	my_logger->setLevel(ros::console::g_level_lookup[ros::console::levels::Info]);
	ros::NodeHandle nh;

	signal(SIGINT, rosShutdownHandler);

	// Declare variables that can be modified by launch file or command line.
	std::string output_topic, limb, input_topic;
	std::vector<std::string> joint_list;
	std::string control_topic, gravity_topic;
	bool disable_gravity_compensation;
	double sampling_frequency;

	stop_node = false;

	// Initialize node parameters from launch file or command line.
	// Use a private node handle so that multiple instances of the node can be run simultaneously
	// while using different parameters.
	ros::NodeHandle private_node_handle_("~");
	private_node_handle_.getParam("output_topic", output_topic);
	private_node_handle_.getParam("limb", limb);
	private_node_handle_.getParam("input_topic", input_topic);
	private_node_handle_.getParam("joint_list", joint_list);
	private_node_handle_.getParam("control_topic", control_topic);
	private_node_handle_.getParam("sampling_frequency", sampling_frequency);
	private_node_handle_.getParam("disable_gravity_compensation", disable_gravity_compensation);
	private_node_handle_.getParam("gravity_topic", gravity_topic);

	Synchronizer objSynchronizer =Synchronizer(output_topic, limb, joint_list, disable_gravity_compensation, gravity_topic);

	// Create the subscriber
	ros::CallbackQueue CallbackQueue;
	nh.setCallbackQueue(&CallbackQueue);
	ros::Subscriber input_subscriber = nh.subscribe(input_topic, 1, &Synchronizer::EventCallback, &objSynchronizer);
	ros::Subscriber control_subscriber = nh.subscribe(control_topic, 1, &Synchronizer::ControlEventCallback, &objSynchronizer);
	ros::Publisher unpause_publisher = nh.advertise<std_msgs::Bool>("/unpause_gazebo", 1);

	ROS_INFO("Baxter Torque Command node initialized: reading from topic %s and writing to topic %s with sampling frequency %f", input_topic.c_str(), output_topic.c_str(), sampling_frequency);

	ros::Rate rate(sampling_frequency);

	while (!stop_node){
		CallbackQueue.callAvailable(ros::WallDuration(0.001));
		objSynchronizer.PublishCommand();
		rate.sleep();
	}

	//publish message for the plugin to send baxter to 0.0.0.0.0.0 position
	std_msgs::Bool msg_bool;
	msg_bool.data = true;
	unpause_publisher.publish(msg_bool);
	ROS_INFO("Ending Baxter Torque Command node");

	ros::shutdown();
	return 0;
} // end main()
