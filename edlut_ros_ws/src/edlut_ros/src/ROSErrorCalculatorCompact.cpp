/***************************************************************************
 *                           ROSErrorCalculatorCompact.cpp                           *
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

// This node computes the error comparing the current position and velocity
// topics. Eeach joint's position and velocity has a gain associated for the
// computation of the error.

#include <vector>
#include <iostream>
#include <iterator>
#include <cmath>

#include "edlut_ros/ROSErrorCalculatorCompact.h"
#include "edlut_ros/AnalogCompact.h"

void ROSErrorCalculatorCompact::PositionPlusCallback(const edlut_ros::AnalogCompact::ConstPtr& msg){
	// Check if the spike should have been processed previously
	if (msg->header.stamp.toSec()<this->last_time_pos_plus){
		ROS_WARN("Error Calculator: Positive position received from a past time. Discarded. Time: %f, Current time: %f.", msg->header.stamp.toSec(), ros::Time::now().toSec());
	} else {
		edlut_ros::AnalogCompact newMessage;
		newMessage.header.stamp = msg->header.stamp;
		newMessage.data = msg->data;
		newMessage.names = msg->names;
		this->activity_queue_pos_plus.push(newMessage);
	}
}

void ROSErrorCalculatorCompact::PositionMinusCallback(const edlut_ros::AnalogCompact::ConstPtr& msg){
	// Check if the spike should have been processed previously
	if (msg->header.stamp.toSec()<this->last_time_pos_minus){
		ROS_WARN("Error Calculator: Actual position received from a past time. Discarded. Time: %f, Current time: %f.", msg->header.stamp.toSec(), ros::Time::now().toSec());
	} else {
		edlut_ros::AnalogCompact newMessage;
		newMessage.header.stamp = msg->header.stamp;
		newMessage.data = msg->data;
		newMessage.names = msg->names;
		this->activity_queue_pos_minus.push(newMessage);
	}
}

void ROSErrorCalculatorCompact::VelocityPlusCallback(const edlut_ros::AnalogCompact::ConstPtr& msg){
	// Check if the spike should have been processed previously
	if (msg->header.stamp.toSec()<this->last_time_vel_plus){
		ROS_WARN("Error Calculator: Desired velocity received from a past time. Discarded. Time: %f, Current time: %f.", msg->header.stamp.toSec(), ros::Time::now().toSec());
	} else {
		edlut_ros::AnalogCompact newMessage;
		newMessage.header.stamp = msg->header.stamp;
		newMessage.data = msg->data;
		newMessage.names = msg->names;
		this->activity_queue_vel_plus.push(newMessage);
	}
}

void ROSErrorCalculatorCompact::VelocityMinusCallback(const edlut_ros::AnalogCompact::ConstPtr& msg){
	// Check if the spike should have been processed previously
	if (msg->header.stamp.toSec()<this->last_time_vel_minus){
		ROS_WARN("Error Calculator: Actual velocity received from a past time. Discarded. Time: %f, Current time: %f.", msg->header.stamp.toSec(), ros::Time::now().toSec());
	} else {
		edlut_ros::AnalogCompact newMessage;
		newMessage.header.stamp = msg->header.stamp;
		newMessage.data = msg->data;
		newMessage.names = msg->names;
		this->activity_queue_vel_minus.push(newMessage);
	}
}

int ROSErrorCalculatorCompact::FindJointIndex(std::vector<std::string> strvector, std::string name){
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

ROSErrorCalculatorCompact::ROSErrorCalculatorCompact(
	std::string pos_plus_topic_name,
	std::string vel_plus_topic_name,
	std::string pos_minus_topic_name,
	std::string vel_minus_topic_name,
	std::string output_topic_name,
	std::vector<std::string> joint_list,
	std::vector<double> position_gain,
	std::vector<double> velocity_gain):
				NodeHandler(),
				CallbackQueue(),
				activity_queue_pos_plus(),
				activity_queue_pos_minus(),
				activity_queue_vel_plus(),
				activity_queue_vel_minus(),
				joint_list(joint_list),
				position_gain(position_gain),
				velocity_gain(velocity_gain),
				last_time_pos_plus(0.0),
				last_time_pos_minus(0.0),
				last_time_vel_plus(0.0),
				last_time_vel_minus(0.0)
{

	this->position_plus = std::vector<double> (joint_list.size(), 0.0);
	this->position_minus = std::vector<double> (joint_list.size(), 0.0);
	this->velocity_plus = std::vector<double> (joint_list.size(), 0.0);
	this->velocity_minus = std::vector<double> (joint_list.size(), 0.0);
	this->output_var = std::vector<double> (joint_list.size(), 0.0);

	this->NodeHandler.setCallbackQueue(&this->CallbackQueue);
	this->subscriber_pos_plus = this->NodeHandler.subscribe(pos_plus_topic_name, 10.0, &ROSErrorCalculatorCompact::PositionPlusCallback, this);
	this->subscriber_pos_minus = this->NodeHandler.subscribe(pos_minus_topic_name, 10.0, &ROSErrorCalculatorCompact::PositionMinusCallback, this);
	this->subscriber_vel_plus = this->NodeHandler.subscribe(vel_plus_topic_name, 10.0, &ROSErrorCalculatorCompact::VelocityPlusCallback, this);
	this->subscriber_vel_minus = this->NodeHandler.subscribe(vel_minus_topic_name, 10.0, &ROSErrorCalculatorCompact::VelocityMinusCallback, this);
	this->publisher = this->NodeHandler.advertise<edlut_ros::AnalogCompact>(output_topic_name, 1.0);
}

ROSErrorCalculatorCompact::~ROSErrorCalculatorCompact() {
	// TODO Auto-generated destructor stub
}

void ROSErrorCalculatorCompact::UpdateError(ros::Time current_time){

	// Process all the spikes in the queue
	this->CallbackQueue.callAvailable();

	double end_time = current_time.toSec();

	this->CleanQueue(this->activity_queue_pos_plus, this->position_plus, this->last_time_pos_plus, end_time);
	this->CleanQueue(this->activity_queue_vel_plus, this->velocity_plus, this->last_time_vel_plus, end_time);
	this->CleanQueue(this->activity_queue_pos_minus, this->position_minus, this->last_time_pos_minus, end_time);
	this->CleanQueue(this->activity_queue_vel_minus, this->velocity_minus, this->last_time_vel_minus, end_time);

	std::vector<double> ErrorPos(this->joint_list.size());
	std::vector<double> ErrorVel(this->joint_list.size());
	for (unsigned int i=0; i<this->joint_list.size(); ++i){
		ErrorPos[i] = this->position_plus[i] - this->position_minus[i];
		ErrorVel[i] = this->velocity_plus[i] - this->velocity_minus[i];
		this->output_var[i] = this->position_gain[i]*ErrorPos[i] + this->velocity_gain[i]*ErrorVel[i];
	}
	// Create the message and publish it
	edlut_ros::AnalogCompact newMsg;
	newMsg.header.stamp = current_time;
	newMsg.names = this->joint_list;
	newMsg.data = this->output_var;
	this->publisher.publish(newMsg);

	return;
}

void ROSErrorCalculatorCompact::CleanQueue(std::priority_queue<edlut_ros::AnalogCompact, std::vector<edlut_ros::AnalogCompact>, analog_comparison > & queue,
		std::vector<double> & updateVar,
		double & lastTime,
		double end_time){
	edlut_ros::AnalogCompact top_value;
	// Clean the queue (just in case any spike has to be discarded in the queue)
	if (!queue.empty()){
		top_value= queue.top();
		while (!(queue.empty()) && top_value.header.stamp.toSec()<lastTime){
			queue.pop();
			ROS_WARN("Error calculator: Discarded analog value from queue with time %f. Current time: %f", top_value.header.stamp.toSec(), ros::Time::now().toSec());
			if (!queue.empty()){
				top_value = queue.top();
			}
		}
	}

	if (!queue.empty()){
		while (!(queue.empty()) && top_value.header.stamp.toSec()<=end_time){
			for (unsigned int i=0; i<top_value.names.size(); ++i){
				int index = this->FindJointIndex(this->joint_list, top_value.names[i]);

				if (index!=-1) {
					updateVar[index] = top_value.data[i];
				}
			}
			lastTime = top_value.header.stamp.toSec();
			queue.pop();
			if (!queue.empty()){
				top_value = queue.top();
			}
		}
	}
}
