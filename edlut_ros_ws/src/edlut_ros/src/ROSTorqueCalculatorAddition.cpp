/***************************************************************************
 *                           ROSTorqueCalculatorAddition.cpp                           *
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

#include <vector>
#include <iostream>
#include <iterator>
#include <cmath>

#include "edlut_ros/ROSTorqueCalculatorAddition.h"
#include "edlut_ros/AnalogCompact.h"

void ROSTorqueCalculatorAddition::Torque1Callback(const edlut_ros::AnalogCompact::ConstPtr& msg){
	// Check if the spike should have been processed previously
	if (msg->header.stamp.toSec()<this->last_time_torque_1){
		ROS_WARN("Torque Calculator: Positive torque received from a past time. Discarded. Time: %f, Current time: %f.", msg->header.stamp.toSec(), ros::Time::now().toSec());
	} else {
		std::ostringstream oss;
    	std::copy(msg->data.begin(), msg->data.end(), std::ostream_iterator<double>(oss, ","));
		ROS_DEBUG("Torque Calculator: Received positive torque: time %f and value %s.", msg->header.stamp.toSec(), oss.str().c_str());
		edlut_ros::AnalogCompact newMessage;
		newMessage.header.stamp = msg->header.stamp;
		newMessage.data = msg->data;
		newMessage.names = msg->names;
		this->activity_queue_torque_1.push(newMessage);
	}
}

void ROSTorqueCalculatorAddition::Torque2Callback(const edlut_ros::AnalogCompact::ConstPtr& msg){
	// Check if the spike should have been processed previously
	if (msg->header.stamp.toSec()<this->last_time_torque_2){
		ROS_WARN("Torque Calculator: Negative torque received from a past time. Discarded. Time: %f, Current time: %f.", msg->header.stamp.toSec(), ros::Time::now().toSec());
	} else {
		std::ostringstream oss;
    	std::copy(msg->data.begin(), msg->data.end(), std::ostream_iterator<double>(oss, ","));
		ROS_DEBUG("Torque Calculator: Received positive torque: time %f and value %s.", msg->header.stamp.toSec(), oss.str().c_str());
		edlut_ros::AnalogCompact newMessage;
		newMessage.header.stamp = msg->header.stamp;
		newMessage.data = msg->data;
		newMessage.names = msg->names;
		this->activity_queue_torque_2.push(newMessage);
	}
}

int ROSTorqueCalculatorAddition::FindJointIndex(std::vector<std::string> strvector, std::string name){
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

ROSTorqueCalculatorAddition::ROSTorqueCalculatorAddition(
	std::string torque_1_topic_name,
	std::string torque_2_topic_name,
	std::string output_topic_name,
	std::vector<std::string> joint_list):
				NodeHandler(),
				CallbackQueue(),
				activity_queue_torque_1(),
				activity_queue_torque_2(),
				joint_list(joint_list),
				last_time_torque_1(0.0),
				last_time_torque_2(0.0)
{

	this->torque_1 = std::vector<double> (joint_list.size(), 0.0);
	this->torque_2 = std::vector<double> (joint_list.size(), 0.0);
	this->output_var  = std::vector<double> (joint_list.size(), 0.0);

	this->NodeHandler.setCallbackQueue(&this->CallbackQueue);
	this->subscriber_torque_1 = this->NodeHandler.subscribe(torque_1_topic_name, 10.0, &ROSTorqueCalculatorAddition::Torque1Callback, this);
	this->subscriber_torque_2 = this->NodeHandler.subscribe(torque_2_topic_name, 10.0, &ROSTorqueCalculatorAddition::Torque2Callback, this);
	this->publisher = this->NodeHandler.advertise<edlut_ros::AnalogCompact>(output_topic_name, 1.0);
}

ROSTorqueCalculatorAddition::~ROSTorqueCalculatorAddition() {
	// TODO Auto-generated destructor stub
}

void ROSTorqueCalculatorAddition::UpdateTorque(ros::Time current_time){

	// Process all the spikes in the queue
	this->CallbackQueue.callAvailable();

	double end_time = current_time.toSec();

	this->CleanQueue(this->activity_queue_torque_1, this->torque_1, this->last_time_torque_1, end_time);
	this->CleanQueue(this->activity_queue_torque_2, this->torque_2, this->last_time_torque_2, end_time);

	std::vector<double> Torque(this->joint_list.size());
	for (unsigned int i=0; i<this->joint_list.size(); ++i){
		this->output_var[i] = this->torque_1[i] + this->torque_2[i];
	}

	std::ostringstream oss;
    std::copy(this->output_var.begin(), this->output_var.end(), std::ostream_iterator<double>(oss, ","));
    std::ostringstream oss2;
    std::copy(this->torque_1.begin(), this->torque_1.end(), std::ostream_iterator<double>(oss2, ","));
    std::ostringstream oss3;
    std::copy(this->torque_2.begin(), this->torque_2.end(), std::ostream_iterator<double>(oss3, ","));
    ROS_DEBUG("Torque Calculator: Calculated torque value: time %f and value %s. Torque 1 %s, Torque 2 %s", current_time.toSec(), oss.str().c_str(), oss2.str().c_str(), oss3.str().c_str());


	// Create the message and publish it
	edlut_ros::AnalogCompact newMsg;
	newMsg.header.stamp = current_time;
	newMsg.names = this->joint_list;
	newMsg.data = this->output_var;
	this->publisher.publish(newMsg);

	return;
}

void ROSTorqueCalculatorAddition::CleanQueue(std::priority_queue<edlut_ros::AnalogCompact, std::vector<edlut_ros::AnalogCompact>, analog_comparison > & queue,
		std::vector<double> & updateVar,
		double & lastTime,
		double end_time){
	edlut_ros::AnalogCompact top_value;

	// Clean the queue (just in case any spike has to be discarded in the queue)
	if (!queue.empty()){
		top_value= queue.top();
		while (!(queue.empty()) &&
			top_value.header.stamp.toSec()<lastTime){

			queue.pop();
			ROS_WARN("Torque calculator: Discarded analog value from queue with time %f. Current time: %f", top_value.header.stamp.toSec(), ros::Time::now().toSec());
			if (!queue.empty()){
				top_value = queue.top();
			}
		}
	}

	if (!queue.empty()){
		while (!(queue.empty()) &&
				top_value.header.stamp.toSec()<=end_time){
			std::ostringstream oss;
    		std::copy(top_value.data.begin(), top_value.data.end(), std::ostream_iterator<double>(oss, ","));
    		std::ostringstream oss2;
    		std::copy(top_value.names.begin(), top_value.names.end(), std::ostream_iterator<std::string>(oss2, ","));
			ROS_DEBUG("Torque Calculator: Processing value with time %f. End time: %f. Current time: %f. Value: %s. Names: %s", top_value.header.stamp.toSec(), end_time, ros::Time::now().toSec(), oss.str().c_str(), oss2.str().c_str());

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
