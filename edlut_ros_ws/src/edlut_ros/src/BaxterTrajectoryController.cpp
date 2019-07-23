/***************************************************************************
 *                           BaxterTrajectoryController.cpp                *
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

#include "edlut_ros/BaxterTrajectoryController.h"
#include "edlut_ros/SinTrajectoryGenerator.h"
#include "edlut_ros/AnalogCompact.h"
#include "edlut_ros/LearningState.h"

#include <ros/ros.h>
#include <baxter_core_msgs/JointCommand.h>
#include <sensor_msgs/JointState.h>
#include <vector>
#include <cmath>

#define JOINT_ANGLE_TOLERANCE 0.008726646

int BaxterTrajectoryController::FindJointIndex(std::vector<std::string> strvector, std::string name){
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
}

void BaxterTrajectoryController::BaxterStateCallback(const sensor_msgs::JointState::ConstPtr& msg){

	for (unsigned int i=0; i<this->joint_list.size(); ++i){
		std::string cur_joint = this->joint_list[i];

		int index = this->FindJointIndex(msg->name, cur_joint);

		if (index!=-1) {
			this->baxter_last_state[i] = msg->position[index];
		} else {
			ROS_WARN("Baxter Trajectory Controller: %s joint name does not exist in BaxterJointState. Ignoring", cur_joint.c_str());
		}
	}

	return;
}


void BaxterTrajectoryController::MoveToStartingPoint(const bool & StopMovement){

	ROS_INFO("Baxter Trajectory Controller: Moving to the starting point");

	// Publish the control signal (learning=false)
	edlut_ros::LearningState msg;
	CallbackQueue.callAvailable(ros::WallDuration(0.001));
	if (this->use_sim_time){
		msg.header.stamp = ext_clock.GetLastConfirmedTime();
	}
	else{
		msg.header.stamp = ros::Time::now();
	}
	msg.learning = false;
	this->control_signal_pub.publish(msg);

	// Move to the starting point
	std::vector<double> StartingPoint = this->trajectory_generator->GetStartingPoint();

	std::ostringstream oss;
    std::copy(StartingPoint.begin(), StartingPoint.end(), std::ostream_iterator<double>(oss, ","));

	baxter_core_msgs::JointCommand newMsg;
	newMsg.mode = 1; // Position control mode
	newMsg.names = this->joint_list;
	newMsg.command = StartingPoint;
	this->joint_command_pub.publish(newMsg);

	// Check and block the function until the robot has reached the point
	bool reached;

	do {
		this->CallbackQueue.callAvailable(ros::WallDuration(0.001));

		reached = true;
		for (unsigned int i=0; i<baxter_last_state.size() & reached; ++i){
			reached &= std::abs(this->baxter_last_state[i]-StartingPoint[i])<JOINT_ANGLE_TOLERANCE;
		}

		// Publish the position again
		this->joint_command_pub.publish(newMsg);

		this->rate.sleep();

	} while (!reached && !StopMovement);

	return;
}

void BaxterTrajectoryController::LearningTrial(const bool & StopMovement){

	ROS_INFO("Baxter Trajectory Controller: Performing learning trial");

	// Publish the control signal (learning=true)
	edlut_ros::LearningState msg;
	CallbackQueue.callAvailable(ros::WallDuration(0.001));
	if (this->use_sim_time){
		msg.header.stamp = ext_clock.GetLastConfirmedTime();
	}
	else{
		msg.header.stamp = ros::Time::now();
	}
	msg.learning = true;
	this->control_signal_pub.publish(msg);

	// Check the initial time
	this->trial_init_time = this->trajectory_generator->ResetGenerator();


	std::vector<double> DesiredPosition(this->joint_list.size()), DesiredVelocity(this->joint_list.size());

	ros::Time current_time = this->trial_init_time;

	do {
		// Get the desired position and velocity
		this->trajectory_generator->GetState(DesiredPosition, DesiredVelocity);

		edlut_ros::AnalogCompact msgPosition;
		msgPosition.header.stamp = current_time;
		msgPosition.names = this->joint_list;
		msgPosition.data.resize(DesiredPosition.size());

		// Send the desired position and velocity
		for (unsigned int i=0; i<DesiredPosition.size(); ++i){
			msgPosition.data[i] = DesiredPosition[i];
		}
		this->desired_position_pub.publish(msgPosition);

		std::ostringstream oss;
    	std::copy(DesiredPosition.begin(), DesiredPosition.end(), std::ostream_iterator<double>(oss, ","));

		edlut_ros::AnalogCompact msgVelocity;
		msgVelocity.header.stamp = current_time;
		msgVelocity.names = this->joint_list;
		msgVelocity.data.resize(DesiredVelocity.size());

		// Send the desired position and velocity
		for (unsigned int i=0; i<DesiredVelocity.size(); ++i){
			msgVelocity.data[i] = DesiredVelocity[i];
		}
		this->desired_velocity_pub.publish(msgVelocity);

		std::ostringstream oss2;
    	std::copy(DesiredVelocity.begin(), DesiredVelocity.end(), std::ostream_iterator<double>(oss2, ","));


		this->rate.sleep();

		CallbackQueue.callAvailable(ros::WallDuration(0.001));
		if (this->use_sim_time){
			current_time = ext_clock.GetLastConfirmedTime();
		}
		else{
			current_time = ros::Time::now();
		}
	} while ((current_time-this->trial_init_time).toSec() < this->trial_length && !StopMovement);


	return;
}


BaxterTrajectoryController::BaxterTrajectoryController(std::vector<std::string> joint_list,
			std::string desired_position_topic,
			std::string desired_velocity_topic,
			std::string control_topic,
			std::string joint_command_topic,
			std::string joint_state_topic,
			SinTrajectoryGenerator * trajectory_generator,
			unsigned int total_number_of_trials,
			double trial_length,
			double update_frequency,
			bool sim_time):
			NodeHandler(),
			CallbackQueue(),
			joint_list(joint_list),
			rate(update_frequency),
			trajectory_generator(trajectory_generator),
			trial_length(trial_length),
			total_number_of_trials(total_number_of_trials),
			use_sim_time(sim_time) {

	this->NodeHandler.setCallbackQueue(&this->CallbackQueue);

	// Initialize the vectors of publishers
	this->desired_position_pub = this->NodeHandler.advertise<edlut_ros::AnalogCompact>(desired_position_topic, 10);
	ROS_DEBUG("Baxter Trajectory Controller: Advertised topic %s for desired position", desired_position_topic.c_str());
	this->desired_velocity_pub = this->NodeHandler.advertise<edlut_ros::AnalogCompact>(desired_velocity_topic, 10);
	ROS_DEBUG("Baxter Trajectory Controller: Advertised topic %s for desired velocity", desired_velocity_topic.c_str());

	this->baxter_last_state = std::vector<double>(this->joint_list.size(),-1000.0);

	this->joint_command_pub = this->NodeHandler.advertise<baxter_core_msgs::JointCommand>(joint_command_topic, 1);
	ROS_DEBUG("Baxter Trajectory Controller: Advertised topic %s for joint command", joint_command_topic.c_str());
	this->control_signal_pub = this->NodeHandler.advertise<edlut_ros::LearningState>(control_topic, 1);
	ROS_DEBUG("Baxter Trajectory Controller: Advertised topic %s for control signal (learning)", control_topic.c_str());

	this->joint_state_sub = this->NodeHandler.subscribe(joint_state_topic, 1, &BaxterTrajectoryController::BaxterStateCallback, this);
	ROS_DEBUG("Baxter Trajectory Controller: Subscribed to topic %s for joint state", joint_state_topic.c_str());

	this->clock_subscriber = this->NodeHandler.subscribe("/clock_sync", 1000, &ExternalClock::ClockCallback, &ext_clock);
	this->number_of_trials = 0;

	bool in_learning = false;


	return;
}


BaxterTrajectoryController::~BaxterTrajectoryController() {
	// TODO Auto-generated destructor stub
}


/*
 * This function executes a controller step. It returns
 * true if the simulation is not ended yet and false otherwise.
 */
bool BaxterTrajectoryController::NextControlStep(const bool & StopMovement){

	ROS_INFO("Baxter Trajectory Controller: Starting learning trial %d", this->number_of_trials);

	// Move to the position
	this->MoveToStartingPoint(StopMovement);

	// Make the learning trial
	this->LearningTrial(StopMovement);

	this->number_of_trials++;

	return this->number_of_trials<this->total_number_of_trials;


}
