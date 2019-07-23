/***************************************************************************
 *                           ROErrorCalculatorCompact.h                             *
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

#ifndef ROSERRORCALCULATORCOMPACT_H_
#define ROSERRORCALCULATORCOMPACT_H_

#include <queue>
#include <ros/ros.h>
#include <ros/callback_queue.h>


#include <edlut_ros/AnalogCompact.h>


class analog_comparison {
public:
	bool operator() (const edlut_ros::AnalogCompact lsp, const edlut_ros::AnalogCompact rsp) const{
		if (lsp.header.stamp>rsp.header.stamp) {
			return true;
		} else {
			return false;
		}
	}
};

/*
 * This class defines a spike-to-analog decoder.
 */
class ROSErrorCalculatorCompact {
private:

	// ROS Node Handler
	ros::NodeHandle NodeHandler;

	// Desired position subscriber
	ros::Subscriber subscriber_pos_plus;

	// Actual position subscriber
	ros::Subscriber subscriber_pos_minus;

	// Desired velocity subscriber
	ros::Subscriber subscriber_vel_plus;

	// Actual velocity subscriber
	ros::Subscriber subscriber_vel_minus;

	// Output publisher positive and negative error
	ros::Publisher publisher;

	// ROS Callback queue
	ros::CallbackQueue CallbackQueue;

	// Queue of desired position analog signals not processed yet
	std::priority_queue<edlut_ros::AnalogCompact, std::vector<edlut_ros::AnalogCompact>, analog_comparison > activity_queue_pos_plus;

	// Queue of actual position analog signals not processed yet
	std::priority_queue<edlut_ros::AnalogCompact, std::vector<edlut_ros::AnalogCompact>, analog_comparison > activity_queue_pos_minus;

	// Queue of desired velocity analog signals not processed yet
	std::priority_queue<edlut_ros::AnalogCompact, std::vector<edlut_ros::AnalogCompact>, analog_comparison > activity_queue_vel_plus;

	// Queue of actual velocity analog signals not processed yet
	std::priority_queue<edlut_ros::AnalogCompact, std::vector<edlut_ros::AnalogCompact>, analog_comparison > activity_queue_vel_minus;

	// Position error gain
	std::vector<double> position_gain;

	// Velocity error gain
	std::vector<double> velocity_gain;

	// Joint list
	std::vector<std::string> joint_list;

	// Current values
	std::vector<double> position_plus, velocity_plus, position_minus, velocity_minus;

	// Value of the output variable (positive and negative)
	std::vector<double> output_var;


	// Last time when the decoder has been updated
	double last_time_pos_plus, last_time_vel_plus, last_time_pos_minus, last_time_vel_minus;

	// Callback function for reading input activity
	void PositionPlusCallback(const edlut_ros::AnalogCompact::ConstPtr& msg);

	// Callback function for reading input activity
	void VelocityPlusCallback(const edlut_ros::AnalogCompact::ConstPtr& msg);

	// Callback function for reading input activity
	void PositionMinusCallback(const edlut_ros::AnalogCompact::ConstPtr& msg);

	// Callback function for reading input activity
	void VelocityMinusCallback(const edlut_ros::AnalogCompact::ConstPtr& msg);

	// Clean the input value queue and retrieve the last value
	void CleanQueue(std::priority_queue<edlut_ros::AnalogCompact, std::vector<edlut_ros::AnalogCompact>, analog_comparison > & queue,
			std::vector<double> & updateVar,
			double & lastTime,
			double end_time);

	// Find the index of name in vector strvector. -1 if not found
	int FindJointIndex(std::vector<std::string> strvector, std::string name);

public:

	/*
	 * This function initializes a spike decoder taking the activity from the specified ros topic.
	 */
	ROSErrorCalculatorCompact(std::string pos_plus_topic_name,
			std::string vel_plus_topic_name,
			std::string pos_minus_topic_name,
			std::string vel_minus_topic_name,
			std::string output_topic_name,
			std::vector<std::string> joint_list,
			std::vector<double> position_gain,
			std::vector<double> velocity_gain);

	/*
	 * Update the output variables with the current time and the output activity and send the output message
	 */
	void UpdateError(ros::Time current_time);

	/*
	 * Destructor of the class
	 */
	virtual ~ROSErrorCalculatorCompact();

};

#endif /* ROSERRORCALCULATORCOMPACT_H_ */
