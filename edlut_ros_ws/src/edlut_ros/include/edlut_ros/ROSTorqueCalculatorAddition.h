/***************************************************************************
 *                           ROSTorqueCalculatorAddition.h                             *
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

#ifndef ROSTorqueCalculatorAddition_H_
#define ROSTorqueCalculatorAddition_H_

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
class ROSTorqueCalculatorAddition {
private:

	// ROS Node Handler
	ros::NodeHandle NodeHandler;

	// Desired position subscriber
	ros::Subscriber subscriber_torque_1;

	// Actual position subscriber
	ros::Subscriber subscriber_torque_2;

	// Output publisher
	ros::Publisher publisher;

	// ROS Callback queue
	ros::CallbackQueue CallbackQueue;

	// Queue of desired position analog signals not processed yet
	std::priority_queue<edlut_ros::AnalogCompact, std::vector<edlut_ros::AnalogCompact>, analog_comparison > activity_queue_torque_1;

	// Queue of actual position analog signals not processed yet
	std::priority_queue<edlut_ros::AnalogCompact, std::vector<edlut_ros::AnalogCompact>, analog_comparison > activity_queue_torque_2;

	// Joint list
	std::vector<std::string> joint_list;

	// Current values
	std::vector<double> torque_1, torque_2;

	// Value of the output variables
	std::vector<double> output_var;

	// Last time when the decoder has been updated
	double last_time_torque_1, last_time_torque_2;

	// Callback function for reading input activity
	void Torque1Callback(const edlut_ros::AnalogCompact::ConstPtr& msg);

	// Callback function for reading input activity
	void Torque2Callback(const edlut_ros::AnalogCompact::ConstPtr& msg);

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
	ROSTorqueCalculatorAddition(std::string torque_1_topic_name,
			std::string torque_2_topic_name,
			std::string output_topic_name,
			std::vector<std::string> joint_list);

	/*
	 * Update the output variables with the current time and the output activity and send the output message
	 */
	void UpdateTorque(ros::Time current_time);

	/*
	 * Destructor of the class
	 */
	virtual ~ROSTorqueCalculatorAddition();

};

#endif /* ROSTorqueCalculatorAddition_H_ */
