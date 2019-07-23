/***************************************************************************
 *                           poisson_generator_node_delay_compact.cpp      *
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

// This node performs a conversion from spikes to analogue signals using a
// Poisson generator approach. 


#include <edlut_ros/AnalogCompact.h>
#include <edlut_ros/Spike_group.h>
#include <edlut_ros/LearningState.h>
#include <edlut_ros/ROSPoissonGenerator.h>
#include <ros/callback_queue.h>
#include <ros/ros.h>
#include "log4cxx/logger.h"
#include <cstring>
#include <ctime>
#include <limits>
#include <signal.h>
#include "edlut_ros/ExternalClock.h"
#include "rosgraph_msgs/Clock.h"


static bool stop_node;

void rosShutdownHandler(int sig)
{
	stop_node = true;
}

/*
 * Create a class
 */
class SensorState {
private:
	std::vector<double> SensorValue;

	bool in_learning;

	std::vector<std::string> joint_list;

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

	SensorState(std::vector<std::string> joint_list): joint_list(joint_list) {
		this->SensorValue.resize(joint_list.size());
	}

	void AnalogCallback(const edlut_ros::AnalogCompact::ConstPtr& msg){
		for (unsigned int i=0; i<this->joint_list.size(); ++i){
			int index = this->FindJointIndex(msg->names, this->joint_list[i]);
			this->SensorValue[i] = msg->data[index];
		}
		return;
	};

	void ControlCallback(const edlut_ros::LearningState::ConstPtr& lear){
		this->in_learning = lear->learning;
		return;
	}

	double GetSensorValue(int index){
		return SensorValue[index];
	};

	bool IsInLearning(){
		return this->in_learning;
	}
};

int main(int argc, char **argv)
{
	// Set up ROS.
	ros::init(argc, argv, "spike_decoder", ros::init_options::NoSigintHandler);
	log4cxx::LoggerPtr my_logger = log4cxx::Logger::getLogger(ROSCONSOLE_DEFAULT_NAME);
	my_logger->setLevel(ros::console::g_level_lookup[ros::console::levels::Info]);
	ros::NodeHandle nh;

	signal(SIGINT, rosShutdownHandler);

	// Declare variables that can be modified by launch file or command line.
	std::string input_topic, output_topic, control_topic, clock_topic;
	std::vector<int> min_neuron_index_pos, max_neuron_index_pos, min_neuron_index_neg, max_neuron_index_neg;
	int seed;
	std::vector<double> error_center, max_spike_frequency, min_spike_frequency;
	double sampling_frequency, checking_frequency;
	std::vector<std::string> joint_list;
  bool use_sim_time;
	double delay;

	std::vector<double> min_value, max_value, overlapping_factor;	//

	ros::Publisher time_publisher;
	ros::Subscriber clock_subscriber;
	ExternalClock ext_clock;

	stop_node = false;

	// Initialize node parameters from launch file or command line.
	// Use a private node handle so that multiple instances of the node can be run simultaneously
	// while using different parameters.
	ros::NodeHandle private_node_handle_("~");
	private_node_handle_.getParam("input_topic", input_topic);
	private_node_handle_.getParam("output_topic", output_topic);
	private_node_handle_.getParam("joint_list", joint_list);
	private_node_handle_.getParam("control_topic", control_topic);
	private_node_handle_.getParam("clock_topic", clock_topic);
	private_node_handle_.getParam("min_neuron_index_list_pos", min_neuron_index_pos);
	private_node_handle_.getParam("max_neuron_index_list_pos", max_neuron_index_pos);
	private_node_handle_.getParam("min_neuron_index_list_neg", min_neuron_index_neg);
	private_node_handle_.getParam("max_neuron_index_list_neg", max_neuron_index_neg);
	private_node_handle_.getParam("min_value_list", min_value);		//
	private_node_handle_.getParam("max_value_list", max_value);		//
	private_node_handle_.getParam("sampling_frequency", sampling_frequency);
	private_node_handle_.getParam("checking_frequency", checking_frequency);
	private_node_handle_.getParam("max_spike_frequency_list", max_spike_frequency);
	private_node_handle_.getParam("min_spike_frequency_list", min_spike_frequency);
	private_node_handle_.getParam("overlapping_factor_list", overlapping_factor);    //
	private_node_handle_.getParam("seed", seed);
	private_node_handle_.getParam("delay", delay);


  nh.getParam("use_sim_time", use_sim_time);

	ros::CallbackQueue CallbackQueue;
	nh.setCallbackQueue(&CallbackQueue);

	if (use_sim_time){
		ROS_DEBUG("Poisson Generator: Subscribing to topic /clock_sync");   //change from "clock" to "clock_ext"
		clock_subscriber = nh.subscribe("/clock_sync", 1000, &ExternalClock::ClockCallback, &ext_clock);   //change from "clock" to "clock_ext"
		// Initialize the publisher in clock topic
		time_publisher  = nh.advertise<rosgraph_msgs::Clock>(clock_topic, 1000);
		ROS_DEBUG("Poisson Generator: Publishing simulation time to topic %s",clock_topic.c_str());
	}

	// Create the publisher
	ros::Publisher output_publisher_group = nh.advertise<edlut_ros::Spike_group>(output_topic, 1e9);
	SensorState objSensorState = SensorState(joint_list);

	std::vector<unsigned int> num_filters_pos, num_filters_neg;
	for (unsigned int i= 0; i<joint_list.size(); ++i){
		num_filters_pos.push_back(max_neuron_index_pos[i] - min_neuron_index_pos[i] + 1);
		num_filters_neg.push_back(max_neuron_index_neg[i] - min_neuron_index_neg[i] + 1);	//
	}

	std::vector<unsigned int> u_min_neuron_index_pos (joint_list.size());
	std::vector<unsigned int> u_max_neuron_index_pos (joint_list.size());
	std::vector<unsigned int> u_min_neuron_index_neg (joint_list.size());
	std::vector<unsigned int> u_max_neuron_index_neg (joint_list.size());
	for (unsigned int i=0; i<joint_list.size(); ++i){
		u_min_neuron_index_pos[i] = min_neuron_index_pos[i];
		u_max_neuron_index_pos[i] = max_neuron_index_pos[i];
		u_min_neuron_index_neg[i] = min_neuron_index_neg[i];
		u_max_neuron_index_neg[i] = max_neuron_index_neg[i];
	}

	ROSPoissonGenerator objPoissonGeneratorPos(
			num_filters_pos,
			min_value,
			max_value,
			u_min_neuron_index_pos,
			u_max_neuron_index_pos,
			sampling_frequency,
			max_spike_frequency,
			min_spike_frequency,
			overlapping_factor,
			use_sim_time,
			seed,
			&nh);

	ROSPoissonGenerator objPoissonGeneratorNeg(
			num_filters_neg,
			min_value,
			max_value,
			u_min_neuron_index_neg,
			u_max_neuron_index_neg,
			sampling_frequency,
			max_spike_frequency,
			min_spike_frequency,
			overlapping_factor,
			use_sim_time,
			seed,
			&nh);


	// Create the subscriber
	ros::Subscriber input_subscriber = nh.subscribe(input_topic, 1, &SensorState::AnalogCallback, &objSensorState);
	ros::Subscriber control_subscriber = nh.subscribe(control_topic, 1, &SensorState::ControlCallback, &objSensorState);

	ROS_INFO("Poisson Generator node initialized: reading from topic %s and writing to topic %s", input_topic.c_str(), output_topic.c_str());

	ros::Rate rate(sampling_frequency);

	ros::Time time(0.0), last_sent_time(0.0);
	ros::WallRate sim_rate(checking_frequency);
	ros::WallRate init_rate(1.0);

	if (use_sim_time){
		// Wait until the first signal has been received from the synchronizer node
		while(!ext_clock.FirstReceived()){
			CallbackQueue.callAvailable(ros::WallDuration(0.001));
			rosgraph_msgs::Clock current_time;
			current_time.clock = ros::Time(0.0);
			ROS_DEBUG("Poisson Generator: Publishing simulation time %f", time.toSec());
			time_publisher.publish(current_time);
			init_rate.sleep();
		}

		time = ext_clock.GetLastConfirmedTime();

		ROS_DEBUG("Poisson Generator: Node synchronized");
	}else{
		time = ros::Time::now();
	}

	while (!stop_node){
		CallbackQueue.callAvailable(ros::WallDuration(0.001));

		if (use_sim_time){
			ros::Time new_time = ext_clock.GetLastConfirmedTime();
			if (new_time>time){
				ros::Duration new_sampling_period = new_time - time;
				time = new_time;
				ROS_DEBUG("Poisson Generator node: Generating activity at time %f", time.toSec());
				if (objSensorState.IsInLearning()){
					edlut_ros::Spike_group spike_group;
					ROS_DEBUG("Poisson Generator node: 1");
					for (unsigned int i= 0; i<joint_list.size(); ++i){
						//ROS_DEBUG("Poisson Generator node: Generating activity in joint %d", i);
						if (objSensorState.GetSensorValue(i) > 0){
							objPoissonGeneratorPos.GenerateActivity(objSensorState.GetSensorValue(i), time.toSec(), i, new_sampling_period.toSec());
						}
						else{
							objPoissonGeneratorNeg.GenerateActivity(-objSensorState.GetSensorValue(i), time.toSec(), i, new_sampling_period.toSec());
						}
					}
					// Poisitive error spikes
					for (unsigned int j=0; j<objPoissonGeneratorPos.neuron_index_array.size(); j++){
						spike_group.neuron_index.push_back(objPoissonGeneratorPos.neuron_index_array[j]);
						spike_group.time.push_back(objPoissonGeneratorPos.time_array[j] + delay );
					}
					objPoissonGeneratorPos.neuron_index_array.clear();
					objPoissonGeneratorPos.time_array.clear();

					// Negative error spikes
					for (unsigned int j=0; j<objPoissonGeneratorNeg.neuron_index_array.size(); j++){
						spike_group.neuron_index.push_back(objPoissonGeneratorNeg.neuron_index_array[j]);
						spike_group.time.push_back(objPoissonGeneratorNeg.time_array[j] + delay );
					}
					objPoissonGeneratorNeg.neuron_index_array.clear();
					objPoissonGeneratorNeg.time_array.clear();

					// Publish positive and negative error spikes
					output_publisher_group.publish(spike_group);
				}
				//ROS_DEBUG("Poisson Generator node: Generated activity at time %f", time.toSec());
			}
			if (time!=last_sent_time){
				// Publish the simulation time
				rosgraph_msgs::Clock current_time;
				current_time.clock = time;
				ROS_DEBUG("Poisson Generator: Publishing simulation time %f", time.toSec());
				time_publisher.publish(current_time);
				last_sent_time = time;
			}
			sim_rate.sleep();
		}else{
			ros::Duration new_sampling_period = ros::Time::now() - time;
			time = ros::Time::now();
			ROS_DEBUG("Poisson Generator node: Generating activity at time %f", time.toSec());
			if (objSensorState.IsInLearning()){
				edlut_ros::Spike_group spike_group;
				for (unsigned int i=0; i<joint_list.size(); ++i){
					if (objSensorState.GetSensorValue(i) > 0){
						objPoissonGeneratorPos.GenerateActivity(objSensorState.GetSensorValue(i), time.toSec(), i, new_sampling_period.toSec());
					}
					else{
						objPoissonGeneratorNeg.GenerateActivity(-objSensorState.GetSensorValue(i), time.toSec(), i, new_sampling_period.toSec());
					}
				}
				// Poisitive error spikes
				for (unsigned int j=0; j<objPoissonGeneratorPos.neuron_index_array.size(); j++){
					spike_group.neuron_index.push_back(objPoissonGeneratorPos.neuron_index_array[j]);
					spike_group.time.push_back(objPoissonGeneratorPos.time_array[j] + delay );
				}
				objPoissonGeneratorPos.neuron_index_array.clear();
				objPoissonGeneratorPos.time_array.clear();

				// Negative error spikes
				for (unsigned int j=0; j<objPoissonGeneratorNeg.neuron_index_array.size(); j++){
					spike_group.neuron_index.push_back(objPoissonGeneratorNeg.neuron_index_array[j]);
					spike_group.time.push_back(objPoissonGeneratorNeg.time_array[j] + delay );
				}
				objPoissonGeneratorNeg.neuron_index_array.clear();
				objPoissonGeneratorNeg.time_array.clear();

				// Publish positive and negative error spikes
				output_publisher_group.publish(spike_group);
			}
			rate.sleep();
		}
	}

	ROS_INFO("Ending Poisson Generator node");

	ros::shutdown();
	return 0;
} // end main()
