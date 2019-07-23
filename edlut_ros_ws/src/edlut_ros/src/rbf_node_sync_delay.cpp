/***************************************************************************
 *                           rbf_node.cpp                                  *
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

 // This is the RBF node. It transforms an input analog signal to spikes, relating
 // the analog data to neurons in the RBF bank

 #include <ros/ros.h>
 #include <ros/callback_queue.h>
 #include "rosgraph_msgs/Clock.h"
 #include "edlut_ros/ExternalClock.h"
 #include <edlut_ros/AnalogCompact.h>
 #include <edlut_ros/ROSRBFBank_delay.h>
 #include <edlut_ros/Spike_group.h>
 #include <edlut_ros/Spike.h>
 #include <edlut_ros/LearningState.h>
 #include <cstring>
 #include <ctime>
 #include <limits>
 #include <signal.h>
 #include <iostream>
 #include <vector>


 static bool stop_node;

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

  std::vector<double> GetSensorValue(){
    return SensorValue;
  };

 	bool IsInLearning(){
 		return this->in_learning;
 	}
 };

 void rosShutdownHandler(int sig)
 {
 	stop_node = true;
 }



 int main(int argc, char **argv)
 {
 	// Set up ROS.
 	ros::init(argc, argv, "rbf", ros::init_options::NoSigintHandler);
  if( ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info) ) {
   		ros::console::notifyLoggerLevelsChanged();
	}
 	ros::NodeHandle nh;

 	signal(SIGINT, rosShutdownHandler);

 	// Declare variables that can be modified by launch file or command line.
 	std::string input_topic, output_topic, control_topic, clock_topic;
 	std::vector<int> min_neuron_index, max_neuron_index;
 	std::vector<double> min_values, max_values, max_spike_frequency, overlapping_factor;
 	double sampling_frequency, checking_frequency;
 	std::vector<std::string> joint_list;
  double delay;

  bool use_sim_time;

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
 	private_node_handle_.getParam("min_neuron_index_list", min_neuron_index);
 	private_node_handle_.getParam("max_neuron_index_list", max_neuron_index);
 	private_node_handle_.getParam("min_value_list", min_values);
 	private_node_handle_.getParam("max_value_list", max_values);
 	private_node_handle_.getParam("sampling_frequency", sampling_frequency);
	private_node_handle_.getParam("checking_frequency", checking_frequency);
 	private_node_handle_.getParam("max_spike_frequency_list", max_spike_frequency);
 	private_node_handle_.getParam("overlapping_factor_list", overlapping_factor);
  private_node_handle_.getParam("delay", delay);


	nh.getParam("use_sim_time", use_sim_time);

  // Create the subscriber
  ros::CallbackQueue CallbackQueue;
  nh.setCallbackQueue(&CallbackQueue);

  // Subscribe to synchronizer clock signal when using simulated time and publish
  // RBF node clock.
  if (use_sim_time){
		ROS_DEBUG("RBF: Subscribing to topic /clock_sync");
		clock_subscriber = nh.subscribe("/clock_sync", 1000, &ExternalClock::ClockCallback, &ext_clock);
		time_publisher  = nh.advertise<rosgraph_msgs::Clock>(clock_topic, 1000);
		ROS_DEBUG("RBF: Publishing simulation time to topic %s",clock_topic.c_str());
	}

  // Create the output spike publisher
 	ros::Publisher output_publisher = nh.advertise<edlut_ros::Spike>(output_topic+"OLD", 1e9);
  ros::Publisher output_publisher_group = nh.advertise<edlut_ros::Spike_group>(output_topic, 1e9);
 	SensorState objSensorState = SensorState(joint_list);
 	std::vector<ROSRBFBank_delay> objRBFBankVector;
  ROS_DEBUG("RBF: Publisher created");

  std::vector<unsigned int> num_filters;
  std::vector<unsigned int> u_min_neuron_index (joint_list.size());
  std::vector<unsigned int> u_max_neuron_index (joint_list.size());
  for (unsigned int i= 0; i<joint_list.size(); ++i){
 		num_filters.push_back(max_neuron_index[i] - min_neuron_index[i] + 1);
    u_min_neuron_index[i] = min_neuron_index[i];
    u_max_neuron_index[i] = max_neuron_index[i];
  }
  ROSRBFBank_delay objRBFBank(
      num_filters,
      &output_publisher,
      min_values,
      max_values,
      u_min_neuron_index,
			u_max_neuron_index,
      sampling_frequency,
      max_spike_frequency,
      overlapping_factor,
      delay);


 	ros::Subscriber input_subscriber = nh.subscribe(input_topic, 1, &SensorState::AnalogCallback, &objSensorState);

 	ros::Subscriber control_subscriber = nh.subscribe(control_topic, 1, &SensorState::ControlCallback, &objSensorState);

  ROS_DEBUG("RBF: Subscriber created");

 	ROS_INFO("RBF node initialized: reading from topic %s and writing to topic %s", input_topic.c_str(), output_topic.c_str());

 	ros::Rate rate(sampling_frequency);
	ros::Time time(0.0), last_sent_time(0.0);
	ros::WallRate sim_rate(checking_frequency);
	ros::WallRate init_rate(1.0);

	if (use_sim_time){
    // Wait until the first signal has been received from the synchronizer node
		while(!ext_clock.FirstReceived()){
      ROS_DEBUG("RBF: Synchronizing");
			CallbackQueue.callAvailable(ros::WallDuration(0.001));
			rosgraph_msgs::Clock current_time;
			current_time.clock = ros::Time(0.0);
			ROS_DEBUG("RBF: Publishing simulation time %f", time.toSec());
			time_publisher.publish(current_time);
			init_rate.sleep();
		}

		time = ext_clock.GetLastConfirmedTime();

		ROS_DEBUG("RBF: Node synchronized");
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
				if (objSensorState.IsInLearning()){
					ROS_DEBUG("RBF node: Generating activity at time %f ", time.toSec());
          edlut_ros::Spike_group spike_group;
					objRBFBank.GenerateActivity(objSensorState.GetSensorValue(), time.toSec(), new_sampling_period.toSec());
          spike_group.neuron_index = objRBFBank.neuron_index_array;
          spike_group.time = objRBFBank.time_array;
          objRBFBank.neuron_index_array.clear();
          objRBFBank.time_array.clear();

          output_publisher_group.publish(spike_group);
				}
			}

			if (time!=last_sent_time){
				// Publish the simulation time
				rosgraph_msgs::Clock current_time;
				current_time.clock = time;
				ROS_DEBUG("RBF: Publishing simulation time %fs", time.toSec());
				time_publisher.publish(current_time);
				last_sent_time = time;
			}

			sim_rate.sleep();
		} else{
      ros::Duration new_sampling_period = ros::Time::now() - time;
			time = ros::Time::now();
			if (objSensorState.IsInLearning()){
				ROS_DEBUG("RBF node: Generating activity at time %f", time.toSec());
        edlut_ros::Spike_group spike_group;
        objRBFBank.GenerateActivity(objSensorState.GetSensorValue(), time.toSec(), new_sampling_period.toSec());
        spike_group.neuron_index = objRBFBank.neuron_index_array;
        spike_group.time = objRBFBank.time_array;
        objRBFBank.neuron_index_array.clear();
        objRBFBank.time_array.clear();

        output_publisher_group.publish(spike_group);
			}
			rate.sleep();
		}
 	}

 	ROS_INFO("Ending RBF node");

 	ros::shutdown();
 	return 0;
} // end main()
