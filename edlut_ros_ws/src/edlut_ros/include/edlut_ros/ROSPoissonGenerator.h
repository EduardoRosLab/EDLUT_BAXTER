/***************************************************************************
 *                           ROSPoissonGenerator.h                         *
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

#ifndef ROSPOISSONGENERATOR_H_
#define ROSPOISSONGENERATOR_H_

#include <vector>
#include <ros/ros.h>
#include "edlut_ros/ExternalClock.h"
#include <ros/callback_queue.h>


/*
 * This class defines a Poisson generator with tanh function.
 */
class ROSPoissonGenerator {
private:
	// ROS Node Handler
	ros::NodeHandle * NodeHandler;

	//Subscriber and external clock for synchronizer clock
	ros::Subscriber clock_subscriber;
	ExternalClock ext_clock;

	// Array with the centers of the RBFs //
	std::vector<std::vector<double> > centers; //

	// Width of the RBFs //
	std::vector<std::vector<double> > width; //

	// Vector with the last spike times //
	std::vector<double> last_spike_times; //

	//Simulation time active
	bool use_sim_time;

	// First neuron to generate the activity
	std::vector<unsigned int> init_neuron;

	// Last neuron to generate the activity
	std::vector<unsigned int> end_neuron;

	// Data sampling frequency
	double sampling_period;

	// Input value where the medium firing rate is reached
	//double error_center;

	// Slope of the tanh function
	//double slope;

	// Maximum speed frequency
	std::vector<double> max_spike_frequency;

	// Minimum speed frequency
	std::vector<double> min_spike_frequency;

	// Maximum value
	std::vector<double> max_values;

	// Minimum value
	std::vector<double> min_values;

public:
	//Spike neuron index array
	std::vector<unsigned int> neuron_index_array;

	//Spike time array
	std::vector<double> time_array;
	/*
	 * This function initializes a Poisson generator to generate activity to the network.
	 */
	ROSPoissonGenerator(std::vector<unsigned int> num_filters_row,
			std::vector<double> min_values,
			std::vector<double> max_values,
			std::vector<unsigned int> init_neuron,
			std::vector<unsigned int> end_neuron,
			double sampling_frequency,
			std::vector<double> max_spike_frequency,
			std::vector<double> min_spike_frequency,
			std::vector<double> overlapping_factor, 
			bool simTime,
			unsigned int seed,
			ros::NodeHandle * NodeHandler);

	// new //
	/*
	* This function returns the centers of the RBFs
	*/
	std::vector<std::vector<double> > GetCenters();



	/*
	 * This function generates and inject the activity produced by the input signal. It must
	 * be called every 1/sampling_frequency s. to achieve accurate firing rate generation.
	 */
	void GenerateActivity(double input_signal_value, double current_time, int joint, double new_sampling_period);
	//void GenerateActivity(std::vector<double> current_values, double current_time, int joint); //

	/*
	 * Destructor of the class
	 */
	virtual ~ROSPoissonGenerator();

// new //
private:

	/*
	 * Calculate the amplitude of the gaussian funcion
	 */
	void calculate_gaussian_amplitude();
// end new //
};

#endif /* RBFBANK_H_ */
