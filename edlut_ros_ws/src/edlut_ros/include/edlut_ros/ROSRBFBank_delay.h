/***************************************************************************
 *                           ROSRBFBank_delay.h                                  *
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

#ifndef ROSRBFBANK_DELAY_H_
#define ROSRBFBANK_DELAY_H_

#include <vector>
#include <ros/ros.h>

/*
 * This class defines a Radial-basis function bank to generate spiking activity from multidimensional data.
 */
class ROSRBFBank_delay {
private:
	// Array with the centers of the RBFs
	std::vector<std::vector<double> > centers;

	// Width of the RBFs
	std::vector<std::vector<double> > width;

	// Publisher where the spikes will be published
	ros::Publisher * Publisher;

	// Index of the rbf input neurons (num_filters)
	std::vector<std::vector<unsigned int>> neuron_indexes;

	// Number of filters
	std::vector<unsigned int> num_filters_row;

	// Vector with the last spike times
	std::vector<std::vector<double>> last_spike_times;

	// Data sampling frequency
	double sampling_period;

	// Amplitude of the Gaussian function to be precalculated
	double amplitude;

	// Maximum speed frequency
	std::vector<double> max_spike_frequency;

	// Spike delay
	double delay;



public:
	//Spike neuron index array
	std::vector<unsigned int> neuron_index_array;

	//Spike time array
	std::vector<double> time_array;
	/*
	 * This function initializes a bank of RBFs to generate spike activity for the network. Constructor of the class.
	 */
	ROSRBFBank_delay(std::vector<unsigned int> num_filters_row,
			ros::Publisher * publisher,
			std::vector<double> min_values,
			std::vector<double> max_values,
			std::vector<unsigned int> min_neuron_index,
			std::vector<unsigned int> max_neuron_index,
			double sampling_frequency,
			std::vector<double> max_spike_frequency,
			std::vector<double> overlapping_factor,
			double delay);

	/*
	 * This function returns the centers of the RBFs
	 */
	std::vector<std::vector<double> > GetRBFCenters();

	/*
	 * This function generates and inject the activity produced by the training data from the current simulation time.
	 */
	void GenerateActivity(std::vector<double> current_values, double cur_simulation_time, double new_sampling_period);

	/*
	 * This function set the simulation to a new object
	 */
	void SetPublisher(ros::Publisher * newPublisher);

	/*
	 * Destructor of the class
	 */
	virtual ~ROSRBFBank_delay();

private:
	/*
	 * Calculate the value of the gaussian function
	 */
	void gaussian_function(double x, std::vector<double> centers, std::vector<double> witdhs, std::vector<double> &output_values);

	/*
	 * Calculate the amplitude of the gaussian funcion
	 */
	void calculate_gaussian_amplitude();

};

#endif /* RBFBANK_H_ */
