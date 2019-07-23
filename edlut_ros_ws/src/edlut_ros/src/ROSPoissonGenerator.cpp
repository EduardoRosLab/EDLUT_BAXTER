/***************************************************************************
 *                           ROSPoissonGenerator.cpp                       *
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

// This node performs an analogue-to-spike translation using a Poisson generator
// approach.

#include <vector>
#include <iostream>
#include <iterator>
#include <cmath>
#include <cstdlib>

#include "edlut_ros/ROSPoissonGenerator.h"
#include "edlut_ros/Spike_group.h"

#include "omp.h"

ROSPoissonGenerator::ROSPoissonGenerator(std::vector<unsigned int> num_filters_row, //
		std::vector<double> min_values,
		std::vector<double> max_values,
		std::vector<unsigned int> init_neuron,
		std::vector<unsigned int> end_neuron,
		double sampling_frequency,
		std::vector<double> max_spike_frequency,
		std::vector<double> min_spike_frequency,
		std::vector<double> overlapping_factor,
		bool sim_time,
		unsigned int seed,
		ros::NodeHandle * NodeHandler):
			NodeHandler(NodeHandler),
			init_neuron(init_neuron),
			end_neuron(end_neuron),
			max_spike_frequency(max_spike_frequency),
			min_spike_frequency(min_spike_frequency),
			max_values(max_values),
			min_values(min_values),
			use_sim_time(sim_time){

	this->sampling_period = 1./sampling_frequency;

	srand(seed);

	this->clock_subscriber = this->NodeHandler->subscribe("/clock_sync", 1000, &ExternalClock::ClockCallback, &ext_clock);

	this->centers = std::vector<std::vector<double> > ();
	this->width = std::vector<std::vector<double> > ();

	// Calculate centers and widths for each dimension
	for (unsigned int i=0; i<num_filters_row.size(); ++i){
		if (max_spike_frequency[i] > sampling_frequency){
			ROS_WARN("Poisson Generator: Max spike frequency (%f) is higher than sampling frequency (%f). Activity will be cut.", max_spike_frequency[i], sampling_frequency);
		}

		std::vector<double> c;
		std::vector<double> w;

		double min_value = min_values[i];
		double max_value = max_values[i];
		double step = (num_filters_row[i]>1)?(max_value - min_value)/(num_filters_row[i]-1.):(max_value - min_value);
		for(unsigned int j=0; j<num_filters_row[i]; ++j){
			double cen = min_value + step*j;
			double wid = step*overlapping_factor[i];
			c.push_back(cen);
			w.push_back(wid);
		}

		this->centers.push_back(c);
		this->width.push_back(w);
	}

	this->last_spike_times = std::vector<double> (this->centers.size(), -1000.0);
	return;
}

ROSPoissonGenerator::~ROSPoissonGenerator() {
	// TODO Auto-generated destructor stub
}


std::vector<std::vector<double> > ROSPoissonGenerator::GetCenters(){
	return this->centers;
}

/*
 * This function generates and inject the activity produced by the input signal. It must
 * be called every 1/sampling_frequency s. to achieve accurate firing rate generation.
 */
void ROSPoissonGenerator::GenerateActivity(double input_signal_value, double current_time, int joint, double new_sampling_period){
	//ROS_INFO("%f",sp_firing_rate);
	for (unsigned int index=0, id_neuron=this->init_neuron[joint]; id_neuron<=this->end_neuron[joint]; ++index, ++id_neuron){
		double norm_firing_rate = (tanh((input_signal_value-this->centers[joint][index])/this->width[joint][index])+1.0)*0.5;

		double sp_firing_rate = min_spike_frequency[joint] + norm_firing_rate * (max_spike_frequency[joint]-min_spike_frequency[joint]);

		if (rand()/double(RAND_MAX) < sp_firing_rate*new_sampling_period){

			int n_spikes = input_signal_value / this->centers[joint][index];
			if(n_spikes < 1){
				n_spikes=1;
			}
			if(n_spikes > 6){
				n_spikes=6;
			}
			for (int i=0; i< n_spikes; i++){

				if(use_sim_time){
					this->time_array.push_back(current_time + i * 0.002);
				}
				else{
					this->time_array.push_back(ros::Time::now().toSec() + i * 0.002);
				}
				this->neuron_index_array.push_back(id_neuron);
			}

		}
	}

	return;
}
