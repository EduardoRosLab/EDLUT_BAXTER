/***************************************************************************
 *                           ROSRBFBank_delay.cpp                                  *
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
 // the analog data to neurons in the RBF bank.

#include <vector>
#include <iostream>
#include <iterator>
#include <cmath>

#include "edlut_ros/ROSRBFBank_delay.h"
#include "edlut_ros/Spike_group.h"

#include "omp.h"


ROSRBFBank_delay::ROSRBFBank_delay(std::vector<unsigned int> num_filters_row,
		ros::Publisher * publisher,
		std::vector<double> min_values,
		std::vector<double> max_values,
		std::vector<unsigned int> min_neuron_index,
		std::vector<unsigned int> max_neuron_index,
		double sampling_frequency,
		std::vector<double> max_spike_frequency,
		std::vector<double> overlapping_factor,
		double delay):
			num_filters_row(num_filters_row), Publisher(publisher), amplitude(0), max_spike_frequency(max_spike_frequency), delay(delay){

	this->neuron_indexes = std::vector<std::vector<unsigned int>> (num_filters_row.size());
	for (int i=0; i<num_filters_row.size(); i++){
		for (unsigned int j = min_neuron_index[i]; j<= max_neuron_index[i]; ++j){
			this->neuron_indexes[i].push_back(j);
		}
	}

	this->centers = std::vector<std::vector<double> > ();
	this->width = std::vector<std::vector<double> > ();

	// Calculate centers and widths for each dimension
	for (unsigned int i=0; i<num_filters_row.size(); ++i){
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

	this->calculate_gaussian_amplitude();

	this->last_spike_times = std::vector<std::vector<double>> (num_filters_row.size());
	for (int i= 0; i< this->num_filters_row.size(); i++){
		this->last_spike_times[i]=std::vector<double>(num_filters_row[i], -1000);
	}

	this->sampling_period = 1./sampling_frequency;

	return;
}

ROSRBFBank_delay::~ROSRBFBank_delay() {
	// TODO Auto-generated destructor stub
}

std::vector<std::vector<double> > ROSRBFBank_delay::GetRBFCenters(){
	return this->centers;
}

void ROSRBFBank_delay::SetPublisher(ros::Publisher * newPublisher){
	this->Publisher = newPublisher;
}

void ROSRBFBank_delay::gaussian_function(double x, std::vector<double> centers, std::vector<double> witdhs, std::vector<double> &output_values)
{
	 // SQUARE
	 for (std::vector<double>::const_iterator itcenter = centers.begin(), itwitdh = witdhs.begin(); itcenter!=centers.end(); ++itcenter, ++itwitdh){
			double value = 1.0 - fabs((x-*itcenter)/(*itwitdh));
			if (value <= 0.0){
				value = 0.0;
			}else{
				value = 1.0;
			}
			output_values.push_back(this->amplitude*value);
	 }

}

void ROSRBFBank_delay::calculate_gaussian_amplitude(){
	this->amplitude = 1.0;
	return;
}

/*
 * This function generates and inject the activity produced by the training data from the current simulation time.
 */
void ROSRBFBank_delay::GenerateActivity(std::vector<double> current_values, double cur_simulation_time, double new_sampling_period){

		// #pragma omp parallel for
		for (int i=0; i< this->num_filters_row.size(); i++){
			unsigned int num_centers = this->num_filters_row[i];
			if (current_values[i]<this->centers[i][0]){
				current_values[i] = this->centers[i][0];
			}
			else if (current_values[i]>this->centers[i][num_centers-1]){
				current_values[i] = this->centers[i][num_centers-1];
			}

			std::vector<double> it_centers = this->centers[i];
			std::vector<double> it_widths = this->width[i];
			std::vector<unsigned int> it_neuron = this->neuron_indexes[i];
			std::vector<double> it_spike_time = this->last_spike_times[i];
			std::vector<double> output_current;

			this->gaussian_function(current_values[i], it_centers, it_widths, output_current);


			std::vector<double> spike_period (this->num_filters_row[i]);
			std::vector<double> spike_time (this->num_filters_row[i]);
			for (int j=0; j< this->num_filters_row[i]; j++){

				spike_period[j] = 1./(this->max_spike_frequency[i]*output_current[j]);
				spike_time[j] = (it_spike_time[j])+spike_period[j];

				if(spike_time[j]<cur_simulation_time){
					spike_time[j]=cur_simulation_time;
				}

				for(;spike_time[j]<cur_simulation_time+new_sampling_period;spike_time[j]+=spike_period[j]){
					this->neuron_index_array.push_back(it_neuron[j]);
					this->time_array.push_back(spike_time[j] + delay);


					this->last_spike_times[i][j]=spike_time[j];

				}
			}
		}
	return;
}
