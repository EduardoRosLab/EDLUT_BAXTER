/***************************************************************************
 *                           Q_TrajectoryGenerator.cpp                     *
 *                           -------------------                           *
 * copyright            : (C) 2018 by Ignacio Abadia                       *
 * email                : iabadia@ugr.es		                               *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

// This is the trajectory generator node.
// It generates trajectory position and velocity signals for every joint out of
// .txt files in the format (one file for position, one for velocity):
// J0 J1 J2 J3 J4 J5 J6
// ...
// J0 J1 J2 J3 J4 J5 J6


#include "edlut_ros/Q_TrajectoryGenerator.h"

#include <cmath>
#include <vector>
#include <string>

Q_TrajectoryGenerator::Q_TrajectoryGenerator(bool sim_time, std::vector<double> max_pos_amplitude, std::vector<double> min_pos_amplitude,
std::vector<double> max_vel_amplitude, std::vector<double> min_vel_amplitude, int samples, std::string positions_file_name, std::string velocities_file_name,
double trajectory_frequency):
use_sim_time(sim_time), max_pos_amplitude(max_pos_amplitude), min_pos_amplitude(min_pos_amplitude), max_vel_amplitude(max_vel_amplitude),
min_vel_amplitude(min_vel_amplitude), samples(samples) , trajectory_frequency(trajectory_frequency){
	this->NodeHandler.setCallbackQueue(&CallbackQueue);
	if (use_sim_time){
		this->clock_subscriber = this->NodeHandler.subscribe("/clock_sync", 1000, &ExternalClock::ClockCallback, &ext_clock);
	}
	this->index = 0;
	this->aux_index_computation = this->trajectory_frequency*(this->samples-1);
	this->inverse_trajectory_frequency=1.0/this->trajectory_frequency;
	this->last_time = 0.0;
	this->first_iteration = true;
	this->file_positions.open(positions_file_name);
	this->file_velocities.open(velocities_file_name);
	std::string word;
	std::vector<double> row_positions;
	std::vector<double> row_velocities;
	double vel;
	double dt = (1.0/trajectory_frequency) / samples ;
	int x=0;
	int joint=0;

	std::cout << "----LOADING POSITION FILE---- " << "\n";
	while (file_positions >> word){
		row_positions.push_back(std::stod(word));
		joint++;
		if (joint == 7){
			this->Q_positions.push_back(row_positions);
			row_positions.clear();
			joint = 0;
		}
	}

	std::cout << "----LOADING VELOCITY FILE---- " << "\n";

	while (file_velocities >> word){
		row_velocities.push_back(std::stod(word));
		joint++;
		if (joint == 7){
			this->Qd_velocities.push_back(row_velocities);
			row_velocities.clear();
			joint = 0;
		}
	}

	std::cout << "----FINISH LOADING----" << "\n";

	this->file_positions.close();
	this->file_velocities.close();

}


Q_TrajectoryGenerator::~Q_TrajectoryGenerator() {
	// TODO Auto-generated destructor stub
}

ros::Time Q_TrajectoryGenerator::ResetGenerator(){
	CallbackQueue.callAvailable(ros::WallDuration(0.001));
	if(use_sim_time){
		this->InitTime = ext_clock.GetLastConfirmedTime();
	}
	else{
		this->InitTime = ros::Time::now();
	}
	return this->InitTime;
}

void Q_TrajectoryGenerator::GetState(std::vector<double> & CurrentPosition, std::vector<double> & CurrentVelocity, double current_time){
	double ElapsedTime = current_time - this->InitTime.toSec();
	if(ElapsedTime > this->last_time || this->first_iteration){
		this->last_time = ElapsedTime;
		this->index = fmod(ElapsedTime, this->inverse_trajectory_frequency) * this->aux_index_computation;
		for (unsigned int i=0; i<7; ++i){
			CurrentPosition[i] = this->Q_positions[this->index][i];
			CurrentVelocity[i] = this->Qd_velocities[this->index][i];
		}
		this->first_iteration = false;
	}

	return;
}

std::vector<double> Q_TrajectoryGenerator::GetStartingPoint(){
	std::vector<double> CurrentPosition(7);
	std::vector<double> CurrentVelocity(7);
	this->first_iteration = true;
	this->GetState(CurrentPosition, CurrentVelocity, 0.0);

	return CurrentPosition;
}
