/***************************************************************************
 *                           SinTrajectoryGenerator.cpp                    *
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

// This is the sinusoidal trajectory generator node. It generates position and
// velocity trajectory signals for every joint according to the amplitude,
// frequency and phase parameters. 


#include "edlut_ros/SinTrajectoryGenerator.h"

#include <cmath>
#include <vector>

SinTrajectoryGenerator::SinTrajectoryGenerator(std::vector<double> Amplitude, std::vector<double> Frequency, std::vector<double> Phase, bool sim_time, std::vector<double> max_pos_amplitude,
std::vector<double> min_pos_amplitude, std::vector<double> max_vel_amplitude, std::vector<double> min_vel_amplitude):
Amplitude(Amplitude), Frequency(Frequency), Phase(Phase), use_sim_time(sim_time), max_pos_amplitude(max_pos_amplitude), min_pos_amplitude(min_pos_amplitude), max_vel_amplitude(max_vel_amplitude),
min_vel_amplitude(min_vel_amplitude) {
	this->NodeHandler.setCallbackQueue(&CallbackQueue);
	this->clock_subscriber = this->NodeHandler.subscribe("/clock_sync", 1000, &ExternalClock::ClockCallback, &ext_clock);
}

SinTrajectoryGenerator::~SinTrajectoryGenerator() {
	// TODO Auto-generated destructor stub
}

ros::Time SinTrajectoryGenerator::ResetGenerator(){
	CallbackQueue.callAvailable(ros::WallDuration(0.001));
	if(use_sim_time){
		this->InitTime = ext_clock.GetLastConfirmedTime();
	}
	else{
		this->InitTime = ros::Time::now();
	}
	return this->InitTime;
}

void SinTrajectoryGenerator::GetState(std::vector<double> & CurrentPosition, std::vector<double> & CurrentVelocity){
	CallbackQueue.callAvailable(ros::WallDuration(0.001));
	double ElapsedTime;
	if (use_sim_time){
		ElapsedTime = (ext_clock.GetLastConfirmedTime()-this->InitTime).toSec();
	}
	else{
		ElapsedTime = (ros::Time::now()-this->InitTime).toSec();
	}
	this->GetState(CurrentPosition, CurrentVelocity, ElapsedTime);
	return;
}

void SinTrajectoryGenerator::GetState(std::vector<double> & CurrentPosition, std::vector<double> & CurrentVelocity, double ElapsedTime){
	const double pi = 3.1415926535897;
	for (unsigned int i=0; i<this->Amplitude.size(); ++i){
		CurrentPosition[i] = this->Amplitude[i] * sin(2*pi*this->Frequency[i]*ElapsedTime + this->Phase[i]);
		CurrentVelocity[i] = 2*pi*this->Frequency[i]*this->Amplitude[i]* cos(2*pi*this->Frequency[i]*ElapsedTime + this->Phase[i]);
		if (CurrentPosition[i]>max_pos_amplitude[i] || CurrentPosition[i]<min_pos_amplitude[i] || CurrentVelocity[i]>max_vel_amplitude[i] || CurrentVelocity[i]<min_vel_amplitude[i]){
			std::cout<<"ERROR: the generated trajectory for joint " << i << " exceeds joint position or velocity hardware limitations." << std::endl;
		}
	}
	return;
}

std::vector<double> SinTrajectoryGenerator::GetStartingPoint(){
	std::vector<double> CurrentPosition(this->Frequency.size());
	std::vector<double> CurrentVelocity(this->Frequency.size());

	this->GetState(CurrentPosition, CurrentVelocity, 0.0);

	return CurrentPosition;
}
