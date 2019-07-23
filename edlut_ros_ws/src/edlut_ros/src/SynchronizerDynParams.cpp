/***************************************************************************
 *                           SynchronizerDynParams.cpp                     *
 *                           -------------------                           *
 * copyright            : (C) 2018 by Ignacio Abadia                       *
 * email                : iabadia@ugr.es                                   *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

// Dynamic parameters for the synchronizer node.


#include "edlut_ros/SynchronizerDynParams.h"
#include <iostream>


SynchronizerDynParams::SynchronizerDynParams(ros::NodeHandle *nh, ros::NodeHandle *pnh) : nh_(nh), pnh(pnh), dr_srv_(*pnh), paused(false), stop_ts(false), time_stamp(0.0) {
	dynamic_reconfigure::Server<edlut_ros::SynchronizerDynParametersConfig>::CallbackType cb;
  cb = boost::bind(&SynchronizerDynParams::callback, this, _1, _2);
  dr_srv_.setCallback(cb);
}

SynchronizerDynParams::~SynchronizerDynParams() {
	// TODO Auto-generated destructor stub
}

void SynchronizerDynParams::callback(edlut_ros::SynchronizerDynParametersConfig &config, uint32_t level) {
	this->paused = config.Pause;
	this->stop_ts = config.Stop_at_time_stamp;
	this->time_stamp = config.Time_stamp;
	//std::cout<<"SYNCHRONIZER DYN Callback";
}

bool SynchronizerDynParams::GetPaused(){
	//std::cout<<"DYN PARAM GET PAUSED";
	return this->paused;
}

bool SynchronizerDynParams::GetStopTS(){
	//std::cout<<"DYN PARAM GET STOP TS";
	return this->stop_ts;
}

double SynchronizerDynParams::GetTimeStamp(){
	//std::cout<<"DYN PARAM GET TIME STAMP";
	return this->time_stamp;
}
