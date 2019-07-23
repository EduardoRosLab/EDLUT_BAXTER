/***************************************************************************
 *                           ExternalClock.h                                  *
 *                           -------------------                           *
 * copyright            : (C) 2017 by Jesus Garrido                        *
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

#ifndef EXTERNALCLOCK_H_
#define EXTERNALCLOCK_H_

#include "ros/ros.h"
#include "rosgraph_msgs/Clock.h"

//#define MIN_TIME ros::Duration(1.0e-6)

class ExternalClock{
private:
	ros::Time received_time;

	bool first_received;

public:
	ExternalClock(): received_time(0.0), first_received(false) {}

	void ClockCallback (const rosgraph_msgs::Clock& msg){
		this->first_received = true;
		if (this->received_time<msg.clock){
			this->received_time = msg.clock;
		}
		return;
	}

	ros::Time GetLastConfirmedTime(){
		return this->received_time;
	}

	bool FirstReceived(){
		return this->first_received;
	}
};

#endif
