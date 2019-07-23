/***************************************************************************
 *                           EnableRobot.h                                  *
 *                           -------------------                           *
 * copyright            : (C) 2018 by Ignacio Abadia                        *
 * email                : iabadia@ugr.es                              *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef ENABLEROBOT_H_
#define ENABLEROBOT_H_

#include "ros/ros.h"
#include "std_msgs/Bool.h"


class EnableRobot{
private:

	bool enabled;

public:
	EnableRobot(): enabled(false) {}

	void EnabledCallback (const std_msgs::Bool::ConstPtr& msg){
		ROS_INFO("ENABLEDCALLBACK");
		if (msg->data){
			this->enabled = true;
		}
		return;
	}

	bool GetEnabled(){
		return this->enabled;
	}
};

#endif
