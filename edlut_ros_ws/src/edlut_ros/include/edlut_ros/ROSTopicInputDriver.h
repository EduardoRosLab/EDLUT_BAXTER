
/***************************************************************************
 *                           ROSTopicInputDriver.h                         *
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

#ifndef ROSTOPICINPUTDRIVER_H_
#define ROSTOPICINPUTDRIVER_H_

/*!
 * \file ROSTopicInputDriver.h
 *
 * \author Jesus Garrido
 * \date August 2016
 *
 * This file declares a class for getting external input spikes from a ROS topic.
 */

#include <ros/ros.h>
#include <ros/callback_queue.h>

#include <edlut_ros/Spike.h>
#include <edlut_ros/Spike_group.h>


#include "spike/EDLUTFileException.h"
#include "communication/InputSpikeDriver.h"

class Network;
class EventQueue;

/*!
 * \class ROSTopicInputSpikeDriver
 *
 * \brief Class for getting input spikes from a ROS topic.
 *
 * This class abstract methods for getting the input spikes to the network. Its subclasses
 * implements the input source and methods.
 *
 * \author Jesus Garrido
 * \date August 2016
 */
class ROSTopicInputDriver: public InputSpikeDriver {

private:
	/*!
	 * Node handler
	 */
	ros::NodeHandle NodeHandler;

	/*!
	 * Topic subscriber
	 */
	ros::Subscriber Subscriber;

	/*!
	 * ROS Callback queue
	 */
	ros::CallbackQueue CallbackQueue;

	/*!
	 * ROS Init Simulation time (reference for upcoming spikes)
	 */
	double InitSimulationRosTime;

	/*!
	 * Simulated network pointer
	 */
	Network * Net;

	/*!
	 * Simulation queue
	 */
	EventQueue * Queue;


	/*!
	 * Callback function for reading input activity
	 */
	void SpikeCallback(const edlut_ros::Spike_group::ConstPtr& msg);


public:
	/*!
	 * \brief Class constructor.
	 *
	 * It creates a new object to introduce spikes.
	 */
	ROSTopicInputDriver(string TopicName, unsigned int MaxSpikeBuffered, double InitSimulationRosTime);

	/*!
	 * \brief Class desctructor.
	 *
	 * Class desctructor.
	 */
	~ROSTopicInputDriver();

	/*!
	 * \brief This method is only a stub.
	 *
	 * This method is only a stub.
	 *
	 * \param Queue The event queue where the input spikes are inserted.
	 * \param Net The network associated to the input spikes.
	 *
	 * \throw EDLUTException If something wrong happens in the input process.
	 */
	void LoadInputs(EventQueue * Queue, Network * Net) throw (EDLUTFileException);

	/*!
	 * \brief It prints the information of the object.
	 *
	 * It prints the information of the object.
	 *
	 * \param out The output stream where it prints the object to.
	 * \return The output stream.
	 */
	virtual ostream & PrintInfo(ostream & out);

};

#endif /* ROSTOPICINPUTDRIVER_H_ */
