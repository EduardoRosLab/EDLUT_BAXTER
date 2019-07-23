/***************************************************************************
 *                           ROSTopicInputDriver.cpp                      *
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

// This node works as an auxiliar node for the EDLUT simulator node.

#include "edlut_ros/ROSTopicInputDriver.h"

#include "simulation/EventQueue.h"

#include "spike/InputSpike.h"
#include "spike/Network.h"
#include "spike/Neuron.h"

#include <string>

void ROSTopicInputDriver::SpikeCallback(const edlut_ros::Spike_group::ConstPtr& msg){
	for (int i=0; i<msg->neuron_index.size(); i++){
		ROS_DEBUG("EDLUT: Received spike: time %f and neuron %d. Current time: %f. Reference time: %f", msg->time[i], msg->neuron_index[i], ros::Time::now().toSec(), this->InitSimulationRosTime);
		InputSpike * NewSpike = new InputSpike(msg->time[i] - this->InitSimulationRosTime, this->Net->GetNeuronAt(msg->neuron_index[i])->get_OpenMP_queue_index(), this->Net->GetNeuronAt(msg->neuron_index[i]));
		this->Queue->InsertEvent(NewSpike->GetSource()->get_OpenMP_queue_index(),NewSpike);
	}
}

ROSTopicInputDriver::ROSTopicInputDriver(string TopicName, unsigned int MaxSpikeBuffered, double InitSimulationRosTime): NodeHandler(), CallbackQueue(), InitSimulationRosTime(InitSimulationRosTime), Net(0), Queue(0) {
	// TODO Auto-generated constructor stub
	this->NodeHandler.setCallbackQueue(&this->CallbackQueue);
	this->Subscriber = this->NodeHandler.subscribe(TopicName, MaxSpikeBuffered, &ROSTopicInputDriver::SpikeCallback, this);

}

ROSTopicInputDriver::~ROSTopicInputDriver() {
	// TODO Auto-generated destructor stub
}

void ROSTopicInputDriver::LoadInputs(EventQueue * Queue, Network * Net) throw (EDLUTFileException){
//ros::spinOnce();
	this->Net = Net;
	this->Queue = Queue;
	this->CallbackQueue.callAvailable();
	return;
}

ostream & ROSTopicInputDriver::PrintInfo(ostream & out){

	out << "- ROS Topic Input Spike Driver" << endl;

	return out;
}
