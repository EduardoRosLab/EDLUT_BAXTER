/***************************************************************************
 *                           ROSTopicOutputDriver.cpp                      *
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

#include "edlut_ros/ROSTopicOutputDriver.h"

#include "spike/Spike.h"
#include "spike/Neuron.h"

#include <edlut_ros/Spike.h>
#include <edlut_ros/Spike_group.h>


ROSTopicOutputDriver::ROSTopicOutputDriver(string TopicName, unsigned int MaxSpikeBuffered, double InitSimulationRosTime, double output_delay):
	NodeHandler(), InitSimulationRosTime(InitSimulationRosTime), output_delay(output_delay) {
	this->Publisher_group = this->NodeHandler.advertise<edlut_ros::Spike_group>(TopicName, MaxSpikeBuffered);
	this->neuron_index_array = new unsigned int[1000];
	this->time_array = new double[1000];
	index = 0;
}

ROSTopicOutputDriver::~ROSTopicOutputDriver() {
	// TODO Auto-generated destructor stub
}

void ROSTopicOutputDriver::WriteSpike(const Spike * NewSpike) throw (EDLUTException){
	if(index < 1000){
		this->neuron_index_array[index]=(NewSpike->GetSource()->GetIndex());
		this->time_array[index]=( NewSpike->GetTime()+this->InitSimulationRosTime);
		index++;
		if(index == 1000){
			this->SendSpikeGroup();
		}
	}

	return;
}

void ROSTopicOutputDriver::WriteState(float Time, Neuron * Source) throw (EDLUTException){
	return;
}

bool ROSTopicOutputDriver::IsBuffered() const{
	return false;
}

bool ROSTopicOutputDriver::IsWritePotentialCapable() const{
	return false;
}

void ROSTopicOutputDriver::FlushBuffers() throw (EDLUTException){

	return;
}

int ROSTopicOutputDriver::GetBufferedSpikes(double *& Times, long int *& Cells){
	return 0;

}

bool ROSTopicOutputDriver::RemoveBufferedSpike(double & Time, long int & Cell){
	return true;
}

void ROSTopicOutputDriver::SendSpikeGroup(){
	if(index>0){
		edlut_ros::Spike_group spike_group;
		spike_group.time = std::vector<double> (this->index);
		spike_group.neuron_index = std::vector<unsigned int> (this->index);
		for (int i=0; i<this->index; i++){
			spike_group.time[i] = this->time_array[i] + this->output_delay;
			spike_group.neuron_index[i] = this->neuron_index_array[i];
		}
		this->Publisher_group.publish(spike_group);
		index=0;
	}
}

ostream & ROSTopicOutputDriver::PrintInfo(ostream & out){

	out << "- Array Output Spike Driver" << endl;

	return out;
}
