/***************************************************************************
 *                           NeuronModel.cpp                               *
 *                           -------------------                           *
 * copyright            : (C) 2012 by Jesus Garrido and Francisco Naveros  *
 * email                : jgarrido@atc.ugr.es, fnaveros@ugr.es             *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include "../../include/neuron_model/NeuronModel.h"

#include "../../include/neuron_model/VectorNeuronState.h"
#include "../../include/spike/NeuronModelPropagationDelayStructure.h"
#include "../../include/neuron_model/CurrentSynapseModel.h"

#include "../../include/spike/Neuron.h"
#include "../../include/spike/Interconnection.h"


NeuronModel::NeuronModel(string NeuronTypeID, string NeuronModelID) : TypeID(NeuronTypeID), ModelID(NeuronModelID), State(0), CurrentSynapses(0){
	// TODO Auto-generated constructor stub
	PropagationStructure=new NeuronModelPropagationDelayStructure();

}

NeuronModel::NeuronModel(string NeuronTypeID, string NeuronModelID, TimeScale new_timeScale) : TypeID(NeuronTypeID), ModelID(NeuronModelID), State(0), CurrentSynapses(0),
			timeScale(new_timeScale), inv_timeScale(1.0f/new_timeScale) {
	// TODO Auto-generated constructor stub
	PropagationStructure=new NeuronModelPropagationDelayStructure();

}


NeuronModel::~NeuronModel() {
	// TODO Auto-generated destructor stub
	if (this->State!=0){
		delete this->State;
	}

	if (this->CurrentSynapses != 0){
		delete this->CurrentSynapses;
	}

	if(this->PropagationStructure){
		delete this->PropagationStructure;
	}
}

string NeuronModel::GetTypeID(){
	return this->TypeID;
}

string NeuronModel::GetModelID(){
	return this->ModelID;
}

//VectorNeuronState * NeuronModel::GetVectorNeuronState(){
//	return this->InitialState;
//}

NeuronModelPropagationDelayStructure * NeuronModel::GetNeuronModelPropagationDelayStructure(){
	return PropagationStructure;
}


void NeuronModel::SetTimeScale(float new_timeScale){
	this->timeScale=new_timeScale;
	this->inv_timeScale=1.0f/new_timeScale;
}

float NeuronModel::GetTimeScale(){
	return timeScale;
}

void NeuronModel::InitializeInputCurrentSynapseStructure(){
	if (this->CurrentSynapses != 0){
		this->CurrentSynapses->InitializeInputCurrentPerSynapseStructure();
	}
}

void NeuronModel::ProcessInputCurrent(Interconnection * inter, Neuron * target, float current){
	this->CurrentSynapses->SetInputCurrent(target->GetIndex_VectorNeuronState(), inter->GetSubindexType(), current);
}