/***************************************************************************
 *                           CurrentSynapseModel.cpp                      *
 *                           -------------------                           *
 * copyright            : (C) 2018 by Francisco Naveros                    *
 * email                : fnaveros@ugr.es                                  *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include "../../include/neuron_model/CurrentSynapseModel.h"

using namespace std;

CurrentSynapseModel::CurrentSynapseModel(): N_target_neurons(0){
	// TODO Auto-generated constructor stub
}

CurrentSynapseModel::CurrentSynapseModel(int size): N_target_neurons(size){
	InitializeNInputCurrentSynapsesPerNeuron();
}

CurrentSynapseModel::~CurrentSynapseModel() {
	for (int i = 0; i < this->N_target_neurons; i++){
		if (N_current_synapses_per_target_neuron[i] > 0){
			delete input_current_per_synapse[i];
		}
	}
	delete input_current_per_synapse;
	delete N_current_synapses_per_target_neuron;
}


void CurrentSynapseModel::SetNTargetNeurons(int size){
	N_target_neurons = size;
}


int CurrentSynapseModel::GetNTargetNeurons(){
	return N_target_neurons;
}



void CurrentSynapseModel::InitializeNInputCurrentSynapsesPerNeuron(){
	//Neuron number insed the model
	N_current_synapses_per_target_neuron = new int[this->N_target_neurons]();
}

void CurrentSynapseModel::IncrementNInputCurrentSynapsesPerNeuron(int neuron_index){
	N_current_synapses_per_target_neuron[neuron_index]++;
}

int CurrentSynapseModel::GetNInputCurrentSynapsesPerNeuron(int neuron_index){
	return N_current_synapses_per_target_neuron[neuron_index];
}

void CurrentSynapseModel::InitializeInputCurrentPerSynapseStructure(){
	input_current_per_synapse = (float **) new float *[this->N_target_neurons];
	for (int i = 0; i < this->N_target_neurons; i++){
		if (N_current_synapses_per_target_neuron[i] > 0){
			input_current_per_synapse[i] = new float[N_current_synapses_per_target_neuron[i]]();
		}
	}
}

void CurrentSynapseModel::SetInputCurrent(int neuron_index, int synapse_index, float current){
	input_current_per_synapse[neuron_index][synapse_index] = current;
}

float CurrentSynapseModel::GetTotalCurrent(int neuron_index){
	float current = 0;
	for (int i = 0; i < N_current_synapses_per_target_neuron[neuron_index]; i++){
		current += input_current_per_synapse[neuron_index][i];
	}
	return current;
}

