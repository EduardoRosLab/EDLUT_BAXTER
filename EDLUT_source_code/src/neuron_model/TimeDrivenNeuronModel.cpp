/***************************************************************************
 *                           TimeDrivenNeuronModel.cpp                     *
 *                           -------------------                           *
 * copyright            : (C) 2011 by Jesus Garrido                        *
 * email                : jgarrido@atc.ugr.es                              *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include "../../include/neuron_model/TimeDrivenNeuronModel.h"
#include "../../include/neuron_model/TimeDrivenModel.h"
#include "../../include/neuron_model/VectorNeuronState.h"

#include "../../include/openmp/openmp.h"

TimeDrivenNeuronModel::TimeDrivenNeuronModel(string NeuronTypeID, string NeuronModelID, TimeScale timeScale) : TimeDrivenModel(NeuronTypeID, NeuronModelID, timeScale), conductance_exp_values(0), N_conductances(0){
	// TODO Auto-generated constructor stub

}

TimeDrivenNeuronModel::~TimeDrivenNeuronModel() {
	delete integrationMethod;

	if(N_conductances!=0){
		delete conductance_exp_values;
	}
}


enum NeuronModelSimulationMethod TimeDrivenNeuronModel::GetModelSimulationMethod(){
	return TIME_DRIVEN_MODEL_CPU;
}

enum NeuronModelType TimeDrivenNeuronModel::GetModelType(){
	return NEURAL_LAYER;
}



void TimeDrivenNeuronModel::CheckValidIntegration(double CurrentTime){
	float valid_integration = 0.0f;
	for (int i = 0; i < this->State->SizeStates; i++){
		valid_integration += this->State->GetStateVariableAt(i, 0);
	}
	if (valid_integration != valid_integration){
		for (int i = 0; i < this->State->SizeStates; i++){
			if (this->State->GetStateVariableAt(i, 0) != this->State->GetStateVariableAt(i, 0)){

				cout << CurrentTime << ": Integration error in " << this->GetTypeID() << ", " << this->GetModelID() << endl;
				for (int z = 0; z < this->GetVectorNeuronState()->NumberOfVariables; z++){
					cout << this->State->GetStateVariableAt(i, z) << " ";
				}cout << endl;

				State->ResetState(i);
			}
		}
	}
}

void TimeDrivenNeuronModel::CheckValidIntegration(double CurrentTime, float valid_integration){
	if (valid_integration != valid_integration){
		for (int i = 0; i < this->State->SizeStates; i++){
			if (this->State->GetStateVariableAt(i, 0) != this->State->GetStateVariableAt(i, 0)){

				cout << CurrentTime << ": Integration error in " << this->GetTypeID() << ", " << this->GetModelID() << endl;
				for (int z = 0; z < this->GetVectorNeuronState()->NumberOfVariables; z++){
					cout << this->State->GetStateVariableAt(i, z) << " ";
				}cout << endl;

				State->ResetState(i);
			}
		}
	}
}




void TimeDrivenNeuronModel::Initialize_conductance_exp_values(int N_conductances, int N_elapsed_times){
	conductance_exp_values = new float[N_conductances *  N_elapsed_times]();
	this->N_conductances=N_conductances;
}

void TimeDrivenNeuronModel::Set_conductance_exp_values(int elapsed_time_index, int conductance_index, float value){
	conductance_exp_values[elapsed_time_index*N_conductances+conductance_index]=value;

}

float TimeDrivenNeuronModel::Get_conductance_exponential_values(int elapsed_time_index, int conductance_index){
	return conductance_exp_values[elapsed_time_index*N_conductances+conductance_index];
}

float * TimeDrivenNeuronModel::Get_conductance_exponential_values(int elapsed_time_index){
	return conductance_exp_values + elapsed_time_index*N_conductances;
}


