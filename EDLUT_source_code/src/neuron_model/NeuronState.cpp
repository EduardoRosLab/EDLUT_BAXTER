/***************************************************************************
 *                           NeuronState.cpp                               *
 *                           -------------------                           *
 * copyright            : (C) 2010 by Jesus Garrido                        *
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

#include "../../include/neuron_model/NeuronState.h"

NeuronState::NeuronState(unsigned int NumVariables): NumberOfVariables(NumVariables), LastUpdate(0), PredictedSpike(-1), PredictionEnd(-1), LastSpikeTime(100){
	// TODO Auto-generated constructor stub
	this->StateVars = (float *) new float [NumVariables];
}

NeuronState::NeuronState(const NeuronState & OldState): NumberOfVariables(OldState.NumberOfVariables),
		LastUpdate(OldState.LastUpdate), PredictedSpike(OldState.PredictedSpike), PredictionEnd(OldState.PredictionEnd), LastSpikeTime(OldState.LastSpikeTime) {
	this->StateVars = (float *) new float [this->NumberOfVariables];

	for (unsigned int i=0; i<this->NumberOfVariables; ++i){
		this->StateVars[i] = OldState.StateVars[i];
	}
}

NeuronState::~NeuronState() {
	// TODO Auto-generated destructor stub
	if (this->StateVars!=0){
		delete [] this->StateVars;
	}
}

void NeuronState::SetStateVariableAt(unsigned int position,float NewValue){
	this->StateVars[position] = NewValue;
}

void NeuronState::SetLastUpdateTime(double NewTime){
	this->LastUpdate = NewTime;
}

void NeuronState::SetNextPredictedSpikeTime(double NextPredictedTime){
	this->PredictedSpike = NextPredictedTime;
}

void NeuronState::SetEndRefractoryPeriod(double NextRefractoryPeriod){
	this->PredictionEnd = NextRefractoryPeriod;
}

unsigned int NeuronState::GetNumberOfVariables(){
	return this->NumberOfVariables;
}

float NeuronState::GetStateVariableAt(unsigned int position){
	return *(this->StateVars+position);
}

double NeuronState::GetLastUpdateTime(){
	return this->LastUpdate;
}

double NeuronState::GetNextPredictedSpikeTime(){
	return this->PredictedSpike;
}

double NeuronState::GetEndRefractoryPeriod(){
	return this->PredictionEnd;
}

unsigned int NeuronState::GetNumberOfPrintableValues(){
	return this->GetNumberOfVariables()+3;
}

double NeuronState::GetPrintableValuesAt(unsigned int position){
	if (position<this->GetNumberOfVariables()){
		return this->GetStateVariableAt(position);
	} else if (position==this->GetNumberOfVariables()) {
		return this->GetLastUpdateTime();
	} else if (position==this->GetNumberOfVariables()+1){
		return this->GetNextPredictedSpikeTime();
	} else if (position==this->GetNumberOfVariables()+2){
		return this->GetEndRefractoryPeriod();
	} else return -1;
}

double NeuronState::GetLastSpikeTime(){
	return this->LastSpikeTime;
}

void NeuronState::NewFiredSpike(){
	this->LastSpikeTime = 0;
}

void NeuronState::AddElapsedTime(float ElapsedTime){

	this->LastSpikeTime += ElapsedTime;
}



