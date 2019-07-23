/***************************************************************************
 *                           VectorNeuronState.cpp                         *
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

#include "../../include/neuron_model/VectorNeuronState.h"
#include <string.h>
#include <stdio.h>

#include <iostream>
using namespace std;

VectorNeuronState::VectorNeuronState(unsigned int NumVariables, bool isTimeDriven): NumberOfVariables(NumVariables), TimeDriven(isTimeDriven),Is_Monitored(false),Is_GPU(false){
}

VectorNeuronState::VectorNeuronState(unsigned int NumVariables, bool isTimeDriven, bool isGPU): NumberOfVariables(NumVariables), TimeDriven(isTimeDriven),Is_Monitored(false),Is_GPU(isGPU){
}

VectorNeuronState::VectorNeuronState(const VectorNeuronState & OldState): NumberOfVariables(OldState.NumberOfVariables), SizeStates(OldState.SizeStates), TimeDriven(OldState.TimeDriven),Is_Monitored(OldState.Is_Monitored), Is_GPU(OldState.Is_GPU) {

	VectorNeuronStates = new float[GetNumberOfVariables()*GetSizeState()];
	memcpy(VectorNeuronStates, OldState.VectorNeuronStates, GetNumberOfVariables()*GetSizeState()*sizeof(float));

	LastUpdate=new double[GetSizeState()];
	memcpy(LastUpdate, OldState.LastUpdate, GetSizeState()*sizeof(double));

	LastSpikeTime=new double[GetSizeState()];
	memcpy(LastSpikeTime, OldState.LastSpikeTime, GetSizeState()*sizeof(double));

	InitialState=new float [GetNumberOfVariables()];
	memcpy(InitialState, OldState.InitialState, GetNumberOfVariables()*sizeof(float));
	
	if(!GetTimeDriven()){
		PredictedSpike=new double[GetSizeState()];
		memcpy(PredictedSpike, OldState.PredictedSpike, GetSizeState()*sizeof(double));

		PredictionEnd=new double[GetSizeState()];
		memcpy(PredictionEnd, OldState.PredictionEnd, GetSizeState()*sizeof(double));
	}
	else{
		if (Is_GPU){
			InternalSpike = new bool[1];
		}
		else{
			InternalSpikeIndexs = new int[1];
		}
	}
}


VectorNeuronState::VectorNeuronState(const VectorNeuronState & OldState, int index): NumberOfVariables(OldState.NumberOfVariables), SizeStates(1), TimeDriven(OldState.TimeDriven),Is_Monitored(OldState.Is_Monitored), Is_GPU(OldState.Is_GPU) {

	VectorNeuronStates = new float[GetNumberOfVariables()];
	for(int i=0; i<GetNumberOfVariables(); i++){
		VectorNeuronStates[i]=OldState.VectorNeuronStates[index*GetNumberOfVariables()+i];
	}

	LastUpdate=new double[1];
	LastUpdate[0]=OldState.LastUpdate[index];

	LastSpikeTime=new double[1];
	LastSpikeTime[0]=OldState.LastSpikeTime[index];
	
	InitialState=new float [GetNumberOfVariables()];
	memcpy(InitialState, OldState.InitialState, GetNumberOfVariables()*sizeof(float));

	if(!GetTimeDriven()){
		PredictedSpike=new double[1];
		PredictedSpike[0]=OldState.PredictedSpike[index];

		PredictionEnd=new double[1];
		PredictionEnd[0]=OldState.PredictionEnd[index];
	}else{
		if (Is_GPU){
			InternalSpike = new bool[1];
		}
		else{
			InternalSpikeIndexs = new int[1];
		}
	}
}

VectorNeuronState::~VectorNeuronState() {
	delete [] this->VectorNeuronStates;
	delete [] this->LastUpdate;
	delete [] this->LastSpikeTime;
	delete [] this->InitialState;

	if (!TimeDriven){
		delete [] this->PredictedSpike;
		delete [] this->PredictionEnd;
	}else{
		if (Is_GPU){
			delete [] InternalSpike;
		}
		else{
			delete [] InternalSpikeIndexs;
		}
	}
}

void VectorNeuronState::SetStateVariableAt(int index, int position, float NewValue){
	if(Is_GPU==false){
		this->VectorNeuronStates[index*NumberOfVariables + position] = NewValue;
	}else{
		this->VectorNeuronStates[this->SizeStates*position + index] = NewValue;
	}
}

void VectorNeuronState::IncrementStateVariableAt(int index, int position, float Increment){
	if(Is_GPU==false){
		this->VectorNeuronStates[index*NumberOfVariables + position]+= Increment;
	}else{
		this->VectorNeuronStates[this->SizeStates*position + index]+= Increment;
	}
}

//void VectorNeuronState::IncrementStateVariableAtCPU(int index, int position, float Increment){
//	this->VectorNeuronStates[index*NumberOfVariables + position]+= Increment;
//}

//void VectorNeuronState::IncrementStateVariableAtGPU(int index, int position, float Increment){
//	this->VectorNeuronStates[this->SizeStates*position + index]+= Increment;
//}

void VectorNeuronState::SetLastUpdateTime(int index, double NewTime){
	this->LastUpdate[index] = NewTime;
}

void VectorNeuronState::SetLastUpdateTime(double NewTime){
	for (int i = 0; i < this->SizeStates; i++){
		this->LastUpdate[i] = NewTime;
	}
}

void VectorNeuronState::SetNextPredictedSpikeTime(int index, double NextPredictedTime){
	this->PredictedSpike[index] = NextPredictedTime;
}

void VectorNeuronState::SetEndRefractoryPeriod(int index, double NextRefractoryPeriod){
	this->PredictionEnd[index] = NextRefractoryPeriod;
}

unsigned int VectorNeuronState::GetNumberOfVariables(){
	return this->NumberOfVariables;
}

//float VectorNeuronState::GetStateVariableAt(int index, int position){
//	if(Is_GPU==false){
//		return VectorNeuronStates[index*NumberOfVariables + position];
//	}else{
//		return VectorNeuronStates[this->SizeStates*position + index];
//	}
//}

//float * VectorNeuronState::GetStateVariableAt(int index){
//	return VectorNeuronStates+(index*NumberOfVariables);
//}

//double VectorNeuronState::GetLastUpdateTime(int index){
//	return this->LastUpdate[index];
//}

double VectorNeuronState::GetNextPredictedSpikeTime(int index){
	return this->PredictedSpike[index];
}

//double VectorNeuronState::GetEndRefractoryPeriod(int index){
//	return this->PredictionEnd[index];
//}

unsigned int VectorNeuronState::GetNumberOfPrintableValues(){
	return this->GetNumberOfVariables()+3;
}

double VectorNeuronState::GetPrintableValuesAt(int index, int position){
	if (position<this->GetNumberOfVariables()){
		return this->GetStateVariableAt(index, position);
	} else if (position==this->GetNumberOfVariables()) {
		return this->GetLastUpdateTime(index);
	} else if(GetTimeDriven()==true){
		return -1;
	}else if (position==this->GetNumberOfVariables()+1){
		return this->GetNextPredictedSpikeTime(index);
	} else if (position==this->GetNumberOfVariables()+2){
		return this->GetEndRefractoryPeriod(index);
	} else return -1;
}

//double VectorNeuronState::GetLastSpikeTime(int index){
//	return this->LastSpikeTime[index];
//}

void VectorNeuronState::NewFiredSpike(int index){
	this->LastSpikeTime[index] = 0;
	//this->PredictedSpike[index] = -1;
}


void VectorNeuronState::AddElapsedTime(int index, double ElapsedTime){

	this->LastSpikeTime[index] += ElapsedTime;
}


void VectorNeuronState::SetSizeState(int size){
	SizeStates=size;
}

int VectorNeuronState::GetSizeState(){
	return SizeStates;
}

void VectorNeuronState::SetTimeDriven(bool isTimeDriven){
	TimeDriven=isTimeDriven;
}

bool VectorNeuronState::GetTimeDriven(){
	return TimeDriven;
}


void VectorNeuronState::InitializeStates(int size, float * initialization){
	SetSizeState(size);

	VectorNeuronStates = new float[GetNumberOfVariables()*GetSizeState()]();
	LastUpdate=new double[GetSizeState()]();
	LastSpikeTime=new double[GetSizeState()]();
	InitialState=new float [GetNumberOfVariables()];
	
	if(!TimeDriven){
		PredictedSpike=new double[GetSizeState()]();
		PredictionEnd=new double[GetSizeState()]();
	}else{
		if (Is_GPU){
			InternalSpike = new bool[GetSizeState()]();
		}
		else{
			InternalSpikeIndexs = new int[GetSizeState()]();
		}
	}
	

	//For the CPU, we store all the variables of a neuron in adjacent memory positions to
	//improve the spatial location of the data.
	for(int z=0; z<GetSizeState()*GetNumberOfVariables(); z+=GetNumberOfVariables()){
		for (int j=0; j<GetNumberOfVariables(); j++){ 
			VectorNeuronStates[z+j]=initialization[j];
		}
	}

	for (int j=0; j<GetNumberOfVariables(); j++){ 
		InitialState[j]=initialization[j];
	}

	for(int z=0; z<GetSizeState(); z++){
		LastSpikeTime[z]=100.0;
	}

}


bool * VectorNeuronState::getInternalSpike(){
	return InternalSpike;
}

int * VectorNeuronState::getInternalSpikeIndexs(){
	return InternalSpikeIndexs;
}

int VectorNeuronState::getNInternalSpikeIndexs(){
	return NInternalSpikeIndexs;
}


void VectorNeuronState::Set_Is_Monitored(bool monitored){
	Is_Monitored=monitored;
}


bool VectorNeuronState::Get_Is_Monitored(){
	return Is_Monitored;
}


void VectorNeuronState::ResetState(int index){
	for (int j=0; j<GetNumberOfVariables(); j++){ 
		VectorNeuronStates[index*GetNumberOfVariables()+j]=InitialState[j];
	}
}


