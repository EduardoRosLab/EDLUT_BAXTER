/***************************************************************************
 *                           ConnectionState.cpp                           *
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

#include "../../include/learning_rules/ConnectionState.h"

#include "../../include/simulation/ExponentialTable.h"

ConnectionState::ConnectionState(unsigned int NumSynapses, int NumVariables): NumberOfSynapses(NumSynapses), NumberOfVariables(NumVariables){
	// TODO Auto-generated constructor stub
	this->LastUpdate = (double *) new double [NumSynapses]();
	this->StateVars = (float *) new float [NumSynapses*NumVariables]();
}

ConnectionState::~ConnectionState() {
	// TODO Auto-generated destructor stub
	if (this->LastUpdate!=0){
		delete [] this->LastUpdate;
	}

	if (this->StateVars!=0){
		delete [] this->StateVars;
	}
}

//void ConnectionState::void SetStateVariableAt(unsigned int index, unsigned int position,float NewValue){
//	*(this->StateVars + index*NumberOfVariables + position) = NewValue;
//}

unsigned int ConnectionState::GetNumberOfVariables(){
	return this->NumberOfVariables;
}

//float ConnectionState::GetStateVariableAt(unsigned int index, unsigned int position){
//	return *(this->StateVars + index*NumberOfVariables + position);
//}

//double ConnectionState::GetLastUpdateTime(unsigned int index){
//	return *(this->LastUpdate + index);
//}

//void ConnectionState::SetLastUpdateTime(unsigned int index, double NewUpdateTime){
//	*(this->LastUpdate+index) = NewUpdateTime;
//}

unsigned int ConnectionState::GetNumberOfPrintableValues(){
	return this->GetNumberOfVariables()+1;
}

double ConnectionState::GetPrintableValuesAt(unsigned int index, unsigned int position){
	if (position<this->GetNumberOfVariables()){
		return this->GetStateVariableAt(index, position);
	} else if (position==this->GetNumberOfVariables()) {
		return this->GetLastUpdateTime(index);
	} else return -1;
}




