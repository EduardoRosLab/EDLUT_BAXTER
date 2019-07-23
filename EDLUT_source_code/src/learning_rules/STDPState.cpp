/***************************************************************************
 *                           STDPState.cpp                                 *
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

#include "../../include/learning_rules/STDPState.h"

#include "../../include/simulation/ExponentialTable.h"

#include <cmath>
#include <stdio.h>
#include <float.h>

STDPState::STDPState(int NumSynapses, float NewLTPValue, float NewLTDValue): ConnectionState(NumSynapses, 2), LTPTau(NewLTPValue), LTDTau(NewLTDValue){
	inv_LTPTau=1.0f/NewLTPValue;
	inv_LTDTau=1.0f/NewLTDValue;
}

STDPState::~STDPState() {
}

unsigned int STDPState::GetNumberOfPrintableValues(){
	return ConnectionState::GetNumberOfPrintableValues()+2;
}

double STDPState::GetPrintableValuesAt(unsigned int index, unsigned int position){
	if (position<ConnectionState::GetNumberOfPrintableValues()){
		return ConnectionState::GetStateVariableAt(index, position);
	} else if (position==ConnectionState::GetNumberOfPrintableValues()) {
		return this->LTPTau;
	} else if (position==ConnectionState::GetNumberOfPrintableValues()+1) {
		return this->LTDTau;
	} else return -1;
}

//float STDPState::GetPresynapticActivity(unsigned int index){
//	return this->GetStateVariableAt(index, 0);
//}

//float STDPState::GetPostsynapticActivity(unsigned int index){
//	return this->GetStateVariableAt(index, 1);
//}


//void STDPState::SetNewUpdateTime(unsigned int index, double NewTime, bool pre_post){
//	float PreActivity = this->GetPresynapticActivity(index);
//	float PostActivity = this->GetPostsynapticActivity(index);
//
//	float ElapsedTime=(float)(NewTime - this->GetLastUpdateTime(index));
//
//	//// Accumulate activity since the last update time
//	PreActivity *= exp(-ElapsedTime*this->inv_LTPTau);
//	PostActivity *= exp(-ElapsedTime*this->inv_LTDTau);
//
//	// Store the activity in state variables
//	//this->SetStateVariableAt(index, 0, PreActivity);
//	//this->SetStateVariableAt(index, 1, PostActivity);
//	this->SetStateVariableAt(index, 0, PreActivity, PostActivity);
//
//	this->SetLastUpdateTime(index, NewTime);
//}

void STDPState::SetNewUpdateTime(unsigned int index, double NewTime, bool pre_post){
	float ElapsedTime=(float)(NewTime - this->GetLastUpdateTime(index));

    //Accumulate activity since the last update time
	this->multiplyStateVaraibleAt(index,0,ExponentialTable::GetResult(-ElapsedTime*this->inv_LTPTau));
    //Accumulate activity since the last update time
	this->multiplyStateVaraibleAt(index,1,ExponentialTable::GetResult(-ElapsedTime*this->inv_LTDTau));
	
	this->SetLastUpdateTime(index, NewTime);
}



void STDPState::ApplyPresynapticSpike(unsigned int index){
	// Increment the activity in the state variable
	//this->SetStateVariableAt(index, 0, this->GetPresynapticActivity(index)+1.0f);
	this->incrementStateVaraibleAt(index, 0, 1.0f);
}

void STDPState::ApplyPostsynapticSpike(unsigned int index){
	// Increment the activity in the state variable
	//this->SetStateVariableAt(index, 1, this->GetPostsynapticActivity(index)+1.0f);
	this->incrementStateVaraibleAt(index, 1, 1.0f); 
}

