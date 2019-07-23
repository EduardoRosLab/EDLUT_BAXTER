/***************************************************************************
 *                           TimeDrivenSinCurrentGenerator.cpp             *
 *                           -------------------                           *
 * copyright            : (C) 2013 by Jesus Garrido and Francisco Naveros  *
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

#include "../../include/neuron_model/TimeDrivenSinCurrentGenerator.h"
#include "../../include/neuron_model/VectorNeuronState.h"

#include <iostream>
#include <cmath>
#include <string>

#include "../../include/openmp/openmp.h"

#include "../../include/spike/EDLUTFileException.h"
#include "../../include/spike/Neuron.h"
#include "../../include/spike/PropagatedCurrent.h"
#include "../../include/spike/Interconnection.h"

#include "../../include/simulation/Utils.h"

#include "../../include/openmp/openmp.h"


void TimeDrivenSinCurrentGenerator::LoadNeuronModel(string ConfigFile) throw (EDLUTFileException){
	FILE *fh;
	long Currentline = 0L;
	fh=fopen(ConfigFile.c_str(),"rt");
	if(fh){
		Currentline=1L;
		skip_comments(fh,Currentline);
		if(fscanf(fh,"%f",&this->frequency)==1){
			skip_comments(fh,Currentline);

			if (fscanf(fh,"%f",&this->phase)==1){
				skip_comments(fh,Currentline);

				if(fscanf(fh,"%f",&this->amplitude)==1){
					skip_comments(fh,Currentline);

					if(fscanf(fh,"%f",&this->offset)==1){
						skip_comments(fh,Currentline);
						this->State = (VectorNeuronState *) new VectorNeuronState(1, true);

					} else {
						throw EDLUTFileException(TASK_TIME_DRIVEN_SIN_CURRENT_GENERATOR_LOAD, ERROR_TIME_DRIVEN_SIN_CURRENT_GENERATOR_OFFSET, REPAIR_NEURON_MODEL_VALUES, Currentline, ConfigFile.c_str(), true);
					}
				} else {
					throw EDLUTFileException(TASK_TIME_DRIVEN_SIN_CURRENT_GENERATOR_LOAD, ERROR_TIME_DRIVEN_SIN_CURRENT_GENERATOR_AMPLITUDE, REPAIR_NEURON_MODEL_VALUES, Currentline, ConfigFile.c_str(), true);
				}
			} else {
				throw EDLUTFileException(TASK_TIME_DRIVEN_SIN_CURRENT_GENERATOR_LOAD, ERROR_TIME_DRIVEN_SIN_CURRENT_GENERATOR_PHASE, REPAIR_NEURON_MODEL_VALUES, Currentline, ConfigFile.c_str(), true);
			}
		} else {
			throw EDLUTFileException(TASK_TIME_DRIVEN_SIN_CURRENT_GENERATOR_LOAD, ERROR_TIME_DRIVEN_SIN_CURRENT_GENERATOR_FREQUENCY, REPAIR_NEURON_MODEL_VALUES, Currentline, ConfigFile.c_str(), true);
		}
	
		//LOAD TIME-DRIVEN STEP SIZE
		double step = LoadTimeDrivenStepSize(this->GetModelID(), fh, Currentline);
		
		//SET TIME-DRIVEN STEP SIZE
		this->SetTimeDrivenStepSize(step);

	}else{
		throw EDLUTFileException(TASK_TIME_DRIVEN_SIN_CURRENT_GENERATOR_LOAD, ERROR_NEURON_MODEL_OPEN, REPAIR_NEURON_MODEL_NAME, Currentline, ConfigFile.c_str(), true);
	}
	fclose(fh);
}


//this neuron model is implemented in a second scale.
TimeDrivenSinCurrentGenerator::TimeDrivenSinCurrentGenerator(string NeuronTypeID, string NeuronModelID): TimeDrivenInputDevice(NeuronTypeID, NeuronModelID, SecondScale), frequency(0), phase(0), amplitude(0), offset(0){
}

TimeDrivenSinCurrentGenerator::~TimeDrivenSinCurrentGenerator(void)
{
}

void TimeDrivenSinCurrentGenerator::LoadNeuronModel() throw (EDLUTFileException){
	this->LoadNeuronModel(this->GetModelID()+".cfg");
}

VectorNeuronState * TimeDrivenSinCurrentGenerator::InitializeState(){
	return this->GetVectorNeuronState();
}


bool TimeDrivenSinCurrentGenerator::UpdateState(int index, double CurrentTime){
	//float * NeuronState;
	//NeuronState[0] --> current 

	for (int i = 0; i< State->GetSizeState(); i++){

		float current = offset + amplitude * sin(CurrentTime * frequency * 2 * 3.141592 + phase);
		State->SetStateVariableAt(i, 0, current);
	}

	return false;
}



enum NeuronModelOutputActivityType TimeDrivenSinCurrentGenerator::GetModelOutputActivityType(){
	return OUTPUT_CURRENT;
}


ostream & TimeDrivenSinCurrentGenerator::PrintInfo(ostream & out){
	out << "- Time-Driven Sinusoidal Current Generator: " << this->GetModelID() << endl;

	out << "\tFrequency: " << this->frequency << "Hz\tPhase: " << this->phase << "rad\tAmplitude: " << this->amplitude << "////////REVISAR////////" << endl;

	out << "\tOffset: " << this->offset << "///////REVISAR////////" << endl;

	return out;
}	


void TimeDrivenSinCurrentGenerator::InitializeStates(int N_neurons, int OpenMPQueueIndex){
	//Initialize neural state variables.
	float initialization[] = {0.0f};
	State->InitializeStates(N_neurons, initialization);
}

















