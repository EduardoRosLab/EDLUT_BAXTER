/***************************************************************************
 *                           FixedStep.cpp                                 *
 *                           -------------------                           *
 * copyright            : (C) 2013 by Francisco Naveros                    *
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


#include "../../include/integration_method/FixedStep.h"
#include "../../include/neuron_model/TimeDrivenNeuronModel.h"


FixedStep::FixedStep(string integrationMethodType):IntegrationMethod(integrationMethodType){

}

FixedStep::~FixedStep(){

}


void FixedStep::loadParameter(TimeDrivenNeuronModel * model, FILE *fh, long * Currentline, string fileName) throw (EDLUTFileException){

	skip_comments(fh,*Currentline);
	if(fscanf(fh,"%f",&elapsedTimeInSeconds)==1){
		if(elapsedTimeInSeconds<=0.0){
			throw EDLUTFileException(TASK_FIXED_STEP_LOAD, ERROR_FIXED_STEP_STEP_SIZE, REPAIR_FIXED_STEP, *Currentline, fileName.c_str(), true);
		}
	}else{
		throw EDLUTFileException(TASK_FIXED_STEP_LOAD, ERROR_FIXED_STEP_READ_STEP, REPAIR_FIXED_STEP, *Currentline, fileName.c_str(), true);
	}

	//Calculate the elapsed time size in neuron model time scale.
	elapsedTimeInNeuronModelScale=elapsedTimeInSeconds*model->GetTimeScale();
}