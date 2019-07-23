/***************************************************************************
 *                           IntegratoinMethod_GPU.cu                      *
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

#include "../../include/integration_method/FixedStep_GPU.h"
//#include "../../include/integration_method/FixedStep_GPU2.h"
#include "../../include/neuron_model/TimeDrivenNeuronModel_GPU.h"

#include "../../include/cudaError.h"
//Library for CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


FixedStep_GPU::FixedStep_GPU(TimeDrivenNeuronModel_GPU * NewModel, char * integrationMethodType, int N_neuronStateVariables, int N_differentialNeuronState, int N_timeDependentNeuronState):IntegrationMethod_GPU(NewModel, integrationMethodType, N_neuronStateVariables, N_differentialNeuronState, N_timeDependentNeuronState){
}

FixedStep_GPU::~FixedStep_GPU(){
}


void FixedStep_GPU::loadParameter(FILE *fh, long * Currentline, string fileName) throw (EDLUTFileException){

	skip_comments(fh, *Currentline);
	if (fscanf(fh, "%f", &elapsedTimeInSeconds) == 1){
		if (elapsedTimeInSeconds <= 0.0){
			throw EDLUTFileException(TASK_FIXED_STEP_LOAD, ERROR_FIXED_STEP_STEP_SIZE, REPAIR_FIXED_STEP, *Currentline, fileName.c_str(), true);
		}
	}
	else{
		throw EDLUTFileException(TASK_FIXED_STEP_LOAD, ERROR_FIXED_STEP_READ_STEP, REPAIR_FIXED_STEP, *Currentline, fileName.c_str(), true);
	}

	//Calculate the elapsed time size in neuron model time scale.
	elapsedTimeInNeuronModelScale = elapsedTimeInSeconds*model->GetTimeScale();
}
		


