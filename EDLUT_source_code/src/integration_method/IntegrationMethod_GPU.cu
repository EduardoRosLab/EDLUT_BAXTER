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

#include "../../include/integration_method/IntegrationMethod_GPU.h"
#include "../../include/integration_method/IntegrationMethod_GPU2.h"
#include "../../include/neuron_model/TimeDrivenNeuronModel_GPU.h"

#include "../../include/cudaError.h"
//Library for CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


IntegrationMethod_GPU::IntegrationMethod_GPU(TimeDrivenNeuronModel_GPU * NewModel, char * integrationMethodType, int N_neuronStateVariables, int N_differentialNeuronState, int N_timeDependentNeuronState):model(NewModel), N_NeuronStateVariables(N_neuronStateVariables), N_DifferentialNeuronState(N_differentialNeuronState), N_TimeDependentNeuronState(N_timeDependentNeuronState){
	IntegrationMethodType=new char [strlen(integrationMethodType)];
	strncpy(IntegrationMethodType,integrationMethodType,strlen(integrationMethodType));
}

IntegrationMethod_GPU::~IntegrationMethod_GPU(){
	delete [] IntegrationMethodType;
	HANDLE_ERROR(cudaFree(Buffer_GPU));
}

char * IntegrationMethod_GPU::GetType(){
	return this->IntegrationMethodType;
}

ostream & IntegrationMethod_GPU::PrintInfo(ostream & out){
	out << "Integration Method Type: " << this->GetType() << endl;

	return out;
}	
		


