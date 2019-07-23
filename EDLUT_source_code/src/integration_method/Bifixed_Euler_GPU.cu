/***************************************************************************
 *                           Euler_GPU.cu                                  *
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

#include "../../include/integration_method/Bifixed_Euler_GPU.h"
#include "../../include/integration_method/Bifixed_Euler_GPU2.h"
#include "../../include/neuron_model/TimeDrivenNeuronModel_GPU2.h"

//Library for CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


Bifixed_Euler_GPU::Bifixed_Euler_GPU(TimeDrivenNeuronModel_GPU * NewModel, int N_neuronStateVariables, int N_differentialNeuronState, int N_timeDependentNeuronState):BiFixedStep_GPU(NewModel, "Bifixed_Euler", N_neuronStateVariables, N_differentialNeuronState, N_timeDependentNeuronState){
}

Bifixed_Euler_GPU::~Bifixed_Euler_GPU(){
	cudaFree(AuxNeuronState);
}

__global__ void Bifixed_Euler_GPU_position(void ** vector, float * integration_method_parameters_GPU, float * element1){
	vector[0]=integration_method_parameters_GPU;
	vector[1]=element1;
}

void Bifixed_Euler_GPU::InitializeMemoryGPU(int N_neurons, int Total_N_thread){
	int size=2*sizeof(float *);

	cudaMalloc((void **)&Buffer_GPU, size);

	float integration_method_parameters_CPU[3];
	integration_method_parameters_CPU[0]=this->elapsedTimeInSeconds;
	integration_method_parameters_CPU[1]=this->bifixedElapsedTimeInSeconds;
	integration_method_parameters_CPU[2]=((float)this->N_BiFixedSteps);
	float * integration_method_parameters_GPU;
	cudaMalloc((void**)&integration_method_parameters_GPU, 3*sizeof(float));
	cudaMemcpy(integration_method_parameters_GPU,integration_method_parameters_CPU,3*sizeof(float),cudaMemcpyHostToDevice);

	cudaMalloc((void**)&AuxNeuronState, N_NeuronStateVariables*Total_N_thread*sizeof(float));

	Bifixed_Euler_GPU_position<<<1,1>>>(Buffer_GPU, integration_method_parameters_GPU, AuxNeuronState);

	cudaFree(integration_method_parameters_GPU);
}
		





