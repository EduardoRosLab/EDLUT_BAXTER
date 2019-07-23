/***************************************************************************
 *                           RK4_GPU.cu                                    *
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

#include "../../include/integration_method/RK4_GPU.h"
#include "../../include/integration_method/RK4_GPU2.h"
#include "../../include/neuron_model/TimeDrivenNeuronModel_GPU2.h"

//Library for CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


RK4_GPU::RK4_GPU(TimeDrivenNeuronModel_GPU * NewModel, int N_neuronStateVariables, int N_differentialNeuronState, int N_timeDependentNeuronState):FixedStep_GPU(NewModel, "RK4", N_neuronStateVariables, N_differentialNeuronState, N_timeDependentNeuronState){
}

RK4_GPU::~RK4_GPU(){
	cudaFree(AuxNeuronState);
	cudaFree(AuxNeuronState1);
	cudaFree(AuxNeuronState2);
	cudaFree(AuxNeuronState3);
	cudaFree(AuxNeuronState4);
}

__global__ void RK4_GPU_position(void ** vector, float * integration_method_parameters_GPU, float * element1, float * element2, float * element3, float * element4, float * element5){
	vector[0]=integration_method_parameters_GPU;
	vector[1]=element1;
	vector[2]=element2;
	vector[3]=element3;
	vector[4]=element4;
	vector[5]=element5;
}

void RK4_GPU::InitializeMemoryGPU(int N_neurons, int Total_N_thread){
	int size=6*sizeof(float *);
	cudaMalloc((void **)&Buffer_GPU, size);

	float integration_method_parameters_CPU[1];
	integration_method_parameters_CPU[0]=this->elapsedTimeInSeconds;
	float * integration_method_parameters_GPU;
	cudaMalloc((void**)&integration_method_parameters_GPU, 1*sizeof(float));
	cudaMemcpy(integration_method_parameters_GPU,integration_method_parameters_CPU,1*sizeof(float),cudaMemcpyHostToDevice);

	cudaMalloc((void**)&AuxNeuronState, N_NeuronStateVariables*Total_N_thread*sizeof(float));
	cudaMalloc((void**)&AuxNeuronState1, N_NeuronStateVariables*Total_N_thread*sizeof(float));
	cudaMalloc((void**)&AuxNeuronState2, N_NeuronStateVariables*Total_N_thread*sizeof(float));
	cudaMalloc((void**)&AuxNeuronState3, N_NeuronStateVariables*Total_N_thread*sizeof(float));
	cudaMalloc((void**)&AuxNeuronState4, N_NeuronStateVariables*Total_N_thread*sizeof(float));

	RK4_GPU_position<<<1,1>>>(Buffer_GPU, integration_method_parameters_GPU, AuxNeuronState, AuxNeuronState1, AuxNeuronState2, AuxNeuronState3, AuxNeuronState4);
	cudaFree(integration_method_parameters_GPU);
}
		





