/***************************************************************************
 *                           BDFn_GPU.cu                                   *
 *                           -------------------                           *
 * copyright            : (C) 2012 by Francisco Naveros                    *
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

#include "../../include/integration_method/BDFn_GPU.h"
#include "../../include/integration_method/BDFn_GPU2.h"
#include "../../include/neuron_model/TimeDrivenNeuronModel_GPU2.h"

//Library for CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

const float BDFn_GPU::Coeficient_CPU [7*7]={1.0f,1.0f,0.0f,0.0f,0.0f,0.0f,0.0f,
						1.0f,1.0f,0.0f,0.0f,0.0f,0.0f,0.0f,
						2.0f/3.0f,4.0f/3.0f,-1.0f/3.0f,0.0f,0.0f,0.0f,0.0f,
						6.0f/11.0f,18.0f/11.0f,-9.0f/11.0f,2.0f/11.0f,0.0f,0.0f,0.0f,
						12.0f/25.0f,48.0f/25.0f,-36.0f/25.0f,16.0f/25.0f,-3.0f/25.0f,0.0f,0.0f,
						60.0f/137.0f,300.0f/137.0f,-300.0f/137.0f,200.0f/137.0f,-75.0f/137.0f,12.0f/137.0f,0.0f,
						60.0f/147.0f,360.0f/147.0f,-450.0f/147.0f,400.0f/147.0f,-225.0f/147.0f,72.0f/147.0f,-10.0f/147.0f};


BDFn_GPU::BDFn_GPU(TimeDrivenNeuronModel_GPU * NewModel, int N_neuronStateVariables, int N_differentialNeuronState, int N_timeDependentNeuronState, int BDFOrder, char * intergrationMethod):FixedStep_GPU(NewModel, intergrationMethod, N_neuronStateVariables, N_differentialNeuronState, N_timeDependentNeuronState),BDForder(BDFOrder){
}

BDFn_GPU::~BDFn_GPU(){
	cudaFree(AuxNeuronState);
	cudaFree(AuxNeuronState_p);
	cudaFree(AuxNeuronState_p1);
	cudaFree(AuxNeuronState_c);
	cudaFree(jacnum);
	cudaFree(J);
	cudaFree(inv_J);

	if(BDForder>1){
		cudaFree(PreviousNeuronState);
	}

	cudaFree(D);
	cudaFree(state);

	cudaFree(AuxNeuronState2);
	cudaFree(AuxNeuronState_pos);
	cudaFree(AuxNeuronState_neg);
}

__global__ void BDFn_GPU_position(void ** vector, float * integration_method_parameters_GPU, float * element1, float * element2, float * element3, float * element4, float * element5, float * element6, float * element7, float * element8, float * element9, float * element10, int * element11, float * element12, float * element13, float * element14){
	vector[0]=integration_method_parameters_GPU;
	vector[1]=element1;
	vector[2]=element2;
	vector[3]=element3;
	vector[4]=element4;
	vector[5]=element5;
	vector[6]=element6;
	vector[7]=element7;
	vector[8]=element8;
	vector[9]=element9;
	vector[10]=element10;
	vector[11]=element11;
	vector[12]=element12;
	vector[13]=element13;
	vector[14]=element14;
}
		

void BDFn_GPU::InitializeMemoryGPU(int N_neurons, int Total_N_thread){

	int size=15*sizeof(float *);

	cudaMalloc((void **)&Buffer_GPU, size);

	float integration_method_parameters_CPU[1];
	integration_method_parameters_CPU[0]=this->elapsedTimeInSeconds;
	float * integration_method_parameters_GPU;
	cudaMalloc((void**)&integration_method_parameters_GPU, 1*sizeof(float));
	cudaMemcpy(integration_method_parameters_GPU,integration_method_parameters_CPU,1*sizeof(float),cudaMemcpyHostToDevice);

	cudaMalloc((void**)&AuxNeuronState, N_NeuronStateVariables*Total_N_thread*sizeof(float));
	cudaMalloc((void**)&AuxNeuronState_p, N_NeuronStateVariables*Total_N_thread*sizeof(float));
	cudaMalloc((void**)&AuxNeuronState_p1, N_NeuronStateVariables*Total_N_thread*sizeof(float));
	cudaMalloc((void**)&AuxNeuronState_c, N_NeuronStateVariables*Total_N_thread*sizeof(float));
	cudaMalloc((void**)&jacnum, N_DifferentialNeuronState*N_DifferentialNeuronState*Total_N_thread*sizeof(float));
	cudaMalloc((void**)&J, N_DifferentialNeuronState*N_DifferentialNeuronState*Total_N_thread*sizeof(float));
	cudaMalloc((void**)&inv_J, N_DifferentialNeuronState*N_DifferentialNeuronState*Total_N_thread*sizeof(float));

	cudaMalloc((void**)&Coeficient, 7*7*sizeof(float));
	cudaMemcpy(Coeficient, Coeficient_CPU, 7*7*sizeof(float), cudaMemcpyHostToDevice);

	if(BDForder>1){
		cudaMalloc((void**)&PreviousNeuronState, (BDForder-1)*N_neurons*N_DifferentialNeuronState*sizeof(float));
	}

	cudaMalloc((void**)&D, BDForder*N_neurons*N_DifferentialNeuronState*sizeof(float));

	cudaMalloc((void**)&state, N_neurons*sizeof(int));
	cudaMemset(state,0,N_neurons*sizeof(int));

	cudaMalloc((void**)&AuxNeuronState2, N_NeuronStateVariables*Total_N_thread*sizeof(float));
	cudaMalloc((void**)&AuxNeuronState_pos, N_NeuronStateVariables*Total_N_thread*sizeof(float));
	cudaMalloc((void**)&AuxNeuronState_neg, N_NeuronStateVariables*Total_N_thread*sizeof(float));

	BDFn_GPU_position<<<1,1>>>(Buffer_GPU, integration_method_parameters_GPU, AuxNeuronState, AuxNeuronState_p, AuxNeuronState_p1, AuxNeuronState_c, jacnum, J, inv_J, Coeficient, PreviousNeuronState, D, state, AuxNeuronState2, AuxNeuronState_pos, AuxNeuronState_neg);

	cudaFree(integration_method_parameters_GPU);
}





