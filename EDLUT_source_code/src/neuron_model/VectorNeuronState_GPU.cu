/***************************************************************************
 *                           VectorNeuronState_GPU.cpp                     *
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

#include "../../include/neuron_model/VectorNeuronState_GPU.h"
#include "../../include/cudaError.h"

#include <iostream>
//Library for CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

using namespace std;

	VectorNeuronState_GPU::VectorNeuronState_GPU(unsigned int NumVariables):VectorNeuronState(NumVariables, true, true){
	};

	VectorNeuronState_GPU::~VectorNeuronState_GPU(){
		HANDLE_ERROR(cudaFree(VectorNeuronStates_GPU));
		HANDLE_ERROR(cudaFree(LastUpdateGPU));
		HANDLE_ERROR(cudaFree(LastSpikeTimeGPU));
		HANDLE_ERROR(cudaFree(InitialStateGPU));
		
		HANDLE_ERROR(cudaFreeHost(AuxStateCPU));
		HANDLE_ERROR(cudaFreeHost(InternalSpikeCPU));

		//GPU can use MapHostMemory
		if(!prop.canMapHostMemory){
			HANDLE_ERROR(cudaFree(AuxStateGPU));
			HANDLE_ERROR(cudaFree(InternalSpikeGPU));
		}
	}


void VectorNeuronState_GPU::InitializeStatesGPU(int N_Neurons, float * initialization, int N_AuxNeuronStates, cudaDeviceProp NewProp){
	prop=NewProp;

	//Initilize State in CPU
	SetSizeState(N_Neurons);
	
	VectorNeuronStates = new float[GetNumberOfVariables()*GetSizeState()]();
	LastUpdate=new double[GetSizeState()]();
	LastSpikeTime=new double[GetSizeState()]();
	InitialState=new float [GetNumberOfVariables()];
	
	if(!TimeDriven){
		PredictedSpike=new double[GetSizeState()]();
		PredictionEnd=new double[GetSizeState()]();
	}else{
		InternalSpike=new bool[GetSizeState()]();
	}
	
	//For the GPU, we store all the variables of the same type in adjacent memory positions
	//to perform coalescent access to data.
	for(int z=0; z<GetNumberOfVariables(); z++){
		for (int j=0; j<GetSizeState(); j++){ 
			VectorNeuronStates[z*GetSizeState() + j]=initialization[z];
		}
	}
	
	for (int j=0; j<GetNumberOfVariables(); j++){ 
		InitialState[j]=initialization[j];
	}

	for(int z=0; z<GetSizeState(); z++){
		LastSpikeTime[z]=100.0;
	}
	

	
	//Memory in CPU uses as buffer.
	//cudaHostAlloc allocate CPU memory with specific properties.
	HANDLE_ERROR(cudaHostAlloc((void**)&AuxStateCPU, N_AuxNeuronStates*GetSizeState()*sizeof(float),cudaHostAllocMapped));
	memset(AuxStateCPU,0,N_AuxNeuronStates*GetSizeState()*sizeof(float));
	HANDLE_ERROR(cudaHostAlloc((void**)&InternalSpikeCPU, GetSizeState()*sizeof(bool),cudaHostAllocMapped));


	//GPU can use MapHostMemory
	if(prop.canMapHostMemory){
		HANDLE_ERROR ( cudaHostGetDevicePointer( (void**)&AuxStateGPU,AuxStateCPU, 0 ) );
		HANDLE_ERROR ( cudaHostGetDevicePointer( (void**)&InternalSpikeGPU,InternalSpikeCPU, 0 ) );
	}
	//GPU can not use MapHostMemory.
	else{
		HANDLE_ERROR(cudaMalloc((void**)&AuxStateGPU, N_AuxNeuronStates*GetSizeState()*sizeof(float)));
		HANDLE_ERROR(cudaMalloc((void**)&InternalSpikeGPU, GetSizeState()*sizeof(bool)));
	}
	HANDLE_ERROR(cudaMalloc((void**)&VectorNeuronStates_GPU, GetNumberOfVariables()*GetSizeState()*sizeof(float)));
	HANDLE_ERROR(cudaMalloc((void**)&LastUpdateGPU, GetSizeState()*sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void**)&LastSpikeTimeGPU, GetSizeState()*sizeof(double)));
	HANDLE_ERROR(cudaMalloc((void**)&InitialStateGPU, GetNumberOfVariables()*sizeof(float)));

	
	//Copy initial state from CPU to GPU
	HANDLE_ERROR(cudaMemcpy(VectorNeuronStates_GPU,VectorNeuronStates,GetNumberOfVariables()*GetSizeState()*sizeof(float),cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(LastUpdateGPU,LastUpdate,GetSizeState()*sizeof(double),cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(LastSpikeTimeGPU,LastSpikeTime,GetSizeState()*sizeof(double),cudaMemcpyHostToDevice));
	HANDLE_ERROR(cudaMemcpy(InitialStateGPU,InitialState,GetNumberOfVariables()*sizeof(float),cudaMemcpyHostToDevice));


}


bool * VectorNeuronState_GPU::getInternalSpike(){
	return InternalSpikeCPU;
}

