/***************************************************************************
 *                           NeuronModelPropagationDelayStructure.cpp      *
 *                           -------------------                           *
 * copyright            : (C) 2015 by Francisco Naveros					   *
 * email                : fnaveros@ugr.es		                           *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include "../../include/spike/NeuronModelPropagationDelayStructure.h"

#include "../../include/openmp/openmp.h"

#include <cstring>
#include <iostream>

using namespace std;

NeuronModelPropagationDelayStructure::NeuronModelPropagationDelayStructure(){
	//Allocate the initial memory.
	AllocatedSize=new int[NumberOfOpenMPQueues]();
	size=new int[NumberOfOpenMPQueues]();
	SynapseDelays= (double**)new double*[NumberOfOpenMPQueues];

	//Allocate one element in the SynapseDelays vector for each target OpenMP queue.
	for(int i=0; i<NumberOfOpenMPQueues; i++){
		AllocatedSize[i]=1;
		SynapseDelays[i]=new double[AllocatedSize[i]];
	}
}

NeuronModelPropagationDelayStructure::~NeuronModelPropagationDelayStructure(){
	for(int i=0; i<NumberOfOpenMPQueues; i++){
		delete SynapseDelays[i];
	}
	delete AllocatedSize;
	delete size;
	delete SynapseDelays;
}

void NeuronModelPropagationDelayStructure::IncludeNewDelay(int queueIndex, double newDelay){
	int i=0;
	//Check if the newDelay is already stored in SynapseDelays.
	while(i<size[queueIndex] && SynapseDelays[queueIndex][i]!=newDelay){
		i++;
	}

	//The newDelay must be stored.
	if(i==size[queueIndex]){
		//Check if the allocated size in SynapseDelays for each targe OpenMP queue is enough for a new delay.
		if(size[queueIndex]==AllocatedSize[queueIndex]){
			//Doubled the allocated size.
			AllocatedSize[queueIndex]*=2;
			double * auxDelays=SynapseDelays[queueIndex];
			SynapseDelays[queueIndex]= new double[AllocatedSize[queueIndex]];
			memcpy(SynapseDelays[queueIndex],auxDelays,sizeof(double)*size[queueIndex]);
			delete auxDelays;

		}

		//Store the new delay.
		SynapseDelays[queueIndex][size[queueIndex]]=newDelay;
		size[queueIndex]++;
		//Reorder the delays after the new insertion.
		for(int j =size[queueIndex]-1; j>0; j--){
			if(SynapseDelays[queueIndex][j]<SynapseDelays[queueIndex][j-1]){
				double aux=SynapseDelays[queueIndex][j];
				SynapseDelays[queueIndex][j]=SynapseDelays[queueIndex][j-1];
				SynapseDelays[queueIndex][j-1]=aux;
			}
		}
	}
}

int NeuronModelPropagationDelayStructure::GetAllocatedSize(int queueIndex){
	return AllocatedSize[queueIndex];
}

int NeuronModelPropagationDelayStructure::GetSize(int queueIndex){
	return size[queueIndex];
}

double NeuronModelPropagationDelayStructure::GetDelayAt(int queueIndex, int index){
	return SynapseDelays[queueIndex][index];
}

double * NeuronModelPropagationDelayStructure::GetDelays(int queueIndex){
	return SynapseDelays[queueIndex];
}


