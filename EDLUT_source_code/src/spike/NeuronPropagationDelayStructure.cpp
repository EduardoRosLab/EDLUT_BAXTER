/***************************************************************************
 *                           NeuronPropagationDelayStructure.cpp           *
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

#include "../../include/spike/NeuronPropagationDelayStructure.h"
#include "../../include/spike/Neuron.h"
#include "../../include/spike/NeuronModelPropagationDelayStructure.h"
#include "../../include/spike/Interconnection.h"

#include "../../include/openmp/openmp.h"
#include <iostream>
using namespace std;

NeuronPropagationDelayStructure::NeuronPropagationDelayStructure(Neuron * neuron){
	//Allocate the initial memory for each targe OpenMP queue.
	NDifferentDelays=new int[NumberOfOpenMPQueues]();
	NSynapsesWithEqualDelay=(int **) new int *[NumberOfOpenMPQueues];
	SynapseDelay=(double **) new double *[NumberOfOpenMPQueues];
	OutputConnectionsWithEquealDealy=(Interconnection***)new Interconnection**[NumberOfOpenMPQueues];
	IndexSynapseDelay=(int **) new int *[NumberOfOpenMPQueues];

	double delay1, delay2;
	//For each target OpenMP queue
	for(int i=0; i<NumberOfOpenMPQueues; i++){
		//If the neuron has output synapses for that target OpenMP queue.
		if (neuron->GetOutputNumber(i)!=0){
			//Increment the number of different delays and store the first delay.
			NDifferentDelays[i]++;
			delay1=neuron->GetOutputConnectionAt(i,0)->GetDelay();
			//For the remaining output synapses (that have been previously ordered in function of their propagation delays).
			for(int j=1; j<neuron->GetOutputNumber(i); j++){
				//Get the propagation delay for the next synapse.
				delay2=neuron->GetOutputConnectionAt(i,j)->GetDelay();
				//If this synapse and the previous one have different delays, the number of different delays is incremented.
				if(delay1!=delay2){
					delay1=delay2;
					NDifferentDelays[i]++;
				}
			}
		}
	}

	//Once the number of synapses that implement each propagation delay has been calculated, the remaining variables are
	//calculated.
	//For each target OpenMP queue
	for(int i=0; i<NumberOfOpenMPQueues; i++){
		//If the neuron has output synapses for that target OpenMP queue.
		if(NDifferentDelays[i]!=0){
			//Allocate the initial memory for each delay.
			NSynapsesWithEqualDelay[i]=(int *) new int [NDifferentDelays[i]]();
			SynapseDelay[i]=(double *) new double [NDifferentDelays[i]]();
			OutputConnectionsWithEquealDealy[i]=(Interconnection**)new Interconnection*[NDifferentDelays[i]];
			IndexSynapseDelay[i]=(int *) new int [NDifferentDelays[i]]();
			
			//Store the first synapse.
			int index=0;
			delay1=neuron->GetOutputConnectionAt(i,0)->GetDelay();

			NSynapsesWithEqualDelay[i][index]++;
			SynapseDelay[i][index]=delay1;
			OutputConnectionsWithEquealDealy[i][index]=neuron->GetOutputConnectionAt(i,0);

			//For the remaining output synapses (that have been previously ordered in function of their propagation delays).
			for(int j=1; j<neuron->GetOutputNumber(i); j++){
				//Get the propagation delay for the next synapse.
				delay2=neuron->GetOutputConnectionAt(i,j)->GetDelay();
				//If this synapse and the previous one have different delays, this one is stored in a new position.
				if(delay1!=delay2){
					delay1=delay2;
					index++;
					NSynapsesWithEqualDelay[i][index]++;
					SynapseDelay[i][index]=delay1;
					OutputConnectionsWithEquealDealy[i][index]=neuron->GetOutputConnectionAt(i,j);
				}else{
					//Increment the number of synapses with this delay.
					NSynapsesWithEqualDelay[i][index]++;
				}
			}
		}
	}
}

NeuronPropagationDelayStructure::~NeuronPropagationDelayStructure(){
	for(int i=0; i<NumberOfOpenMPQueues; i++){
		if(NDifferentDelays[i]!=0){
			delete NSynapsesWithEqualDelay[i];
			delete SynapseDelay[i];
			delete OutputConnectionsWithEquealDealy[i];
			delete IndexSynapseDelay[i];
		}
	}
	delete NDifferentDelays;
	delete NSynapsesWithEqualDelay;
	delete SynapseDelay;
	delete OutputConnectionsWithEquealDealy;
	delete IndexSynapseDelay;
}


void NeuronPropagationDelayStructure::CalculateOutputDelayIndex(NeuronModelPropagationDelayStructure * PropagationStructure){
	//For each target OpenMP queue.
	for(int i=0; i<NumberOfOpenMPQueues; i++){
		//For each propagation delay.
		for(int j=0; j<NDifferentDelays[i]; j++){
			int z=0;
			//Calculate in which index of PropagationStructure object is stored the propagation delay.
			while(SynapseDelay[i][j]!=PropagationStructure->GetDelayAt(i,z)){
				z++;
			}
			//Store the index
			IndexSynapseDelay[i][j]=z;
		}
	}
}
