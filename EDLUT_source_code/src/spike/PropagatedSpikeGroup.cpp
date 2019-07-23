/***************************************************************************
 *                           PropagatedSpikeGroup.cpp                      *
 *                           -------------------                           *
 * copyright            : (C) 2015 by Francisco Naveros                    *
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

#include "../../include/spike/PropagatedSpikeGroup.h"
#include "../../include/spike/Interconnection.h"
#include "../../include/spike/Neuron.h"
#include "../../include/spike/InternalSpike.h"

#include "../../include/neuron_model/NeuronModel.h"

#include "../../include/simulation/Simulation.h"
#include "../../include/simulation/EventQueue.h"

#include "../../include/learning_rules/LearningRule.h"
#include "../../include/learning_rules/AdditiveKernelChange.h"

#include "../../include/spike/Neuron.h"

#include "../../include/openmp/openmp.h"


PropagatedSpikeGroup::PropagatedSpikeGroup(double NewTime, int NewQueueIndex) :Spike(NewTime, NewQueueIndex, NULL), N_Elements(0) {
}
   	
	
PropagatedSpikeGroup::~PropagatedSpikeGroup(){
}

int PropagatedSpikeGroup::GetN_Elementes(){
	return N_Elements;
}

int PropagatedSpikeGroup::GetMaxSize(){
	return MaxSize;
}

bool PropagatedSpikeGroup::IncludeNewSource(int NewN_connections, Interconnection * NewConnections){
	N_ConnectionsWithEqualDelay[N_Elements]=NewN_connections;
	ConnectionsWithEqualDelay[N_Elements]=NewConnections;

	N_Elements++;
	if(N_Elements<MaxSize){
		return false;
	}
	return true;
}



   	

//Optimized function. This function propagates all neuron output spikes that have the same delay.
void PropagatedSpikeGroup::ProcessEvent(Simulation * CurrentSimulation, RealTimeRestrictionLevel RealTimeRestriction){

	if (RealTimeRestriction < SPIKES_DISABLED){
		CurrentSimulation->IncrementTotalPropagateEventCounter(this->GetQueueIndex()); 
		Interconnection * inter;
		for(int i=0; i<N_Elements; i++){
			inter=ConnectionsWithEqualDelay[i];
			CurrentSimulation->IncrementTotalPropagateCounter(this->GetQueueIndex(), N_ConnectionsWithEqualDelay[i]);
			for (int j=0; j<N_ConnectionsWithEqualDelay[i]; j++){

				InternalSpike * Generated = inter->GetTargetNeuronModel()->ProcessInputSpike(inter, this->time);

				if (Generated!=0){
					CurrentSimulation->GetQueue()->InsertEvent(this->GetQueueIndex(),Generated);
				}

				if (CurrentSimulation->monitore_neurons){
					Neuron * TargetNeuron = inter->GetTarget();  // target of the spike
					CurrentSimulation->WriteState(this->time, TargetNeuron);
				}

				if (RealTimeRestriction < LEARNING_RULES_DISABLED){
					LearningRule * ConnectionRule = inter->GetWeightChange_withoutPost();
					// If learning, change weights
					if(ConnectionRule != 0){
						ConnectionRule->ApplyPreSynapticSpike(inter,this->time);
					}
					ConnectionRule = inter->GetWeightChange_withPost();
					// If learning, change weights
					if(ConnectionRule != 0){
						ConnectionRule->ApplyPreSynapticSpike(inter,this->time);
					}
				}
				inter++;
			}
		}
	}
}


//Optimized function. This function propagates all neuron output spikes that have the same delay.
void PropagatedSpikeGroup::ProcessEvent(Simulation * CurrentSimulation){
	CurrentSimulation->IncrementTotalPropagateEventCounter(this->GetQueueIndex());
	Interconnection * inter;
	for(int i=0; i<N_Elements; i++){
		inter=ConnectionsWithEqualDelay[i];
		CurrentSimulation->IncrementTotalPropagateCounter(this->GetQueueIndex(), N_ConnectionsWithEqualDelay[i]);
		for (int j=0; j<N_ConnectionsWithEqualDelay[i]; j++){

			InternalSpike * Generated = inter->GetTargetNeuronModel()->ProcessInputSpike(inter, this->time);

			if (Generated != 0){
				CurrentSimulation->GetQueue()->InsertEvent(this->GetQueueIndex(), Generated);
			}

			if (CurrentSimulation->monitore_neurons){
				Neuron * TargetNeuron = inter->GetTarget();  // target of the spike
				CurrentSimulation->WriteState(this->time, TargetNeuron);
			}

			LearningRule * ConnectionRule = inter->GetWeightChange_withoutPost();
			// If learning, change weights
			if(ConnectionRule != 0){
				ConnectionRule->ApplyPreSynapticSpike(inter,this->time);
			}
			ConnectionRule = inter->GetWeightChange_withPost();
			// If learning, change weights
			if(ConnectionRule != 0){
				ConnectionRule->ApplyPreSynapticSpike(inter,this->time);
			}
			inter++;
		}
	}
}



void PropagatedSpikeGroup::PrintType(){
	cout<<"PropagatedSpikeGroup"<<endl;
}


enum EventPriority PropagatedSpikeGroup::ProcessingPriority(){
	return PROPAGATEDSPIKE;
}

