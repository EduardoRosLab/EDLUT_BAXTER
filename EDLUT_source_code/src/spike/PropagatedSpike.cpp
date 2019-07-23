/***************************************************************************
 *                           PropagatedSpike.cpp                           *
 *                           -------------------                           *
 * copyright            : (C) 2009 by Jesus Garrido and Richard Carrillo   *
 * email                : jgarrido@atc.ugr.es                              *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include "../../include/spike/PropagatedSpike.h"
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


PropagatedSpike::PropagatedSpike():Spike() {
}
   	
PropagatedSpike::PropagatedSpike(double NewTime, int NewQueueIndex, Neuron * NewSource, int NewPropagationDelayIndex, int NewUpperPropagationDelayIndex) : Spike(NewTime, NewQueueIndex, NewSource), propagationDelayIndex(NewPropagationDelayIndex), UpperPropagationDelayIndex(NewUpperPropagationDelayIndex){
	inter=NewSource->PropagationStructure->OutputConnectionsWithEquealDealy[this->GetQueueIndex()][NewPropagationDelayIndex];
	NSynapses=NewSource->PropagationStructure->NSynapsesWithEqualDelay[this->GetQueueIndex()][NewPropagationDelayIndex];
}

PropagatedSpike::PropagatedSpike(double NewTime, int NewQueueIndex, Neuron * NewSource, int NewPropagationDelayIndex, int NewUpperPropagationDelayIndex, Interconnection * NewInter) : Spike(NewTime, NewQueueIndex, NewSource), propagationDelayIndex(NewPropagationDelayIndex), UpperPropagationDelayIndex(NewUpperPropagationDelayIndex), inter(NewInter){
	NSynapses=NewSource->PropagationStructure->NSynapsesWithEqualDelay[this->GetQueueIndex()][NewPropagationDelayIndex];
}
   		
PropagatedSpike::~PropagatedSpike(){
}

int PropagatedSpike::GetPropagationDelayIndex (){
	return this->propagationDelayIndex;
}
 
int PropagatedSpike::GetUpperPropagationDelayIndex (){
	return this->UpperPropagationDelayIndex;
}

   	



//Optimized function. This function propagates all neuron output spikes that have the same delay.
void PropagatedSpike::ProcessEvent(Simulation * CurrentSimulation, RealTimeRestrictionLevel RealTimeRestriction){

	if (RealTimeRestriction < SPIKES_DISABLED){

		CurrentSimulation->IncrementTotalPropagateEventCounter(this->GetQueueIndex()); 
		CurrentSimulation->IncrementTotalPropagateCounter(this->GetQueueIndex(),NSynapses);

		Neuron * TargetNeuron;
		InternalSpike * Generated;
		LearningRule * ConnectionRule;


		for(int i=0; i<NSynapses; i++){

			InternalSpike * Generated = inter->GetTargetNeuronModel()->ProcessInputSpike(inter, this->time);

			if (Generated != 0){
				CurrentSimulation->GetQueue()->InsertEvent(this->GetQueueIndex(), Generated);
			}

			if (CurrentSimulation->monitore_neurons){
				Neuron * TargetNeuron = inter->GetTarget();  // target of the spike
				CurrentSimulation->WriteState(this->time, TargetNeuron);
			}

			if (RealTimeRestriction < LEARNING_RULES_DISABLED){
				ConnectionRule = inter->GetWeightChange_withoutPost();
				// If learning, change weights
				if (ConnectionRule != 0){
					ConnectionRule->ApplyPreSynapticSpike(inter, this->time);
				}
				ConnectionRule = inter->GetWeightChange_withPost();
				// If learning, change weights
				if (ConnectionRule != 0){
					ConnectionRule->ApplyPreSynapticSpike(inter, this->time);
				}
			}

			inter++;
		}
		if(UpperPropagationDelayIndex>(propagationDelayIndex+1)){
			PropagatedSpike * spike = new PropagatedSpike(this->GetTime() - this->GetSource()->PropagationStructure->SynapseDelay[this->GetQueueIndex()][propagationDelayIndex] + this->GetSource()->PropagationStructure->SynapseDelay[this->GetQueueIndex()][propagationDelayIndex + 1], this->GetQueueIndex(), this->GetSource(), propagationDelayIndex + 1, UpperPropagationDelayIndex);
			CurrentSimulation->GetQueue()->InsertEvent(this->GetQueueIndex(),spike);
		} 

	}
}




//Optimized function. This function propagates all neuron output spikes that have the same delay.
void PropagatedSpike::ProcessEvent(Simulation * CurrentSimulation){
	CurrentSimulation->IncrementTotalPropagateEventCounter(this->GetQueueIndex()); 
	CurrentSimulation->IncrementTotalPropagateCounter(this->GetQueueIndex(),NSynapses);

	for(int i=0; i<NSynapses; i++){

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
	if(UpperPropagationDelayIndex>(propagationDelayIndex+1)){
		PropagatedSpike * spike = new PropagatedSpike(this->GetTime() - this->GetSource()->PropagationStructure->SynapseDelay[this->GetQueueIndex()][propagationDelayIndex] + this->GetSource()->PropagationStructure->SynapseDelay[this->GetQueueIndex()][propagationDelayIndex + 1], this->GetQueueIndex(), this->GetSource(), propagationDelayIndex + 1, UpperPropagationDelayIndex);
		CurrentSimulation->GetQueue()->InsertEvent(this->GetQueueIndex(),spike);
	} 
}


void PropagatedSpike::PrintType(){
	cout<<"PropagatedSpike"<<endl;
}


enum EventPriority PropagatedSpike::ProcessingPriority(){
	return PROPAGATEDSPIKE;
}
