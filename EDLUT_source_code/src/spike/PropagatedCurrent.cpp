/***************************************************************************
 *                           PropagatedCurrent.cpp                         *
 *                           -------------------                           *
 * copyright            : (C) 2018 by Francisco Naveros                    *
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

#include "../../include/spike/PropagatedCurrent.h"
#include "../../include/spike/Interconnection.h"
#include "../../include/spike/Neuron.h"
//#include "../../include/spike/InternalSpike.h"

#include "../../include/neuron_model/NeuronModel.h"
//#include "../../include/neuron_model/TimeDrivenInputCurrentNeuronModel.h"

#include "../../include/simulation/Simulation.h"
#include "../../include/simulation/EventQueue.h"

#include "../../include/learning_rules/LearningRule.h"
#include "../../include/learning_rules/AdditiveKernelChange.h"

#include "../../include/spike/Neuron.h"

#include "../../include/openmp/openmp.h"


PropagatedCurrent::PropagatedCurrent():Current() {
}

   	
PropagatedCurrent::PropagatedCurrent(double NewTime, int NewQueueIndex, Neuron * NewSource, int NewPropagationDelayIndex, int NewUpperPropagationDelayIndex, float NewCurrent) : Current(NewTime, NewQueueIndex, NewSource, NewCurrent), propagationDelayIndex(NewPropagationDelayIndex), UpperPropagationDelayIndex(NewUpperPropagationDelayIndex){
	inter = NewSource->PropagationStructure->OutputConnectionsWithEquealDealy[NewQueueIndex][NewPropagationDelayIndex];
	NSynapses = NewSource->PropagationStructure->NSynapsesWithEqualDelay[NewQueueIndex][NewPropagationDelayIndex];
}

PropagatedCurrent::PropagatedCurrent(double NewTime, int NewQueueIndex, Neuron * NewSource, int NewPropagationDelayIndex, int NewUpperPropagationDelayIndex, float NewCurrent, Interconnection * NewInter) : Current(NewTime, NewQueueIndex, NewSource, NewCurrent), propagationDelayIndex(NewPropagationDelayIndex), UpperPropagationDelayIndex(NewUpperPropagationDelayIndex), inter(NewInter){
	NSynapses = NewSource->PropagationStructure->NSynapsesWithEqualDelay[NewQueueIndex][NewPropagationDelayIndex];
}
   		
PropagatedCurrent::~PropagatedCurrent(){
}

int PropagatedCurrent::GetPropagationDelayIndex (){
	return this->propagationDelayIndex;
}
 
int PropagatedCurrent::GetUpperPropagationDelayIndex (){
	return this->UpperPropagationDelayIndex;
}

   	



//Optimized function. This function propagates all neuron output spikes that have the same delay.
void PropagatedCurrent::ProcessEvent(Simulation * CurrentSimulation, RealTimeRestrictionLevel RealTimeRestriction){

	if (RealTimeRestriction < SPIKES_DISABLED){

		CurrentSimulation->IncrementTotalPropagateEventCounter(this->GetQueueIndex());
		CurrentSimulation->IncrementTotalPropagateCounter(this->GetQueueIndex(), NSynapses);

		Neuron * TargetNeuron;
		InternalSpike * Generated;
		LearningRule * ConnectionRule;


		for(int i=0; i<NSynapses; i++){
			TargetNeuron = inter->GetTarget();  // target of the spike

			TargetNeuron->GetNeuronModel()->ProcessInputCurrent(inter, TargetNeuron, this->GetCurrent());

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
			PropagatedCurrent * spike = new PropagatedCurrent(this->GetTime() - this->GetSource()->PropagationStructure->SynapseDelay[this->GetQueueIndex()][propagationDelayIndex] + this->GetSource()->PropagationStructure->SynapseDelay[this->GetQueueIndex()][propagationDelayIndex + 1], this->GetQueueIndex(), this->GetSource(), propagationDelayIndex + 1, UpperPropagationDelayIndex, this->GetCurrent());
			CurrentSimulation->GetQueue()->InsertEvent(this->GetQueueIndex(),spike);
		} 

	}
}




//Optimized function. This function propagates all neuron output spikes that have the same delay.
void PropagatedCurrent::ProcessEvent(Simulation * CurrentSimulation){
	CurrentSimulation->IncrementTotalPropagateEventCounter(this->GetQueueIndex()); 
	CurrentSimulation->IncrementTotalPropagateCounter(this->GetQueueIndex(), NSynapses);

	for(int i=0; i<NSynapses; i++){
		Neuron * TargetNeuron = inter->GetTarget();  // target of the spike

		TargetNeuron->GetNeuronModel()->ProcessInputCurrent(inter, TargetNeuron, this->GetCurrent());


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
		PropagatedCurrent * spike = new PropagatedCurrent(this->GetTime() - this->GetSource()->PropagationStructure->SynapseDelay[this->GetQueueIndex()][propagationDelayIndex] + this->GetSource()->PropagationStructure->SynapseDelay[this->GetQueueIndex()][propagationDelayIndex + 1], this->GetQueueIndex(), this->GetSource(), propagationDelayIndex + 1, UpperPropagationDelayIndex, this->GetCurrent());
		CurrentSimulation->GetQueue()->InsertEvent(this->GetQueueIndex(), spike);
	} 
}


void PropagatedCurrent::PrintType(){
	cout<<"PropagatedCurrent"<<endl;
}


enum EventPriority PropagatedCurrent::ProcessingPriority(){
	return PROPAGATEDCURRENT;
}
