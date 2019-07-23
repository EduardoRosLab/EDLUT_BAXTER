/***************************************************************************
 *                           InternalSpike.cpp                             *
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

#include "../../include/spike/InternalSpike.h"
#include "../../include/spike/EndRefractoryPeriodEvent.h"
#include "../../include/spike/Interconnection.h"

#include "../../include/spike/Neuron.h"
#include "../../include/spike/PropagatedSpike.h"

#include "../../include/neuron_model/VectorNeuronState.h"
#include "../../include/neuron_model/NeuronModel.h"
#include "../../include/neuron_model/EventDrivenNeuronModel.h"

#include "../../include/simulation/Simulation.h"
#include "../../include/simulation/EventQueue.h"

#include "../../include/communication/OutputSpikeDriver.h"

#include "../../include/learning_rules/LearningRule.h"

#include "../../include/spike/Neuron.h"

#include "../../include/openmp/openmp.h"

InternalSpike::InternalSpike():Spike() {
}
   	
InternalSpike::InternalSpike(double NewTime, int NewQueueIndex, Neuron * NewSource) : Spike(NewTime, NewQueueIndex, NewSource){
}
   		
InternalSpike::~InternalSpike(){
}

void InternalSpike::ProcessEvent(Simulation * CurrentSimulation, RealTimeRestrictionLevel RealTimeRestriction){
	if (RealTimeRestriction < ALL_UNESSENTIAL_EVENTS_DISABLED){
	
		Neuron * neuron=this->GetSource();  // source of the spike
		/////////REVISAR////////////
		if (neuron->GetNeuronModel()->GetModelSimulationMethod() == EVENT_DRIVEN_MODEL){
			EventDrivenNeuronModel * Model = (EventDrivenNeuronModel *) neuron->GetNeuronModel();
			if(!Model->DiscardSpike(this)){

				// Add the spike to simulation spike counter
				CurrentSimulation->IncrementTotalSpikeCounter(neuron->get_OpenMP_queue_index());
				neuron->GetVectorNeuronState()->NewFiredSpike(neuron->GetIndex_VectorNeuronState());

				// Update the neuron state and generate a new event that will check if an spike must be generated after the refractory period.
				EndRefractoryPeriodEvent * endRefractoryPeriodEvent = Model->ProcessInternalSpike(this);

				if(endRefractoryPeriodEvent!=0){
					CurrentSimulation->GetQueue()->InsertEvent(neuron->get_OpenMP_queue_index(),endRefractoryPeriodEvent);
				}
				
				CurrentSimulation->WriteSpike(this);
				CurrentSimulation->WriteState(this->GetTime(), neuron);

				// Generate the output activity
				for(int i=0; i<NumberOfOpenMPQueues; i++){
					if (neuron->IsOutputConnected(i)){
						PropagatedSpike * spike = new PropagatedSpike(this->GetTime() + neuron->PropagationStructure->SynapseDelay[i][0], i, neuron, 0, neuron->PropagationStructure->NDifferentDelays[i]);
						if(i==neuron->get_OpenMP_queue_index()){
							CurrentSimulation->GetQueue()->InsertEvent(i,spike);
						}else{
							CurrentSimulation->GetQueue()->InsertEventInBuffer(omp_get_thread_num(),i,spike);
						}
					}
				}

				if(neuron->GetInputNumberWithPostSynapticLearning()>0){
					if (RealTimeRestriction < LEARNING_RULES_DISABLED){
						neuron->GetInputConnectionWithPostSynapticLearningAt(0)->GetWeightChange_withPost()->ApplyPostSynapticSpike(neuron, this->time);
					}
					//Interconnection * inter;
					//if(RealTimeRestriction < LEARNING_RULES_DISABLED){
					//	for (int i=0; i<neuron->GetInputNumberWithPostSynapticLearning(); ++i){
					//		inter = neuron->GetInputConnectionWithPostSynapticLearningAt(i);
					//		inter->GetWeightChange_withPost()->ApplyPostSynapticSpike(inter,this->time);
					//	}
					//}


				}

			}
		} else { // Time-driven model (no check nor update needed
			// Add the spike to simulation spike counter
			CurrentSimulation->IncrementTotalSpikeCounter(neuron->get_OpenMP_queue_index()); 

			CurrentSimulation->WriteSpike(this);
			CurrentSimulation->WriteState(this->GetTime(), neuron);

			
			// Generate the output activity
			for(int i=0; i<NumberOfOpenMPQueues; i++){
				if (neuron->IsOutputConnected(i)){
					PropagatedSpike * spike = new PropagatedSpike(this->GetTime() + neuron->PropagationStructure->SynapseDelay[i][0], i, neuron, 0, neuron->PropagationStructure->NDifferentDelays[i]);
					if(i==neuron->get_OpenMP_queue_index()){
						CurrentSimulation->GetQueue()->InsertEvent(i,spike);
					}else{
						CurrentSimulation->GetQueue()->InsertEventInBuffer(omp_get_thread_num(),i,spike);
					}
				}
			}

			if(neuron->GetInputNumberWithPostSynapticLearning()>0){
				if (RealTimeRestriction < LEARNING_RULES_DISABLED){
					neuron->GetInputConnectionWithPostSynapticLearningAt(0)->GetWeightChange_withPost()->ApplyPostSynapticSpike(neuron, this->time);
				}
				//Interconnection * inter;
				//if(RealTimeRestriction < LEARNING_RULES_DISABLED){
				//	for (int i=0; i<neuron->GetInputNumberWithPostSynapticLearning(); ++i){
				//		inter = neuron->GetInputConnectionWithPostSynapticLearningAt(i);
				//		inter->GetWeightChange_withPost()->ApplyPostSynapticSpike(inter,this->time);
				//	}
				//}
			}
			
		}
	}

}


void InternalSpike::ProcessEvent(Simulation * CurrentSimulation){
	Neuron * neuron=this->GetSource();  // source of the spike
	
	if (neuron->GetNeuronModel()->GetModelSimulationMethod() == EVENT_DRIVEN_MODEL){
		EventDrivenNeuronModel * Model = (EventDrivenNeuronModel *) neuron->GetNeuronModel();
		if(!Model->DiscardSpike(this)){
			// Add the spike to simulation spike counter
			CurrentSimulation->IncrementTotalSpikeCounter(neuron->get_OpenMP_queue_index());

			neuron->GetVectorNeuronState()->NewFiredSpike(neuron->GetIndex_VectorNeuronState());

			// Update the neuron state and generate a new event that will check if an spike must be generated after the refractory period.
			EndRefractoryPeriodEvent * endRefractoryPeriodEvent = Model->ProcessInternalSpike(this);

			if(endRefractoryPeriodEvent!=0){
				CurrentSimulation->GetQueue()->InsertEvent(neuron->get_OpenMP_queue_index(),endRefractoryPeriodEvent);
			}
			
			CurrentSimulation->WriteSpike(this);
			CurrentSimulation->WriteState(this->GetTime(), neuron);


			// Generate the output activity
			for(int i=0; i<NumberOfOpenMPQueues; i++){
				if (neuron->IsOutputConnected(i)){
					PropagatedSpike * spike = new PropagatedSpike(this->GetTime() + neuron->PropagationStructure->SynapseDelay[i][0], i, neuron, 0, neuron->PropagationStructure->NDifferentDelays[i]);
					if(i==neuron->get_OpenMP_queue_index()){
						CurrentSimulation->GetQueue()->InsertEvent(i,spike);
					}else{
						CurrentSimulation->GetQueue()->InsertEventInBuffer(omp_get_thread_num(),i,spike);
					}
				}
			}

			if(neuron->GetInputNumberWithPostSynapticLearning()>0){
				neuron->GetInputConnectionWithPostSynapticLearningAt(0)->GetWeightChange_withPost()->ApplyPostSynapticSpike(neuron, this->time);

				//Interconnection * inter;
				//for (int i=0; i<neuron->GetInputNumberWithPostSynapticLearning(); ++i){
				//	inter = neuron->GetInputConnectionWithPostSynapticLearningAt(i);
				//	inter->GetWeightChange_withPost()->ApplyPostSynapticSpike(inter,this->time);
				//}
			}

		}
	} else { // Time-driven model (no check nor update needed
		// Add the spike to simulation spike counter
		CurrentSimulation->IncrementTotalSpikeCounter(neuron->get_OpenMP_queue_index()); 

		CurrentSimulation->WriteSpike(this);
		CurrentSimulation->WriteState(this->GetTime(), neuron);

		
		// Generate the output activity
		for(int i=0; i<NumberOfOpenMPQueues; i++){
			if (neuron->IsOutputConnected(i)){
				PropagatedSpike * spike = new PropagatedSpike(this->GetTime() + neuron->PropagationStructure->SynapseDelay[i][0], i, neuron, 0, neuron->PropagationStructure->NDifferentDelays[i]);
				if(i==neuron->get_OpenMP_queue_index()){
					CurrentSimulation->GetQueue()->InsertEvent(i,spike);
				}else{
					CurrentSimulation->GetQueue()->InsertEventInBuffer(omp_get_thread_num(),i,spike);
				}
			}
		}

		if(neuron->GetInputNumberWithPostSynapticLearning()>0){
			neuron->GetInputConnectionWithPostSynapticLearningAt(0)->GetWeightChange_withPost()->ApplyPostSynapticSpike(neuron, this->time);

			//Interconnection * inter;
			//for (int i=0; i<neuron->GetInputNumberWithPostSynapticLearning(); ++i){
			//	inter = neuron->GetInputConnectionWithPostSynapticLearningAt(i);
			//	inter->GetWeightChange_withPost()->ApplyPostSynapticSpike(inter,this->time);
			//}
		}
		
	}
}

void InternalSpike::PrintType(){
	cout<<"InternalSpike"<<endl;
}


enum EventPriority InternalSpike::ProcessingPriority(){
	return INTERNALSPIKE;
}
