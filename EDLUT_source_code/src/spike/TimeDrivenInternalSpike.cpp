/***************************************************************************
 *                           TimeDrivenInternalSpike.cpp                   *
 *                           -------------------                           *
 * copyright            : (C) 2014 by Francisco Naveros                    *
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

#include "../../include/spike/TimeDrivenInternalSpike.h"
#include "../../include/spike/Interconnection.h"

#include "../../include/spike/Neuron.h"
#include "../../include/spike/PropagatedSpike.h"
#include "../../include/spike/PropagatedSpikeGroup.h"
#include "../../include/spike/NeuronModelPropagationDelayStructure.h"

#include "../../include/neuron_model/VectorNeuronState.h"
#include "../../include/neuron_model/NeuronModel.h"
#include "../../include/neuron_model/EventDrivenNeuronModel.h"

#include "../../include/simulation/Simulation.h"
#include "../../include/simulation/EventQueue.h"

#include "../../include/communication/OutputSpikeDriver.h"

#include "../../include/learning_rules/LearningRule.h"

#include "../../include/spike/Neuron.h"

#include "../../include/openmp/openmp.h"

TimeDrivenInternalSpike::TimeDrivenInternalSpike(double NewTime, int NewQueueIndex, VectorNeuronState * NewState, NeuronModelPropagationDelayStructure * NewPropagationStructure, Neuron ** NewNeurons) : InternalSpike(NewTime, NewQueueIndex, NULL), State(NewState), PropagationStructure(NewPropagationStructure), Neurons(NewNeurons){
	propagatedSpikeGroup=(PropagatedSpikeGroup***) new PropagatedSpikeGroup ** [NumberOfOpenMPQueues];
	for(int i=0; i<NumberOfOpenMPQueues; i++){
		propagatedSpikeGroup[i]=(PropagatedSpikeGroup**)new PropagatedSpikeGroup**[PropagationStructure->GetSize(i)];
		for(int j=0; j<PropagationStructure->GetSize(i); j++){
			propagatedSpikeGroup[i][j]=new PropagatedSpikeGroup(NewTime+PropagationStructure->GetDelayAt(i,j),i);
		}
	}
}
   		
TimeDrivenInternalSpike::~TimeDrivenInternalSpike(){
	for(int i=0; i<NumberOfOpenMPQueues; i++){
		delete propagatedSpikeGroup[i];
	}
	delete propagatedSpikeGroup;
}

void TimeDrivenInternalSpike::ProcessInternalSpikeEvent(Simulation * CurrentSimulation, int index){
	Neuron * neuron = Neurons[index];
	this->SetSource(neuron);
	CurrentSimulation->IncrementTotalSpikeCounter( neuron->get_OpenMP_queue_index()); 

	CurrentSimulation->WriteSpike(this);
	CurrentSimulation->WriteState(this->GetTime(), neuron);


	// Generate the output activity
	for (int i = 0; i<NumberOfOpenMPQueues; i++){
		for (int j=0; j<neuron->PropagationStructure->NDifferentDelays[i]; j++){
			int index=neuron->PropagationStructure->IndexSynapseDelay[i][j];
			//Include the new source and checks if the PropagatedSpikeGroup event is full
			if(propagatedSpikeGroup[i][index]->IncludeNewSource(neuron->PropagationStructure->NSynapsesWithEqualDelay[i][j],neuron->PropagationStructure->OutputConnectionsWithEquealDealy[i][j])){
				if (i == this->GetQueueIndex()){
					CurrentSimulation->GetQueue()->InsertEvent(i, propagatedSpikeGroup[i][index]);
				}
				else{
					CurrentSimulation->GetQueue()->InsertEventInBuffer(this->GetQueueIndex(), i, propagatedSpikeGroup[i][index]);
				}

				//Create a new PropagatedSpikeGroup event
				propagatedSpikeGroup[i][index]=new PropagatedSpikeGroup(propagatedSpikeGroup[i][index]->GetTime(),i);
			}
		}
	}

	if (neuron->GetInputNumberWithPostSynapticLearning()>0){
		neuron->GetInputConnectionWithPostSynapticLearningAt(0)->GetWeightChange_withPost()->ApplyPostSynapticSpike(neuron, this->time);
		//for (int i = 0; i < neuron->GetInputNumberWithPostSynapticLearning(); ++i){
		//	Interconnection * inter = neuron->GetInputConnectionWithPostSynapticLearningAt(i);
		//	inter->GetWeightChange_withPost()->ApplyPostSynapticSpike(inter, this->time);
		//}
	}
}

void TimeDrivenInternalSpike::ProcessEvent(Simulation * CurrentSimulation, RealTimeRestrictionLevel RealTimeRestriction){

	if (RealTimeRestriction < SPIKES_DISABLED){

		if (!State->Is_GPU){
			int * generateInternalSpikeIndexs = State->getInternalSpikeIndexs();
			int generateNInternalSpikeIndexs = State->getNInternalSpikeIndexs();

			Neuron * Cell;


			for (int t = 0; t < generateNInternalSpikeIndexs; t++){
				Cell = Neurons[generateInternalSpikeIndexs[t]];
				ProcessInternalSpikeEvent(CurrentSimulation, generateInternalSpikeIndexs[t]);
			}


			//We check if some neuron inside the model is monitored
			if (State->Get_Is_Monitored()){
				for (int t = 0; t < State->GetSizeState(); t++){
					Cell = Neurons[t];
					if (Cell->IsMonitored()){
						CurrentSimulation->WriteState(this->GetTime(), Cell);
					}
				}
			}
		}
		else{
			bool * generateInternalSpike = State->getInternalSpike();

			Neuron * Cell;

			//We check if some neuron inside the model is monitored
			if (State->Get_Is_Monitored()){
				for (int t = 0; t<State->GetSizeState(); t++){
					Cell = Neurons[t];
					if (generateInternalSpike[t] == true){
						generateInternalSpike[t] = false;
						ProcessInternalSpikeEvent(CurrentSimulation, t);
					}
					if (Cell->IsMonitored()){
						CurrentSimulation->WriteState(this->GetTime(), Cell);
					}
				}
			}
			else{
				for (int t = 0; t<State->GetSizeState(); t++){
					if (generateInternalSpike[t] == true){
						generateInternalSpike[t] = false;
						Cell = Neurons[t];
						ProcessInternalSpikeEvent(CurrentSimulation, t);
					}
				}

			}
		}


		for (int i = 0; i<NumberOfOpenMPQueues; i++){
			for (int j = 0; j<this->PropagationStructure->GetSize(i); j++){
				//Check if the each PropagatedSpikeGroup event is empty.
				if (propagatedSpikeGroup[i][j]->GetN_Elementes()>0){
					if (i == this->GetQueueIndex()){
						CurrentSimulation->GetQueue()->InsertEvent(i, propagatedSpikeGroup[i][j]);
					}
					else{
						CurrentSimulation->GetQueue()->InsertEventInBuffer(this->GetQueueIndex(), i, propagatedSpikeGroup[i][j]);
					}
				}
				else{
					delete propagatedSpikeGroup[i][j];
				}
			}
		}
	}
}

//void TimeDrivenInternalSpike::ProcessEvent(Simulation * CurrentSimulation){
//
//	//CPU write in this array if an internal spike must be generated.
//	bool * generateInternalSpike = State->getInternalSpike();
//
//	Neuron * Cell;
//
//
//
//	//We check if some neuron inside the model is monitored
//	if (State->Get_Is_Monitored()){
//		for (int t = 0; t<State->GetSizeState(); t++){
//			Cell = Neurons[t];
//			if (generateInternalSpike[t] == true){
//generateInternalSpike[t] = false;
//				ProcessInternalSpikeEvent(CurrentSimulation, t);
//			}
//			if (Cell->IsMonitored()){
//				CurrentSimulation->WriteState(this->GetTime(), Cell);
//			}
//		}
//	}
//	else{
//		for (int t = 0; t<State->GetSizeState(); t++){
//			if (generateInternalSpike[t] == true){
//generateInternalSpike[t] = false;
//				Cell = Neurons[t];
//				ProcessInternalSpikeEvent(CurrentSimulation, t);
//			}
//		}
//
//	}
//
//	for (int i = 0; i<NumberOfOpenMPQueues; i++){
//		for (int j=0; j<this->PropagationStructure->GetSize(i); j++){
//			//Check if the each PropagatedSpikeGroup event is empty.
//			if(propagatedSpikeGroup[i][j]->GetN_Elementes()>0){
//				if (i == this->GetQueueIndex()){
//					CurrentSimulation->GetQueue()->InsertEvent(i, propagatedSpikeGroup[i][j]);
//				}
//				else{
//					CurrentSimulation->GetQueue()->InsertEventInBuffer(this->GetQueueIndex(), i, propagatedSpikeGroup[i][j]);
//				}
//			}else{
//				delete propagatedSpikeGroup[i][j];
//			}	
//		}
//	}
//}


void TimeDrivenInternalSpike::ProcessEvent(Simulation * CurrentSimulation){
	if (!State->Is_GPU){
		int * generateInternalSpikeIndexs = State->getInternalSpikeIndexs();
		int generateNInternalSpikeIndexs = State->getNInternalSpikeIndexs();

		Neuron * Cell;


		for (int t = 0; t < generateNInternalSpikeIndexs; t++){
			Cell = Neurons[generateInternalSpikeIndexs[t]];
			ProcessInternalSpikeEvent(CurrentSimulation, generateInternalSpikeIndexs[t]);
		}


		//We check if some neuron inside the model is monitored
		if (State->Get_Is_Monitored()){
			for (int t = 0; t < State->GetSizeState(); t++){
				Cell = Neurons[t];
				if (Cell->IsMonitored()){
					CurrentSimulation->WriteState(this->GetTime(), Cell);
				}
			}
		}
	}
	else{
		bool * generateInternalSpike = State->getInternalSpike();
		
		Neuron * Cell;
				
		//We check if some neuron inside the model is monitored
		if (State->Get_Is_Monitored()){
			for (int t = 0; t<State->GetSizeState(); t++){
				Cell = Neurons[t];
				if (generateInternalSpike[t] == true){
					generateInternalSpike[t] = false;
					ProcessInternalSpikeEvent(CurrentSimulation, t);
				}
				if (Cell->IsMonitored()){
					CurrentSimulation->WriteState(this->GetTime(), Cell);
				}
			}
		}
		else{
			for (int t = 0; t<State->GetSizeState(); t++){
				if (generateInternalSpike[t] == true){
					generateInternalSpike[t] = false;
					Cell = Neurons[t];
					ProcessInternalSpikeEvent(CurrentSimulation, t);
				}
			}
		
		}
	}


	for (int i = 0; i<NumberOfOpenMPQueues; i++){
		for (int j = 0; j<this->PropagationStructure->GetSize(i); j++){
			//Check if the each PropagatedSpikeGroup event is empty.
			if (propagatedSpikeGroup[i][j]->GetN_Elementes()>0){
				if (i == this->GetQueueIndex()){
					CurrentSimulation->GetQueue()->InsertEvent(i, propagatedSpikeGroup[i][j]);
				}
				else{
					CurrentSimulation->GetQueue()->InsertEventInBuffer(this->GetQueueIndex(), i, propagatedSpikeGroup[i][j]);
				}
			}
			else{
				delete propagatedSpikeGroup[i][j];
			}
		}
	}
}



   	
void TimeDrivenInternalSpike::PrintType(){
	cout<<"TimeDrivenInternalSpike"<<endl;
}

enum EventPriority TimeDrivenInternalSpike::ProcessingPriority(){
	return INTERNALSPIKE;
}

