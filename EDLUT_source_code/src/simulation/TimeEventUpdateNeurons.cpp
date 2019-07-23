/***************************************************************************
 *                           TimeEventUpdateNeurons.cpp                    *
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

#include "../../include/simulation/TimeEventUpdateNeurons.h"
#include "../../include/simulation/Simulation.h"
#include "../../include/simulation/EventQueue.h"

#include "../../include/neuron_model/NeuronModel.h"
#include "../../include/neuron_model/TimeDrivenModel.h"
#include "../../include/neuron_model/VectorNeuronState.h"

#include "../../include/spike/TimeDrivenInternalSpike.h"
#include "../../include/spike/PropagatedCurrent.h"
#include "../../include/spike/Interconnection.h"
#include "../../include/spike/Network.h"
#include "../../include/spike/Neuron.h"
#include "../../include/spike/NeuronModelPropagationDelayStructure.h"

#include "../../include/openmp/openmp.h"

TimeEventUpdateNeurons::TimeEventUpdateNeurons(double NewTime, int NewQueueIndex, TimeDrivenModel * newNeuronModel, Neuron ** newNeurons) : Event(NewTime, NewQueueIndex), neuronModel(newNeuronModel), neurons(newNeurons) {

}

TimeEventUpdateNeurons::~TimeEventUpdateNeurons(){

}



//Optimized version which executes the internal spikes instead of insert them in the queue.
void TimeEventUpdateNeurons::ProcessEvent(Simulation * CurrentSimulation, RealTimeRestrictionLevel RealTimeRestriction){
	double CurrentTime = this->GetTime();
	
	VectorNeuronState * State = neuronModel->GetVectorNeuronState();
	neuronModel->UpdateState(NULL, CurrentTime);

	if (RealTimeRestriction < SPIKES_DISABLED){
		//If this model is a spike generator
		if (neuronModel->GetModelOutputActivityType() == OUTPUT_SPIKE){
			TimeDrivenInternalSpike NewEvent(CurrentTime, this->GetQueueIndex(), State, neuronModel->PropagationStructure, neurons);
			NewEvent.ProcessEvent(CurrentSimulation, RealTimeRestriction);
		}

		//If this model is a current generator
		if (neuronModel->GetModelOutputActivityType() == OUTPUT_CURRENT){
			for (int j = 0; j < neuronModel->GetVectorNeuronState()->GetSizeState(); j++){
				float outputCurrent = this->GetModel()->GetVectorNeuronState()->GetStateVariableAt(j, 0);
				for (int i = 0; i < NumberOfOpenMPQueues; i++){
					if (neurons[j]->IsOutputConnected(i)){
						PropagatedCurrent * current = new PropagatedCurrent(this->GetTime() + neurons[j]->GetOutputConnectionAt(i, 0)->GetDelay(), i, neurons[j], 0, neurons[j]->PropagationStructure->NDifferentDelays[i], outputCurrent);
						if (i == omp_get_thread_num()){
							CurrentSimulation->GetQueue()->InsertEvent(i, current);
						}
						else{
							CurrentSimulation->GetQueue()->InsertEventInBuffer(omp_get_thread_num(), i, current);
						}
					}
				}
			}
		}
	}

	//Next TimeEvent for all cell
	CurrentSimulation->GetQueue()->InsertEvent(new TimeEventUpdateNeurons(CurrentTime + neuronModel->GetTimeDrivenStepSize(), this->GetQueueIndex(), GetModel(), GetNeurons()));
}

//Optimized version which executes the internal spikes instead of insert them in the queue.
void TimeEventUpdateNeurons::ProcessEvent(Simulation * CurrentSimulation){

	double CurrentTime = this->GetTime();

	VectorNeuronState * State=neuronModel->GetVectorNeuronState();

	neuronModel->UpdateState(NULL, CurrentTime);

	//If this model is a spike generator
	if (neuronModel->GetModelOutputActivityType() == OUTPUT_SPIKE){
		TimeDrivenInternalSpike NewEvent(CurrentTime, this->GetQueueIndex(), State, neuronModel->PropagationStructure, neurons);
		NewEvent.ProcessEvent(CurrentSimulation);
	}

	//If this model is a current generator
	if (neuronModel->GetModelOutputActivityType() == OUTPUT_CURRENT){
		for (int j = 0; j < neuronModel->GetVectorNeuronState()->GetSizeState(); j++){
			float outputCurrent = this->GetModel()->GetVectorNeuronState()->GetStateVariableAt(j, 0);
			for (int i = 0; i < NumberOfOpenMPQueues; i++){
				if (neurons[j]->IsOutputConnected(i)){
					PropagatedCurrent * current = new PropagatedCurrent(this->GetTime() + neurons[j]->GetOutputConnectionAt(i, 0)->GetDelay(), i, neurons[j], 0, neurons[j]->PropagationStructure->NDifferentDelays[i], outputCurrent);
					if (i == omp_get_thread_num()){
						CurrentSimulation->GetQueue()->InsertEvent(i, current);
					}
					else{
						CurrentSimulation->GetQueue()->InsertEventInBuffer(omp_get_thread_num(), i, current);
					}
				}
			}
		}
	}

	//Next TimeEvent for all cell
	CurrentSimulation->GetQueue()->InsertEvent(new TimeEventUpdateNeurons(CurrentTime + neuronModel->GetTimeDrivenStepSize(), this->GetQueueIndex(), GetModel(), GetNeurons()));
}

TimeDrivenModel * TimeEventUpdateNeurons::GetModel(){
	return neuronModel;
}

Neuron ** TimeEventUpdateNeurons::GetNeurons(){
	return neurons;
}


void TimeEventUpdateNeurons::PrintType(){
	cout<<"TimeEventUpdateNeurons"<<endl;
}

enum EventPriority TimeEventUpdateNeurons::ProcessingPriority(){
	return TIMEEVENT;
}
