/***************************************************************************
 *                           SynchronousTableBasedModelEven.cpp            *
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

#include "../../include/spike/SynchronousTableBasedModelEvent.h"
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

   	
SynchronousTableBasedModelEvent::SynchronousTableBasedModelEvent(double NewTime, int NewQueueIndex, int NewMaxSize) : InternalSpike(NewTime, NewQueueIndex, NULL), MaxSize(NewMaxSize), NElements(0){
	Neurons= (Neuron**) new Neuron * [MaxSize];
}
   		
SynchronousTableBasedModelEvent::~SynchronousTableBasedModelEvent(){
	delete Neurons;
}

void SynchronousTableBasedModelEvent::ProcessEvent(Simulation * CurrentSimulation, RealTimeRestrictionLevel RealTimeRestriction){

	if (RealTimeRestriction < ALL_UNESSENTIAL_EVENTS_DISABLED){
		NeuronModel * Model = Neurons[0]->GetNeuronModel();

		for(int i=0; i<NElements; i++){
			this->SetSource(Neurons[i]);

			InternalSpike * Generated = ((EventDrivenNeuronModel *)Model)->ProcessActivityAndPredictSpike(Neurons[i],this->GetTime());
			if(Generated!=0){
				CurrentSimulation->GetQueue()->InsertEvent(Neurons[i]->get_OpenMP_queue_index(),Generated);
			}
		}
	}
}

void SynchronousTableBasedModelEvent::ProcessEvent(Simulation * CurrentSimulation){
	NeuronModel * Model = Neurons[0]->GetNeuronModel();

	for(int i=0; i<NElements; i++){
		this->SetSource(Neurons[i]);

		InternalSpike * Generated = ((EventDrivenNeuronModel *)Model)->ProcessActivityAndPredictSpike(Neurons[i], this->GetTime());
		if(Generated!=0){
			CurrentSimulation->GetQueue()->InsertEvent(Neurons[i]->get_OpenMP_queue_index(),Generated);
		}
	}
}

void SynchronousTableBasedModelEvent::IncludeNewNeuron(Neuron * neuron){
	Neurons[NElements]=neuron;
	NElements++;
}

   	
void SynchronousTableBasedModelEvent::PrintType(){
	cout<<"SynchronousTableBasedModelEvent"<<endl;
}


enum EventPriority SynchronousTableBasedModelEvent::ProcessingPriority(){
	return SYNCHRONOUSTABLEBASEDMODELEVENT;
}