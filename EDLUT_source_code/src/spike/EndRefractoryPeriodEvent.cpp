/***************************************************************************
 *                           EndRefractoryPeriodEvent.cpp                  *
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

#include "../../include/spike/EndRefractoryPeriodEvent.h"
#include "../../include/spike/InternalSpike.h"
#include "../../include/spike/Neuron.h"

#include "../../include/neuron_model/VectorNeuronState.h"
#include "../../include/neuron_model/NeuronModel.h"
#include "../../include/neuron_model/EventDrivenNeuronModel.h"

#include "../../include/simulation/Simulation.h"
#include "../../include/simulation/EventQueue.h"

#include "../../include/openmp/openmp.h"

 	
EndRefractoryPeriodEvent::EndRefractoryPeriodEvent(double NewTime, int NewQueueIndex, Neuron * NewSource) : Spike(NewTime, NewQueueIndex, NewSource){
}
   		
EndRefractoryPeriodEvent::~EndRefractoryPeriodEvent(){
}

void EndRefractoryPeriodEvent::ProcessEvent(Simulation * CurrentSimulation, RealTimeRestrictionLevel RealTimeRestriction){
	//Check if this event must be executed in function of the real time restriction level for real time simulations.
	if (RealTimeRestriction < SPIKES_DISABLED){
		Neuron * neuron=this->GetSource();  // source of the spike
		EventDrivenNeuronModel * Model = (EventDrivenNeuronModel *) neuron->GetNeuronModel();
		InternalSpike * spike = Model->GenerateNextSpike(this->GetTime(),neuron);
		if(spike!=0){
			CurrentSimulation->GetQueue()->InsertEvent(neuron->get_OpenMP_queue_index(),spike);
		}
	}
}


void EndRefractoryPeriodEvent::ProcessEvent(Simulation * CurrentSimulation){
	Neuron * neuron=this->GetSource();  // source of the spike
	EventDrivenNeuronModel * Model = (EventDrivenNeuronModel *) neuron->GetNeuronModel();
	InternalSpike * spike = Model->GenerateNextSpike(this->GetTime(),neuron);
	if(spike!=0){
		CurrentSimulation->GetQueue()->InsertEvent(neuron->get_OpenMP_queue_index(),spike);
	}
}

void EndRefractoryPeriodEvent::PrintType(){
	cout<<"EndRefractoryPeriod"<<endl;
}


enum EventPriority EndRefractoryPeriodEvent::ProcessingPriority(){
	return INTERNALSPIKE;
}
