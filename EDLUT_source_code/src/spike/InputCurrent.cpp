/***************************************************************************
 *                           InputCurrent.cpp                              *
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

#include "../../include/spike/InputCurrent.h"

#include "../../include/spike/Neuron.h"
#include "../../include/spike/PropagatedCurrent.h"
#include "../../include/spike/Interconnection.h"

#include "../../include/simulation/EventQueue.h"
#include "../../include/simulation/Simulation.h"

#include "../../include/neuron_model/NeuronState.h"
#include "../../include/neuron_model/NeuronModel.h"

#include "../../include/openmp/openmp.h"


InputCurrent::InputCurrent():Current() {
}
   	
InputCurrent::InputCurrent(double NewTime, int NewQueueIndex, Neuron * NewSource, float NewCurrent) : Current(NewTime, NewQueueIndex, NewSource, NewCurrent){
}
   		
InputCurrent::~InputCurrent(){
}

void InputCurrent::ProcessEvent(Simulation * CurrentSimulation, RealTimeRestrictionLevel RealTimeRestriction){

	if (RealTimeRestriction < SPIKES_DISABLED){
		if (source->GetNeuronModel()->GetModelOutputActivityType() == OUTPUT_CURRENT){

			for (int i = 0; i < NumberOfOpenMPQueues; i++){
				if (source->IsOutputConnected(i)){
					PropagatedCurrent * current = new PropagatedCurrent(this->GetTime() + source->GetOutputConnectionAt(i, 0)->GetDelay(), i, source, 0, source->PropagationStructure->NDifferentDelays[i], this->GetCurrent());
					if (i == omp_get_thread_num()){
						CurrentSimulation->GetQueue()->InsertEvent(i, current);
					}
					else{
						CurrentSimulation->GetQueue()->InsertEventInBuffer(omp_get_thread_num(), i, current);
					}

				}
			}
		}
		else{
			cout << "Neuron " << source->GetIndex() << " can not generate an output current." << endl;
		}
	}
}

void InputCurrent::ProcessEvent(Simulation * CurrentSimulation){

	if (source->GetNeuronModel()->GetModelOutputActivityType() == OUTPUT_CURRENT){

		for (int i = 0; i < NumberOfOpenMPQueues; i++){
			if (source->IsOutputConnected(i)){
				PropagatedCurrent * current = new PropagatedCurrent(this->GetTime() + source->GetOutputConnectionAt(i, 0)->GetDelay(), i, source, 0, source->PropagationStructure->NDifferentDelays[i], this->GetCurrent());
				if (i == omp_get_thread_num()){
					CurrentSimulation->GetQueue()->InsertEvent(i, current);
				}
				else{
					CurrentSimulation->GetQueue()->InsertEventInBuffer(omp_get_thread_num(), i, current);
				}

			}
		}
	}
	else{
		cout << "Neuron " << source->GetIndex() << " can not generate an output current." << endl;
	}
}

void InputCurrent::PrintType(){
	cout<<"InputCurrent"<<endl;
}

enum EventPriority InputCurrent::ProcessingPriority(){
	return PROPAGATEDSPIKE;
}   	
