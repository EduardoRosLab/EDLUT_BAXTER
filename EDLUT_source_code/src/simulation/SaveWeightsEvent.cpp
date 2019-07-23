/***************************************************************************
 *                           SaveWeightsEvent.cpp                          *
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

#include "../../include/simulation/SaveWeightsEvent.h"

#include "../../include/simulation/Simulation.h"
#include "../../include/simulation/EventQueue.h"

#include "../../include/simulation/SynchronizeSimulationEvent.h"

#include "../../include/openmp/openmp.h"

   	
SaveWeightsEvent::SaveWeightsEvent(double NewTime, Simulation * CurrentSimulation) : Event(NewTime){
	for(int i=0; i<CurrentSimulation->GetNumberOfQueues(); i++){
		SynchronizeSimulationEvent * NewEvent = new SynchronizeSimulationEvent(NewTime, i);
		CurrentSimulation->GetQueue()->InsertEvent(i,NewEvent);
	}
}
   		
SaveWeightsEvent::~SaveWeightsEvent(){
}

void SaveWeightsEvent::ProcessEvent(Simulation * CurrentSimulation, RealTimeRestrictionLevel RealTimeRestriction){
	CurrentSimulation->SaveWeights();
	
	if (CurrentSimulation->GetSaveStep() > 0.0){
		SaveWeightsEvent * NewEvent = new SaveWeightsEvent(this->GetTime() + CurrentSimulation->GetSaveStep(), CurrentSimulation);
		CurrentSimulation->GetQueue()->InsertEventWithSynchronization(NewEvent);
	}
}
   
void SaveWeightsEvent::ProcessEvent(Simulation * CurrentSimulation){
	CurrentSimulation->SaveWeights();

	if (CurrentSimulation->GetSaveStep()>0.0){
		SaveWeightsEvent * NewEvent = new SaveWeightsEvent(this->GetTime()+CurrentSimulation->GetSaveStep(), CurrentSimulation);
		CurrentSimulation->GetQueue()->InsertEventWithSynchronization(NewEvent);
	}
}

void SaveWeightsEvent::PrintType(){
	cout<<"SaveWeightsEvent"<<endl;
}

enum EventPriority SaveWeightsEvent::ProcessingPriority(){
	return SAVEWEIGHTEVENT;
}

