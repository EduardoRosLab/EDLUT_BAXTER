/***************************************************************************
 *                           CommunicationEvent.cpp                        *
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

#include "../../include/simulation/CommunicationEvent.h"

#include "../../include/simulation/Simulation.h"
#include "../../include/simulation/EventQueue.h"

#include "../../include/simulation/SynchronizeSimulationEvent.h"

#include "../../include/openmp/openmp.h"

   	
CommunicationEvent::CommunicationEvent(double NewTime, Simulation * CurrentSimulation):Event(NewTime){
	for(int i=0; i<CurrentSimulation->GetNumberOfQueues(); i++){
		SynchronizeSimulationEvent * NewEvent = new SynchronizeSimulationEvent(NewTime, i);
		CurrentSimulation->GetQueue()->InsertEvent(i,NewEvent);
	}
}
   		
CommunicationEvent::~CommunicationEvent(){
}

void CommunicationEvent::ProcessEvent(Simulation * CurrentSimulation, RealTimeRestrictionLevel RealTimeRestriction){
	// Send the outputs
	CurrentSimulation->SendOutput();
	
	// Get the inputs
	CurrentSimulation->GetInput();
	
	if (CurrentSimulation->GetSimulationStep() > 0.0){
		CommunicationEvent * NewEvent = new CommunicationEvent(this->GetTime() + CurrentSimulation->GetSimulationStep(), CurrentSimulation);
		CurrentSimulation->GetQueue()->InsertEventWithSynchronization(NewEvent);
	}
}

void CommunicationEvent::ProcessEvent(Simulation * CurrentSimulation){
	// Send the outputs
	CurrentSimulation->SendOutput();
		
	// Get the inputs
	CurrentSimulation->GetInput();
	
	if (CurrentSimulation->GetSimulationStep()>0.0){
		CommunicationEvent * NewEvent = new CommunicationEvent(this->GetTime()+CurrentSimulation->GetSimulationStep(), CurrentSimulation);
		CurrentSimulation->GetQueue()->InsertEventWithSynchronization(NewEvent);
	}
}

void CommunicationEvent::PrintType(){
	cout<<"CommunicationEvent"<<endl;
}

enum EventPriority CommunicationEvent::ProcessingPriority(){
	return COMMUNICATIONEVENT;
}
   	

