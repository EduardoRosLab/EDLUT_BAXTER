/***************************************************************************
 *                           StopSimulationEvent.cpp                        *
 *                           -------------------                           *
 * copyright            : (C) 2009 by Jesus Garrido                        *
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

#include "../../include/simulation/StopSimulationEvent.h"

#include "../../include/simulation/Simulation.h"
#include "../../include/openmp/openmp.h"

StopSimulationEvent::StopSimulationEvent():Event(0, 0){
}
   	
StopSimulationEvent::StopSimulationEvent(double NewTime, int NewQueueIndex) : Event(NewTime, NewQueueIndex){
}
   		
StopSimulationEvent::~StopSimulationEvent(){
}

void StopSimulationEvent::ProcessEvent(Simulation * CurrentSimulation, RealTimeRestrictionLevel RealTimeRestriction){
	CurrentSimulation->StopSimulation(this->GetQueueIndex());
}

void StopSimulationEvent::ProcessEvent(Simulation * CurrentSimulation){
	CurrentSimulation->StopSimulation(this->GetQueueIndex());		
}

void StopSimulationEvent::PrintType(){
	cout<<"StopSimulationEvent"<<endl;
}
   	
enum EventPriority StopSimulationEvent::ProcessingPriority(){
	return STOPSIMULATIONEVENT;
}


