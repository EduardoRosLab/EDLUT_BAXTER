/***************************************************************************
 *                           SynchronizeSimulationEvent.cpp                *
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

#include "../../include/simulation/SynchronizeSimulationEvent.h"

#include "../../include/simulation/Simulation.h"
#include "../../include/simulation/EventQueue.h"
#include "../../include/openmp/openmp.h"

  	
SynchronizeSimulationEvent::SynchronizeSimulationEvent(double NewTime, int NewQueueIndex) : Event(NewTime, NewQueueIndex){
}
   		
SynchronizeSimulationEvent::~SynchronizeSimulationEvent(){
}

void SynchronizeSimulationEvent::ProcessEvent(Simulation * CurrentSimulation, RealTimeRestrictionLevel RealTimeRestriction){
	CurrentSimulation->SetSynchronizeSimulationEvent(this->GetQueueIndex());
}

void SynchronizeSimulationEvent::ProcessEvent(Simulation * CurrentSimulation){
	CurrentSimulation->SetSynchronizeSimulationEvent(this->GetQueueIndex());
}


void SynchronizeSimulationEvent::PrintType(){
	cout<<"SynchronizeSimulationEvent"<<endl;
}
   

enum EventPriority SynchronizeSimulationEvent::ProcessingPriority(){
	return SYNCHRONIZESIMULATIONEVENT;
}


