/***************************************************************************
 *                           Event.cpp                                     *
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

#include "../../include/simulation/Event.h"

#include "../../include/simulation/Simulation.h"

Event::Event() :time(0), queueIndex(0){
}
  
Event::Event(double NewTime): time(NewTime), queueIndex(0){

}
   	
Event::Event(double NewTime, int NewQueueIndex): time(NewTime), queueIndex(NewQueueIndex){

}
   		
Event::~Event(){
}
   	
double Event::GetTime() const{
	return time;
}
   		


bool Event::IsSpikeOrCurrent() const{
	return false;
}


void Event::PrintType(){
	cout<<"Event"<<endl;
}


int Event::GetQueueIndex(){
	return this->queueIndex;
}