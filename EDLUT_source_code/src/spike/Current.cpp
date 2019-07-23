/***************************************************************************
 *                           Current.cpp                                   *
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

#include "../../include/spike/Current.h"

#include "../../include/spike/Neuron.h"

Current::Current():Event(0, 0), source(0), current(0){
}
   	
Current::Current(double NewTime, int NewQueueIndex, Neuron * NewSource, float NewCurrent) : Event(NewTime, NewQueueIndex), source(NewSource), current(NewCurrent){
}
   		
Current::~Current(){
}
   	
Neuron * Current::GetSource () const{
	return source;
}

 void Current::SetSource (Neuron * NewSource){
	source=NewSource;
}

float Current::GetCurrent() const{
	 return current;
 }

void Current::SetCurrent(float NewCurrent){
	 current = NewCurrent;
 }


   		
bool Current::IsSpikeOrCurrent() const{
	return true;
}

void Current::PrintType(){
	cout<<"Current"<<endl;
}



