/***************************************************************************
 *                           Spike.cpp                                     *
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

#include "../../include/spike/Spike.h"

#include "../../include/spike/Neuron.h"

Spike::Spike():Event(0, 0), source(0){
}
   	
Spike::Spike(double NewTime, int NewQueueIndex, Neuron * NewSource) : Event(NewTime, NewQueueIndex), source(NewSource){
}
   		
Spike::~Spike(){
}
   	
Neuron * Spike::GetSource () const{
	return source;
}

 void Spike::SetSource (Neuron * NewSource){
	source=NewSource;
}
   		
bool Spike::IsSpikeOrCurrent() const{
	return true;
}

void Spike::PrintType(){
	cout<<"Spike"<<endl;
}



