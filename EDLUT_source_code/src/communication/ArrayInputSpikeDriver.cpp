/***************************************************************************
 *                           ArrayInputSpikeDriver.cpp                     *
 *                           -------------------                           *
 * copyright            : (C) 2010 by Jesus Garrido                        *
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

#include "../../include/communication/ArrayInputSpikeDriver.h"

#include "../../include/simulation/EventQueue.h"

#include "../../include/spike/InputSpike.h"
#include "../../include/spike/Network.h"
#include "../../include/spike/Neuron.h"

ArrayInputSpikeDriver::ArrayInputSpikeDriver() {
	// TODO Auto-generated constructor stub

}

ArrayInputSpikeDriver::~ArrayInputSpikeDriver() {
	// TODO Auto-generated destructor stub
}

void ArrayInputSpikeDriver::LoadInputs(EventQueue * Queue, Network * Net) throw (EDLUTFileException){
	return;
}

void ArrayInputSpikeDriver::LoadInputs(EventQueue * Queue, Network * Net, int NumSpikes, double * Times, long int * Cells) throw (EDLUTFileException){
	if (NumSpikes>0){
		for (int i=0; i<NumSpikes; ++i){
			InputSpike * NewSpike = new InputSpike(Times[i], Net->GetNeuronAt(Cells[i])->get_OpenMP_queue_index(), Net->GetNeuronAt(Cells[i]));

			Queue->InsertEvent(NewSpike->GetQueueIndex(),NewSpike);
		}

	}
}

ostream & ArrayInputSpikeDriver::PrintInfo(ostream & out){

	out << "- Array Input Spike Driver" << endl;

	return out;
}
