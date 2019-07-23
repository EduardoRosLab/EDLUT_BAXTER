/***************************************************************************
 *                           InputBooleanArrayDriver.cpp                   *
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

#include "../../include/communication/InputBooleanArrayDriver.h"

#include "../../include/simulation/EventQueue.h"

#include "../../include/spike/InputSpike.h"
#include "../../include/spike/Network.h"

#include "../../include/spike/Neuron.h"

InputBooleanArrayDriver::InputBooleanArrayDriver(unsigned int InputLines, int * Associated):AssociatedCells(0),NumInputLines(InputLines){
	AssociatedCells = new int [NumInputLines];

	for (unsigned int i=0; i<this->NumInputLines; ++i){
		AssociatedCells[i] = Associated[i];
	}

	return;
}

InputBooleanArrayDriver::~InputBooleanArrayDriver() {
	delete [] AssociatedCells;
}

void InputBooleanArrayDriver::LoadInputs(EventQueue * Queue, Network * Net, bool * InputLines, double CurrentTime) throw (EDLUTFileException){
	for (unsigned int i=0; i<NumInputLines; ++i){
		if (InputLines[i]){
			InputSpike * NewSpike = new InputSpike(CurrentTime, Net->GetNeuronAt(this->AssociatedCells[i])->get_OpenMP_queue_index(), Net->GetNeuronAt(this->AssociatedCells[i]));

			Queue->InsertEvent(NewSpike->GetQueueIndex(),NewSpike);
		}
	}
}

ostream & InputBooleanArrayDriver::PrintInfo(ostream & out){

	out << "- Input Boolean Array Driver" << endl;

	return out;
}
