/***************************************************************************
 *                           ArrayInputCurrentDriver.cpp                   *
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

#include "../../include/communication/ArrayInputCurrentDriver.h"

#include "../../include/simulation/EventQueue.h"

#include "../../include/spike/InputCurrent.h"
#include "../../include/spike/Network.h"
#include "../../include/spike/Neuron.h"

ArrayInputCurrentDriver::ArrayInputCurrentDriver() {
	// TODO Auto-generated constructor stub

}

ArrayInputCurrentDriver::~ArrayInputCurrentDriver() {
	// TODO Auto-generated destructor stub
}

void ArrayInputCurrentDriver::LoadInputs(EventQueue * Queue, Network * Net) throw (EDLUTFileException){
	return;
}

void ArrayInputCurrentDriver::LoadInputs(EventQueue * Queue, Network * Net, int NumCurrents, double * Times, long int * Cells, float * Currents) throw (EDLUTFileException){
	if (NumCurrents>0){
		for (int i=0; i<NumCurrents; ++i){
			InputCurrent * NewCurrent = new InputCurrent(Times[i], Net->GetNeuronAt(Cells[i])->get_OpenMP_queue_index(), Net->GetNeuronAt(Cells[i]), Currents[i]);

			Queue->InsertEvent(NewCurrent->GetQueueIndex(),NewCurrent);
		}

	}
}

ostream & ArrayInputCurrentDriver::PrintInfo(ostream & out){

	out << "- Array Input Current Driver" << endl;

	return out;
}
