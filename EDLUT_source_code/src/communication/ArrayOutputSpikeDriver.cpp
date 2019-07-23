/***************************************************************************
 *                           ArrayOutputSpikeDriver.cpp                    *
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

#include "../../include/communication/ArrayOutputSpikeDriver.h"

#include "../../include/spike/Spike.h"
#include "../../include/spike/Neuron.h"

ArrayOutputSpikeDriver::ArrayOutputSpikeDriver() {
	// TODO Auto-generated constructor stub

}

ArrayOutputSpikeDriver::~ArrayOutputSpikeDriver() {
	// TODO Auto-generated destructor stub
}

void ArrayOutputSpikeDriver::WriteSpike(const Spike * NewSpike) throw (EDLUTException){
	OutputSpike spike(NewSpike->GetSource()->GetIndex(),NewSpike->GetTime());
	#pragma omp critical (ArrayOutputSpikeDriver)
	{
		this->OutputBuffer.push_back(spike);
	}
}

void ArrayOutputSpikeDriver::WriteState(float Time, Neuron * Source) throw (EDLUTException){
	return;
}

bool ArrayOutputSpikeDriver::IsBuffered() const{
	return true;
}

bool ArrayOutputSpikeDriver::IsWritePotentialCapable() const{
	return false;
}

void ArrayOutputSpikeDriver::FlushBuffers() throw (EDLUTException){

	return;
}

int ArrayOutputSpikeDriver::GetBufferedSpikes(double *& Times, long int *& Cells){
	int size = this->OutputBuffer.size();

	if (size>0){
		Times = (double *) new double [size];

		if (!Times){
			cerr << "Error: Not enough memory" << endl;
		}

		Cells = (long int *) new long int [size];

		if (!Cells){
			cerr << "Error: Not enough memory" << endl;
		}

		for (int i=0; i<size; ++i){
			Times[i] = this->OutputBuffer[i].Time;
			Cells[i] = this->OutputBuffer[i].Neuron;
		}

		this->OutputBuffer.clear();
	}

	return size;

}

bool ArrayOutputSpikeDriver::RemoveBufferedSpike(double & Time, long int & Cell){
	bool noempty = !this->OutputBuffer.empty();
	if (noempty){
		Time = this->OutputBuffer[0].Time;
		Cell = this->OutputBuffer[0].Neuron;
		this->OutputBuffer.erase(this->OutputBuffer.begin());
	}
	return noempty;
}


ostream & ArrayOutputSpikeDriver::PrintInfo(ostream & out){

	out << "- Array Output Spike Driver" << endl;

	return out;
}

