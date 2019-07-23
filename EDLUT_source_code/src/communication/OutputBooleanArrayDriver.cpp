/***************************************************************************
 *                           OutputBooleanArrayDriver.cpp                  *
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

#include "../../include/communication/OutputBooleanArrayDriver.h"

#include "../../include/spike/InputSpike.h"
#include "../../include/spike/Network.h"

OutputBooleanArrayDriver::OutputBooleanArrayDriver(unsigned int OutputLines, int * Associated):AssociatedCells(0),NumOutputLines(OutputLines){
	AssociatedCells = new int [this->NumOutputLines];

	for (unsigned int i=0; i<this->NumOutputLines; ++i){
		AssociatedCells[i] = Associated[i];
	}

	return;
}

OutputBooleanArrayDriver::~OutputBooleanArrayDriver() {
	delete [] AssociatedCells;
}

void OutputBooleanArrayDriver::GetBufferedSpikes(bool * OutputLines){
	memset(OutputLines,0,this->NumOutputLines*sizeof(bool));

	int size = this->OutputBuffer.size();

	if (size>0){

		for (int i=0; i<size; ++i){
			int CellNumber = this->OutputBuffer[i].Neuron;

			for (unsigned int j=0; j<this->NumOutputLines; ++j){
				if (CellNumber==this->AssociatedCells[j]){
					OutputLines[j] = true;
				}
			}
		}

		this->OutputBuffer.clear();
	}

	return;
}

ostream & OutputBooleanArrayDriver::PrintInfo(ostream & out){

	out << "- Output Boolean Array Driver" << endl;

	return out;
}
