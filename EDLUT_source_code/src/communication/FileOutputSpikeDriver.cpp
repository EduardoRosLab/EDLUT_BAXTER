/***************************************************************************
 *                           FileOutputSpikeDriver.cpp                     *
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
 
#include "../../include/communication/FileOutputSpikeDriver.h"
#include "../../include/spike/EDLUTFileException.h"

#include "../../include/neuron_model/VectorNeuronState.h"

#include "../../include/spike/Spike.h"
#include "../../include/spike/Neuron.h"

FileOutputSpikeDriver::FileOutputSpikeDriver(const char * NewFileName, bool WritePotential) throw (EDLUTException): FileName(NewFileName){
	this->PotentialWriteable = WritePotential;
	this->Handler = fopen(NewFileName,"wt");
	if (!this->Handler){
		throw EDLUTFileException(TASK_FILE_OUTPUT_SPIKE_DRIVER, ERROR_FILE_OUTPUT_SPIKE_DRIVER_OPEN_FILE, REPAIR_OPEN_FILE_WRITE, 0, NewFileName, true);
	}
}
		
FileOutputSpikeDriver::~FileOutputSpikeDriver(){
	if (this->Handler){
		fclose(this->Handler);
		this->Handler=NULL;
	}
}

void FileOutputSpikeDriver::WriteSpike(const Spike * NewSpike) throw (EDLUTException){
	#pragma omp critical (FileOutputSpikeDriver) 
	{
	if(fprintf(this->Handler,"%f\t%li\n",NewSpike->GetTime(),NewSpike->GetSource()->GetIndex())<0)
		throw EDLUTException(TASK_FILE_OUTPUT_SPIKE_DRIVER, ERROR_FILE_OUTPUT_SPIKE_DRIVER_WRITE, REPAIR_OPEN_FILE_WRITE);
	}
}
		
void FileOutputSpikeDriver::WriteState(float Time, Neuron * Source) throw (EDLUTException){
	#pragma omp critical (FileOutputSpikeDriver)
	{
	if(fprintf(this->Handler,"%f\t%li",Time,Source->GetIndex()) < 0)
		throw EDLUTException(TASK_FILE_OUTPUT_SPIKE_DRIVER, ERROR_FILE_OUTPUT_SPIKE_DRIVER_WRITE, REPAIR_OPEN_FILE_WRITE);

	for (unsigned int i=0; i<Source->GetVectorNeuronState()->GetNumberOfPrintableValues(); ++i){
		if(fprintf(this->Handler,"\t%1.12f",Source->GetVectorNeuronState()->GetPrintableValuesAt(Source->GetIndex_VectorNeuronState(),i)) < 0)
			throw EDLUTException(TASK_FILE_OUTPUT_SPIKE_DRIVER, ERROR_FILE_OUTPUT_SPIKE_DRIVER_WRITE, REPAIR_OPEN_FILE_WRITE);
	}

	if(fprintf(this->Handler,"\n") < 0)
		throw EDLUTException(TASK_FILE_OUTPUT_SPIKE_DRIVER, ERROR_FILE_OUTPUT_SPIKE_DRIVER_WRITE, REPAIR_OPEN_FILE_WRITE);
	}
}

bool FileOutputSpikeDriver::IsBuffered() const{
	return false;	
}

bool FileOutputSpikeDriver::IsWritePotentialCapable() const{
	return this->PotentialWriteable;	
}

void FileOutputSpikeDriver::FlushBuffers() throw (EDLUTException){
	return;
}

ostream & FileOutputSpikeDriver::PrintInfo(ostream & out){

	out << "- File Output Spike Driver: " << this->FileName << endl;

	if (this->PotentialWriteable) out << "\tWriteable Potential" << endl;
	else out << "\tNon-writeable Potential" << endl;

	return out;
}
	
