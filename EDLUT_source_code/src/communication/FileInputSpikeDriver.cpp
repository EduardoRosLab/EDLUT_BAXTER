/***************************************************************************
 *                           FileInputSpikeDriver.cpp                      *
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
 
#include "../../include/communication/FileInputSpikeDriver.h"

#include "../../include/simulation/Utils.h"
#include "../../include/simulation/EventQueue.h"

#include "../../include/spike/EDLUTFileException.h"
#include "../../include/spike/Network.h"
#include "../../include/spike/InputSpike.h"
#include "../../include/spike/Neuron.h"

#include "../../include/openmp/openmp.h"


FileInputSpikeDriver::FileInputSpikeDriver(const char * NewFileName) throw (EDLUTException): FileName(NewFileName), Currentline(1L){
	this->Finished = false;
	this->Handler = fopen(NewFileName,"rt");
	if (!this->Handler){
		throw EDLUTFileException(TASK_FILE_INPUT_SPIKE_DRIVER, ERROR_FILE_INPUT_SPIKE_DRIVER_OPEN_FILE, REPAIR_OPEN_FILE_READ, 0, NewFileName, true);
	}
}
		
FileInputSpikeDriver::~FileInputSpikeDriver(){
	if (this->Handler){
		fclose(this->Handler);
		this->Handler=NULL;
	}
}
	
void FileInputSpikeDriver::LoadInputs(EventQueue * Queue, Network * Net) throw (EDLUTFileException){
	if (this->Handler){
		int ninputs,i;
		skip_comments(this->Handler,Currentline);
		if(fscanf(this->Handler,"%i",&ninputs)==1){
			int nspikes,nneuron,nreps,ineuron,itime;
			float time,interv;
			
			for(i=0;i<ninputs;){
				skip_comments(this->Handler,Currentline);
				if(fscanf(this->Handler,"%f",&time)==1 && fscanf(this->Handler,"%i",&nspikes)==1 && fscanf(this->Handler,"%f",&interv)==1 && fscanf(this->Handler,"%i",&nneuron)==1 && fscanf(this->Handler,"%i",&nreps)==1){
					if(nneuron+nreps<=Net->GetNeuronNumber() && nneuron >= 0){
						i+=nspikes*nreps;
						
						if(i<=ninputs){
							for(itime=0;itime<nspikes;itime++){
								for(ineuron=0;ineuron<nreps;ineuron++){
									InputSpike * ispike = new InputSpike(time + itime*interv, Net->GetNeuronAt(nneuron + ineuron)->get_OpenMP_queue_index(), Net->GetNeuronAt(nneuron + ineuron));
									Queue->InsertEvent(ispike->GetQueueIndex(), ispike);
								}
							}
						}else{
							throw EDLUTFileException(TASK_FILE_INPUT_SPIKE_DRIVER, ERROR_FILE_INPUT_SPIKE_DRIVER_TOO_MUCH_SPIKES, REPAIR_FILE_INPUT_SPIKE_DRIVER_TOO_MUCH_SPIKES, Currentline, this->FileName.c_str(), true);
						}
					}else{
						throw EDLUTFileException(TASK_FILE_INPUT_SPIKE_DRIVER, ERROR_FILE_INPUT_SPIKE_DRIVER_NEURON_INDEX, REPAIR_FILE_INPUT_SPIKE_DRIVER_NEURON_INDEX, Currentline, this->FileName.c_str(), true);
						break;
					}
				}else{
					throw EDLUTFileException(TASK_FILE_INPUT_SPIKE_DRIVER, ERROR_FILE_INPUT_SPIKE_DRIVER_FEW_SPIKES, REPAIR_FILE_INPUT_SPIKE_DRIVER_FEW_SPIKES, Currentline, this->FileName.c_str(), true);
					break;
				}
			}			
		}else{
			throw EDLUTFileException(TASK_FILE_INPUT_SPIKE_DRIVER, ERROR_FILE_INPUT_SPIKE_DRIVER_N_SPIKES, REPAIR_FILE_INPUT_SPIKE_DRIVER_N_SPIKES, Currentline, this->FileName.c_str(), true);
		}		
		this->Finished = true;
	}
}



ostream & FileInputSpikeDriver::PrintInfo(ostream & out){

	out << "- File Input Spike Driver: " << this->FileName << endl;

	return out;
}
