/***************************************************************************
 *                           FileInputCurrentDriver.cpp                    *
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
 
#include "../../include/communication/FileInputCurrentDriver.h"

#include "../../include/simulation/Utils.h"
#include "../../include/simulation/EventQueue.h"

#include "../../include/spike/EDLUTFileException.h"
#include "../../include/spike/Network.h"
#include "../../include/spike/InputCurrent.h"
#include "../../include/spike/Neuron.h"

#include "../../include/openmp/openmp.h"



FileInputCurrentDriver::FileInputCurrentDriver(const char * NewFileName) throw (EDLUTException): FileName(NewFileName), Currentline(1L){
	this->Finished = false;
	this->Handler = fopen(NewFileName,"rt");
	if (!this->Handler){
		throw EDLUTFileException(TASK_FILE_INPUT_CURRENT_DRIVER, ERROR_FILE_INPUT_CURRENT_DRIVER_OPEN_FILE, REPAIR_OPEN_FILE_READ, 0, NewFileName, true);
	}
}
		
FileInputCurrentDriver::~FileInputCurrentDriver(){
	if (this->Handler){
		fclose(this->Handler);
		this->Handler=NULL;
	}
}
	
void FileInputCurrentDriver::LoadInputs(EventQueue * Queue, Network * Net) throw (EDLUTFileException){
	if (this->Handler){
		int ninputs,i;
		skip_comments(this->Handler,Currentline);
		if(fscanf(this->Handler,"%i",&ninputs)==1){
			int ncurrents,nneuron,nreps,ineuron,itime;
			float time, interv, icurrent;
			
			for(i=0;i<ninputs;){
				skip_comments(this->Handler,Currentline);
				if (fscanf(this->Handler, "%f", &time) == 1 && fscanf(this->Handler, "%i", &ncurrents) == 1 && fscanf(this->Handler, "%f", &interv) == 1 && fscanf(this->Handler, "%i", &nneuron) == 1 && fscanf(this->Handler, "%i", &nreps) == 1 && fscanf(this->Handler, "%f", &icurrent) == 1){
					if(nneuron+nreps<=Net->GetNeuronNumber() && nneuron >= 0){
						i+=ncurrents*nreps;
						
						if(i<=ninputs){
							for(itime=0;itime<ncurrents;itime++){
								for(ineuron=0;ineuron<nreps;ineuron++){
									InputCurrent * newcurrent = new InputCurrent(time + itime*interv, Net->GetNeuronAt(nneuron + ineuron)->get_OpenMP_queue_index(), Net->GetNeuronAt(nneuron + ineuron), icurrent);
									Queue->InsertEvent(newcurrent->GetQueueIndex(), newcurrent);
								}
							}
						}else{
							throw EDLUTFileException(TASK_FILE_INPUT_CURRENT_DRIVER, ERROR_FILE_INPUT_CURRENT_DRIVER_TOO_MUCH_CURRENTS, REPAIR_FILE_INPUT_CURRENT_DRIVER_TOO_MUCH_CURRENTS, Currentline, this->FileName.c_str(), true);
						}
					}else{
						throw EDLUTFileException(TASK_FILE_INPUT_CURRENT_DRIVER, ERROR_FILE_INPUT_CURRENT_DRIVER_NEURON_INDEX, REPAIR_FILE_INPUT_CURRENT_DRIVER_NEURON_INDEX, Currentline, this->FileName.c_str(), true);
						break;
					}
				}else{
					throw EDLUTFileException(TASK_FILE_INPUT_CURRENT_DRIVER, ERROR_FILE_INPUT_CURRENT_DRIVER_FEW_CURRENTS, REPAIR_FILE_INPUT_CURRENT_DRIVER_FEW_CURRENTS, Currentline, this->FileName.c_str(), true);
					break;
				}
			}			
		}else{
			throw EDLUTFileException(TASK_FILE_INPUT_CURRENT_DRIVER, ERROR_FILE_INPUT_CURRENT_DRIVER_N_CURRENTS, REPAIR_FILE_INPUT_CURRENT_DRIVER_N_CURRENTS, Currentline, this->FileName.c_str(), true);
		}		
		this->Finished = true;
	}
}



ostream & FileInputCurrentDriver::PrintInfo(ostream & out){

	out << "- File Input Current Driver: " << this->FileName << endl;

	return out;
}
