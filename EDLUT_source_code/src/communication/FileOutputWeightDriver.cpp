/***************************************************************************
 *                           FileOutputWeightDriver.cpp                    *
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
 
#include "../../include/communication/FileOutputWeightDriver.h"

#include "../../include/spike/Network.h"

FileOutputWeightDriver::FileOutputWeightDriver(const char * NewFileName) throw (EDLUTException): FileName(NewFileName){
}
		
FileOutputWeightDriver::~FileOutputWeightDriver(){
}

void FileOutputWeightDriver::WriteWeights(Network * Net, float SimulationTime) throw (EDLUTException){
	string Name = FileName;
	
	char* str = new char[30];

    sprintf(str, "%.6g", SimulationTime );

	Name = Name.insert(Name.find_last_of('.'),string(str));		
	
	Net->SaveWeights(Name.c_str());

	delete [] str;
	
	return;
}	

void FileOutputWeightDriver::WriteWeights(Network * Net) throw (EDLUTException){
	Net->SaveWeights(FileName.c_str());
}	

ostream & FileOutputWeightDriver::PrintInfo(ostream & out){

	out << "- File Output Weight Driver: " << this->FileName << endl;

	return out;
}

