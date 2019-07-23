/***************************************************************************
 *                           EDLUTFileException.cpp                        *
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

#include "../../include/spike/EDLUTFileException.h"

EDLUTFileException::EDLUTFileException(TASK_CODE task, ERROR_CODE error, REPAIR_CODE repair, long line, const char * file, bool show) : EDLUTException(task, error, repair), Currentline(line), FileName(file), ShowLine(show){
}

long EDLUTFileException::GetErrorLine() const {
	return this->Currentline;
}

const char * EDLUTFileException::GetFileName() const {
	return this->FileName;
}

bool EDLUTFileException::GetShowLine() const{
	return this->ShowLine;
}


void EDLUTFileException::display_error() const {

	char msgbuf[1024];
	if(this->GetErrorNum()){
		cerr << "Error while: " << this->GetTaskMsg() << endl;
		if (this->GetShowLine()){
			cerr << "In file "<< this->GetFileName() <<", line: " << this->GetErrorLine() << endl;
		}
		
		sprintf(msgbuf,"Error message (%016llX): %s",this->GetErrorNum(),this->GetErrorMsg());
		cerr << msgbuf << endl;
		cerr << "Try to: " << this->GetRepairMsg() << endl;
	}
}

ostream & operator<< (ostream & out, EDLUTFileException Exception){
	if(Exception.GetErrorNum()){
		out << "Error while: " << Exception.GetTaskMsg() << endl;
		if (Exception.GetShowLine()){
			out << "In file " << Exception.GetFileName() << ", line: " << Exception.GetErrorLine() << endl;
		}
		
		out << "Error message " << Exception.GetErrorNum() << ": " << Exception.GetErrorMsg() << endl;
		out << "Try to: " << Exception.GetRepairMsg() << endl;
	}
	
	return out;
}

