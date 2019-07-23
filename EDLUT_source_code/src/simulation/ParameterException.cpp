/***************************************************************************
 *                           ParameterException.cpp                        *
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

#include "../../include/simulation/ParameterException.h"

ParameterException::ParameterException(string ErrorParameter, string ErrorMessage):Parameter(ErrorParameter), Message(ErrorMessage){
}
		
string ParameterException::GetParameter() const{
	return this->Parameter;
}
		
string ParameterException::GetErrorMsg() const{
	return this->Message;
}
		
void ParameterException::display_error() const{
	cerr << "Invalid parameter " << this->Parameter << endl;
	cerr << "Error message: " << this->Message << endl;
}

ostream & operator<< (ostream & out, ParameterException Exception){
	out << "Invalid parameter " << Exception.GetParameter() << endl;
	out << "Error message: " << Exception.GetErrorMsg() << endl;
	return out;	
}