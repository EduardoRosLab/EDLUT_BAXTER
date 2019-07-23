/***************************************************************************
 *                           ConnectionException.cpp                       *
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

#include "../../include/communication/ConnectionException.h"

ConnectionException::ConnectionException(string TargetAddress, unsigned short TargetPort, string ErrorMessage):Address(TargetAddress), Port(TargetPort), Message(ErrorMessage){
}

string ConnectionException::GetAddress() const{
	return this->Address;
}
		
unsigned short ConnectionException::GetPort() const{
	return this->Port;
}
		
string ConnectionException::GetErrorMsg() const{
	return this->Message;
}
		
void ConnectionException::display_error() const{
	cerr << "TCP Connection Error " << this->Address << ":" << this->Port << endl;
	cerr << "Error message: " << this->Message << endl;
}

ostream & operator<< (ostream & out, ConnectionException Exception){
	out << "TCP Connection Error " << Exception.GetAddress() << ":" << Exception.GetPort() << endl;
	out << "Error message: " << Exception.GetErrorMsg() << endl;
	return out;	
}