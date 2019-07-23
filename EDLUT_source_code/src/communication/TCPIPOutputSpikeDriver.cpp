/***************************************************************************
 *                           TCPIPOutputSpikeDriver.cpp                     *
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

#include "../../include/communication/TCPIPOutputSpikeDriver.h"

#include "../../include/communication/ServerSocket.h"
#include "../../include/communication/ClientSocket.h"

#include "../../include/spike/Spike.h"
#include "../../include/spike/Neuron.h"


TCPIPOutputSpikeDriver::TCPIPOutputSpikeDriver(enum TCPIPConnectionType Type, string server_address,unsigned short tcp_port){
	if (Type == SERVER){
		this->Socket = new ServerSocket(tcp_port);
	} else {
		this->Socket = new ClientSocket(server_address,tcp_port);
	}
}
		
TCPIPOutputSpikeDriver::~TCPIPOutputSpikeDriver(){
	delete this->Socket;
}
	
void TCPIPOutputSpikeDriver::WriteSpike(const Spike * NewSpike) throw (EDLUTException){
	OutputSpike spike(NewSpike->GetSource()->GetIndex(),NewSpike->GetTime());
	#pragma omp critical (TCPIPOutputSpikeDriver)
	{		
	this->OutputBuffer.push_back(spike);
	}
}
		
void TCPIPOutputSpikeDriver::WriteState(float Time, Neuron * Source) throw (EDLUTException){
	return;	
}
		
bool TCPIPOutputSpikeDriver::IsBuffered() const{
	return true;
}

bool TCPIPOutputSpikeDriver::IsWritePotentialCapable() const{
	return false;
}
		 
void TCPIPOutputSpikeDriver::FlushBuffers() throw (EDLUTException){
	
	unsigned short size = (unsigned short)this->OutputBuffer.size();
	this->Socket->sendBuffer(&size,sizeof(unsigned short));
	if (size>0){
		OutputSpike * Array = new OutputSpike [size];
		
		for (unsigned short int i=0; i<size; ++i){
			Array[i] = this->OutputBuffer[i];	
		}
		
		this->Socket->sendBuffer((char *) Array, sizeof(OutputSpike)*size);
					
		delete [] Array;
		
		this->OutputBuffer.clear();
	}
}		

ostream & TCPIPOutputSpikeDriver::PrintInfo(ostream & out){

	out << "- TCP/IP Output Spike Driver" << endl;

	return out;
}
