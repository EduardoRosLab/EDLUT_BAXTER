/***************************************************************************
 *                           TCPIPInputSpikeDriver.cpp                     *
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
 
#include "../../include/communication/TCPIPInputSpikeDriver.h"

#include "../../include/communication/ServerSocket.h"
#include "../../include/communication/ClientSocket.h"

#include "../../include/communication/CdSocket.h"

#include "../../include/simulation/EventQueue.h"

#include "../../include/spike/InputSpike.h"
#include "../../include/spike/Network.h"

#include "../../include/spike/Neuron.h"



TCPIPInputSpikeDriver::TCPIPInputSpikeDriver(enum TCPIPConnectionType Type, string server_address,unsigned short tcp_port){
	if (Type == SERVER){
		this->Socket = new ServerSocket(tcp_port);
	} else {
		this->Socket = new ClientSocket(server_address,tcp_port);
	}
	this->Finished = false;
}
		
TCPIPInputSpikeDriver::~TCPIPInputSpikeDriver(){
	delete this->Socket;
}
	
void TCPIPInputSpikeDriver::LoadInputs(EventQueue * Queue, Network * Net) throw (EDLUTFileException){
	unsigned short csize;
	
	this->Socket->receiveBuffer(&csize, sizeof(unsigned short));


	if (csize>0){
		OutputSpike * InputSpikes = new OutputSpike [csize];
	
		this->Socket->receiveBuffer(InputSpikes,sizeof(OutputSpike)*(int) csize);
		
		for (int c=0; c<csize; ++c){
			InputSpike * NewSpike = new InputSpike(InputSpikes[c].Time, Net->GetNeuronAt(InputSpikes[c].Neuron)->get_OpenMP_queue_index(), Net->GetNeuronAt(InputSpikes[c].Neuron));
			
			Queue->InsertEvent(NewSpike->GetQueueIndex(),NewSpike);				
		}

		delete [] InputSpikes;
	}
}

ostream & TCPIPInputSpikeDriver::PrintInfo(ostream & out){

	out << "- TCP/IP Input Spike Driver" << endl;

	return out;
}
