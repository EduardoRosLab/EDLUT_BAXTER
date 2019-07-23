/***************************************************************************
 *                           TCPIPInputOutputSpikeDriver.cpp               *
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

#include "../../include/communication/TCPIPInputOutputSpikeDriver.h"

#include "../../include/communication/ServerSocket.h"
#include "../../include/communication/ClientSocket.h"

#include "../../include/simulation/EventQueue.h"

#include "../../include/spike/InputSpike.h"
#include "../../include/spike/Network.h"
#include "../../include/spike/Neuron.h"

TCPIPInputOutputSpikeDriver::TCPIPInputOutputSpikeDriver(enum TCPIPConnectionType Type, string server_address,unsigned short tcp_port){
	if (Type == SERVER){
		this->Socket = new ServerSocket(tcp_port);
	} else {
		this->Socket = new ClientSocket(server_address,tcp_port);
	}
	this->Finished = false;	
}
		
TCPIPInputOutputSpikeDriver::~TCPIPInputOutputSpikeDriver(){
	delete this->Socket;
}

void TCPIPInputOutputSpikeDriver::LoadInputs(EventQueue * Queue, Network * Net) throw (EDLUTFileException){
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

	
void TCPIPInputOutputSpikeDriver::WriteSpike(const Spike * NewSpike) throw (EDLUTException){
	OutputSpike spike(NewSpike->GetSource()->GetIndex(),NewSpike->GetTime());
	#pragma omp critical (TCPIPInputOutputSpikeDriver)
	{	
	this->OutputBuffer.push_back(spike);
	}
}
		
void TCPIPInputOutputSpikeDriver::WriteState(float Time, Neuron * Source) throw (EDLUTException){
	return;	
}
		
bool TCPIPInputOutputSpikeDriver::IsBuffered() const{
	return true;
}

bool TCPIPInputOutputSpikeDriver::IsWritePotentialCapable() const{
	return false;
}
		 
void TCPIPInputOutputSpikeDriver::FlushBuffers() throw (EDLUTException){
	
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

ostream & TCPIPInputOutputSpikeDriver::PrintInfo(ostream & out){

	out << "- TCP/IP Input/Output Spike Driver" << endl;

	return out;
}
