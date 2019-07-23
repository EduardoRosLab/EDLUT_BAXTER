/***************************************************************************
 *                           TCPIPInputOutputSpikeDriver.h                 *
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

#ifndef TCPIPINPUTOUTPUTSPIKEDRIVER_H_
#define TCPIPINPUTOUTPUTSPIKEDRIVER_H_

/*!
 * \file TCPIPInputOutputSpikeDriver.h
 *
 * \author Jesus Garrido
 * \author Richard Carrido
 * \date August 2008
 *
 * This file declares a class for getting external input spikes from a TCPIP connection.
 */
 
#include <vector>

#include "./InputSpikeDriver.h"
#include "./OutputSpikeDriver.h"

#include "./TCPIPConnectionType.h"


#include "../spike/EDLUTFileException.h"

using namespace std;

class CdSocket;

/*!
 * \class TCPIPInputOutputSpikeDriver
 *
 * \brief Class for getting input spikes and send output spikes from an only TCPIP connection. 
 *
 * This class abstract methods for getting the input spikes to the network and send output spikes. Its subclasses
 * implements the input and output source and methods.
 *
 * \author Jesus Garrido
 * \author Richard Carrillo
 * \date August 2008
 */
class TCPIPInputOutputSpikeDriver: public InputSpikeDriver, public OutputSpikeDriver {
	
	private:
	
		/*!
		 * Output spike struct.
		 */
		struct OutputSpike {
			/*!
			 * Number of neuron.
			 */
			long int Neuron;
			
			/*!
			 * Time of the spike.
			 */
			float Time;	
			
			OutputSpike(){};
			
			OutputSpike(int NewNeuron, float NewTime):Neuron(NewNeuron), Time(NewTime){};
		};
	
		/*!
		 * The TCP IP device.
		 */
		CdSocket * Socket;
		
		/*!
		 * Spike buffer
		 */
		vector<OutputSpike> OutputBuffer;
	
	public:
	
		/*!
		 * \brief Class constructor.
		 * 
		 * It creates a new object from the socket data.
		 * 
		 * \param Type Client or Server
		 * \param server_address address of the server host. If Type==Server, server_address is not used.
		 * \param tcp_port tcp_port to connect
		 * 
		 */
		TCPIPInputOutputSpikeDriver(enum TCPIPConnectionType Type, string server_address,unsigned short tcp_port);
		
		/*!
		 * \brief Class desctructor.
		 * 
		 * Class desctructor.
		 */
		~TCPIPInputOutputSpikeDriver();
	
		/*!
		 * \brief It introduces the input activity in the simulation event queue from the connection.
		 * 
		 * This method introduces the cumulated input activity in the simulation event queue.
		 * 
		 * \param Queue The event queue where the input spikes are inserted.
		 * \param Net The network associated to the input spikes.
		 * 
		 * \throw EDLUTException If something wrong happens in the input process.
		 */
		virtual void LoadInputs(EventQueue * Queue, Network * Net) throw (EDLUTFileException);
		
		/*!
		 * \brief It adds the spike to the buffer.
		 * 
		 * This method introduces the output spikes to the output buffer. If the object isn't buffered,
		 * then the spike will be automatically sent.
		 * 
		 * \param NewSpike The spike for send.
		 * 
		 * \see FlushBuffers()
		 * 
		 * \throw EDLUTException If something wrong happens in the output process.
		 */
		virtual void WriteSpike(const Spike * NewSpike) throw (EDLUTException);
		
		/*!
		 * \brief This function isn't implemented in TCPIPOutputDriver.
		 * 
		 * This function isn't implemented in TCPIPOutputDriver.
		 * 
		 * \param Time Time of the event (potential value).
		 * \param Source Source neuron of the potential.
		 * 
		 * \throw EDLUTException If something wrong happens in the output process.
		 */		
		virtual void WriteState(float Time, Neuron * Source) throw (EDLUTException);
		
		/*!
		 * \brief It checks if the current output driver is buffered.
		 * 
		 * This method checks if the current output driver has an output buffer.
		 * 
		 * \return True if the current driver has an output buffer. False in other case.
		 */
		 virtual bool IsBuffered() const;
		 
		/*!
		 * \brief It checks if the current output driver can write neuron potentials.
		 * 
		 * This method checks if the current output driver can write neuron potentials.
		 * 
		 * \return True if the current driver can write neuron potentials. False in other case.
		 */
		 virtual bool IsWritePotentialCapable() const;
		 
		/*!
		 * \brief It writes the existing spikes in the output buffer.
		 * 
		 * This method writes the existing spikes in the output buffer.
		 * 
		 * \throw EDLUTException If something wrong happens in the output process.
		 */
		 void FlushBuffers() throw (EDLUTException);

		/*!
		 * \brief It prints the information of the object.
		 *
		 * It prints the information of the object.
		 *
		 * \param out The output stream where it prints the object to.
		 * \return The output stream.
		 */
		virtual ostream & PrintInfo(ostream & out);
	
};


#endif /*TCPIPINPUTDRIVER_H_*/
