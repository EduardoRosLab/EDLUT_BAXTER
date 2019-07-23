/***************************************************************************
 *                           OutputSpikeDriver.h                           *
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

#ifndef OUTPUTSPIKEDRIVER_H_
#define OUTPUTSPIKEDRIVER_H_

/*!
 * \file OutputSpikeDriver.h
 *
 * \author Jesus Garrido
 * \author Richard Carrido
 * \date August 2008
 *
 * This file declares a class for communicate output spikes.
 */
 
#include "../spike/EDLUTException.h"

#include "../simulation/PrintableObject.h"

class Neuron;
class Spike;
 

/*!
 * \class OutputSpikeDriver
 *
 * \brief Class for communicate output spikes. 
 *
 * This class abstract methods for communicate the output spikes to the external system. Its subclasses
 * implements the output target and methods.
 *
 * \author Jesus Garrido
 * \author Richard Carrillo
 * \date August 2008
 */
class OutputSpikeDriver : public PrintableObject {

	public:
	
		/*!
		 * \brief Default destructor.
		 * 
		 * Default destructor.
		 */
		virtual ~OutputSpikeDriver();
	
		/*!
		 * \brief It communicates the output activity to the external system.
		 * 
		 * This method introduces the output spikes to the output target (file, tcpip system...).
		 * 
		 * \param NewSpike The spike for print.
		 * 
		 * \throw EDLUTException If something wrong happens in the output process.
		 */
		virtual void WriteSpike(const Spike * NewSpike) throw (EDLUTException) = 0;

		/*!
		 * \brief It communicates the neuron state to the external system.
		 * 
		 * This method introduces the neuron state to the output target (file, tcpip system...).
		 * 
		 * \param Time Time of the event (potential value).
		 * \param Source Source neuron of the potential.
		 * 
		 * \throw EDLUTException If something wrong happens in the output process.
		 */		
		virtual void WriteState(float Time, Neuron * Source) throw (EDLUTException) = 0;
		
		/*!
		 * \brief It checks if the current output driver is buffered.
		 * 
		 * This method checks if the current output driver has an output buffer.
		 * 
		 * \return True if the current driver has an output buffer. False in other case.
		 */
		 virtual bool IsBuffered() const = 0;
		 
		 /*!
		 * \brief It checks if the current output driver can write neuron potentials.
		 * 
		 * This method checks if the current output driver can write neuron potentials.
		 * 
		 * \return True if the current driver can write neuron potentials. False in other case.
		 */
		 virtual bool IsWritePotentialCapable() const = 0;
		 
		/*!
		 * \brief It writes the existing spikes in the output buffer.
		 * 
		 * This method writes the existing spikes in the output buffer.
		 * 
		 * \throw EDLUTException If something wrong happens in the output process.
		 */
		 virtual void FlushBuffers() throw (EDLUTException) = 0;
	
};

#endif /*OUTPUTDRIVER_H_*/
