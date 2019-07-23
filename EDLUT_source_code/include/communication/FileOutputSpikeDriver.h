/***************************************************************************
 *                           FileOutputSpikeDriver.h                       *
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

#ifndef FILEOUTPUTSPIKEDRIVER_H_
#define FILEOUTPUTSPIKEDRIVER_H_

/*!
 * \file FileOutputSpikeDriver.h
 *
 * \author Jesus Garrido
 * \author Richard Carrido
 * \date August 2008
 *
 * This file declares a class for write output spikes in a file.
 */
#include <cstdlib>
#include <string>
 
#include "./OutputSpikeDriver.h"

#include "../spike/EDLUTException.h"



/*!
 * \class FileOutputSpikeDriver
 *
 * \brief Class for communicate output spikes in a output file. 
 *
 * This class abstract methods for communicate the output spikes to the target file.
 *
 * \author Jesus Garrido
 * \author Richard Carrillo
 * \date August 2008
 */
class FileOutputSpikeDriver: public OutputSpikeDriver {
	
	private:
	
		/*!
		 * The file handler.
		 */
		FILE * Handler;
		
		/*!
		 * The file name.
		 */
		string FileName;
		
		/*!
		 * Write potential events.
		 */
		bool PotentialWriteable;
		
	public:
	
		/*!
		 * \brief Class constructor.
		 * 
		 * It creates a new object from the file target.
		 * 
		 * \param NewFileName Name of the target output file.
		 * \param WritePotential If true, the potential events will be saved.
		 * 
		 * \throw EDLUTException If something wrong happens when the file is been wrotten.
		 */
		FileOutputSpikeDriver(const char * NewFileName, bool WritePotential) throw (EDLUTException);
		
		/*!
		 * \brief Class desctructor.
		 * 
		 * Class desctructor.
		 */
		~FileOutputSpikeDriver();
	
		/*!
		 * \brief It communicates the output activity to the output file.
		 * 
		 * This method introduces the output spikes to the output file.
		 * 
		 * \param NewSpike The spike for print.
		 * 
		 * \throw EDLUTException If something wrong happens in the output process.
		 */
		virtual void WriteSpike(const Spike * NewSpike) throw (EDLUTException);
		
		/*!
		 * \brief It communicates the membrane potential to the output file.
		 * 
		 * This method introduces the membrane potential to the output file.
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
		 virtual void FlushBuffers() throw (EDLUTException);

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

#endif /*FILEOUTPUTDRIVER_H_*/
