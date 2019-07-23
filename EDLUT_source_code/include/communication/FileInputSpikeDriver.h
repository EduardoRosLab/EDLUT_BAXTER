/***************************************************************************
 *                           FileInputSpikeDriver.h                        *
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

#ifndef FILEINPUTSPIKEDRIVER_H_
#define FILEINPUTSPIKEDRIVER_H_

/*!
 * \file FileInputSpikeDriver.h
 *
 * \author Jesus Garrido
 * \author Richard Carrido
 * \date August 2008
 *
 * This file declares a class for getting external input spikes from a file.
 */
#include <cstdlib>
#include <string>

#include "./InputSpikeDriver.h"

#include "../spike/EDLUTFileException.h"
 
class EventQueue;
class Network;



/*!
 * \class FileInputSpikeDriver
 *
 * \brief Class for getting input spikes from a file. 
 *
 * This class abstract methods for getting the input spikes to the network.
 *
 * \author Jesus Garrido
 * \author Richard Carrillo
 * \date August 2008
 */
class FileInputSpikeDriver: public InputSpikeDriver {
	
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
		 * The current line in the file.
		 */
		long Currentline; 
	
	public:
	
		/*!
		 * \brief Class constructor.
		 * 
		 * It creates a new object from the file source.
		 * 
		 * \param NewFileName Name of the source input file.
		 * 
		 * \throw EDLUTException If something wrong happens when the file is been read.
		 */
		FileInputSpikeDriver(const char * NewFileName) throw (EDLUTException);
		
		/*!
		 * \brief Class desctructor.
		 * 
		 * Class desctructor.
		 */
		~FileInputSpikeDriver();
	
		/*!
		 * \brief It introduces the input activity in the simulation event queue from the file.
		 * 
		 * This method introduces the cumulated input activity in the simulation event queue.
		 * 
		 * \param Queue The event queue where the input spikes are inserted.
		 * \param Net The network associated to the input spikes.
		 * 
		 * \throw EDLUTException If something wrong happens in the input process.
		 */
		void LoadInputs(EventQueue * Queue, Network * Net) throw (EDLUTFileException);

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

#endif /*FILEINPUTDRIVER_H_*/
