/***************************************************************************
 *                           FileInputCurrentDriver.h                      *
 *                           -------------------                           *
 * copyright            : (C) 2018 by Francisco Naveros                    *
 * email                : fnaveros@ugr.es                                  *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef FILEINPUTCURRENTDRIVER_H_
#define FILEINPUTCURRENTDRIVER_H_

/*!
 * \file FileInputCurrentDriver.h
 *
 * \author Francisco Naveros
 * \date April 2018
 *
 * This file declares a class for getting external input currents from a file.
 */
#include <cstdlib>
#include <string>

#include "./InputCurrentDriver.h"

#include "../spike/EDLUTFileException.h"
 
class EventQueue;
class Network;



/*!
 * \class FileInputCurrentDriver
 *
 * \brief Class for getting input currents from a file. 
 *
 * This class abstract methods for getting the input currents to the network.
 *
 * \author Francisco Naveros
 * \date April 2018
 */
class FileInputCurrentDriver: public InputCurrentDriver {
	
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
		FileInputCurrentDriver(const char * NewFileName) throw (EDLUTException);
		
		/*!
		 * \brief Class desctructor.
		 * 
		 * Class desctructor.
		 */
		~FileInputCurrentDriver();
	
		/*!
		 * \brief It introduces the input activity in the simulation event queue from the file.
		 * 
		 * This method introduces the cumulated input activity in the simulation event queue.
		 * 
		 * \param Queue The event queue where the input currents are inserted.
		 * \param Net The network associated to the input currents.
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

#endif /*FILEINPUTCURRENTDRIVER_H_*/
