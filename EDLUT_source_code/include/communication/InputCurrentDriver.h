/***************************************************************************
 *                           InputCurrentDriver.h                          *
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

#ifndef INPUTCURRENTDRIVER_H_
#define INPUTCURRENTDRIVER_H_

/*!
 * \file InputCurrentDriver.h
 *
 * \author Francisco Naveros
 * \date April 2018
 *
 * This file declares a class for getting external input currents.
 */
#include "../spike/EDLUTException.h"

#include "../simulation/PrintableObject.h"
 
class EventQueue;
class Network;

/*!
 * \class InputCurrentDriver
 *
 * \brief Class for getting input currents. 
 *
 * This class abstract methods for getting the input currents to the network. Its subclasses
 * implements the input source and methods.
 *
 * \author Francisco Naveros
 * \date April 2018
 */
class InputCurrentDriver : public PrintableObject{
	
	protected:
	
		/*!
		 * Is the input proccess finished?
		 */
		bool Finished;

	public:
	
		/*!
		 * \brief Default destructor.
		 * 
		 * Default destructor.
		 */
		virtual ~InputCurrentDriver();
	
		/*!
		 * \brief It introduces the input activity in the simulation event queue.
		 * 
		 * This method introduces the cumulated input activity in the simulation event queue.
		 * 
		 * \param Queue The event queue where the input current are inserted.
		 * \param Net The network associated to the input currents.
		 * 
		 * \throw EDLUTException If something wrong happens in the input process.
		 */
		virtual void LoadInputs(EventQueue * Queue, Network * Net) throw (EDLUTException) = 0;
		
		/*!
		 * \brief It checks if the input process is finished.
		 * 
		 * It checks if the input process is finished.
		 * 
		 * \return True if the inputs have been finished. False in other case.
		 */
		bool IsFinished() const;
};


#endif /*INPUTCURRENTDRIVER_H_*/
