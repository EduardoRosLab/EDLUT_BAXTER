/***************************************************************************
 *                           InputCurrent.h                                *
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

#ifndef INPUTCURRENT_H_
#define INPUTCURRENT_H_

/*!
 * \file InputCurrent.h
 *
 * \author Francisco Naveros
 * \date April 2018
 *
 * This file declares a class which abstracts an input current event.
 */
 
#include <iostream>
#include <list>

#include "./Current.h"

using namespace std;

class Neuron;
class Simulation;

/*!
 * \class InputCurrent
 *
 * \brief Neural network current. Input external current.
 *
 * This class abstract the concept of input current.
 *
 * \author Francisco Naveros
 * \date April 2018
 */
class InputCurrent: public Current{
	
	public:
   		
   		/*!
   		 * \brief Default constructor.
   		 * 
   		 * It creates and initializes a new InputCurrent object.
   		 */
   		InputCurrent();
   	
   		/*!
   		 * \brief Constructor with parameters.
   		 * 
   		 * It creates and initializes a new InputCurrent object with the parameters.
   		 * 
		 * \param NewTime Time of the new current.
		 * \param NewQueueIndex Queue index where the event is stored
		 * \param NewSource Source neuron of the current.
		 * \param NewCurrent Current generated in a current generator
   		 */
		InputCurrent(double NewTime, int NewQueueIndex, Neuron * NewSource, float NewCurrent);
   		
   		/*!
   		 * \brief Class destructor.
   		 * 
   		 * It destroies an object of this class.
   		 */
   		~InputCurrent();
   		

   		/*!
   		 * \brief It process an event in the simulation with the option of real time available.
   		 * 
   		 * It process an event in the simulation with the option of real time available.
   		 * 
   		 * \param CurrentSimulation The simulation object where the event is working.
		 * \param RealTimeRestriction watchdog variable executed in a parallel OpenMP thread that
		 * control the consumed time in each slot.
   		 */
		void ProcessEvent(Simulation * CurrentSimulation, RealTimeRestrictionLevel RealTimeRestriction);

		/*!
   		 * \brief It process an event in the simulation without the option of real time available.
   		 * 
   		 * It process an event in the simulation without the option of real time available.
   		 * 
   		 * \param CurrentSimulation The simulation object where the event is working.
   		 */
		void ProcessEvent(Simulation * CurrentSimulation);

   		/*!
   		 * \brief this method print the event type.
   		 * 
   		 * This method print the event type..
		 */
		virtual void PrintType();


		/*!
   		 * \brief The event queue uses this preference variable to sort the events with the same time stamp.
   		 * 
   		 * The event queue uses this preference variable to sort the events with the same time stamp.
		 */
		virtual enum EventPriority ProcessingPriority();
};

#endif /*INPUTCURRENT_H_*/
