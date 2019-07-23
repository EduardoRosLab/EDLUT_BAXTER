/***************************************************************************
 *                           InputSpike.h                                  *
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

#ifndef INPUTSPIKE_H_
#define INPUTSPIKE_H_

/*!
 * \file InputSpike.h
 *
 * \author Jesus Garrido
 * \author Richard Carrido
 * \date November 2008
 *
 * This file declares a class which abstracts a neural network spike.
 */
 
#include <iostream>
#include <list>

#include "./Spike.h"

using namespace std;

class Neuron;
class Simulation;

/*!
 * \class InputSpike
 *
 * \brief Neural network spike. Input external spike.
 *
 * This class abstract the concept of input spike.
 *
 * \author Jesus Garrido
 * \author Richard Carrillo
 * \date November 2008
 */
class InputSpike: public Spike{
	
	public:
   		
   		/*!
   		 * \brief Default constructor.
   		 * 
   		 * It creates and initializes a new spike object.
   		 */
   		InputSpike();
   	
   		/*!
   		 * \brief Constructor with parameters.
   		 * 
   		 * It creates and initializes a new spike with the parameters.
   		 * 
   		 * \param NewTime Time of the new spike.
		 * \param NewQueueIndex Queue index where the event is stored
   		 * \param NewSource Source neuron of the spike.
   		 */
		InputSpike(double NewTime, int NewQueueIndex, Neuron * NewSource);
   		
   		/*!
   		 * \brief Class destructor.
   		 * 
   		 * It destroies an object of this class.
   		 */
   		~InputSpike();
   		

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

#endif /*INPUTSPIKE_H_*/
