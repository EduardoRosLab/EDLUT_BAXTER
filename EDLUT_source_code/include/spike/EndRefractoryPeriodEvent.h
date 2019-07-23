/***************************************************************************
 *                           EndRefractoryPeriodEvent.h                    *
 *                           -------------------                           *
 * copyright            : (C) 2015 by Francisco Naveros                    *
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

#ifndef ENDREFRACTORYPERIODEVENT_H_
#define ENDREFRACTORYPERIODEVENT_H_

/*!
 * \file EndRefractoryPeriodEvent.h
 *
 * \author Francisco Naveros
 * \date July 2015
 *
 * This file declares a class which implement and event that control the generation of internal spikes after a 
 * refractory period in an event driven neuron model
 */
 
#include <iostream>

#include "./Spike.h"

using namespace std;

class Neuron;
class Simulation;

/*!
 * \class EndRefractoryPeriodEvent
 *
 * \brief Event that process the end of a refractory period.
 *
 * This file declares a class which implement and event that control the generation of internal spikes after a 
 * refractory period in an event driven neuron model
 *
 * \author Francisco Naveros
 * \date July 2015
 */
class EndRefractoryPeriodEvent: public Spike{
	
	public:
   		
   		/*!
   		 * \brief Constructor with parameters.
   		 * 
   		 * It creates and initializes a new spike with the parameters.
   		 * 
   		 * \param NewTime Time of the new spike.
		 * \param NewQueueIndex Queue index where the event is stored
   		 * \param NewSource Source neuron of the spike.
   		 */
		EndRefractoryPeriodEvent(double NewTime, int NewQueueIndex, Neuron * NewSource);
   		
   		/*!
   		 * \brief Class destructor.
   		 * 
   		 * It destroies an object of this class.
   		 */
   		virtual ~EndRefractoryPeriodEvent();
   	

   		/*!
   		 * \brief It process an event in the simulation with the option of real time available.
   		 * 
   		 * It process an event in the simulation with the option of real time available.
   		 * 
   		 * \param CurrentSimulation The simulation object where the event is working.
		 * \param RealTimeRestriction watchdog variable executed in a parallel OpenMP thread that
		 * control the consumed time in each slot.
   		 */
		virtual void ProcessEvent(Simulation * CurrentSimulation, RealTimeRestrictionLevel RealTimeRestriction);

		/*!
   		 * \brief It process an event in the simulation without the option of real time available.
   		 * 
   		 * It process an event in the simulation without the option of real time available.
   		 * 
   		 * \param CurrentSimulation The simulation object where the event is working.
   		 */
		virtual void ProcessEvent(Simulation * CurrentSimulation);
   		 
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

#endif /*ENDREFRACTORYPERIODEVENT_H_*/
