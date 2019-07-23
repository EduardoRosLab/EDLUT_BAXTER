/***************************************************************************
 *                           SynchronizeSimulationEvent.h                  *
 *                           -------------------                           *
 * copyright            : (C) 2014 by Francisco Naveros                    *
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

#ifndef SYNCHRONIZESIMULATIONEVENT_H_
#define SYNCHRONIZESIMULATIONEVENT_H_

/*!
 * \file SynchronizeSimulationEvent.h
 *
 * \author Francisco Naveros
 * \date February 2014
 *
 * This file declares a class which abstracts a simulation event to synchronize all the OpenMP threads. This
 * synchronization is needed to process properly some events such as TimeEventSpikingNeurons_GPU, SaveWeightsEvent,
 * CommunicationEvent and SynchronizeActivityEvent.
 */
 
#include <iostream>

#include "./Event.h"

using namespace std;

class Simulation;

/*!
 * \class SynchronizeSimulationEvent
 *
 * \brief Simulation abstract event to synchronize all the OpenMP threads.
 *
 * This file declares a class which abstracts a simulation event to synchronize all the OpenMP threads. This
 * synchronization is needed to process properly some events such as TimeEventSpikingNeurons_GPU, SaveWeightsEvent,
 * CommunicationEvent and SynchronizeActivityEvent.
 *
 * \author Francisco Naveros
 * \date February 2014
 */
class SynchronizeSimulationEvent: public Event{
	
	public:

  		
   	
   		/*!
   		 * \brief Constructor with parameters.
   		 * 
   		 * It creates and initializes a new event with the parameters.
   		 * 
   		 * \param NewTime Time of the new event.
		 * \param NewQueueIndex Queue index where the event is stored
   		 */
		SynchronizeSimulationEvent(double NewTime, int NewQueueIndex);
   		
   		/*!
   		 * \brief Class destructor.
   		 * 
   		 * It destroies an object of this class.
   		 */
   		~SynchronizeSimulationEvent();
   	

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

#endif /*SYNCHRONIZESIMULATIONEVENT_H_*/
