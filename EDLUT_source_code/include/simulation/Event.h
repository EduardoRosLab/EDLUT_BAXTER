/***************************************************************************
 *                           Event.h                                       *
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

#ifndef EVENT_H_
#define EVENT_H_

/*!
 * \file Event.h
 *
 * \author Jesus Garrido
 * \date November 2008
 *
 * This file declares a class which abstracts a simulation event.
 */
 
#include <iostream>
#include "./RealTimeRestriction.h"

using namespace std;

class Simulation;

//This event priority variable is used in the event queue to sort properly the events with the same time stamp. The higher the priority is, the sooner this event is processed.
enum EventPriority { ENDSIMULATIONEVENT, STOPSIMULATIONEVENT, SYNCHRONIZEACTIVITYEVENT, SYNCHRONIZESIMULATIONEVENT, SAVEWEIGHTEVENT, COMMUNICATIONEVENT, INTERNALSPIKE, SYNCHRONOUSTABLEBASEDMODELEVENT, PROPAGATEDSPIKE, PROPAGATEDCURRENT, TIMEEVENT };

/*!
 * \class Event
 *
 * \brief Simulation abstract event.
 *
 * This class abstract the concept of event.
 *
 * \author Jesus Garrido
 * \date November 2008
 */
class Event{
	
	protected: 
   		/*!
   		 * Time when the event happens.
   		 */
   		const double time;


		/*!
		* Queue index where the event is stored.
		*/
		const int queueIndex;
   
   	public:
   		
   		/*!
   		 * \brief Default constructor.
   		 * 
   		 * It creates and initializes a new event object.
   		 */
   		Event();
 
 
    	/*!
   		 * \brief Constructor with parameters.
   		 * 
   		 * It creates and initializes a new event with the parameters.
   		 * 
   		 * \param NewTime Time of the new event.
		 */
   		Event(double NewTime);
		
		  	
   		/*!
   		 * \brief Constructor with parameters.
   		 * 
   		 * It creates and initializes a new event with the parameters.
   		 * 
   		 * \param NewTime Time of the new event.
		 * \param NewQueueIndex Queue index where the event is stored.
   		 */
   		Event(double NewTime, int NewQueueIndex);
   		
   		/*!
   		 * \brief Class destructor.
   		 * 
   		 * It destroies an object of this class.
   		 */
   		virtual ~Event();
   	
   		/*!
   		 * \brief It gets the event time.
   		 * 
   		 * It gets the event time.
   		 * 
   		 * \return The event time.
   		 */
   		double GetTime() const;
   		
   	
   		/*!
   		 * \brief It process an event in the simulation with the option of real time available.
   		 * 
   		 * It process an event in the simulation with the option of real time available.
   		 * 
   		 * \param CurrentSimulation The simulation object where the event is working.
		 * \param RealTimeRestriction watchdog variable executed in a parallel OpenMP thread that
		 * control the consumed time in each slot.
   		 */
		virtual void ProcessEvent(Simulation * CurrentSimulation, RealTimeRestrictionLevel RealTimeRestriction) = 0;

		/*!
   		 * \brief It process an event in the simulation without the option of real time available.
   		 * 
   		 * It process an event in the simulation without the option of real time available.
   		 * 
   		 * \param CurrentSimulation The simulation object where the event is working.
   		 */
		virtual void ProcessEvent(Simulation * CurrentSimulation) = 0;

   		/*!
   		 * \brief this method indicates if this event is and spike or current event.
   		 * 
   		 * This method indicates if this event is and spike or current event.
		 */
		virtual bool IsSpikeOrCurrent() const;

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
		virtual enum EventPriority ProcessingPriority()=0;


		/*!
		* \brief It returns the queue index where the event is stored.
		*
		* It returns the queue index where the event is stored.
		*
		* \ return queueIndex
		*/
		int GetQueueIndex();
};

#endif /*EVENT_H_*/
