/***************************************************************************
 *                           Current.h                                     *
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

#ifndef CURRENT_H_
#define CURRENT_H_

/*!
 * \file Current.h
 *
 * \author Francisco Naveros
 * \date April 2018
 *
 * This file declares a class which abstracts a neural network current.
 */
 
#include <iostream>

#include "../simulation/Event.h"

using namespace std;

class Neuron;
class Simulation;

/*!
 * \class Current
 *
 * \brief Neural network current.
 *
 * This class abstract the concept of current. A current is an event which generates a current generator.
 *
 * \author Francisco Naveros
 * \date April 2018
 */
class Current: public Event{
	
	protected: 
   		/*!
   		 * Source neuron of the spike.
   		 */
   		Neuron * source;

		/*!
		* Generated current.
		*/
		float current;
     		
   	public:
   		
   		/*!
   		 * \brief Default constructor.
   		 * 
   		 * It creates and initializes a new current object.
   		 */
   		Current();
   	
   		/*!
   		 * \brief Constructor with parameters.
   		 * 
   		 * It creates and initializes a new current event with the parameters.
   		 * 
   		 * \param NewTime Time of the new current.
		 * \param NewQueueIndex Queue index where the event is stored
   		 * \param NewSource Source neuron of the current.
		 * \param NewCurrent Current generated in a current generator
   		 */
		Current(double NewTime, int NewQueueIndex, Neuron * NewSource, float NewCurrent);
   		
   		/*!
   		 * \brief Class destructor.
   		 * 
   		 * It destroies an object of this class.
   		 */
   		~Current();
   	
   		/*!
   		 * \brief It gets the current source neuron.
   		 * 
   		 * It gets the current source neuron.
   		 * 
   		 * \return The current source neuron.
   		 */
   		Neuron * GetSource () const;

		/*!
   		 * \brief It sets the current source neuron.
   		 * 
   		 * It sets the current source neuron.
   		 * 
   		 * \return The current source neuron.
   		 */
   		void SetSource (Neuron * NewSource);

		/*!
		* \brief It gets the current.
		*
		* It gets the current.
		*
		* \return The current.
		*/
		float GetCurrent() const;

		/*!
		* \brief It sets the current.
		*
		* It sets the current.
		*
		* \return The current.
		*/
		void SetCurrent(float NewCurrent);
   		
	
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
		bool IsSpikeOrCurrent() const;

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
   	
};



#endif /*CURRENT_H_*/
