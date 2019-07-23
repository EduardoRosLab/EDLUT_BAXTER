/***************************************************************************
 *                           Spike.h                                       *
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

#ifndef SPIKE_H_
#define SPIKE_H_

/*!
 * \file Spike.h
 *
 * \author Jesus Garrido
 * \author Richard Carrido
 * \date August 2008
 *
 * This file declares a class which abstracts a neural network spike.
 */
 
#include <iostream>

#include "../simulation/Event.h"

using namespace std;

class Neuron;
class Simulation;

/*!
 * \class Spike
 *
 * \brief Neural network spike.
 *
 * This class abstract the concept of spike. A spike is an event which generates a neuron
 * state update.
 *
 * \author Jesus Garrido
 * \author Richard Carrillo
 * \date August 2008
 */
class Spike: public Event{
	
	protected: 
   		/*!
   		 * Source neuron of the spike.
   		 */
   		Neuron * source;
     		
   	public:
   		
   		/*!
   		 * \brief Default constructor.
   		 * 
   		 * It creates and initializes a new spike object.
   		 */
   		Spike();
   	
   		/*!
   		 * \brief Constructor with parameters.
   		 * 
   		 * It creates and initializes a new spike with the parameters.
   		 * 
   		 * \param NewTime Time of the new spike.
		 * \param NewQueueIndex Queue index where the event is stored
   		 * \param NewSource Source neuron of the spike.
   		 */
		Spike(double NewTime, int NewQueueIndex, Neuron * NewSource);
   		
   		/*!
   		 * \brief Class destructor.
   		 * 
   		 * It destroies an object of this class.
   		 */
   		~Spike();
   	
   		/*!
   		 * \brief It gets the spike source neuron.
   		 * 
   		 * It gets the spike source neuron.
   		 * 
   		 * \return The spike source neuron.
   		 */
   		Neuron * GetSource () const;

		/*!
   		 * \brief It sets the spike source neuron.
   		 * 
   		 * It sets the spike source neuron.
   		 * 
   		 * \return The spike source neuron.
   		 */
   		void SetSource (Neuron * NewSource);
   		
	
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
   		
   		friend ostream & operator<< (ostream & out, Spike * spike);


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

/*!
 * \brief It prints an spike in the output.
 * 
 * It prints an spike in the output.
 * 
 * \param out The output stream.
 * \param spike The spike for printing.
 */
ostream & operator<< (ostream & out, Spike * spike);

#endif /*SPIKE_H_*/
