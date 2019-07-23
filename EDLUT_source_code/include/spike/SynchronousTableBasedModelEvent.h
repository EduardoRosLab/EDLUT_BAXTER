/***************************************************************************
 *                           SynchronousTableBasedModelEvent.h                      *
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

#ifndef SYNCHRONOUSTABLEBASEDMODELEVENT_H_
#define SYNCHRONOUSTABLEBASEDMODELEVENT_H_

/*!
 * \file SynchronousTableBasedModelEvent.h
 *
 * \author Francisco Naveros
 * \date April 2014
 *
 * This file declares a class which abstracts the second part of the SynchronousTableBasedModel neuron model.
 * This object makes the prediction of this neuron model.
 */
 
#include <iostream>

#include "./InternalSpike.h"

using namespace std;

class Neuron;
class Simulation;

/*!
 * \class SynchronousTableBasedModelEvent
 *
 * \brief Second part of the SynchronousTableBasedModel neuron model.
 *
 * This file declares a class which abstracts the second part of the SynchronousTableBasedModel neuron model.
 * This object makes the prediction of this neuron model.
 *
 * \author Francisco Naveros
 * \date April 2014
 */
class SynchronousTableBasedModelEvent: public InternalSpike{
	
	public:

		 /*!
   		 * Neuron that compute this event.
   		 */
		Neuron** Neurons;

		/*!
   		 * Maximum number of neuron that this event can process.
   		 */
		int MaxSize;

		/*!
   		 * Number of neuron stored in this event.
   		 */
		int NElements;
   		
 	
   		/*!
   		 * \brief Constructor with parameters.
   		 * 
   		 * It creates and initializes a new spike with the parameters.
   		 * 
   		 * \param NewTime Time of the new spike.
		 * \param NewQueueIndex Queue index where the event is stored
   		 * \param NewSource Source neuron of the spike.
   		 */
		SynchronousTableBasedModelEvent(double NewTime, int NewQueueIndex, int NewMaxSize);
   		
   		/*!
   		 * \brief Class destructor.
   		 * 
   		 * It destroies an object of this class.
   		 */
   		~SynchronousTableBasedModelEvent();
   	

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
   		 * \brief It includes a new neuron in this event.
   		 * 
   		 * It includes a new neuron in this event.
   		 * 
   		 * \param neuron Neuron.
   		 */
		void IncludeNewNeuron(Neuron * neuron);
   		 
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

#endif /*SYNCHRONOUSTABLEBASEDMODELEVENT_H_*/
