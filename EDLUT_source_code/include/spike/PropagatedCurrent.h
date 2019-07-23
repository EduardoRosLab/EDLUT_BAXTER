/***************************************************************************
 *                           PropagatedCurrent.h                           *
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

#ifndef PROPAGATEDCURRENT_H_
#define PROPAGATEDCURRENT_H_

/*!
 * \file PropagatedCurrent.h
 *
 * \author Francisco Naveros
 * \date April 2018
 *
 * This file declares a class which abstracts a neural network propagated current.
 */
 
#include <iostream>

#include "./Current.h"

using namespace std;

class Neuron;
class Interconnection;

/*!
 * \class PropagatedCurrent
 *
 * \brief Neural network propagated current.
 *
 * A propagated current is an event which propagate a current in the next output connection.
 *
 * \author Francisco Naveros
 * \date April 2018
 */
class PropagatedCurrent: public Current{
	
	protected: 
   		/*!
   		 * Indix of the first output connection that must propagate the current
   		 */
   		int propagationDelayIndex;

   		/*!
   		 * Index of the last output connection that must propagate the current in this event (all output connection 
		 * since propagationDelayIndex to UpperPropagationDelayIndex has the same propagation delay).
   		 */
		int UpperPropagationDelayIndex;

		/*
		 * Number of synapses with equal propagation delay
		 */
		int NSynapses;

   		/*!
   		 * Interconnection.
   		 */
		Interconnection * inter;

  		
   	public:
   		
   		/*!
   		 * \brief Default constructor.
   		 * 
   		 * It creates and initializes a new PropagatedCurrent object.
		 * 
		 */
   		PropagatedCurrent();

  	
   		/*!
   		 * \brief Constructor with parameters.
   		 * 
   		 * It creates and initializes a new PropagatedCurrent object with the parameters.
   		 * 
   		 * \param NewTime Time of the new current.
		 * \param NewQueueIndex Queue index where the event is stored
   		 * \param NewSource Source neuron of the current.
   		 * \param NewTarget >0->Interneurons spike.
		 * \param NewCurrent current.
   		 */
		PropagatedCurrent(double NewTime, int NewQueueIndex, Neuron * NewSource, int NewPropagationDelayIndex, int NewUpperPropagationDelayIndex, float NewCurrent);

   		/*!
   		 * \brief Constructor with parameters.
   		 * 
   		 * It creates and initializes a new PropagatedCurrent object with the parameters.
   		 * 
   		 * \param NewTime Time of the new spike.
		 * \param NewQueueIndex Queue index where the event is stored
   		 * \param NewSource Source neuron of the spike.
   		 * \param NewTarget >0->Interneurons spike.
		 * \param NewCurrent current.
		 * \param NewInter interconnection associated to this propagated spike.
   		 */
		PropagatedCurrent(double NewTime, int NewQueueIndex, Neuron * NewSource, int NewPropagationDelayIndex, int NewUpperPropagationDelayIndex, float NewCurrent, Interconnection * NewInter);
   		
   		/*!
   		 * \brief Class destructor.
   		 * 
   		 * It destroies an object of this class.
   		 */
   		~PropagatedCurrent();
   	
   		/*!
   		 * \brief It gets the spike source type.
   		 * 
   		 * It gets the spike source type.
   		 * 
   		 * \return The spike source type: -1->Input spike, -2->Internal spike and >0->Interneurons spike.
   		 */
   		int GetPropagationDelayIndex();
   		
		int GetUpperPropagationDelayIndex();
  		

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

#endif /*PROPAGATEDCURRENT_H_*/
