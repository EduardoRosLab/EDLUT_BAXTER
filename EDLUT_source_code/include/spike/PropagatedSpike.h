/***************************************************************************
 *                           PropagatedSpike.h                             *
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

#ifndef PROPAGATEDSPIKE_H_
#define PROPAGATEDSPIKE_H_

/*!
 * \file PropagatedSpike.h
 *
 * \author Jesus Garrido
 * \author Richard Carrido
 * \date August 2008
 *
 * This file declares a class which abstracts a neural network propagated spike.
 */
 
#include <iostream>

#include "./Spike.h"

using namespace std;

class Neuron;
class Interconnection;

/*!
 * \class PropagatedSpike
 *
 * \brief Neural network propagated spike.
 *
 * A propagated spike is an event which generates a new spike in the next output connection.
 *
 * \author Jesus Garrido
 * \author Richard Carrillo
 * \date August 2008
 */
class PropagatedSpike: public Spike{
	
	protected: 
   		/*!
   		 * Indix of the first output connection that must propagate a spike
   		 */
   		int propagationDelayIndex;

   		/*!
   		 * Index of the last output connection that must propagate a spike in this event (all output connection 
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
   		 * It creates and initializes a new spike object.
		 */
   		PropagatedSpike();
   	
   		/*!
   		 * \brief Constructor with parameters.
   		 * 
   		 * It creates and initializes a new spike with the parameters.
   		 * 
   		 * \param NewTime Time of the new spike.
		 * \param NewQueueIndex Queue index where the event is stored.
   		 * \param NewSource Source neuron of the spike.
   		 * \param NewTarget >0->Interneurons spike.
   		 */
		PropagatedSpike(double NewTime, int NewQueueIndex, Neuron * NewSource, int NewPropagationDelayIndex, int NewUpperPropagationDelayIndex);

   		/*!
   		 * \brief Constructor with parameters.
   		 * 
   		 * It creates and initializes a new spike with the parameters.
   		 * 
   		 * \param NewTime Time of the new spike.
		 * \param NewQueueIndex Queue index where the event is stored.
   		 * \param NewSource Source neuron of the spike.
   		 * \param NewTarget >0->Interneurons spike.
		 * \param NewInter interconnection associated to this propagated spike.
   		 */
		PropagatedSpike(double NewTime, int NewQueueIndex, Neuron * NewSource, int NewPropagationDelayIndex, int NewUpperPropagationDelayIndex, Interconnection * NewInter);
   		
   		/*!
   		 * \brief Class destructor.
   		 * 
   		 * It destroies an object of this class.
   		 */
   		~PropagatedSpike();
   	
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

#endif /*PROPAGATEDSPIKE_H_*/
