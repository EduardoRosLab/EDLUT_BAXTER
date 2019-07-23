/***************************************************************************
 *                           PropagatedSpikeGroup.h                        *
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

#ifndef PROPAGATEDSPIKEGROUP_H_
#define PROPAGATEDSPIKEGROUP_H_

/*!
 * \file PropagatedSpikeGroup.h
 *
 * \author Francisco Naveros
 * \date May 2015
 *
 * This file declares a class which abstracts a groupe of neural network propagated spike.
 */
 
#include <iostream>

#include "./Spike.h"

using namespace std;

class Neuron;
class Interconnection;

/*!
 * \class PropagatedSpikeGroup
 *
 * \brief Group of neural network propagated spike.
 *
 * A propagated spike is an event which generates a new spike in the next output connection.
 *
 * \author Francisco Naveros
 * \date May 2015
 */
class PropagatedSpikeGroup: public Spike{
	
	protected: 
   		/*!
   		 * Max number of neuron that can manage this event.
   		 */
		static const int MaxSize=1024;

   		/*!
   		 * Number of neuron that can manage this event.
   		 */
		int N_Elements;

   		/*!
   		 * For each neuron, how many output connection must be computed.
   		 */
		int N_ConnectionsWithEqualDelay[MaxSize];

   		/*!
   		 * For each neuron, the first output connection that must be computed.
   		 */
		Interconnection * ConnectionsWithEqualDelay[MaxSize];

 		
   	public:
   		
   		/*!
   		 * \brief Default constructor.
   		 * 
   		 * It creates and initializes a new spike object.
		 * 
		 * \param NewOpenMP_index queue index of the target neuron.
		 * \param NewQueueIndex Queue index where the event is stored.
		 */
		PropagatedSpikeGroup(double NewTime, int NewQueueIndex);
   	
    		
   		/*!
   		 * \brief Class destructor.
   		 * 
   		 * It destroies an object of this class.
   		 */
   		~PropagatedSpikeGroup();

		/*!
   		 * \brief It gest the number of neurons stored in this event.
   		 * 
   		 * It gest the number of neurons stored in this event.
   		 * 
   		 * \return the number of neurons stored in this event.
   		 */
		int GetN_Elementes();

		/*!
   		 * \brief It gest the maximum number of neurons that can be stored in this event.
   		 * 
   		 * It gest the maximum number of neurons that can be stored in this event.
   		 * 
   		 * \return The maximum number of neurons that can be stored in this event.
   		 */
		int GetMaxSize();


		/*!
   		 * \brief It Include a new neuron in the event.
   		 * 
   		 * It Include a new neuron in the event.
		 *
		 * \param NewN_Connections Number of output connection that must be processed.
		 * \param NewConnections First output connection that must be processed.
   		 * 
   		 * \return A Boolean value that is true when the number of neuron stored is equal to the maximum.
   		 */
		bool IncludeNewSource(int NewN_Connections, Interconnection * NewConnections);
   	
 		

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

#endif /*PROPAGATEDSPIKEGROUP_H_*/
