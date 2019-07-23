/***************************************************************************
 *                           NeuronModelPropagationDelayStructure.h        *
 *                           -------------------                           *
 * copyright            : (C) 2015 by Francisco Naveros					   *
 * email                : fnaveros@ugr.es		                           *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef NEURONMODELPROPAGATIONDELAYSTRUCTURE_H_
#define NEURONMODELPROPAGATIONDELAYSTRUCTURE_H_


class Neuron;
class Interconnection;

/*!
 * \file NeuronmodelPropagationDelayStructure.h
 *
 * \author Francisco Naveros
 * \date February 2015
 *
 * This file declares a class which define the propagation delay structure of all neurons that implement
 * the same neuron model in each OpenMP thread. This structure will be used to groupe several propagation
 * events of different neurons in just one propagation event (PropagatedSpikeGroup). The output synapses
 * are ordered in function of the propagation delay and to which OpenMP thread belong the target neurons. 
 */
class NeuronModelPropagationDelayStructure{
	
		/*!
		* \brief For each target OpenMP queue, how many different delays there are in the neuron model.
		*/
		int *size;

		/*!
		 * \brief For each target OpenMP queue, maximum different delays that can be stored (allocated size).
		*/
		int *AllocatedSize;

		/*!
		* \brief For each target OpenMP queue, the different propagation delays.
		*/
		double ** SynapseDelays;



	public:

   		/*!
		 * \brief Default constructor without parameters.
		 */
   		NeuronModelPropagationDelayStructure();


		/*!
		 * \brief Default destructor.
		 */
   		~NeuronModelPropagationDelayStructure();


   		/*!
   		 * \brief It includes a new delay in the event.
   		 * 
   		 * It includes a new delay in the event.
   		 * 
   		 * \param queueIndex Target OpenMP queue.
   		 * \param newDelay propagation delay.
   		 */
		void IncludeNewDelay(int queueIndex, double newDelay);

   		/*!
   		 * \brief It gets the maximum number of different delays that can be stored for a OpenMP queue.
   		 * 
   		 * It gets the maximum number of different delays that can be stored for a OpenMP queue.
   		 * 
   		 * \param queueIndex Target OpenMP queue.
   		 * 
		 * \return The maximum number of different delays that can be stored for a OpenMP queue.
   		 */
		int GetAllocatedSize(int queueIndex);

   		/*!
   		 * \brief It gets the number of different delays stored for a OpenMP queue.
   		 * 
   		 * It gets the number of different delays stored for a OpenMP queue.
   		 * 
   		 * \param queueIndex Target OpenMP queue.
   		 * 
		 * \return The number of different delays stored for a OpenMP queue.
   		 */
		int GetSize(int queueIndex);

   		/*!
   		 * \brief It gets the propagation delay.
   		 * 
   		 * It gets the propagation delay.
   		 * 
   		 * \param queueIndex Target OpenMP queue.
		 * \param index Index inside the SynapseDelays
   		 * 
		 * \return The propagation delay.
   		 */
		double GetDelayAt(int queueIndex, int index);

   		/*!
   		 * \brief It gets the list of propagation delays for a target OpenMP queue.
   		 * 
   		 * It gets the list of propagation delays for a target OpenMP queue.
   		 * 
   		 * \param queueIndex Target OpenMP queue.
   		 * 
		 * \return The list of propagation delays.
   		 */
		double * GetDelays(int queueIndex);

};
  
#endif /*NEURONMODELPROPAGATIONDELAYSTRUCTURE_H_*/
