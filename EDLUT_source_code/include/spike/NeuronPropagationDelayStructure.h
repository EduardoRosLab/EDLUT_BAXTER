/***************************************************************************
 *                           NeuronPropagationDelayStructure.h             *
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

#ifndef NEURONPROPAGATIONDELAYSTRUCTURE_H_
#define NEURONPROPAGATIONDELAYSTRUCTURE_H_


class Neuron;
class Interconnection;
class NeuronModelPropagationDelayStructure;

/*!
 * \file NeuronPropagationDelayStructure.h
 *
 * \author Francisco Naveros
 * \date February 2015
 *
 * This file declares a class which define the propagation delay structure for each individual neuron. This
 * structure will be used to groupe several propagation events of different neurons in just one propagation 
 * event (PropagatedSpikeGroup). The output synapses are ordered in function of the propagation delay and 
 * to which OpenMP thread belong the target neurons. This class use a NeuronModelPropagationDelayStructure
 * object to store the propagation delay structure of all neurons that implement the same neuron model in 
 * each OpenMP thread.
 */
class NeuronPropagationDelayStructure{
	public:

		/*!
		 * \brief For each target OpenMP queue, how many different delays contains the neuron.
		 */
		int * NDifferentDelays;
		
		/*!
		 * \brief For each target OpenMP queue and propagation delay, how many synapses have the same propagation delay.
		 */
		int ** NSynapsesWithEqualDelay;

		/*!
		 * \brief For each target OpenMP queue and propagation delay, the propagation delay.
		 */
		double ** SynapseDelay;

		/*!
		 * \brief For each target OpenMP queue and propagation delay, the first output synapse. The remaining synapses (set 
		 * by NSynapsesWithEqualDelay) are in consecutive memory positions.
		 */
		Interconnection*** OutputConnectionsWithEquealDealy;

		/*!
		 * \brief For each target OpenMP queue and propagation delay, which index corresponds with this delay in the 
		 *  NeuronModelPropagationDelayStructure.
		 */
		int ** IndexSynapseDelay;


   		/*!
		* \brief Constructor with parameters.
		*
		* It creates and initializes this object
		*
		* \param neuron Neuron to compute its output delays
		 */
   		NeuronPropagationDelayStructure(Neuron * neuron);


		/*!
		 * \brief Default destructor.
		 */
   		~NeuronPropagationDelayStructure();


		 /*!
   		 * \brief It computes the value of IndexSynapseDelay
   		 * 
   		 * It computes the value of IndexSynapseDelay using the NeuronModelPropagationDelayStructure object
		 *
		 * \param PropagationStructure NeuronModelPropagationDelayStructure object that store all the propagation delays of all 
		 * neurons that implement the same neuron model.
   		 */  
		void CalculateOutputDelayIndex(NeuronModelPropagationDelayStructure * PropagationStructure);
   		
 

};
  
#endif /*NEURONPROPAGATIONDELAYSTRUCTURE_H_*/
