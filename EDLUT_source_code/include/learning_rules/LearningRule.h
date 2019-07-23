/***************************************************************************
 *                           LearningRule.h                                *
 *                           -------------------                           *
 * copyright            : (C) 2010 by Jesus Garrido                        *
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

#ifndef LEARNINGRULE_H_
#define LEARNINGRULE_H_

#include "../simulation/PrintableObject.h"

#include "../spike/EDLUTFileException.h"

/*!
 * \file LearningRule.h
 *
 * \author Jesus Garrido
 * \date August 2010
 *
 * This file declares a class which abstracts a learning rule.
 */

#include "../../include/learning_rules/ConnectionState.h"
class Interconnection;
class Neuron;


/*!
 * \class LearningRule
 *
 * \brief Learning rule.
 *
 * This class abstract the behaviour of a learning rule.
 *
 * \author Jesus Garrido
 * \date March 2010
 */
class LearningRule : public PrintableObject {

	public:

		/*!
		 * \brief The conection state of the learning rule.
		 */
		ConnectionState * State;

		/*!
		 * \brief An auxiliar variable to manage the asignation of index.
		 */
		int counter;

		/*!
		 * \brief Learning rule index inside the whole simulation. This index will be used to identify each learning rule.
		 */
		int LearningRuleIndex;

		/*!
		 * \brief It initialize the state associated to the learning rule for all the synapses.
		 *
		 * It initialize the state associated to the learning rule for all the synapses.
		 *
		 * \param NumberOfSynapses the number of synapses that implement this learning rule.
		 * \param NumberOfNeurons the total number of neurons in the network
		 */
		virtual void InitializeConnectionState(unsigned int NumberOfSynapses, unsigned int NumberOfNeurons) = 0;


		/*!
		 * \brief It return the state associated to the learning rule for all the synapses.
		 *
		 * It return the state associated to the learning rule for all the synapses.
		 *
		 * \return the learning rule state for all the synapses.
		 */
		ConnectionState * GetConnectionState();

		/*!
		 * \brief Default constructor with parameters.
		 *
		 * It generates a new learning rule with its index.
		 *
		 * \param NewLearningRuleIndex learning rule index.
		 */ 
		LearningRule(int NewLearningRuleIndex);

		/*!
		 * \brief Object destructor.
		 *
		 * It remove a LearningRule object an releases the memory of the ConnectionState.
		 */
		virtual ~LearningRule();

		/*!
		 * \brief It loads the learning rule properties.
		 *
		 * It loads the learning rule properties.
		 *
		 * \param fh A file handler placed where the Learning rule properties are defined.
		 * \param Currentline The file line where the handler is placed.
		 * \param fileName file name.
		 *
		 * \throw EDLUTFileException If something wrong happens in reading the learning rule properties.
		 */
		virtual void LoadLearningRule(FILE * fh, long & Currentline, string fileName) throw (EDLUTFileException)= 0;

   		/*!
   		 * \brief It applies the weight change function when a presynaptic spike arrives.
   		 *
   		 * It applies the weight change function when a presynaptic spike arrives.
   		 *
   		 * \param Connection The connection where the spike happened.
   		 * \param SpikeTime The spike time.
   		 */
   		virtual void ApplyPreSynapticSpike(Interconnection * Connection,double SpikeTime) = 0;

   		/*!
		 * \brief It applies the weight change function when a postsynaptic spike arrives.
		 *
		 * It applies the weight change function when a postsynaptic spike arrives.
		 *
		 * \param Connection The connection where the learning rule happens.
		 * \param SpikeTime The spike time of the postsynaptic spike.
		 */
		virtual void ApplyPostSynapticSpike(Interconnection * Connection,double SpikeTime) = 0;

		/*!
		* \brief It applies the weight change function to all its input synapses when a postsynaptic spike arrives.
		*
		* It applies the weight change function to all its input synapses when a postsynaptic spike arrives.
		*
		* \param neuron The target neuron that manage the postsynaptic spike
		* \param SpikeTime The spike time of the postsynaptic spike.
		*/
		virtual void ApplyPostSynapticSpike(Neuron * neuron, double SpikeTime) = 0;

   		/*!
		 * \brief It prints the learning rule info.
		 *
		 * It prints the current learning rule characteristics.
		 *
		 * \param out The stream where it prints the information.
		 *
		 * \return The stream after the printer.
		 */
		virtual ostream & PrintInfo(ostream & out) = 0;

   		/*!
		 * \brief It returns if this learning rule implements postsynaptic learning.
		 *
		 * It returns if this learning rule implements postsynaptic learning.
		 *
		 * \returns if this learning rule implements postsynaptic learning
		 */
		virtual bool ImplementPostSynaptic() = 0;

};

#endif /* LEARNINGRULE_H_ */
