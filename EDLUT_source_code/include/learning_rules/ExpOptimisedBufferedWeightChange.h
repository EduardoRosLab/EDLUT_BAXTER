/***************************************************************************
 *                           ExpOptimisedBufferedWeightChange.h            *
 *                           -------------------                           *
 * copyright            : (C) 2019 by Francisco Naveros                    *
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

#ifndef EXPOPTIMISEDBUFFEREDWEIGHTCHANGE_H_
#define EXPOPTIMISEDBUFFEREDWEIGHTCHANGE_H_

/*!
 * \file ExpOptimisedBufferedWeightChange.h
 *
 * \author Francisco Naveros
 * \date February 2019
 *
 * This file declares a class which abstracts a exponential additive learning rule precomputed in a look-up table.
 */

#include "./AdditiveKernelChange.h"

class BufferedActivityTimes;

/*!
 * \class SinOptimisedBufferedWeightChange
 *
 * \brief Sinuidal learning rule.
 *
 * This class abstract the behaviour of a exponential-sinusoidal additive learning rule precomputed in a look-up table.
 *
 * \author Francisco Naveros
 * \date April 2016
 */
class ExpOptimisedBufferedWeightChange: public AdditiveKernelChange{
	private:

    /*!
		 * Initial time of the learning rule.
		 */
		float initpos;


		/*!
		* Maximum time calulated in the look-up table.
		*/
		double maxTimeMeasured;
		double inv_maxTimeMeasured;

		/*!
		* Number of elements inside the look-up table.
		*/
		int N_elements;


		/*!
		* Look-up table for the kernel.
		*/
		float * kernelLookupTable;


		/*!
		* Buffer of spikes propagated by "no trigger" synapses
		*/
		BufferedActivityTimes * bufferedActivityTimesNoTrigger;

	public:
		/*!
		 * \brief Default constructor with parameters.
		 *
		 * It generates a new learning rule with its index.
		 *
		 * \param NewLearningRuleIndex learning rule index.
		 */
		ExpOptimisedBufferedWeightChange(int NewLearningRuleIndex);

		/*!
		 * \brief Object destructor.
		 *
		 * It remove the object.
		 */
		virtual ~ExpOptimisedBufferedWeightChange();


		 /* \brief It initialize the state associated to the learning rule for all the synapses.
		 *
		 * It initialize the state associated to the learning rule for all the synapses.
		 *
		 * \param NumberOfSynapses the number of synapses that implement this learning rule.
		 * \param NumberOfNeurons the total number of neurons in the network
		 */
		void InitializeConnectionState(unsigned int NumberOfSynapses, unsigned int NumberOfNeurons);


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
		virtual void LoadLearningRule(FILE * fh, long & Currentline, string fileName) throw (EDLUTFileException);

   		/*!
   		 * \brief It applies the weight change function when a presynaptic spike arrives.
   		 *
   		 * It applies the weight change function when a presynaptic spike arrives.
   		 *
   		 * \param Connection The connection where the spike happened.
   		 * \param SpikeTime The spike time.
   		 */
   		virtual void ApplyPreSynapticSpike(Interconnection * Connection,double SpikeTime);


		/*!
		 * \brief It gets the number of state variables that this learning rule needs.
		 *
		 * It gets the number of state variables that this learning rule needs.
		 *
		 * \return The number of state variables that this learning rule needs.
		 */
   		virtual int GetNumberOfVar() const;

};

#endif /*EXPOPTIMISEDBUFFEREDWEIGHTCHANGE_H_*/
