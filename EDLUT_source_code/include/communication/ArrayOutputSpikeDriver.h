/***************************************************************************
 *                           ArrayOutputSpikeDriver.h                      *
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

#ifndef ARRAYOUTPUTSPIKEDRIVER_H_
#define ARRAYOUTPUTSPIKEDRIVER_H_

/*!
 * \file ArrayOutputSpikeDriver.h
 *
 * \author Jesus Garrido
 * \date August 2010
 *
 * This file declares a class for write output spikes into a vector in memory.
 */

#include <vector>

#include "./OutputSpikeDriver.h"


/*!
 * \class ArrayOutputSpikeDriver
 *
 * \brief Class for getting output spikes only using a function when the simulation is stopped.
 *
 * This class abstract methods for communicate the output spikes when you are simulating step-by-step.
 *
 * \author Jesus Garrido
 * \date May 2010
 */
class ArrayOutputSpikeDriver: public OutputSpikeDriver {

	private:

		/*!
		 * Output spike struct.
		 */
		struct OutputSpike {
			/*!
			 * Number of neuron.
			 */
			long int Neuron;

			/*!
			 * Time of the spike.
			 */
			float Time;

			OutputSpike(){};

			OutputSpike(int NewNeuron, float NewTime):Neuron(NewNeuron), Time(NewTime){};
		};

	public:
		/*!
		 * Spike buffer
		 */
		vector<OutputSpike> OutputBuffer;

	

		/*!
		 * \brief Class constructor.
		 *
		 * It creates a new object.
		 *
		 */
		ArrayOutputSpikeDriver();

		/*!
		 * \brief Class desctructor.
		 *
		 * Class desctructor.
		 */
		~ArrayOutputSpikeDriver();

		/*!
		 * \brief It adds the spike to the buffer.
		 *
		 * This method introduces the output spikes to the output buffer. If the object isn't buffered,
		 * then the spike will be automatically sent.
		 *
		 * \param NewSpike The spike for send.
		 *
		 * \see FlushBuffers()
		 * \see GetBufferedSpikes()
		 *
		 * \throw EDLUTException If something wrong happens in the output process.
		 */
		void WriteSpike(const Spike * NewSpike) throw (EDLUTException);

		/*!
		 * \brief This function isn't implemented in ArrayOutputDriver.
		 *
		 * This function isn't implemented in ArrayOutputDriver.
		 *
		 * \param Time Time of the event (potential value).
		 * \param Source Source neuron of the potential.
		 *
		 * \throw EDLUTException If something wrong happens in the output process.
		 */
		void WriteState(float Time, Neuron * Source) throw (EDLUTException);

		/*!
		 * \brief It checks if the current output driver is buffered.
		 *
		 * This method checks if the current output driver has an output buffer.
		 *
		 * \return True if the current driver has an output buffer. False in other case.
		 */
		 bool IsBuffered() const;

		/*!
		 * \brief It checks if the current output driver can write neuron potentials.
		 *
		 * This method checks if the current output driver can write neuron potentials.
		 *
		 * \return True if the current driver can write neuron potentials. False in other case.
		 */
		 bool IsWritePotentialCapable() const;

		/*!
		 * \brief It writes the existing spikes in the output buffer.
		 *
		 * This method writes the existing spikes in the output buffer. In this class, this function
		 * does not make sense because spikes are read by GetBufferedSpikes.
		 *
		 * \see GetBufferedSpikes
		 *
		 * \throw EDLUTException If something wrong happens in the output process.
		 */
		 void FlushBuffers() throw (EDLUTException);

		/*!
		 * \brief It writes the existing spikes in the output buffer.
		 *
		 * This method writes the existing spikes in the output buffer. Spikes are stored into a
		 * vector of times and another one of number of cells.
		 *
		 * \param Times Variable where spike times are stored.
		 * \param Cells Variable where spike cells are stored.
		 *
		 * \return The number of stored spikes.
		 *
		 * \note This function allocates the necessary memory (spike_number*(sizeof(double)+sizeof(long int)).
		 * \note The buffered will be empty after storing all the spikes.
		 *
		 * \throw EDLUTException If something wrong happens in the output process.
		 */
		 int GetBufferedSpikes(double *& Times, long int *& Cells);

		/*!
		 * \brief It pops the first spike from the output buffer.
		 *
		 * This method returns (as parameters) the first existing 
		 * spike in the output buffer and removes it from the buffer.
		 *
		 * \param Time Variable where spike time is stored.
		 * \param Cell Variable where spike cell os stored.
		 *
		 * \return True if there is at least one spike in the output buffer
		 * \return when the function is called. False when no spike can be
		 * \return returned.
		 *
		 * \throw EDLUTException If something wrong happens in the output process.
		 */
		 bool RemoveBufferedSpike(double & Time, long int & Cell);

		/*!
		 * \brief It prints the information of the object.
		 *
		 * It prints the information of the object.
		 *
		 * \param out The output stream where it prints the object to.
		 * \return The output stream.
		 */
		virtual ostream & PrintInfo(ostream & out);
};

#endif /*ARRAYOUTPUTDRIVER_H_*/
