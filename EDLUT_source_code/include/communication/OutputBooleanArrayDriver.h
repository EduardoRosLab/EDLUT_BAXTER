/***************************************************************************
 *                           OutputBooleanArrayDriver.h                    *
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

#ifndef OUTPUTBOOLEANARRAYDRIVER_H_
#define OUTPUTBOOLEANARRAYDRIVER_H_

/*!
 * \file OutputBooleanArrayDriver.h
 *
 * \author Jesus Garrido
 * \date December 2010
 *
 * This file declares a class for sending external output spikes when implementing a
 * block with boolean outputs (one output line with every output cell).
 */

#include "./ArrayOutputSpikeDriver.h"

#include "../spike/EDLUTFileException.h"

/*!
 * \class OutputBooleanArrayDriver
 *
 * \brief Class for sending external output spikes when implementing a
 * block with boolean outputs
 *
 * This class abstract methods for sending the output spikes to the network. Its subclasses
 * implements the output source and methods.
 *
 * \author Jesus Garrido
 * \date December 2010
 */
class OutputBooleanArrayDriver: public ArrayOutputSpikeDriver {

	private:

		/*!
		 * Output array in order to associate each output cell to an output line
		 */
		int * AssociatedCells;

		/*!
		 * Number of output lines.
		 */
		unsigned int NumOutputLines;

	public:

		/*!
		 * \brief Class constructor.
		 *
		 * It creates a new object to send spikes.
		 */
		OutputBooleanArrayDriver(unsigned int OutputLines, int * Associated);

		/*!
		 * \brief Class destructor.
		 *
		 * Class destructor.
		 */
		~OutputBooleanArrayDriver();

		/*!
		 * \brief This method sends spikes from arrays to a boolean array.
		 *
		 * This method sends spikes from arrays to a boolean array.
		 *
		 * \param OutputLines An array of NumOutputLines boolean values (true->Output spikes to associated neuron).
		 */
		void GetBufferedSpikes(bool * OutputLines);


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

#endif /*OUTPUTBOOLEANARRAYDRIVER_H_*/
