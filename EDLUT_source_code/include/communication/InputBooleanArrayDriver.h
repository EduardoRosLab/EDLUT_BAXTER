/***************************************************************************
 *                           InputBooleanArrayDriver.h                     *
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

#ifndef INPUTBOOLEANARRAYDRIVER_H_
#define INPUTBOOLEANARRAYDRIVER_H_

/*!
 * \file InputBooleanArrayDriver.h
 *
 * \author Jesus Garrido
 * \date December 2010
 *
 * This file declares a class for getting external input spikes when implementing a
 * block with input boolean lines.
 */

#include "./ArrayInputSpikeDriver.h"

#include "../spike/EDLUTFileException.h"

/*!
 * \class InputBooleanArrayDriver
 *
 * \brief Class for getting input spikes from a block with input boolean lines.
 *
 * This class abstract methods for getting the input spikes to the network. Its subclasses
 * implements the input source and methods.
 *
 * \author Jesus Garrido
 * \date December 2010
 */
class InputBooleanArrayDriver: public ArrayInputSpikeDriver {
	private:
		/*!
		 * Input array in order to associate each input line to an input cell
		 */
		int * AssociatedCells;

		/*!
		 * Number of input lines
		 */
		unsigned int NumInputLines;

	public:
		/*!
		 * \brief Class constructor.
		 *
		 * It creates a new object to introduce spikes.
		 */
		InputBooleanArrayDriver(unsigned int InputLines, int * Associated);

		/*!
		 * \brief Class desctructor.
		 *
		 * Class desctructor.
		 */
		~InputBooleanArrayDriver();

		/*!
		 * \brief This method loads spikes from arrays.
		 *
		 * This method loads spikes from arrays.
		 *
		 * \param Queue The event queue where the input spikes are inserted.
		 * \param Net The network associated to the input spikes.
		 * \param InputLines An array of NumInputLines boolean values (true->Input spikes to associated neuron).
		 * \param CurrentTime The current simulation time. Every spike will be inserted in this time.
		 *
		 * \throw EDLUTException If something wrong happens in the input process.
		 */
		void LoadInputs(EventQueue * Queue, Network * Net, bool * InputLines, double CurrentTime) throw (EDLUTFileException);

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

#endif /* INPUTBOOLEANARRAYDRIVER_H_ */
