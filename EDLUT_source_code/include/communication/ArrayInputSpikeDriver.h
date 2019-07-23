/***************************************************************************
 *                           ArrayInputSpikeDriver.h                       *
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

#ifndef ARRAYINPUTSPIKEDRIVER_H_
#define ARRAYINPUTSPIKEDRIVER_H_

/*!
 * \file ArrayInputSpikeDriver.h
 *
 * \author Jesus Garrido
 * \date August 2008
 *
 * This file declares a class for getting external input spikes when simulating step-by-step.
 */

#include "./InputSpikeDriver.h"

#include "../spike/EDLUTFileException.h"

/*!
 * \class ArrayInputSpikeDriver
 *
 * \brief Class for getting input spikes from a TCPIP connection.
 *
 * This class abstract methods for getting the input spikes to the network. Its subclasses
 * implements the input source and methods.
 *
 * \author Jesus Garrido
 * \author Richard Carrillo
 * \date August 2008
 */
class ArrayInputSpikeDriver: public InputSpikeDriver {
	public:
		/*!
		 * \brief Class constructor.
		 *
		 * It creates a new object to introduce spikes.
		 */
		ArrayInputSpikeDriver();

		/*!
		 * \brief Class desctructor.
		 *
		 * Class desctructor.
		 */
		~ArrayInputSpikeDriver();

		/*!
		 * \brief This method is only a stub.
		 *
		 * This method is only a stub.
		 *
		 * \param Queue The event queue where the input spikes are inserted.
		 * \param Net The network associated to the input spikes.
		 *
		 * \throw EDLUTException If something wrong happens in the input process.
		 */
		void LoadInputs(EventQueue * Queue, Network * Net) throw (EDLUTFileException);

		/*!
		 * \brief This method loads spikes from arrays.
		 *
		 * This method loads spikes from arrays.
		 *
		 * \param Queue The event queue where the input spikes are inserted.
		 * \param Net The network associated to the input spikes.
		 * \param NumSpikes The number of spikes to load.
		 * \param Times Times when the spikes will be produced.
		 * \param Cell Indexes of the cells firing spikes.
		 *
		 * \throw EDLUTException If something wrong happens in the input process.
		 */
		void LoadInputs(EventQueue * Queue, Network * Net, int NumSpikes, double * Times, long int * Cells) throw (EDLUTFileException);

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

#endif /* ARRAYINPUTSPIKEDRIVER_H_ */
