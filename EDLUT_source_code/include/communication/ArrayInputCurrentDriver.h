/***************************************************************************
 *                           ArrayInputCurrentDriver.h                     *
 *                           -------------------                           *
 * copyright            : (C) 2018 by Francisco Naveros                    *
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

#ifndef ARRAYINPUTCURRENTDRIVER_H_
#define ARRAYINPUTCURRENTDRIVER_H_

/*!
 * \file ArrayInputCurrentDriver.h
 *
 * \author Francisco Naveros
 * \date April 2018
 *
 * This file declares a class for getting external input currents when simulating step-by-step.
 */

#include "./InputCurrentDriver.h"

#include "../spike/EDLUTFileException.h"

/*!
 * \class ArrayInputCurrentDriver
 *
 * \brief Class for getting external input currents when simulating step-by-step.
 *
 * This class abstract methods for getting the input current to the network. Its subclasses
 * implements the input source and methods.
 *
 * \author Francisco Naveros
 * \date April 2018
 */
class ArrayInputCurrentDriver: public InputCurrentDriver {
	public:
		/*!
		 * \brief Class constructor.
		 *
		 * It creates a new object to introduce spikes.
		 */
		ArrayInputCurrentDriver();

		/*!
		 * \brief Class desctructor.
		 *
		 * Class desctructor.
		 */
		~ArrayInputCurrentDriver();

		/*!
		 * \brief This method is only a stub.
		 *
		 * This method is only a stub.
		 *
		 * \param Queue The event queue where the input currents are inserted.
		 * \param Net The network associated to the input currents.
		 *
		 * \throw EDLUTException If something wrong happens in the input process.
		 */
		void LoadInputs(EventQueue * Queue, Network * Net) throw (EDLUTFileException);

		/*!
		 * \brief This method loads spikes from arrays.
		 *
		 * This method loads spikes from arrays.
		 *
		 * \param Queue The event queue where the input currents are inserted.
		 * \param Net The network associated to the input currents.
		 * \param NumCurrents The number of currents to load.
		 * \param Times Times when the currents will be produced.
		 * \param Cell Indexes of the cells.
		 * \param Currents Currents values.
		 *
		 * \throw EDLUTException If something wrong happens in the input process.
		 */
		void LoadInputs(EventQueue * Queue, Network * Net, int NumCurrents, double * Times, long int * Cells, float * Currents) throw (EDLUTFileException);

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

#endif /* ARRAYINPUTCURRENTDRIVER_H_ */
