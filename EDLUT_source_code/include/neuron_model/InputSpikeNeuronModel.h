/***************************************************************************
 *                           InputSpikeNeuronModel.h                       *
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

#ifndef INPUTSPIKENEURONMODEL_H_
#define INPUTSPIKENEURONMODEL_H_

/*!
 * \file InputSpikeNeuronModel.h
 *
 * \author Francisco Naveros
 * \date July 2015
 *
 * This file declares a neuron model that can emulate an input layer of neurons that propagates InputSpike events.
 */

#include "./EventDrivenInputDevice.h"

#include "../spike/EDLUTFileException.h"

#include <iostream>

using namespace std;

class Neuron;


/*!
 * \class InputSpikeNeuronModel
 *
 * \brief Input neuron model
 *
 * This class abstracts the behavior of an input neuron layer that can propagate spikes. It includes 
 * internal model function which define the behavior of the model.
 *
 * \author Francisco Naveros
 * \date July 2015
 */
class InputSpikeNeuronModel : public EventDrivenInputDevice {
	public:


		/*!
		 * \brief Default constructor with parameters.
		 *
		 * It generates a new neuron model object without being initialized.
		 *
		 * \param NeuronTypeID Neuron model type.
		 * \param NeuronModelID Neuron model description file.
		 */
		InputSpikeNeuronModel(string NeuronTypeID, string NeuronModelID);

		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		virtual ~InputSpikeNeuronModel();


		/*!
		 * \brief It loads the neuron model description and tables (if necessary).
		 *
		 * It loads the neuron model description and tables (if necessary).
		 */
		void LoadNeuronModel() throw (EDLUTFileException){};


		/*!
		 * \brief It initializes the neuron state to defined values.
		 *
		 * It initializes the neuron state to defined values.
		 *
		 */
		VectorNeuronState * InitializeState() {
			return NULL;
		};

		/*!
		* \brief It gets the neuron model generator type (spike or current).
		*
		* It gets the neuron model generator type (spike or current).
		*
		* \return The neuron model generator type (spike or current).
		*/
		enum NeuronModelOutputActivityType GetModelOutputActivityType();



		/*!
		 * \brief It prints the neuron model info.
		 *
		 * It prints the current neuron model characteristics.
		 *
		 * \param out The stream where it prints the information.
		 *
		 * \return The stream after the printer.
		 */
		ostream & PrintInfo(ostream & out);

		/*!
		 * \brief It initialice VectorNeuronState.
		 *
		 * It initialice VectorNeuronState.
		 *
		 * \param N_neurons cell number inside the VectorNeuronState.
		 * \param OpenMPQueueIndex openmp index
		 */
		void InitializeStates(int N_neurons, int OpenMPQueueIndex){
		
		};
};

#endif /* EVENTDRIVENNEURONMODEL_H_ */
