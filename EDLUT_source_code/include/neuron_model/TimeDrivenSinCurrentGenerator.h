/***************************************************************************
 *                           TimeDrivenSinCurrentGenerator.h               *
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

#ifndef TIMEDRIVENSINCURRENTGENERATOR_H_
#define TIMEDRIVENSINCURRENTGENERATOR_H_

/*!
 * \file TimeDrivenSinCurrentGenerator.h
 *
 * \author Francisco Naveros
 * \date April 2018
 *
 * This file declares a class which abstracts a time-driven sinusoidal current generator.
 */

#include "./TimeDrivenInputDevice.h"

#include <string>

using namespace std;

class VectorNeuronState;
class Interconnection;



/*!
 * \class TimeDrivenSinCurrentGenerator
 *
 * \brief Time-driven sinusoidadl current generator.
 *
 * This class abstracts the behavior of a sinusoidal current generator in a time-driven spiking neural network.
 * It includes internal model functions which define the behavior of the model
 * (initialization, update of the state...).
 * This is only a virtual function (an interface) which defines the functions of the
 * inherited classes.
 *
 * \author Francisco Naveros
 * \date April 2018
 */
class TimeDrivenSinCurrentGenerator : public TimeDrivenInputDevice {
	protected:
		/*!
		 * \brief Sinusoidal frequency in Hz units
		 */
		float frequency;

		/*!
		* \brief Sinusoidal phase in radian units
		*/
		float phase;

		/*!
		* \brief Sinusoidal amplitude in XXXXXXXXXXXXX ///////REVISAR////////
		*/
		float amplitude;

		/*!
		* \brief Sinusoidal offset in XXXXXXXXXXXXX ///////REVISAR////////
		*/
		float offset;


		/*!
		 * \brief It loads the current generator description.
		 *
		 * It loads the current generator description from the file .cfg.
		 *
		 * \param ConfigFile Name of the current generator description file (*.cfg).
		 *
		 * \throw EDLUTFileException If something wrong has happened in the file load.
		 */
		void LoadNeuronModel(string ConfigFile) throw (EDLUTFileException);


	public:


		/*!
		 * \brief Default constructor with parameters.
		 *
		 * It generates a new Current generator object without being initialized.
		 *
		 * \param NeuronTypeID Neuron model identificator.
		 * \param NeuronModelID Neuron model configuration file.
		 */
		TimeDrivenSinCurrentGenerator(string NeuronTypeID, string NeuronModelID);


		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		virtual ~TimeDrivenSinCurrentGenerator();


		/*!
		 * \brief It loads the current generator description.
		 *
		 * It loads the current generator description.
 		 *
		 * \throw EDLUTFileException If something wrong has happened in the file load.
		 */
		virtual void LoadNeuronModel() throw (EDLUTFileException);


		/*!
		 * \brief It return the Neuron Model VectorNeuronState 
		 *
		 * It return the Neuron Model VectorNeuronState 
		 *
		 */
		virtual VectorNeuronState * InitializeState();


		/*!
		 * \brief Update the current generator state variables.
		 *
		 * It updates the current generator state variables.
		 *
		 * \param index The cell index inside the VectorNeuronState. if index=-1, updating all cell.
		 * \param CurrentTime Current time.
		 *
		 * \return NOT USED IN THIS MODEL.
		 */
		virtual bool UpdateState(int index, double CurrentTime);


		/*!
		 * \brief It gets the neuron output activity type (spikes or currents).
		 *
		 * It gets the neuron output activity type (spikes or currents).
		 *
		 * \return The neuron output activity type (spikes or currents).
		 */
		enum NeuronModelOutputActivityType GetModelOutputActivityType();

		/*!
		 * \brief It prints the time-driven model info.
		 *
		 * It prints the current time-driven model characteristics.
		 *
		 * \param out The stream where it prints the information.
		 *
		 * \return The stream after the printer.
		 */
		virtual ostream & PrintInfo(ostream & out);


		/*!
		 * \brief It initialice VectorNeuronState.
		 *
		 * It initialice VectorNeuronState.
		 *
		 * \param N_neurons cell number inside the VectorNeuronState.
		 * \param OpenMPQueueIndex openmp index
		 */
		virtual void InitializeStates(int N_neurons, int OpenMPQueueIndex);

};

#endif /* TIMEDRIVENSINCURRENTGENERATOR_H_ */
