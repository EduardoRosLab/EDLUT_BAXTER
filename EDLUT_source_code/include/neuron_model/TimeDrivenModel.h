/***************************************************************************
 *                           TimeDrivenModel.h                             *
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

#ifndef TIMEDRIVENMODEL_H_
#define TIMEDRIVENMODEL_H_

/*!
 * \file TimeDrivenModel.h
 *
 * \author Francisco Naveros
 * \date April 2018
 *
 * This file declares a class which abstracts a time-driven neural model.
 */

#include "./NeuronModel.h"

#include <string>

using namespace std;

class InputSpike;
class VectorNeuronState;

/*!
 * \class TimeDrivenModel
 *
 * \brief Time-driven neuron model
 *
 * This class abstracts the behavior of a time-driven neuron.
 * It includes internal model functions which define the behavior of the model
 * (initialization, update of the state, synapses effect...).
 * This is only a virtual function (an interface) which defines the functions of the
 * inherited classes.
 *
 * \author Francisco Naveros
 * \date April 2018
 */
class TimeDrivenModel : public NeuronModel{

	public:


		/*!
		 * \brief Time-driven model must be updated periodically with this step in seconds.
		 */
		double time_driven_step_size;

		/*!
		 * \brief Default constructor with parameters.
		 *
		 * It generates a new neuron model object without being initialized.
		 *
		 * \param NeuronTypeID Neuron model type.
		 * \param NeuronModelID Neuron model description file.
		 */
		TimeDrivenModel(string NeuronTypeID, string NeuronModelID);

		/*!
		 * \brief Default constructor with parameters.
		 *
		 * It generates a new neuron model object without being initialized.
		 *
		 * \param NeuronTypeID Neuron model type.
		 * \param NeuronModelID Neuron model description file.
		 */
		TimeDrivenModel(string NeuronTypeID, string NeuronModelID, TimeScale new_timeScale);

		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		virtual ~TimeDrivenModel();

		/*!
		 * \brief It loads the neuron model description file.
		 *
		 * It loads the neuron model description file.
		 */
		virtual void LoadNeuronModel() throw (EDLUTFileException) = 0;

		/*!
		 * \brief It initializes the neuron state to defined values.
		 *
		 * It initializes the neuron state to defined values.
		 *
		 */
		virtual VectorNeuronState * InitializeState() = 0;

		/*!
		 * \brief It processes a propagated spike (input spike in the cell).
		 *
		 * It processes a propagated spike (input spike in the cell).
		 *
		 * \note This function doesn't generate the next propagated spike. It must be externally done.
		 *
		 * \param inter the interconection which propagate the spike
		 * \param time the time of the spike.
		 *
		 * \return A new internal spike if someone is predicted. 0 if none is predicted.
		 */
		virtual InternalSpike * ProcessInputSpike(Interconnection * inter, double time) = 0;

		
		/*!
		* \brief Update the neuron state variables.
		*
		* It updates the neuron state variables.
		*
		* \param index The cell index inside the vector. if index=-1, updating all cell.
		* \param CurrentTime Current time.
		*
		* \return True if an output spike have been fired. False in other case.
		*/
		virtual bool UpdateState(int index, double CurrentTime) = 0;

		/*!
		 * \brief It gets the neuron model simulation method (event-driven, time-driven in CPU or time-driven in GPU).
		 *
		 * It gets the neuron model simulation method (event-driven, time-driven in CPU or time-driven in GPU).
		 *
		 * \return The simulation method of the neuron model.
		 */
		virtual enum NeuronModelSimulationMethod GetModelSimulationMethod() = 0;

		/*!
		 * \brief It gets the neuron model type (an input device that inyects activity (spikes or currents) in the neural network or a neuron layer).
		 *
		 * It gets the neuron model type (an input device that inyects activity (spikes or currents) in the neural network or a neuron layer).
		 *
		 * \return The neuron model type
		 */
		virtual enum NeuronModelType GetModelType() = 0;

		/*!
		 * \brief It gets the neuron output activity type (spikes or currents).
		 *
		 * It gets the neuron output activity type (spikes or currents).
		 *
		 * \return The neuron output activity type (spikes or currents).
		 */
		virtual enum NeuronModelOutputActivityType GetModelOutputActivityType() = 0;

		/*!
		 * \brief It gets the neuron input activity types (spikes and/or currents or none).
		 *
		 * It gets the neuron input activity types (spikes and/or currents or none).
		 *
		 * \return The neuron input activity types (spikes and/or currents or none).
		 */
		virtual enum NeuronModelInputActivityType GetModelInputActivityType() = 0;

		/*!
		 * \brief It prints the neuron model info.
		 *
		 * It prints the current neuron model characteristics.
		 *
		 * \param out The stream where it prints the information.
		 *
		 * \return The stream after the printer.
		 */
		virtual ostream & PrintInfo(ostream & out) = 0;

		/*!
		 * \brief It initialice VectorNeuronState.
		 *
		 * It initialice VectorNeuronState.
		 *
		 * \param N_neurons cell number inside the VectorNeuronState.
		 * \param OpenMPQueueIndex openmp index
		 */
		virtual void InitializeStates(int N_neurons, int OpenMPQueueIndex)=0;


		/*!
		 * \brief It checks if the neuron model has this connection type.
		 *
		 * It Checks if the neuron model has this connection type.
		 *
		 * \param Type input connection type.
		 *
		 * \return If the neuron model supports this connection type
		 */
		virtual bool CheckSynapseType(Interconnection * connection)=0;


		/*!
		* \brief It calculates the number of electrical coupling synapses. 
		*
		* It calculates the number for electrical coupling synapses.
		*
		* \param inter synapse that arrive to a neuron.
		*/
		virtual void CalculateElectricalCouplingSynapseNumber(Interconnection * inter)=0;

		/*!
		* \brief It allocate memory for electrical coupling synapse dependencies.
		*
		* It allocate memory for electrical coupling synapse dependencies.
		*/
		virtual void InitializeElectricalCouplingSynapseDependencies()=0;

		/*!
		* \brief It calculates the dependencies for electrical coupling synapses.
		*
		* It calculates the dependencies for electrical coupling synapses.
		*
		* \param inter synapse that arrive to a neuron.
		*/
		virtual void CalculateElectricalCouplingSynapseDependencies(Interconnection * inter) = 0;


		/*!
		* \brief It sets the time-driven step size in seconds.
		*
		* It sets the time-driven step size in seconds.
		*
		* \param step time-driven step size.
		*/
		void SetTimeDrivenStepSize(double step);

		/*!
		* \brief It gets the time-driven step size in seconds.
		*
		* It gets the time-driven step size in seconds.
		*
		* \return time-driven step size.
		*/
		double GetTimeDrivenStepSize();

			


};

#endif /* TIMEDRIVENMODEL_H_ */
