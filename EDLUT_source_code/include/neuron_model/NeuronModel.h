/***************************************************************************
 *                           NeuronModel.h                                 *
 *                           -------------------                           *
 * copyright            : (C) 2011 by Jesus Garrido and Francisco Naveros  *
 * email                : jgarrido@atc.ugr.es, fnaveros@ugr.es             *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef NEURONMODEL_H_
#define NEURONMODEL_H_

/*!
 * \file NeuronModel.h
 *
 * \author Jesus Garrido
 * \author Francisco Naveros
 * \date January 2011
 *
 * \note Modified on January 2012 in order to include time-driven simulation support in GPU.
 * New state variables (TIME_DRIVEN_MODEL_CPU and TIME_DRIVEN_MODEL_GPU)
 *
 * This file declares a class which abstracts an spiking neural model.
 */

#include <string>
#include <string.h>

#include "../spike/EDLUTFileException.h"

#include "../../include/simulation/ExponentialTable.h"

#include "../../include/integration_method/IntegrationMethod.h"

using namespace std;

class VectorNeuronState;
class InternalSpike;
class PropagatedSpike;
class Interconnection;
class Neuron;
class NeuronModelPropagationDelayStructure;
class CurrentSynapseModel;


//This variable indicates if the neuron model is an event driven neuron model (ASYNCHRONOUS UPDATE) or a time driven neuron model in CPU or GPU (SYNCHRONOUS UPDATE).
enum NeuronModelSimulationMethod { EVENT_DRIVEN_MODEL, TIME_DRIVEN_MODEL_CPU, TIME_DRIVEN_MODEL_GPU};

//This variable indicates if the neuron model is an input device that inyects activity (spikes or currents) in the neural network or is a neuron layer.
enum NeuronModelType { INPUT_DEVICE, NEURAL_LAYER };

//This variable indicates if the neuron model generates output spikes or currents.
enum NeuronModelOutputActivityType { OUTPUT_SPIKE, OUTPUT_CURRENT};

//This variable indicates if the neuron model can receive input spikes and/or currents or none (INPUT_DEVICES do not receive input synapses).
enum NeuronModelInputActivityType { INPUT_SPIKE, INPUT_CURRENT, INPUT_SPIKE_AND_CURRENT, NONE_INPUT};





//This variable indicate if a neuron model is defined in a second or milisicond time scale.
enum TimeScale {SecondScale=1, MilisecondScale=1000};


/*!
 * \class NeuronModel
 *
 * \brief Spiking neuron model
 *
 * This class abstracts the behavior of a neuron in a spiking neural network.
 * It includes internal model functions which define the behavior of the model
 * (initialization, update of the state, synapses effect...).
 * This is only a virtual function (an interface) which defines the functions of the
 * inherited classes.
 *
 * \author Jesus Garrido
 * \date January 2011
 */
class NeuronModel {
	protected:


		/*!
		 * \brief Neuron model type ID (LIFTimeDriven_1_2, TableBasedModel, etc.).
		*/
		string TypeID;

		/*!
		 * \brief Neuron model description file.
		 */
		string ModelID;



		/*!
		 * \brief This variable indicate if the neuron model has a time scale of seconds (1) or miliseconds (1000).
		*/
		float timeScale;
		float inv_timeScale;


		/*!
		* \brief Object to store the input currents of synapses that receive currents.
		*/
		CurrentSynapseModel * CurrentSynapses;

	public:

		/*!
		* \brief Initial state of this neuron model
		*/
		VectorNeuronState * State;


		/*!
		 * \brief PropagationStructure Object that include a structure of all the propagation delay of the neuron that composse this neuron model.
		 */
		NeuronModelPropagationDelayStructure * PropagationStructure;

		/*!
		 * \brief Default constructor with parameters.
		 *
		 * It generates a new neuron model object without being initialized.
		 *
		 * \param NeuronTypeID Neuron model type.
		 * \param NeuronModelID Neuron model description file.
		 */
		NeuronModel(string NeuronTypeID,string NeuronModelID);

		/*!
		 * \brief Default constructor with parameters.
		 *
		 * It generates a new neuron model object without being initialized.
		 *
		 * \param NeuronTypeID Neuron model type.
		 * \param NeuronModelID Neuron model description file.
		 */
		NeuronModel(string NeuronTypeID,string NeuronModelID, TimeScale new_timeScale);

		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		virtual ~NeuronModel();

		/*!
		 * \brief It loads the neuron model description and tables (if necessary).
		 *
		 * It loads the neuron model description and tables (if necessary).
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
		* \brief It processes a propagated current (input current in the cell).
		*
		* It processes a propagated current (input current in the cell).
		*
		* \param inter the interconection which propagate the spike
		* \param target the neuron which receives the spike
		* \param Current input current.
		*
		* \return A new internal spike if someone is predicted. 0 if none is predicted.
		*/
		void ProcessInputCurrent(Interconnection * inter, Neuron * target, float current);


		/*!
		 * \brief It gets the neuron type ID.
		 *
		 * It gets the neuron type ID.
		 *
		 * \return The identificator of the neuron type.
		 */
		string GetTypeID();

		/*!
		 * \brief It gets the neuron model ID.
		 *
		 * It gets the neuron model ID.
		 *
		 * \return The identificator of the neuron model.
		 */
		string GetModelID();

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
		 * \brief It gets the VectorNeuronState.
		 *
		 * It gets the VectorNeuronState.
		 *
		 * \return The VectorNeuronState.
		 */
		//VectorNeuronState * GetVectorNeuronState();
		inline VectorNeuronState * GetVectorNeuronState(){
			return this->State;
		}

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
		 * \brief It returns the NeuronModelPropagationDelayStructure object.
		 *
		 * It returns the NeuronModelPropagationDelayStructure object.
		 *
		 * \return the NeuronModelPropagationDelayStructure object.
		 */
		NeuronModelPropagationDelayStructure * GetNeuronModelPropagationDelayStructure();


		/*!
		 * \brief It sets the neuron model time scale.
		 *
		 * It sets the neuron model time scale.
		 *
		 * \param new_timeScale time scale.
		 */
		void SetTimeScale(float new_timeScale);


		/*!
		 * \brief It gets the neuron model time scale.
		 *
		 * It gets the neuron model time scale.
		 *
		 * \return new_timeScale time scale.
		 */
		float GetTimeScale();



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



		void InitializeInputCurrentSynapseStructure();



};

#endif /* NEURONMODEL_H_ */
