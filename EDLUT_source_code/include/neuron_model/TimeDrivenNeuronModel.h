/***************************************************************************
 *                           TimeDrivenNeuronModel.h                       *
 *                           -------------------                           *
 * copyright            : (C) 2012 by Jesus Garrido and Francisco Naveros  *
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

#ifndef TIMEDRIVENNEURONMODEL_H_
#define TIMEDRIVENNEURONMODEL_H_

/*!
 * \file TimeDrivenNeuronModel.h
 *
 * \author Jesus Garrido
 * \author Francisco Naveros
 * \date January 2011
 *
 * This file declares a class which abstracts an time-driven neuron model in a CPU.
 */

#include "./TimeDrivenModel.h"



#include "../integration_method/IntegrationMethod.h"
#include "../integration_method/Euler.h"
#include "../integration_method/RK2.h"
#include "../integration_method/RK4.h"
#include "../integration_method/BDFn.h"

#include "../integration_method/Bifixed_Euler.h"
#include "../integration_method/Bifixed_RK2.h"
#include "../integration_method/Bifixed_RK4.h"
#include "../integration_method/Bifixed_BDFn.h"

#include "../simulation/Utils.h"
#include "../simulation/Configuration.h"


#include <string>
//#include <cstdlib>

using namespace std;

class InputSpike;
class VectorNeuronState;



/*!
 * \class TimeDrivenNeuronModel
 *
 * \brief Time-Driven Spiking neuron model in a CPU
 *
 * This class abstracts the behavior of a neuron in a time-driven spiking neural network.
 * It includes internal model functions which define the behavior of the model
 * (initialization, update of the state, synapses effect, next firing prediction...).
 * This is only a virtual function (an interface) which defines the functions of the
 * inherited classes.
 *
 * \author Jesus Garrido
 * \date January 2011
 */
class TimeDrivenNeuronModel : public TimeDrivenModel {
	public:


		/*!
		 * \brief integration method.
		*/
		IntegrationMethod * integrationMethod;

		/*!
		 * \brief Auxiliar array for time dependente variables.
		*/
		float * conductance_exp_values;

		/*!
		 * \brief Auxiliar variable for time dependente variables.
		*/
		int N_conductances;


		/*!
		 * \brief Constructor with parameters.
		 *
		 * It generates a new neuron model.
		 *
		 * \param NeuronTypeID Neuron model type.
		 * \param NeuronModelID Neuron model description file.
		 * \param timeScale Variable that indicate which time scale implement this neuron model.
		 */
		TimeDrivenNeuronModel(string NeuronTypeID, string NeuronModelID, TimeScale timeScale);


		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		virtual ~TimeDrivenNeuronModel();


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
		enum NeuronModelSimulationMethod GetModelSimulationMethod();

		/*!
		 * \brief It gets the neuron model type (an input device that inyects activity (spikes or currents) in the neural network or a neuron layer).
		 *
		 * It gets the neuron model type (an input device that inyects activity (spikes or currents) in the neural network or a neuron layer).
		 *
		 * \return The neuron model type
		 */
		enum NeuronModelType GetModelType();

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
		 * \brief It Checks if the neuron model has this connection type.
		 *
		 * It Checks if the neuron model has this connection type.
		 *
		 * \param Type input connection type.
		 *
		 * \return If the neuron model supports this connection type
		 */
		virtual bool CheckSynapseType(Interconnection * connection) = 0;


		/*!
		 * \brief It Checks if the integrations has work properly.
		 *
		 * It Checks if the integrations has worked properly.
		 *
		 * \param current time
		 */
		void CheckValidIntegration(double CurrentTime);

		/*!
		* \brief It Checks if the integrations has work properly.
		*
		* It Checks if the integrations has worked properly.
		*
		* \param current time
		* \param valid integration acumulated value of all the membranen potential computed on the integration method.
		*/
		void CheckValidIntegration(double CurrentTime, float valid_integration);


		/*!
		 * \brief It initializes an auxiliar array for time dependente variables.
		 *
		 * It initializes an auxiliar array for time dependente variables.
		 *
		 * \param N_conductances .
		 * \param N_elapsed_times .
		 */
		void Initialize_conductance_exp_values(int N_conductances, int N_elapsed_times);

		/*!
		 * \brief It calculates the conductace exponential value for an elapsed time.
		 *
		 * It calculates the conductace exponential value for an elapsed time.
		 *
		 * \param index elapsed time index .
		 * \param elapses_time elapsed time.
		 */
		virtual void Calculate_conductance_exp_values(int index, float elapsed_time)=0;

		/*!
		 * \brief It sets the conductace exponential value for an elapsed time and a specific conductance.
		 *
		 * It sets the conductace exponential value for an elapsed time and a specific conductance.
		 *
		 * \param elapses_time_index elapsed time index.
		 * \param conductance_index conductance index.
		 * \param value.
		 */
		void Set_conductance_exp_values(int elapsed_time_index, int conductance_index, float value);

		/*!
		 * \brief It gets the conductace exponential value for an elapsed time and a specific conductance.
		 *
		 * It gets the conductace exponential value for an elapsed time and a specific conductance.
		 *
		 * \param elapses_time_index elapsed time index.
		 * \param conductance_index conductance index.
		 *
		 * \return A conductance exponential values.
		 */
		float Get_conductance_exponential_values(int elapsed_time_index, int conductance_index);

		/*!
		 * \brief It gets the conductace exponential value for an elapsed time.
		 *
		 * It gets the conductace exponential value for an elapsed time .
		 *
		 * \param elapses_time_index elapsed time index.
		 * 
		 * \return A pointer to a set of conductance exponential values.
		 */
		float * Get_conductance_exponential_values(int elapsed_time_index);

		/*!
		* \brief It calculates the number of electrical coupling synapses.
		*
		* It calculates the number for electrical coupling synapses.
		*
		* \param inter synapse that arrive to a neuron.
		*/
		virtual void CalculateElectricalCouplingSynapseNumber(Interconnection * inter) = 0;

		/*!
		* \brief It allocate memory for electrical coupling synapse dependencies.
		*
		* It allocate memory for electrical coupling synapse dependencies.
		*/
		virtual void InitializeElectricalCouplingSynapseDependencies() = 0;

		/*!
		* \brief It calculates the dependencies for electrical coupling synapses.
		*
		* It calculates the dependencies for electrical coupling synapses.
		*
		* \param inter synapse that arrive to a neuron.
		*/
		virtual void CalculateElectricalCouplingSynapseDependencies(Interconnection * inter) = 0;


		/*!
		* \brief It loads the integration method from the neuron model configuration file
		*
		* It loads the integration method from the neuron model configuration file
		*
		* \param fileName neuron model configuration file name
		* \fh neuron model configuration file
		* \Currentline current line inside the file
		*/
		virtual void loadIntegrationMethod(string fileName, FILE *fh, long * Currentline)throw (EDLUTFileException) = 0;


};

#endif /* TIMEDRIVENNEURONMODEL_H_ */
