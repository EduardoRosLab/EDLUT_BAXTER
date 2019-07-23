/***************************************************************************
 *                           CurrentSynapseModel.h                        *
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

#ifndef CURRENTSYNAPSEMODEL_H_
#define CURRENTSYNAPSEMODEL_H_

/*!
 * \file CurrentSynapseModel.h
 *
 * \author Francisco Naveros
 * \date April 2018
 *
 * This file declares a class which abstracts an input current synapse.
 */

#include <string>

using namespace std;


/*!
 * \class CurrentSynapseModel
 *
 * \brief this class declares a input current synapse model.
 *
 * This class abstracts the behavior of an input current synapse model. 
 * It includes internal model functions which define the behavior of the model.
 *
 * \author Francisco Naveros
 * \date April 2018
 */
class CurrentSynapseModel {
	public:

		/*!
		* \brief Number of target neurons that include current synapses.
		*/
		int N_target_neurons;

		/*!
		* \brief Number of current synapses per target neuron.
		*/
		int * N_current_synapses_per_target_neuron;

		/*!
		* \brief Current value inyected in each synapse for each neuron.
		*/
		float ** input_current_per_synapse;
		
		
		/*!
		 * \brief Default Constructor without parameters.
		 *
		 * It generates a new current synapse object.
		 *
		 */
		CurrentSynapseModel();

		/*!
		* \brief Constructor with parameters.
		*
		* It generates a new current synapse object with parameters.
		*
		* \param size number of target neurons.
		*
		*/
		CurrentSynapseModel(int size);


		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		~CurrentSynapseModel();


		/*!
		* \brief It sets the number of target neurons.
		*
		* It sets the number of target neurons.
		*
		* \param size number of target neurons.
		*/
		void SetNTargetNeurons(int size);
		
		
		/*!
		* \brief It gets the number of target neurons.
		*
		* It gets the number of target neurons.
		*
		* \return number of target neurons.
		*/
		int GetNTargetNeurons();
		

		/*!
		* \brief It initializes the array that stores the number of current synapses for each neuron in the model.
		*
		* It initializes the array that stores the number of current synapses  for each neuron in the model.
		*/
		void InitializeNInputCurrentSynapsesPerNeuron();


		/*!
		* \brief It increments the number of current synapses for a target neuron.
		*
		* It increments the number of current synapses for a target neuron.
		*
		* \param neuron_index target neuron index.
		*/
		void IncrementNInputCurrentSynapsesPerNeuron(int neuron_index);
		
		
		/*!
		* \brief It gets the number of current synapses for a target neuron.
		*
		* It gets the number of current synapses for a target neuron.
		*
		* \param neuron_index target neuron index.
		*/
		int GetNInputCurrentSynapsesPerNeuron(int neuron_index);


		/*!
		* \brief It initializes the matrix that stores the currents for each input current synapse and target neuron.
		*
		* It initializes the matrix that stores the currents for each input current synapse and target neuron.
		*/
		void InitializeInputCurrentPerSynapseStructure();


		/*!
		* \brief It sets the current in a synapse of a target neuron.
		*
		* It sets the current in a synapse of a target neuron.
		*
		* \param neuron_index target neuron index.
		* \param synapse_index synapse index.
		* \param current input current.
		*/
		void SetInputCurrent(int neuron_index, int synapse_index, float current);
		
		
		/*!
		* \brief It gets the total current of a target neuron.
		*
		* It gets the total current of a target neuron.
		*
		* \param neuron_index target neuron index.
		* 
		* \return total input current.
		*/
		float GetTotalCurrent(int neuron_index);


};

#endif /* CURRENTSYNAPSEMODEL_H_ */
