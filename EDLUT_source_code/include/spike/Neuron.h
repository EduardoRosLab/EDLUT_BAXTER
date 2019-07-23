/***************************************************************************
 *                           Neuron.h                                      *
 *                           -------------------                           *
 * copyright            : (C) 2009 by Jesus Garrido and Richard Carrillo   *
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

#ifndef NEURON_H_
#define NEURON_H_

/*!
 * \file Neuron.h
 *
 * \author Jesus Garrido
 * \author Richard Carrido
 * \date August 2008
 *
 * This file declares a class which abstracts a spiking neuron behaviour.
 */
#include <math.h>
#include <vector>

#include "../simulation/Configuration.h"

#include "../simulation/PrintableObject.h"

#include "../spike/NeuronPropagationDelayStructure.h"
 
using namespace std;

class NeuronModel;
class VectorNeuronState;
class Interconnection;
class EventQueue;
class InternalSpike;
class PropagatedSpike;
class Spike;




/*!
 * \class Neuron
 *
 * \brief Spiking neuron
 *
 * This class abstract the behaviour of a neuron in a spiking neural network.
 * It includes network characteristics as index in the network or input and output 
 * connections and spiking characteristics as the neural model, the current state,
 * the last updated time, or the spike prediction.
 *
 * \author Jesus Garrido
 * \author Richard Carrillo
 * \date August 2008
 */
class Neuron : public PrintableObject {
	private:

		
		/*!
		 * \brief Neuron index into the Neuron State Vector.
		 */
		long int index_VectorNeuronState;

		/*!
		 * \brief Neuron index into the network neurons.
		 */
		long int index;


   		/*!
   		 * It tells if neuron activity will be registered.
   		 */
   		int monitored;

		/*!
		 * \brief OpenMP associated thread.
		 */
		int OpenMP_queue_index;

	
		/*!
		 * \brief Neuron associated model.
		 */
		NeuronModel * type;
		
   		/*!
   		 * \brief Neuron state variables.
   		 */
   		VectorNeuronState * state;
   		
   		/*!
   		 * Output connections (divided in vector for each OpenMP thread).
   		 */
   		Interconnection*** OutputConnections;

		/*!
		 * Output Connection number (divided in vector for each OpenMP thread).
		 */
		unsigned long * OutputConNumber;
   		
   		/*!
   		 * Input connections with asssociated postsynaptic learning.
   		 */
   		Interconnection** InputLearningConnectionsWithPostSynapticLearning;

		/*!
   		 * Input connections without asssociated postsynaptic learning.
   		 */
   		Interconnection** InputLearningConnectionsWithoutPostSynapticLearning;
   		
		/*!
		 * Input Connection number with presynpatic learning.
		 */
		unsigned int InputConLearningNumberWithPostSynaptic;

		/*!
		 * Input Connection number without presynpatic learning.
		 */
		unsigned int InputConLearningNumberWithoutPostSynaptic;

 		
   		/*!
		 * Counts the number of fired spikes.
		 */
		long spikeCounter;

   		/*!
   		 * It tells if neuron is output neuron
   		 */
   		int isOutput;


   		/*!
   		 * Number of input connections with a learning rule of type trigger.
   		 */
		int N_TriggerConnection;

   		/*!
   		 * Connections with a learning rule of type trigger.
   		 */
		Interconnection** TriggerConnection;

   		
   	public:
		/*!
   		 * Propagation delay structure used to mix multiple propagation spike event in only one.
   		 */
		NeuronPropagationDelayStructure * PropagationStructure;

		/*!
		* Learning rule index of input connections (pointer 0 for learning rules with post synaptic learning, pointer 1 for learning rules without post synaptic learning).
		* We use this structure to improve the performance of learning rules after an output spike or a trigger spike.
		*/
		int ** IndexInputLearningConnections;


   		/*!
		 * \brief Default constructor without parameters.
		 *
		 * It generates a new default neuron object without input connections, output connections
		 * or neuron model. The neuron will be initialized with the default values.
		 */
   		Neuron();


		/*!
		 * \brief Default destructor.
		 *
		 * It deletes a Neuron object (but not the associated neuron type).
		 */
   		~Neuron();
   		
   		/*!
		 * \brief It initializes the neuron values.
		 *
		 * It initializes a neuron object with neuron model Type and neuron index NewIndex.
		 * Moreover, it initializes the neuron variables with the model initial values.
		 *
		 * \param NewIndex The neuron index into the network order.
		 * \param Type The neuron type. It can't be null.
		 * \param Monitored If true, the neuron activity will be registered.
		 * \param IsOutput If true, the neuron activity will be send to output driver
		 */
   		void InitNeuron(int NewIndex, int index_VectorNeuronState, NeuronModel * Type, bool Monitored, bool IsOutput, int blockIndex);
   		
   		/*!
		 * \brief It gets the neuron index into the network.
		 * 
		 * It returns the saved neuron index into the network's neurons.
		 * 
		 * \return The neuron index.
		 */
		//long int GetIndex() const;
		long int GetIndex() const{
			return this->index;
		}
   		
   		/*!
   		 * \brief It gets the current neuron state.
   		 * 
   		 * It gets the current neuron state.
   		 * 
   		 * \return The current neuron state.
   		 */
   		//VectorNeuronState * GetVectorNeuronState() const;
		inline VectorNeuronState * GetVectorNeuronState() const{
			return state;
		}
   		
   		/*!
   		 * \brief It gets the number of inputs to the current neuron which have associated learning.
   		 * 
   		 * It returns the number of input connections to the current neuron which have associated learning.
   		 * 
   		 * \return The number of input connections to the current neuron with plasticity.
   		 */
   		unsigned int GetInputNumberWithPostSynapticLearning() const;

		/*!
   		 * \brief It gets the number of inputs to the current neuron which have associated learning.
   		 * 
   		 * It returns the number of input connections to the current neuron which have associated learning.
   		 * 
   		 * \return The number of input connections to the current neuron with plasticity.
   		 */
   		unsigned int GetInputNumberWithoutPostSynapticLearning() const;
   		
   		/*!
   		 * \brief It gets the number of output from the current neuron.
   		 * 
   		 * It returns the number of output connections from the current neuron.
   		 * 
   		 * \return The number of output connections from the current neuron.
   		 */
   		//unsigned int GetOutputNumber(int index) const;
		inline unsigned int GetOutputNumber(int index) const{
			return this->OutputConNumber[index];
		}
   		
   		/*!
   		 * \brief It gets the input connection at an specified index.
   		 * 
   		 * It returns the input connection at index index.
   		 * 
   		 * \param index The index of the input connection what we want to get.
   		 * \return The input connection of index index.
   		 */
   		Interconnection * GetInputConnectionWithPostSynapticLearningAt(unsigned int index) const;

		/*!
   		 * \brief It gets the input connection at an specified index.
   		 * 
   		 * It returns the input connection at index index.
   		 * 
   		 * \param index The index of the input connection what we want to get.
   		 * \return The input connection of index index.
   		 */
   		Interconnection * GetInputConnectionWithoutPostSynapticLearningAt(unsigned int index) const;
   		
   		/*!
   		 * \brief It sets the input connections which have associated learning.
   		 * 
   		 * It sets the input connections which have associated learning.
   		 * 
   		 * \param Connection The input connections to set. The memory will be released within the class destructor.
		 * \param NumberOfConnections The number of input connections in the first parameter.
   		 */
   		void SetInputConnectionsWithPostSynapticLearning(Interconnection ** Connections, unsigned int NumberOfConnections);

		/*!
   		 * \brief It sets the input connections which have associated learning.
   		 * 
   		 * It sets the input connections which have associated learning.
   		 * 
   		 * \param Connection The input connections to set. The memory will be released within the class destructor.
		 * \param NumberOfConnections The number of input connections in the first parameter.
   		 */
   		void SetInputConnectionsWithoutPostSynapticLearning(Interconnection ** Connections, unsigned int NumberOfConnections);
   		
   		
   		/*!
   		 * \brief It gets the output connection at an specified index.
   		 * 
   		 * It returns the output connection at index index.
   		 * 
		 * \param index1 The index of the OpenMP queue.
   		 * \param index2 The index of the output connection what we want to get.
		 *
   		 * \return The output connection of index index.
   		 */
   		//Interconnection * GetOutputConnectionAt(unsigned int index1, unsigned int index2) const;
		inline Interconnection * GetOutputConnectionAt(unsigned int index1, unsigned int index2) const{
			return OutputConnections[index1][index2];
		}

   		/*!
   		 * \brief It gets the output connection connected to a specified OpenMP queue.
   		 * 
   		 * It gets the output connection connected to a specified OpenMP queue.
   		 * 
		 * \param index1 The index of the OpenMP queue.
		 *
   		 * \return The output connection of index index.
   		 */
		inline Interconnection ** GetOutputConnectionAt(unsigned int index1) const{
			return OutputConnections[index1];
		}
   		
   		/*!
   		 * \brief It sets the output connections from this neuron.
   		 * 
   		 * It sets the output connection array.
   		 * 
   		 * \param Connection The output connections to set. The memory will be released within the class destructor.
   		 * \param NumberOfConnections The number of input connections in the first parameter.
		 */
   		void SetOutputConnections(Interconnection *** Connections, unsigned long * NumberOfConnections);
   		
   		/*!
   		 * \brief It checks if the neuron has some output connection.
   		 * 
   		 * It checks if the number of output connections is greater than 0.
   		 * 
   		 * \return True if the neuron has some output connection. False in other case.
   		 */
   		bool IsOutputConnected(int index) const;
   		
   		/*!
   		 * \brief It checks if the neuron is monitored.
   		 * 
   		 * It checks if the neuron activity will be registered.
   		 * 
   		 * \return True if the neuron is monitored. False in other case.
   		 */
		//bool IsMonitored() const;
		inline int IsMonitored() const{
			return this->monitored;	
		}
		
		/*!
   		 * \brief It checks if the neuron is output.
   		 * 
   		 * It checks if the neuron activity is output.
   		 * 
   		 * \return True if the neuron is output. False in other case.
   		 */
		//int IsOutput() const;
		inline int IsOutput() const{
			return this->isOutput;	
		}
		
		/*!
   		 * \brief It gets the neuron type.
   		 * 
   		 * It returns the neuron type of the current object.
   		 * 
   		 * \return The neuron type of the current object.
   		 */
		//NeuronModel * GetNeuronModel() const;
		inline NeuronModel * GetNeuronModel() const{
			return type;
		}

		/*!
		 * \brief It prints the neuron info.
		 *
		 * It prints the current neuron characteristics.
		 *
		 * \param out The stream where it prints the information.
		 *
		 * \return The stream after the printer.
		 */
		virtual ostream & PrintInfo(ostream & out);

		/*!
		 * \brief Sets the spike counter.
		 *
		 * Sets the neuron spike counter. For LSAM.
		 *
		 * \param n is the value that the spike counter should be set to.
		 */
		void SetSpikeCounter(long n);

		/*!
		 * \brief Number of spikes fired.
		 *
		 * Number of spikes fired by this neuron. For LSAM.
		 *
		 * \return The number of spikes fired by the neuron.
		 */
		long GetSpikeCounter();

		/*!
		 * \brief Sets the neuron index into the VectorNeuronState.
		 *
		 * Sets the neuron index into the VectorNeuronState.
		 *
		 * \param Index is the value that Index_VectorNeuronState should be set to.
		 */
		void SetIndex_VectorNeuronState(long int Index);

		/*!
		 * \brief It gets the neuron index into the VectorNeuronState.
		 * 
		 * It returns the saved neuron index into the VectorNeuronState.
		 * 
		 * \return The neuron index.
		 */
		//long GetIndex_VectorNeuronState();
		inline long GetIndex_VectorNeuronState(){
			return index_VectorNeuronState;
		}


		/*!
		 * \brief It sets the OpenMP queue index to which this neuron belong.
		 * 
		 * It sets the OpenMP queue index to which this neuron belong.
		 * 
		 * \paramt OpenMP queue index.
		 */
		void set_OpenMP_queue_index(int index);

		/*!
		 * \brief It gets the OpenMP queue index to which this neuron belong.
		 * 
		 * It gets the OpenMP queue index to which this neuron belong.
		 * 
		 * \return The OpenMP queue index.
		 */
		inline int get_OpenMP_queue_index(){
			return OpenMP_queue_index;
		}

		/*!
		 * \brief It computes part of the output delay structure.
		 * 
		 * It computes part of the output delay structure.
		 */
		void CalculateOutputDelayStructure();

		/*!
		 * \brief It computes part of the output delay structure.
		 * 
		 * It computes part of the output delay structure.
		 */
		void CalculateOutputDelayIndex();


		/*!
		 * \brief It gets the connections with a learning rule of type trigger.
		 * 
		 * It gets the connections with a learning rule of type trigger.
		 * 
		 * \return The connections with a learning rule of type trigger.
		 */
		inline Interconnection ** GetTriggerConnection() const{
			return this->TriggerConnection;
		}

		/*!
		 * \brief It gets the number of connections with a learning rule of type trigger.
		 * 
		 * It gets the number of connections with a learning rule of type trigger.
		 * 
		 * \return The number of connections with a learning rule of type trigger.
		 */
		inline int GetN_TriggerConnection() const{
			return this->N_TriggerConnection;
		}


		/*!
		* \brief Initialize the learning rule index in aech neuron in order to improve cache friendly.
		*
		* Initialize the learning rule index in aech neuron in order to improve cache friendly.
		*/
		void initializeLearningRuleIndex();

};
  
#endif /*NEURON_H_*/
