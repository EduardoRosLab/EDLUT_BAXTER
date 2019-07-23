/***************************************************************************
 *                           Interconnection.h                             *
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

#ifndef INTERCONNECTION_H_
#define INTERCONNECTION_H_

/*!
 * \file Interconnection.h
 *
 * \author Jesus Garrido
 * \author Richard Carrido
 * \date August 2008
 *
 * This file declares a class which abstracts a spiking neural network connection.
 */

#include "../simulation/PrintableObject.h"
 

class Neuron;
class NeuronModel;
class LearningRule;
class ConnectionState;

/*!
 * \class Interconnection
 *
 * \brief Spiking neural network connection
 *
 * This class abstract the behaviour of a spiking neural network connection.
 * It is composed by source and target neuron, an index, a connection delay...
 *
 * \author Jesus Garrido
 * \author Richard Carrillo
 * \date August 2008
 */
class Interconnection : public PrintableObject {
	
	private:
	
		/*!
		 * \brief The target neuron of the connection
		 */
		Neuron* target;

		/*!
		 * \brief The target neuron model of the connection
		 */
		NeuronModel * targetNeuronModel;
		

		/*!
		 * \brief The source neuron of the connection
		 */
		Neuron* source;

		/*!
		* \brief The learning (or weight change) rule of the connection.
		*/
		LearningRule* wchange_withPost;


		/*!
		* \brief The learning (or weight change) rule of the connection.
		*/
		LearningRule* wchange_withoutPost;

		/*!
		* \brief The delay of the spike propagation.
		*/
		double delay;
		
		/*!
		 * \brief The index of the connection in the network connections.
		 */
		long int index;		
		
		/*!
		 * \brief The vector neuron state index of the target neuron in its neuron model
		 */
		int targetNeuronModelIndex;
		
		/*!
		 * \brief The synaptic weight of the connection.
		 */
		float weight;
		
		/*!
		 * \brief The maximum weight of the connection.
		 */
		float maxweight;


		/*!
		* \brief The connection type (excitatory, inhibitory, electrical coupling, input current...)
		*/
		int type;

		/*!
		* \brief For current based synapses (electrical coupling, input current), each current must be stored in a 
		* different position. This index set the position in which the current propagated by this synapses must
		* be stored.
		*/
		int subindex_type;

		/*!
		* \brief Index inside the Learning Rule.
		*/
		int LearningRuleIndex_withPost;
		
		/*!
		 * \brief Index inside the Learning Rule.
		 */
		int LearningRuleIndex_withoutPost;

	public:

		/*!
		* \brief Index inside the target neuron for the index inside the Learning Rule.
		*/
		int LearningRuleIndex_withPost_insideTargetNeuron;

		/*!
		* \brief Index inside the target neuron for the index inside the Learning Rule.
		*/
		int LearningRuleIndex_withoutPost_insideTargetNeuron;

		/*!
		* \brief This connection is a trigger connection for the learning rule without post synpatic effect
		*/
		bool TriggerConnection;
	
		/*!
		 * \brief Default constructor.
		 * 
		 * It creates a new interconnection object without source or target neuron.
		 */ 
		Interconnection();
	
		/*!
		 * \brief Constructor with parameters.
		 * 
		 * It creates and initializes a new interconnection object with the parameters.
		 * 
		 * \param NewIndex The interconnection index in the network configuration file.
		 * \param NewSource Source neuron of this connection.
		 * \param NewTarget Target neuron of this connection.
		 * \param NewDelay Delay of this connection.
		 * \param NewType Connection type (excitatory, inhibitory, electrical coupling...) of this connection. The meaning of a value is depend on the neuron model.
		 * \param NewWeight Synaptic weight of this connection.
		 * \param NewMaxWeight Maximum synaptic weight of this connection.
		 * \param NewWeightChange Learning (or weight change) rule associated to this connection.
		 * \param NewLearningRuleIndex Current learning rule index.
		 */
		Interconnection(int NewIndex, Neuron * NewSource, Neuron * NewTarget, float NewDelay, int NewType, float NewWeight, float NewMaxWeight, LearningRule* NewWeightChange_withPost, unsigned int NewLearningRuleIndex_withPost, LearningRule* NewWeightChange_withoutPost, unsigned int NewLearningRuleIndex_withoutPost);
		
		/*!
		 * \brief Object destructor.
		 *
		 * It remove an interconnetion object an releases the memory of the connection state.
		 */
		~Interconnection();

		/*!
		 * \brief It gets the connection index.
		 * 
		 * It gets the connection index in the network connections.
		 * 
		 * \return The connection index.
		 */
		long int GetIndex() const;
		
		/*!
		 * \brief It sets the connection index.
		 * 
		 * It sets the connection index in the network connections.
		 * 
		 * \param NewIndex The new index of the connection.
		 */
		void SetIndex(long int NewIndex);
		
		/*!
		 * \brief It gets the source neuron.
		 * 
		 * It gets the source neuron of the connection.
		 * 
		 * \return The source neuron of the connection.
		 */	
		//Neuron * GetSource() const;
		inline Neuron * GetSource() const{
			return this->source;	
		}
		
		/*!
		 * \brief It sets the source neuron.
		 * 
		 * It sets the source neuron of the connection.
		 * 
		 * \param NewSource The new source neuron of the connection.
		 */
		void SetSource(Neuron * NewSource);
		
		/*!
		 * \brief It gets the target neuron.
		 * 
		 * It gets the target neuron of the connection.
		 * 
		 * \return The target neuron of the connection.
		 */
		//Neuron * GetTarget() const;
		inline Neuron * GetTarget() const{
			return this->target;
		}

		/*!
		 * \brief It gets the target neuron model.
		 * 
		 * It gets the target neuron model.
		 * 
		 * \return The target neuron model.
		 */
		inline NeuronModel * GetTargetNeuronModel() const{
			return this->targetNeuronModel;
		}

		/*!
		 * \brief It gets the target neuron model index in the vector neuron state.
		 * 
		 * It gets the target neuron model index in the vector neuron state.
		 * 
		 * \return The target neuron model index in the vector neuron state.
		 */
		inline int GetTargetNeuronModelIndex() const{
			return this->targetNeuronModelIndex;
		}
		
		/*!
		 * \brief It sets the target neuron.
		 * 
		 * It sets the target neuron of the connection.
		 * 
		 * \param NewTarget The new target neuron of the connection.
		 */
		void SetTarget(Neuron * NewTarget);

		/*!
		 * \brief It sets the target neuron model.
		 * 
		 * It sets the target neuron model. 
		 * 
		 * \param model The target neuron model.
		 */
		inline void SetTargetNeuronModel(NeuronModel * model){
			this->targetNeuronModel = model;
		}

		/*!
		 * \brief It sets the target neuron model index in the vector neuron state.
		 * 
		 * It sets the target neuron model index in the vector neuron state.
		 * 
		 * \param index The target neuron model index in the vector neuron state.
		 */
		inline void SetTargetNeuronModelIndex(int index){
			this->targetNeuronModelIndex = index;
		}
		
		/*!
		 * \brief It gets the connection delay.
		 * 
		 * It gets the delay in the spike propagation through this connection.
		 * 
		 * \return The connection delay.
		 */
		//inline double GetDelay() const;
		inline double GetDelay() const{
			return delay;
		}
		
		/*!
		 * \brief It sets the connection delay.
		 * 
		 * It sets the delay of the connection.
		 * 
		 * \param NewDelay The new connection delay.
		 */
		void SetDelay(double NewDelay);
		
		/*!
		 * \brief It gets the connection type.
		 * 
		 * It gets the connection type (excitatory, inhibitory, electrical coupling...) of this connection. The meaning of a value is depend on the neuron model. 
		 * 
		 * \return The connection type.
		 */
		//int GetType() const;
		inline int GetType() const{
			return type;
		}
		
		/*!
		 * \brief It sets the connection type.
		 * 
		 * It sets the connection type (excitatory, inhibitory, electrical coupling...) of this connection. The meaning of a value is depend on the neuron model.
		 * 
		 * \param NewType The new connection type.
		 */
		void SetType(int NewType);


		/*!
		* \brief It gets the connection subindex type.
		*
		* It gets the connection subindex type.
		*
		* \return The connection subindex type.
		*/
		//int GetSubindexType() const;
		inline int GetSubindexType() const{
			return subindex_type;
		}

		/*!
		* \brief It sets the connection suindex type.
		*
		* It sets the connection subindex type.
		*
		* \param NewType The new connection subindex type.
		*/
		void SetSubindexType(int index);
		
		/*!
		 * \brief It gets the synaptic weight.
		 * 
		 * It gets the synaptic weight of the connection.
		 * 
		 * \return The synaptic weight.
		 */
		//float GetWeight() const;
		inline float GetWeight() const{
			return weight;
		}
		
		/*!
		 * \brief It sets the synaptic weight.
		 * 
		 * It sets the synaptic weight of the connection.
		 * 
		 * \param NewWeight The new synaptic weight of the connection.
		 */
		//void SetWeight(float NewWeight);
		inline void SetWeight(float NewWeight){
			this->weight = NewWeight;
			if (this->weight > this->GetMaxWeight()){
				this->weight = this->GetMaxWeight();
			}
			else if (this->weight < 0.0f){
				this->weight = 0.0f;
			}
		}

		/*!
		 * \brief It increment the synaptic weight and checks the final value is inside the limits.
		 * 
		 * It increment the synaptic weight and checks the final value is inside the limits.
		 * 
		 * \param Increment The synaptic weight increment of the connection.
		 */
		//void IncrementWeight(float Increment);
		inline void IncrementWeight(float Increment){
			this->weight += Increment;
			if(this->weight > this->GetMaxWeight()){
				this->weight = this->GetMaxWeight();
			}else if(this->weight < 0.0f){
				this->weight = 0.0f;
			}
		}
		
		/*!
		 * \brief It gets the maximum synaptic weight.
		 * 
		 * It gets the maximum synaptic weight of the connection.
		 * 
		 * \return The maximum synaptic weight.
		 */
		//float GetMaxWeight() const;
		inline float GetMaxWeight() const{
			return this->maxweight;
		}

		
		/*!
		 * \brief It sets the maximum synaptic weight.
		 * 
		 * It sets the maximum synaptic weight of the connection.
		 * 
		 * \param NewMaxWeight The new maximum synaptic weight of the connection.
		 */
		void SetMaxWeight(float NewMaxWeight);
		
		/*!
		 * \brief It gets the learning rule of this connection.
		 * 
		 * It gets the learning rule of the connection.
		 * 
		 * \return The learning rule of the connection. 0 if the connection hasn't learning rule.
		 */
		//LearningRule * GetWeightChange() const;
		inline LearningRule * GetWeightChange_withPost() const{
			return this->wchange_withPost;
		}
		
		/*!
		 * \brief It sets the learning rule of this connection.
		 * 
		 * It sets the learning rule of the connection.
		 * 
		 * \param NewWeightChange The new learning rule of the connection. 0 if the connection hasn't learning rule.
		 */
		void SetWeightChange_withPost(LearningRule * NewWeightChange_withPost);

				/*!
		 * \brief It gets the learning rule of this connection.
		 * 
		 * It gets the learning rule of the connection.
		 * 
		 * \return The learning rule of the connection. 0 if the connection hasn't learning rule.
		 */
		//LearningRule * GetWeightChange() const;
		inline LearningRule * GetWeightChange_withoutPost() const{
			return this->wchange_withoutPost;
		}
		
		/*!
		 * \brief It sets the learning rule of this connection.
		 * 
		 * It sets the learning rule of the connection.
		 * 
		 * \param NewWeightChange The new learning rule of the connection. 0 if the connection hasn't learning rule.
		 */
		void SetWeightChange_withoutPost(LearningRule * NewWeightChange_withoutPost);
		
		/*!
		 * \brief It gets the connection learning rule index.
		 * 
		 * It gets the connection learning rule index in the network connections.
		 * 
		 * \return The connection learning rule index.
		 */
		//int GetLearningRuleIndex() const;
		inline int GetLearningRuleIndex_withPost() const{
			return this->LearningRuleIndex_withPost;
		}

		/*!
		 * \brief It gets the connection learning rule index.
		 * 
		 * It gets the connection learning rule index in the network connections.
		 * 
		 * \return The connection learning rule index.
		 */
		//int GetLearningRuleIndex() const;
		inline int GetLearningRuleIndex_withoutPost() const{
			return this->LearningRuleIndex_withoutPost;
		}



		
		/*!
		 * \brief It sets the connection learning rule index.
		 * 
		 * It sets the connection learning rule index in the network connections.
		 * 
		 * \param NewIndex The new learning rule index of the connection.
		 */
		void SetLearningRuleIndex_withPost(int NewIndex);

		/*!
		 * \brief It sets the connection learning rule index.
		 * 
		 * It sets the connection learning rule index in the network connections.
		 * 
		 * \param NewIndex The new learning rule index of the connection.
		 */
		void SetLearningRuleIndex_withoutPost(int NewIndex);


		/*!
		 * \brief It clears the activity register of this connection.
		 * 
		 * It clears the activity register of this connections and sets their values to 0.
		 */
		void ClearActivity();
		
		/*!
		 * \brief It gets the activity in an specified index.
		 * 
		 * It gets the activity in an specified index.
		 * 
		 * \param index The index of the activity.
		 * 
		 * \return The activity in that index.
		 */
		float GetActivityAt(int index) const;
		
		/*!
		 * \brief It sets the activity in an specified index.
		 * 
		 * It sets the activity in an specified index.
		 * 
		 * \param The index of the activity.
		 * \param NewActivity The new activity to be changed.
		 */
		void SetActivityAt(int index, float NewActivity);
		
		/*!
		 * \brief It gets the time of the last propagated spike.
		 * 
		 * It gets the time of the last propagated spike through this connection.
		 * 
		 * \return The time of the last propagated spike.
		 */
		double GetLastSpikeTime() const;	
		
		/*!
		 * \brief It sets the time of the last propagated spike.
		 * 
		 * It sets the time of the last propagated spike through this connection.
		 * 
		 * \param NewTime The new time of the last propagated spike.
		 */
		void SetLastSpikeTime(double NewTime);
		
		/*!
		 * \brief It prints the interconnection info.
		 *
		 * It prints the current interconnection characteristics.
		 *
		 * \param out The stream where it prints the information.
		 *
		 * \return The stream after the printer.
		 */
		virtual ostream & PrintInfo(ostream & out);

		/*!
		 * \brief It sets the trigger connection option to true.
		 * 
		 * It sets the trigger connection option to true.
		 */
		void SetTriggerConnection();

		/*!
		 * \brief It return the trigger connection option.
		 * 
		 * It return the trigger connection option.
		 */
		bool GetTriggerConnection();
};
  
#endif /*INTERCONNECTION_H_*/
