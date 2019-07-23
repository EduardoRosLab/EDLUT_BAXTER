/***************************************************************************
 *                           ConnectionState.h                             *
 *                           -------------------                           *
 * copyright            : (C) 2011 by Jesus Garrido                        *
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

#ifndef CONNECTIONSTATE_H_
#define CONNECTIONSTATE_H_

#include "../../include/simulation/ExponentialTable.h"

/*!
 * \file ConnectionState.h
 *
 * \author Jesus Garrido
 * \date October 2011
 *
 * This file declares a class which abstracts the current state of a synaptic connection.
 */

/*!
 * \class ConnectionState
 *
 * \brief Synaptic connection current state.
 *
 * This class abstracts the state of a synaptic connection and defines the state variables of
 * that connection.
 *
 * \author Jesus Garrido
 * \date October 2011
 */

class ConnectionState {

	protected:

		/*!
	   	 * \brief Neuron state variables.
	   	 */
	   	float * StateVars;

		/*!
	   	 * \brief Last update time for all neurons
	   	 */
	   	double * LastUpdate;

		/*!
		 * \brief Number of synapses that implement this learning rule.
		 */
		unsigned int NumberOfSynapses;

		/*!
		 * \brief Number of state variables.
		 */
		unsigned int NumberOfVariables;


	public:


		/*!
		 * \brief Default constructor with parameters.
		 *
		 * It generates a new state of a connection.
		 *
 		 * \param NumberOfSynapses Number of synapses that implement this learning rule.
		 * \param NumVariables Number of the state variables this model needs.
		 */
		ConnectionState(unsigned int NumberOfSynapses, int NumVariables);

		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		virtual ~ConnectionState();

		/*!
		 * \brief set new time to spikes.
		 *
		 * It set new time to spikes.
		 *
		 * \param index The synapse's index inside the learning rule.
		 * \param NewTime new time.
		 * \param pre_post In some learning rules (i.e. STDPLS) this variable indicate wether the update affects the pre- or post- variables.
		 */
		virtual void SetNewUpdateTime(unsigned int index, double NewTime, bool pre_post) = 0;


		/*!
		 * \brief It sets the state variable in a specified position.
		 *
		 * It sets the state variable in a specified position.
		 *
		 * \param index The synapse's index inside the learning rule.
		 * \param position The position of the state variable.
		 * \param NewValue The new value of that state variable.
		 */
		//void SetStateVariableAt(int index, unsigned int position,float NewValue);
		inline void SetStateVariableAt(unsigned int index, unsigned int position, float NewValue){
			*(this->StateVars + index*NumberOfVariables + position) = NewValue;
		}

		/*!
		 * \brief It sets the state variable in two consecutives position.
		 *
		 * It sets the state variable in a specified position.
		 *
		 * \param index The synapse's index inside the learning rule.
		 * \param position The position of the state variable.
		 * \param NewValue1 The new value of that state variable.
		 * \param NewValue2 The new value of that state variable.
		 */
		inline void SetStateVariableAt(unsigned int index, unsigned int position,float NewValue1, float NewValue2){
			*(this->StateVars + index*NumberOfVariables + position) = NewValue1;
			*(this->StateVars + index*NumberOfVariables + position + 1) = NewValue2;
		}


		/*!
		 * \brief It sets the state variable in two consecutives position.
		 *
		 * It sets the state variable in a specified position.
		 *
		 * \param index The synapse's index inside the learning rule.
		 * \param position The position of the state variable.
		 * \param NewValue1 The new value of that state variable.
		 * \param NewValue2 The new value of that state variable.
		 * \param NewValue3 The new value of that state variable.
		 */
		inline void SetStateVariableAt(unsigned int index, unsigned int position,float NewValue1, float NewValue2, float NewValue3){
			*(this->StateVars + index*NumberOfVariables + position) = NewValue1;
			*(this->StateVars + index*NumberOfVariables + position + 1) = NewValue2;
			*(this->StateVars + index*NumberOfVariables + position + 2) = NewValue3;
			
		}


		/*!
		 * \brief It gets the state variable in a specified position.
		 *
		 * It gets the state variable in a specified position.
		 *
		 * \param index The synapse's index inside the learning rule.
		 * \param position The position of the state variable.
		 * \return The value of the position-th state variable.
		 */
		//float GetStateVariableAt(int index, int position);
		inline float GetStateVariableAt(unsigned int index, unsigned int position){
			return *(this->StateVars + index*NumberOfVariables + position);
		}

	
		/*!
		 * \brief It multiply the state variable in a specified position by factor.
		 *
		 * It multiply the state variable in a specified position by factor.
		 *
		 * \param index The synapse's index inside the learning rule.
		 * \param position The position of the state variable.
		 * \param factor The multiplier of that state variable.
		 */
		inline void multiplyStateVaraibleAt(unsigned int index, unsigned int position, float factor){
			*(this->StateVars + index*NumberOfVariables + position) *= factor;
		}

	   	/*!
		 * \brief It sets the time when the last update happened.
		 *
		 * It sets the time when the last update happened.
		 *
		 * \param NewUpdateTime The time when the last update happened.
		 */
		//void SetLastUpdateTime(int index, double NewUpdateTime);
		inline void SetLastUpdateTime(unsigned int index, double NewUpdateTime){
			*(this->LastUpdate+index) = NewUpdateTime;
		}

		/*!
		 * \brief It gets the time when the last update happened.
		 *
		 * It gets the time when the last update happened.
		 *
		 * \param index The synapse's index inside the learning rule.
		 * \return The time when the last update happened.
		 */
		//double GetLastUpdateTime(unsigned int index);
		inline double GetLastUpdateTime(unsigned int index){
			return *(this->LastUpdate + index);
		}

				/*!
		 * \brief It gets the value of the accumulated presynaptic activity.
		 *
		 * It gets the value of the accumulated presynaptic activity.
		 *
		 * \param index The synapse's index inside the learning rule.
		 * \return The accumulated presynaptic activity.
		 */
		virtual float GetPresynapticActivity(unsigned int index) = 0;

		/*!
		 * \brief It gets the value of the accumulated postsynaptic activity.
		 *
		 * It gets the value of the accumulated postsynaptic activity.
		 *
		 * \param index The synapse's index inside the learning rule.
		 * \return The accumulated postsynaptic activity.
		 */
		virtual float GetPostsynapticActivity(unsigned int index) = 0;


		/*!
		 * \brief It implements the behaviour when it transmits a spike.
		 *
		 * It implements the behaviour when it transmits a spike. It must be implemented
		 * by any inherited class.
		 *
		 * \param index The synapse's index inside the learning rule.
		 */
		virtual void ApplyPresynapticSpike(unsigned int index) = 0;


		/*!
		 * \brief It increment the state variable in a specified position.
		 *
		 * It increment the state variable in a specified position.
		 *
		 * \param index The synapse's index inside the learning rule.
		 * \param position The position of the state variable.
		 * \param increment The increment of that state variable.
		 */
		inline void incrementStateVaraibleAt(unsigned int index, unsigned int position, float increment){
			*(this->StateVars + index*NumberOfVariables + position) += increment;
		}

		/*!
		 * \brief It implements the behaviour when the target cell fires a spike.
		 *
		 * It implements the behaviour when it the target cell fires a spike. It must be implemented
		 * by any inherited class.
		 *
		 * \param index The synapse's index inside the learning rule.
		 */
		virtual void ApplyPostsynapticSpike(unsigned int index) = 0;


		/*!
		 * \brief It gets the number of state variables.
		 *
		 * It gets the number of state variables.
		 *
		 * \return The number of state variables of this model.
		 */
		unsigned int GetNumberOfVariables();


		/*!
		 * \brief It gets the number of variables that you can print in this state.
		 *
		 * It gets the number of variables that you can print in this state.
		 *
		 * \return The number of variables that you can print in this state.
		 */
		virtual unsigned int GetNumberOfPrintableValues();

		/*!
		 * \brief It gets a value to be printed from this state.
		 *
		 * It gets a value to be printed from this state.
		 * 
		 * \param index The synapse's index inside the learning rule.
		 * \param position Position inside each connection.
		 *
		 * \return The value at position-th position in this state.
		 */
		virtual double GetPrintableValuesAt(unsigned int index, unsigned int position);

};

#endif /* CONNECTIONSTATE_H_ */

