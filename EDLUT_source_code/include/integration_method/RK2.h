/***************************************************************************
 *                           RK2.h                                         *
 *                           -------------------                           *
 * copyright            : (C) 2013 by Francisco Naveros                    *
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

#ifndef RK2_H_
#define RK2_H_

/*!
 * \file RK2.h
 *
 * \author Francisco Naveros
 * \date May 2013
 *
 * This file declares a class which implement the second order Runge Kutta integration method. This class implement a fixed step
 * integration method.
 */

#include "./FixedStep.h"


class TimeDrivenNeuronModel;

/*!
 * \class RK2
 *
 * \brief RK2 integration methods in CPU
 *
 * This class abstracts the behavior of a second order Runge Kutta integration method for neurons in a 
 * time-driven spiking neural network.
 * It includes internal model functions which define the behavior of integration methods
 * (initialization, calculate next value, ...).
 *
 * \author Francisco Naveros
 * \date May 2013
 */
template <class Neuron_Model>

class RK2 : public FixedStep {
	public:

		/*
		* Time driven neuron model
		*/
		Neuron_Model * neuron_model;

		/*!
		 * \brief Constructor with parameters.
		 *
		 * It generates a new second order Runge-Kutta object.
		 *
		 * \param NewModel time driven neuron model associated to this integration method.
		 */
		RK2(Neuron_Model * NewModel) :FixedStep("RK2"), neuron_model(NewModel){
		}
		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		~RK2();
		

		/*!
		 * \brief It calculate the new neural state variables for a defined elapsed_time.
		 *
		 * It calculate the new neural state variables for a defined elapsed_time.
		 *
		 * \param index for method with memory (e.g. BDF1ad, BDF2, BDF3, etc.).
		 * \param NeuronState neuron state variables of one neuron.
		 * \return Retrun if the neuron spike
		 */
		void NextDifferentialEquationValues(int index, float * NeuronState){

			float previous_V = NeuronState[0];

			float AuxNeuronState1[MAX_VARIABLES];
			float AuxNeuronState2[MAX_VARIABLES];


			//1st term
			this->neuron_model->EvaluateDifferentialEquation(NeuronState, AuxNeuronState1, index, elapsedTimeInNeuronModelScale);

			//2nd term
			for (int j = 0; j<this->neuron_model->N_DifferentialNeuronState; j++){
				NeuronState[j] += AuxNeuronState1[j] * elapsedTimeInNeuronModelScale;
			}

			this->neuron_model->EvaluateTimeDependentEquation(NeuronState, index, 0);
			this->neuron_model->EvaluateDifferentialEquation(NeuronState, AuxNeuronState2, index, elapsedTimeInNeuronModelScale);


			for (int j = 0; j<this->neuron_model->N_DifferentialNeuronState; j++){
				NeuronState[j] += (-AuxNeuronState1[j] + AuxNeuronState2[j])*elapsedTimeInNeuronModelScale*0.5f;
			}

			//Update the last spike time.
			this->neuron_model->GetVectorNeuronState()->AddElapsedTime(index, elapsedTimeInSeconds);

			this->neuron_model->EvaluateSpikeCondition(previous_V, NeuronState, index, elapsedTimeInNeuronModelScale);

			//Acumulate the membrane potential in a variable
			this->IncrementValidIntegrationVariable(NeuronState[0]);
		}


		/*!
		* \brief It calculate the new neural state variables for a defined elapsed_time and all the neurons.
		*
		* It calculate the new neural state variables for a defined elapsed_time and all the neurons.
		*
		* \return Retrun if the neuron spike
		*/
		void NextDifferentialEquationValues(){
			float AuxNeuronState1[MAX_VARIABLES];
			float AuxNeuronState2[MAX_VARIABLES];

			for (int index = 0; index < this->neuron_model->State->GetSizeState(); index++){
				float * NeuronState = this->neuron_model->State->GetStateVariableAt(index);
				float previous_V = NeuronState[0];

				//1st term
				this->neuron_model->EvaluateDifferentialEquation(NeuronState, AuxNeuronState1, index, elapsedTimeInNeuronModelScale);

				//2nd term
				for (int j = 0; j<this->neuron_model->N_DifferentialNeuronState; j++){
					NeuronState[j] += AuxNeuronState1[j] * elapsedTimeInNeuronModelScale;
				}

				this->neuron_model->EvaluateTimeDependentEquation(NeuronState, index, 0);
				this->neuron_model->EvaluateDifferentialEquation(NeuronState, AuxNeuronState2, index, elapsedTimeInNeuronModelScale);


				for (int j = 0; j<this->neuron_model->N_DifferentialNeuronState; j++){
					NeuronState[j] += (-AuxNeuronState1[j] + AuxNeuronState2[j])*elapsedTimeInNeuronModelScale*0.5f;
				}

				//Update the last spike time.
				this->neuron_model->GetVectorNeuronState()->AddElapsedTime(index, elapsedTimeInSeconds);

				this->neuron_model->EvaluateSpikeCondition(previous_V, NeuronState, index, elapsedTimeInNeuronModelScale);

				//Acumulate the membrane potential in a variable
				this->IncrementValidIntegrationVariable(NeuronState[0]);
			}
		}


		/*!
		* \brief It calculate the new neural state variables for a defined elapsed_time and all the neurons that require integration.
		*
		* It calculate the new neural state variables for a defined elapsed_time and all the neurons that requre integration.
		*
		* \param integration_required array that sets if a neuron must be integrated (for lethargic neuron models)
		* \return Retrun if the neuron spike
		*/
		void NextDifferentialEquationValues(bool * integration_required, double CurrentTime){

			float AuxNeuronState1[MAX_VARIABLES];
			float AuxNeuronState2[MAX_VARIABLES];

			for (int index = 0; index < this->neuron_model->State->GetSizeState(); index++){
				if (integration_required[index] == true){
					float * NeuronState = this->neuron_model->State->GetStateVariableAt(index);
					float previous_V = NeuronState[0];

					//1st term
					this->neuron_model->EvaluateDifferentialEquation(NeuronState, AuxNeuronState1, index, elapsedTimeInNeuronModelScale);

					//2nd term
					for (int j = 0; j<this->neuron_model->N_DifferentialNeuronState; j++){
						NeuronState[j] += AuxNeuronState1[j] * elapsedTimeInNeuronModelScale;
					}

					this->neuron_model->EvaluateTimeDependentEquation(NeuronState, index, 0);
					this->neuron_model->EvaluateDifferentialEquation(NeuronState, AuxNeuronState2, index, elapsedTimeInNeuronModelScale);


					for (int j = 0; j<this->neuron_model->N_DifferentialNeuronState; j++){
						NeuronState[j] += (-AuxNeuronState1[j] + AuxNeuronState2[j])*elapsedTimeInNeuronModelScale*0.5f;
					}

					//Update the last spike time.
					this->neuron_model->GetVectorNeuronState()->AddElapsedTime(index, elapsedTimeInSeconds);

					this->neuron_model->EvaluateSpikeCondition(previous_V, NeuronState, index, elapsedTimeInNeuronModelScale);

					//Set last update time for the analytic resolution of the differential equations in lethargic models 
					this->neuron_model->State->SetLastUpdateTime(index, CurrentTime);

					//Acumulate the membrane potential in a variable
					this->IncrementValidIntegrationVariable(NeuronState[0]);
				}
			}
		}


		/*!
		 * \brief It prints the integration method info.
		 *
		 * It prints the current integration method characteristics.
		 *
		 * \param out The stream where it prints the information.
		 *
		 * \return The stream after the printer.
		 */
		virtual ostream & PrintInfo(ostream & out){
			out << "Integration Method Type: " << this->GetType() << endl;

			return out;
		}


		/*!
		 * \brief It initialize the state of the integration method for method with memory (e.g. BDF1ad, BDF2, BDF3, etc.).
		 *
		 * It initialize the state of the integration method for method with memory (e.g. BDF1ad, BDF2, BDF3, etc.).
		 *
		 * \param N_neuron number of neurons in the neuron model.
		 * \param inicialization vector with initial values.
		 */
		void InitializeStates(int N_neurons, float * initialization){};

		
		/*!
		 * \brief It reset the state of the integration method for method with memory (e.g. BDF1ad, BDF2, BDF3, etc.).
		 *
		 * It reset the state of the integration method for method with memory (e.g. BDF1ad, BDF2, BDF3, etc.).
		 *
		 * \param index indicate which neuron must be reseted.
		 */
		void resetState(int index){};

		/*!
		 * \brief It calculates the conductance exponential values for time driven neuron models.
		 *
		 * It calculates the conductance exponential values for time driven neuron models.
		 */
		void Calculate_conductance_exp_values(){
			this->neuron_model->Initialize_conductance_exp_values(this->neuron_model->N_TimeDependentNeuronState, 1);
			//index 0
			this->neuron_model->Calculate_conductance_exp_values(0, elapsedTimeInNeuronModelScale);
		}
};

#endif /* RK2_H_ */
