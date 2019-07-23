/***************************************************************************
 *                           Bifixed_BDFn.h                                *
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

#ifndef BIFIXED_BDFn_H_
#define BIFIXED_BDFn_H_

/*!
 * \file Bifixed_BDFn.h
 *
 * \author Francisco Naveros
 * \date May 2015
 *
 * This file declares a class which implement two BDF (Backward Differentiation Formulas) integration methods 
 * (first and second order multi step BDF integration method). This method implements a progressive implementation of the
 * higher order integration method using the lower order integration mehtod (BDF1->BDF2). This class 
 * implement a multi step integration method.
 */

#include "./BiFixedStep.h"

class TimeDrivenNeuronModel;

/*!
 * \class Bifixed_BDFn
 *
 * \brief Bifixed_BDFn integration methods in CPU
 *
 * This class abstracts the behavior of BDF1 and BDF2 integration methods for neurons in a 
 * time-driven spiking neural network.
 * It includes internal model functions which define the behavior of integration methods
 * (initialization, calculate next value, ...).
 *
 * \author Francisco Naveros
 * \date May 2015
 */
template <class Neuron_Model>

class Bifixed_BDFn : public BiFixedStep {
	public:

		/*
		* Time driven neuron model
		*/
		Neuron_Model * neuron_model;

		/*!
		 * \brief This vector stores previous neuron state variable for all neurons. This one is used as a memory.
		*/
		float ** PreviousNeuronState;

		/*!
		 * \brief This vector stores the difference between previous neuron state variable for all neurons. This 
		 * one is used as a memory.
		*/
		float ** D;

		/*!
		 * \brief This constant matrix stores the coefficients of each BDF order. Additionally, the coeficiente 
		 * that must be used depend on the previous and actual integration step.
		*/
		float Coeficient [3][3][4];//decrement of h, equal h, increment of h

		/*!
		 * \brief Integration step used in the previous integration
		*/
		float * PreviousIntegrationStep;

		/*!
		 * \brief This variable indicate which BDF coeficients must be used in funcion of the previous and actual integration step.
		*/
		int * CoeficientSelector;

		/*!
		 * \brief This vector contains the state of each neuron (BDF order). When the integration method is reseted (the values of the neuron model variables are
		 * changed outside the integration method, for instance when a neuron spikes and the membrane potential is reseted to the resting potential), the values
		 * store in PreviousNeuronState and D are no longer valid. In this case the order it is set to 0 and must grow in each integration step until it is reache
		 * the target order.
		*/
		int * state;

		/*!
		 * \brief This value stores the order of the integration method.
		*/
		int BDForder;


		/*!
		 * \brief Constructor of the class with parameters.
		 *
		 * It generates a new BDF object indicating the order of the method.
		 *
		 * \param NewModel time driven neuron model associated to this integration method.
		 * \param BDForder BDF order (1 or 2).
		 */
		Bifixed_BDFn(Neuron_Model * NewModel, int BDForder) :BiFixedStep("Bifixed_BDFn"), BDForder(BDForder), neuron_model(NewModel){
		}

		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		~Bifixed_BDFn(){
			if (BDForder>1){
				for (int i = 0; i<BDForder - 1; i++){
					delete[] PreviousNeuronState[i];
				}
				delete[] PreviousNeuronState;
			}

			for (int i = 0; i<BDForder; i++){
				delete[] D[i];
			}
			delete[] D;
			delete[] state;

			delete PreviousIntegrationStep;
			delete CoeficientSelector;
		}
		

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
			
			float AuxNeuronState[MAX_VARIABLES];
			float AuxNeuronState1[MAX_VARIABLES];
			float AuxNeuronState_p[MAX_VARIABLES];
			float AuxNeuronState_p1[MAX_VARIABLES];
			float AuxNeuronState_c[MAX_VARIABLES];
			float jacnum[MAX_VARIABLES*MAX_VARIABLES];
			float J[MAX_VARIABLES*MAX_VARIABLES];
			float inv_J[MAX_VARIABLES*MAX_VARIABLES];

			if (integrationMethodState[index] == 0){
				//If the state of the cell is 0, we use a Euler method to calculate an aproximation of the solution.
				if (state[index] == 0){
					this->neuron_model->EvaluateDifferentialEquation(NeuronState, AuxNeuronState, index, elapsedTimeInNeuronModelScale);
					for (int j = 0; j<this->neuron_model->N_DifferentialNeuronState; j++){
						AuxNeuronState_p[j] = NeuronState[j] + elapsedTimeInNeuronModelScale*AuxNeuronState[j];
					}
				}
				//In this case we use the value of previous states to calculate an aproximation of the solution.
				else{
					float inverseIntegrationStepFactor = PreviousIntegrationStep[index] / elapsedTimeInNeuronModelScale;
					for (int j = 0; j<this->neuron_model->N_DifferentialNeuronState; j++){
						AuxNeuronState_p[j] = NeuronState[j];
						for (int i = 0; i<state[index]; i++){
							AuxNeuronState_p[j] += D[i][index*this->neuron_model->N_DifferentialNeuronState + j]/*/inverseIntegrationStepFactor*/;
						}
					}
				}

				for (int i = this->neuron_model->N_DifferentialNeuronState; i<this->neuron_model->N_NeuronStateVariables; i++){
					AuxNeuronState_p[i] = NeuronState[i];
				}


				this->neuron_model->EvaluateTimeDependentEquation(AuxNeuronState_p, index, 0);



				float epsi = 1.0f;
				int k = 0;

				this->neuron_model->EvaluateDifferentialEquation(NeuronState, AuxNeuronState1, index, elapsedTimeInNeuronModelScale);

				//This integration method is an implicit method. We use this loop to iteratively calculate the implicit value.
				//epsi is the difference between two consecutive aproximation of the implicit method. 
				while (epsi>1e-16 && k<5){
					this->neuron_model->EvaluateDifferentialEquation(AuxNeuronState_p, AuxNeuronState, index, elapsedTimeInNeuronModelScale);

					for (int j = 0; j<this->neuron_model->N_DifferentialNeuronState; j++){
						AuxNeuronState_c[j] = Coeficient[CoeficientSelector[index]][state[index]][0] * elapsedTimeInNeuronModelScale*AuxNeuronState[j] + Coeficient[CoeficientSelector[index]][state[index]][1] * elapsedTimeInNeuronModelScale*AuxNeuronState1[j] + Coeficient[CoeficientSelector[index]][state[index]][2] * NeuronState[j];
						for (int i = 1; i<state[index]; i++){
							AuxNeuronState_c[j] += Coeficient[CoeficientSelector[index]][state[index]][i + 2] * PreviousNeuronState[i - 1][index*this->neuron_model->N_DifferentialNeuronState + j];
						}
					}

					//jacobian.
					Jacobian(AuxNeuronState_p, jacnum, index);

					for (int z = 0; z<this->neuron_model->N_DifferentialNeuronState; z++){
						for (int t = 0; t<this->neuron_model->N_DifferentialNeuronState; t++){
							J[z*this->neuron_model->N_DifferentialNeuronState + t] = Coeficient[CoeficientSelector[index]][state[index]][0] * elapsedTimeInNeuronModelScale * jacnum[z*this->neuron_model->N_DifferentialNeuronState + t];
							if (z == t){
								J[z*this->neuron_model->N_DifferentialNeuronState + t] -= 1;
							}
						}
					}

					this->invermat(J, inv_J);

					for (int z = 0; z<this->neuron_model->N_DifferentialNeuronState; z++){
						float aux = 0.0;
						for (int t = 0; t<this->neuron_model->N_DifferentialNeuronState; t++){
							aux += inv_J[z*this->neuron_model->N_DifferentialNeuronState + t] * (AuxNeuronState_p[t] - AuxNeuronState_c[t]);
						}
						AuxNeuronState_p1[z] = aux + AuxNeuronState_p[z];
					}

					//We calculate the difference between both aproximations.
					float aux = 0.0f;
					float aux2 = 0.0f;
					for (int z = 0; z<this->neuron_model->N_DifferentialNeuronState; z++){
						aux = fabs(AuxNeuronState_p1[z] - AuxNeuronState_p[z]);
						if (aux>aux2){
							aux2 = aux;
						}
					}

					memcpy(AuxNeuronState_p, AuxNeuronState_p1, sizeof(float)* this->neuron_model->N_DifferentialNeuronState);

					epsi = aux2;
					k++;
				}

				if (NeuronState[0]>startVoltageThreshold){
					integrationMethodState[index] = 1;

					//Restore the neuron model state to a previous state.
					this->neuron_model->GetVectorNeuronState()->AddElapsedTime(index, -elapsedTimeInSeconds);

					//Comes form a small step and goes to a small step
					if (CoeficientSelector[index] == 2){
						CoeficientSelector[index] = 1;
					}
					else{//goes to a smaller step.
						CoeficientSelector[index] = 0;
					}
				}
				else{
					//We increase the state of the integration method.
					if (state[index]<BDForder){
						state[index]++;
					}

					//We acumulate these new values for the next step.
					for (int j = 0; j<this->neuron_model->N_DifferentialNeuronState; j++){

						for (int i = (state[index] - 1); i>0; i--){
							D[i][index*this->neuron_model->N_DifferentialNeuronState + j] = -D[i - 1][index*this->neuron_model->N_DifferentialNeuronState + j];
						}
						D[0][index*this->neuron_model->N_DifferentialNeuronState + j] = AuxNeuronState_p[j] - NeuronState[j];
						for (int i = 1; i<state[index]; i++){
							D[i][index*this->neuron_model->N_DifferentialNeuronState + j] += D[i - 1][index*this->neuron_model->N_DifferentialNeuronState + j];
						}
					}

					if (state[index]>1){
						for (int i = state[index] - 2; i>0; i--){
							memcpy(PreviousNeuronState[i] + (index*this->neuron_model->N_DifferentialNeuronState), PreviousNeuronState[i - 1] + (index*this->neuron_model->N_DifferentialNeuronState), sizeof(float)* this->neuron_model->N_DifferentialNeuronState);
						}

						memcpy(PreviousNeuronState[0] + (index*this->neuron_model->N_DifferentialNeuronState), NeuronState, sizeof(float)* this->neuron_model->N_DifferentialNeuronState);
					}
					memcpy(NeuronState, AuxNeuronState_p, sizeof(float)* this->neuron_model->N_DifferentialNeuronState);



					//Finaly, we evaluate the neural state variables with time dependence.
					this->neuron_model->EvaluateTimeDependentEquation(NeuronState, index, 0);

					//update the integration step size.
					PreviousIntegrationStep[index] = elapsedTimeInNeuronModelScale;

					//Set the coeficient selector to 1 for the next iteration.
					CoeficientSelector[index] = 1;

					//Update the last spike time.
					this->neuron_model->GetVectorNeuronState()->AddElapsedTime(index, elapsedTimeInSeconds);

					//Acumulate the membrane potential in a variable
					this->IncrementValidIntegrationVariable(NeuronState[0]);
				}
			}

			if (integrationMethodState[index]>0){

				for (int iteration = 0; iteration<ratioLargerSmallerSteps; iteration++){
					float previous_V = NeuronState[0];

					//If the state of the cell is 0, we use a Euler method to calculate an aproximation of the solution.
					if (state[index] == 0){
						this->neuron_model->EvaluateDifferentialEquation(NeuronState, AuxNeuronState, index, bifixedElapsedTimeInNeuronModelScale);
						for (int j = 0; j<this->neuron_model->N_DifferentialNeuronState; j++){
							AuxNeuronState_p[j] = NeuronState[j] + bifixedElapsedTimeInNeuronModelScale*AuxNeuronState[j];
						}
					}
					//In this case we use the value of previous states to calculate an aproximation of the solution.
					else{
						float inverseIntegrationStepFactor = PreviousIntegrationStep[index] / bifixedElapsedTimeInNeuronModelScale;
						for (int j = 0; j<this->neuron_model->N_DifferentialNeuronState; j++){
							AuxNeuronState_p[j] = NeuronState[j];
							for (int i = 0; i<state[index]; i++){
								AuxNeuronState_p[j] += D[i][index*this->neuron_model->N_DifferentialNeuronState + j]/*/inverseIntegrationStepFactor*/;
							}
						}
					}

					for (int i = this->neuron_model->N_DifferentialNeuronState; i<this->neuron_model->N_NeuronStateVariables; i++){
						AuxNeuronState_p[i] = NeuronState[i];
					}

					this->neuron_model->EvaluateTimeDependentEquation(AuxNeuronState_p, index, 1);

					float epsi = 1.0f;
					int k = 0;


					this->neuron_model->EvaluateDifferentialEquation(NeuronState, AuxNeuronState1, index, bifixedElapsedTimeInNeuronModelScale);

					//This integration method is an implicit method. We use this loop to iteratively calculate the implicit value.
					//epsi is the difference between two consecutive aproximation of the implicit method. 
					while (epsi>1e-16 && k<5){
						this->neuron_model->EvaluateDifferentialEquation(AuxNeuronState_p, AuxNeuronState, index, bifixedElapsedTimeInNeuronModelScale);

						for (int j = 0; j<this->neuron_model->N_DifferentialNeuronState; j++){
							AuxNeuronState_c[j] = Coeficient[CoeficientSelector[index]][state[index]][0] * bifixedElapsedTimeInNeuronModelScale*AuxNeuronState[j] + Coeficient[CoeficientSelector[index]][state[index]][1] * bifixedElapsedTimeInNeuronModelScale*AuxNeuronState1[j] + Coeficient[CoeficientSelector[index]][state[index]][2] * NeuronState[j];
							for (int i = 1; i<state[index]; i++){
								AuxNeuronState_c[j] += Coeficient[CoeficientSelector[index]][state[index]][i + 2] * PreviousNeuronState[i - 1][index*this->neuron_model->N_DifferentialNeuronState + j];
							}
						}

						//jacobian.
						Jacobian(AuxNeuronState_p, jacnum, index);

						for (int z = 0; z<this->neuron_model->N_DifferentialNeuronState; z++){
							for (int t = 0; t<this->neuron_model->N_DifferentialNeuronState; t++){
								J[z*this->neuron_model->N_DifferentialNeuronState + t] = Coeficient[CoeficientSelector[index]][state[index]][0] * bifixedElapsedTimeInNeuronModelScale * jacnum[z*this->neuron_model->N_DifferentialNeuronState + t];
								if (z == t){
									J[z*this->neuron_model->N_DifferentialNeuronState + t] -= 1;
								}
							}
						}

						this->invermat(J, inv_J);

						for (int z = 0; z<this->neuron_model->N_DifferentialNeuronState; z++){
							float aux = 0.0;
							for (int t = 0; t<this->neuron_model->N_DifferentialNeuronState; t++){
								aux += inv_J[z*this->neuron_model->N_DifferentialNeuronState + t] * (AuxNeuronState_p[t] - AuxNeuronState_c[t]);
							}
							AuxNeuronState_p1[z] = aux + AuxNeuronState_p[z];
						}

						//We calculate the difference between both aproximations.
						float aux = 0.0f;
						float aux2 = 0.0f;
						for (int z = 0; z<this->neuron_model->N_DifferentialNeuronState; z++){
							aux = fabs(AuxNeuronState_p1[z] - AuxNeuronState_p[z]);
							if (aux>aux2){
								aux2 = aux;
							}
						}

						memcpy(AuxNeuronState_p, AuxNeuronState_p1, sizeof(float)* this->neuron_model->N_DifferentialNeuronState);

						epsi = aux2;
						k++;
					}

					//We increase the state of the integration method.
					if (state[index]<BDForder){
						state[index]++;
					}


					//We acumulate these new values for the next step.
					for (int j = 0; j<this->neuron_model->N_DifferentialNeuronState; j++){

						for (int i = (state[index] - 1); i>0; i--){
							D[i][index*this->neuron_model->N_DifferentialNeuronState + j] = -D[i - 1][index*this->neuron_model->N_DifferentialNeuronState + j];
						}
						D[0][index*this->neuron_model->N_DifferentialNeuronState + j] = AuxNeuronState_p[j] - NeuronState[j];
						for (int i = 1; i<state[index]; i++){
							D[i][index*this->neuron_model->N_DifferentialNeuronState + j] += D[i - 1][index*this->neuron_model->N_DifferentialNeuronState + j];
						}
					}

					if (state[index]>1){
						for (int i = state[index] - 2; i>0; i--){
							memcpy(PreviousNeuronState[i] + (index*this->neuron_model->N_DifferentialNeuronState), PreviousNeuronState[i - 1] + (index*this->neuron_model->N_DifferentialNeuronState), sizeof(float)* this->neuron_model->N_DifferentialNeuronState);
						}

						memcpy(PreviousNeuronState[0] + (index*this->neuron_model->N_DifferentialNeuronState), NeuronState, sizeof(float)* this->neuron_model->N_DifferentialNeuronState);
					}
					memcpy(NeuronState, AuxNeuronState_p, sizeof(float)* this->neuron_model->N_DifferentialNeuronState);



					//Finaly, we evaluate the neural state variables with time dependence.
					this->neuron_model->EvaluateTimeDependentEquation(NeuronState, index, 1);

					//Update the last spike time.
					this->neuron_model->GetVectorNeuronState()->AddElapsedTime(index, bifixedElapsedTimeInSeconds);

					this->neuron_model->EvaluateSpikeCondition(previous_V, NeuronState, index, bifixedElapsedTimeInNeuronModelScale);

					//Acumulate the membrane potential in a variable
					this->IncrementValidIntegrationVariable(NeuronState[0]);

					//Set the CoeficientSelector to 1.
					CoeficientSelector[index] = 1;
					PreviousIntegrationStep[index] = bifixedElapsedTimeInNeuronModelScale;

					if (NeuronState[0]>startVoltageThreshold && integrationMethodState[index] == 1){
						integrationMethodState[index] = 2;
					}
					else if (NeuronState[0]<endVoltageThreshold && integrationMethodState[index] == 2){
						integrationMethodState[index] = 3;
						integrationMethodCounter[index] = N_postBiFixedSteps;
					}
					if (integrationMethodCounter[index]>0 && integrationMethodState[index] == 3){
						integrationMethodCounter[index]--;
						if (integrationMethodCounter[index] == 0){
							integrationMethodState[index] = 0;
							CoeficientSelector[index] = 2;
						}
					}
				}
				if (integrationMethodState[index] == 1){
					integrationMethodState[index] = 0;
					CoeficientSelector[index] = 2;
				}

			}
		}



		/*!
		* \brief It calculate the new neural state variables for a defined elapsed_time and all the neurons.
		*
		* It calculate the new neural state variables for a defined elapsed_time and all the neurons.
		*
		* \return Retrun if the neuron spike
		*/
		void NextDifferentialEquationValues(){

			float AuxNeuronState[MAX_VARIABLES];
			float AuxNeuronState1[MAX_VARIABLES];
			float AuxNeuronState_p[MAX_VARIABLES];
			float AuxNeuronState_p1[MAX_VARIABLES];
			float AuxNeuronState_c[MAX_VARIABLES];
			float jacnum[MAX_VARIABLES*MAX_VARIABLES];
			float J[MAX_VARIABLES*MAX_VARIABLES];
			float inv_J[MAX_VARIABLES*MAX_VARIABLES];

			for (int index = 0; index < this->neuron_model->State->GetSizeState(); index++){
				float * NeuronState = this->neuron_model->State->GetStateVariableAt(index);
				if (integrationMethodState[index] == 0){
					//If the state of the cell is 0, we use a Euler method to calculate an aproximation of the solution.
					if (state[index] == 0){
						this->neuron_model->EvaluateDifferentialEquation(NeuronState, AuxNeuronState, index, elapsedTimeInNeuronModelScale);
						for (int j = 0; j < this->neuron_model->N_DifferentialNeuronState; j++){
							AuxNeuronState_p[j] = NeuronState[j] + elapsedTimeInNeuronModelScale*AuxNeuronState[j];
						}
					}
					//In this case we use the value of previous states to calculate an aproximation of the solution.
					else{
						float inverseIntegrationStepFactor = PreviousIntegrationStep[index] / elapsedTimeInNeuronModelScale;
						for (int j = 0; j < this->neuron_model->N_DifferentialNeuronState; j++){
							AuxNeuronState_p[j] = NeuronState[j];
							for (int i = 0; i < state[index]; i++){
								AuxNeuronState_p[j] += D[i][index*this->neuron_model->N_DifferentialNeuronState + j]/*/inverseIntegrationStepFactor*/;
							}
						}
					}

					for (int i = this->neuron_model->N_DifferentialNeuronState; i < this->neuron_model->N_NeuronStateVariables; i++){
						AuxNeuronState_p[i] = NeuronState[i];
					}


					this->neuron_model->EvaluateTimeDependentEquation(AuxNeuronState_p, index, 0);



					float epsi = 1.0f;
					int k = 0;

					this->neuron_model->EvaluateDifferentialEquation(NeuronState, AuxNeuronState1, index, elapsedTimeInNeuronModelScale);

					//This integration method is an implicit method. We use this loop to iteratively calculate the implicit value.
					//epsi is the difference between two consecutive aproximation of the implicit method. 
					while (epsi > 1e-16 && k < 5){
						this->neuron_model->EvaluateDifferentialEquation(AuxNeuronState_p, AuxNeuronState, index, elapsedTimeInNeuronModelScale);

						for (int j = 0; j < this->neuron_model->N_DifferentialNeuronState; j++){
							AuxNeuronState_c[j] = Coeficient[CoeficientSelector[index]][state[index]][0] * elapsedTimeInNeuronModelScale*AuxNeuronState[j] + Coeficient[CoeficientSelector[index]][state[index]][1] * elapsedTimeInNeuronModelScale*AuxNeuronState1[j] + Coeficient[CoeficientSelector[index]][state[index]][2] * NeuronState[j];
							for (int i = 1; i < state[index]; i++){
								AuxNeuronState_c[j] += Coeficient[CoeficientSelector[index]][state[index]][i + 2] * PreviousNeuronState[i - 1][index*this->neuron_model->N_DifferentialNeuronState + j];
							}
						}

						//jacobian.
						Jacobian(AuxNeuronState_p, jacnum, index);

						for (int z = 0; z < this->neuron_model->N_DifferentialNeuronState; z++){
							for (int t = 0; t < this->neuron_model->N_DifferentialNeuronState; t++){
								J[z*this->neuron_model->N_DifferentialNeuronState + t] = Coeficient[CoeficientSelector[index]][state[index]][0] * elapsedTimeInNeuronModelScale * jacnum[z*this->neuron_model->N_DifferentialNeuronState + t];
								if (z == t){
									J[z*this->neuron_model->N_DifferentialNeuronState + t] -= 1;
								}
							}
						}

						this->invermat(J, inv_J);

						for (int z = 0; z < this->neuron_model->N_DifferentialNeuronState; z++){
							float aux = 0.0;
							for (int t = 0; t < this->neuron_model->N_DifferentialNeuronState; t++){
								aux += inv_J[z*this->neuron_model->N_DifferentialNeuronState + t] * (AuxNeuronState_p[t] - AuxNeuronState_c[t]);
							}
							AuxNeuronState_p1[z] = aux + AuxNeuronState_p[z];
						}

						//We calculate the difference between both aproximations.
						float aux = 0.0f;
						float aux2 = 0.0f;
						for (int z = 0; z < this->neuron_model->N_DifferentialNeuronState; z++){
							aux = fabs(AuxNeuronState_p1[z] - AuxNeuronState_p[z]);
							if (aux > aux2){
								aux2 = aux;
							}
						}

						memcpy(AuxNeuronState_p, AuxNeuronState_p1, sizeof(float)* this->neuron_model->N_DifferentialNeuronState);

						epsi = aux2;
						k++;
					}

					if (NeuronState[0] > startVoltageThreshold){
						integrationMethodState[index] = 1;

						//Restore the neuron model state to a previous state.
						this->neuron_model->GetVectorNeuronState()->AddElapsedTime(index, -elapsedTimeInSeconds);

						//Comes form a small step and goes to a small step
						if (CoeficientSelector[index] == 2){
							CoeficientSelector[index] = 1;
						}
						else{//goes to a smaller step.
							CoeficientSelector[index] = 0;
						}
					}
					else{
						//We increase the state of the integration method.
						if (state[index] < BDForder){
							state[index]++;
						}

						//We acumulate these new values for the next step.
						for (int j = 0; j < this->neuron_model->N_DifferentialNeuronState; j++){

							for (int i = (state[index] - 1); i > 0; i--){
								D[i][index*this->neuron_model->N_DifferentialNeuronState + j] = -D[i - 1][index*this->neuron_model->N_DifferentialNeuronState + j];
							}
							D[0][index*this->neuron_model->N_DifferentialNeuronState + j] = AuxNeuronState_p[j] - NeuronState[j];
							for (int i = 1; i < state[index]; i++){
								D[i][index*this->neuron_model->N_DifferentialNeuronState + j] += D[i - 1][index*this->neuron_model->N_DifferentialNeuronState + j];
							}
						}

						if (state[index]>1){
							for (int i = state[index] - 2; i > 0; i--){
								memcpy(PreviousNeuronState[i] + (index*this->neuron_model->N_DifferentialNeuronState), PreviousNeuronState[i - 1] + (index*this->neuron_model->N_DifferentialNeuronState), sizeof(float)* this->neuron_model->N_DifferentialNeuronState);
							}

							memcpy(PreviousNeuronState[0] + (index*this->neuron_model->N_DifferentialNeuronState), NeuronState, sizeof(float)* this->neuron_model->N_DifferentialNeuronState);
						}
						memcpy(NeuronState, AuxNeuronState_p, sizeof(float)* this->neuron_model->N_DifferentialNeuronState);



						//Finaly, we evaluate the neural state variables with time dependence.
						this->neuron_model->EvaluateTimeDependentEquation(NeuronState, index, 0);

						//update the integration step size.
						PreviousIntegrationStep[index] = elapsedTimeInNeuronModelScale;

						//Set the coeficient selector to 1 for the next iteration.
						CoeficientSelector[index] = 1;

						//Update the last spike time.
						this->neuron_model->GetVectorNeuronState()->AddElapsedTime(index, elapsedTimeInSeconds);

						//Acumulate the membrane potential in a variable
						this->IncrementValidIntegrationVariable(NeuronState[0]);
					}
				}

				if (integrationMethodState[index] > 0){

					for (int iteration = 0; iteration < ratioLargerSmallerSteps; iteration++){
						float previous_V = NeuronState[0];

						//If the state of the cell is 0, we use a Euler method to calculate an aproximation of the solution.
						if (state[index] == 0){
							this->neuron_model->EvaluateDifferentialEquation(NeuronState, AuxNeuronState, index, bifixedElapsedTimeInNeuronModelScale);
							for (int j = 0; j < this->neuron_model->N_DifferentialNeuronState; j++){
								AuxNeuronState_p[j] = NeuronState[j] + bifixedElapsedTimeInNeuronModelScale*AuxNeuronState[j];
							}
						}
						//In this case we use the value of previous states to calculate an aproximation of the solution.
						else{
							float inverseIntegrationStepFactor = PreviousIntegrationStep[index] / bifixedElapsedTimeInNeuronModelScale;
							for (int j = 0; j < this->neuron_model->N_DifferentialNeuronState; j++){
								AuxNeuronState_p[j] = NeuronState[j];
								for (int i = 0; i < state[index]; i++){
									AuxNeuronState_p[j] += D[i][index*this->neuron_model->N_DifferentialNeuronState + j]/*/inverseIntegrationStepFactor*/;
								}
							}
						}

						for (int i = this->neuron_model->N_DifferentialNeuronState; i < this->neuron_model->N_NeuronStateVariables; i++){
							AuxNeuronState_p[i] = NeuronState[i];
						}

						this->neuron_model->EvaluateTimeDependentEquation(AuxNeuronState_p, index, 1);

						float epsi = 1.0f;
						int k = 0;


						this->neuron_model->EvaluateDifferentialEquation(NeuronState, AuxNeuronState1, index, bifixedElapsedTimeInNeuronModelScale);

						//This integration method is an implicit method. We use this loop to iteratively calculate the implicit value.
						//epsi is the difference between two consecutive aproximation of the implicit method. 
						while (epsi > 1e-16 && k < 5){
							this->neuron_model->EvaluateDifferentialEquation(AuxNeuronState_p, AuxNeuronState, index, bifixedElapsedTimeInNeuronModelScale);

							for (int j = 0; j < this->neuron_model->N_DifferentialNeuronState; j++){
								AuxNeuronState_c[j] = Coeficient[CoeficientSelector[index]][state[index]][0] * bifixedElapsedTimeInNeuronModelScale*AuxNeuronState[j] + Coeficient[CoeficientSelector[index]][state[index]][1] * bifixedElapsedTimeInNeuronModelScale*AuxNeuronState1[j] + Coeficient[CoeficientSelector[index]][state[index]][2] * NeuronState[j];
								for (int i = 1; i < state[index]; i++){
									AuxNeuronState_c[j] += Coeficient[CoeficientSelector[index]][state[index]][i + 2] * PreviousNeuronState[i - 1][index*this->neuron_model->N_DifferentialNeuronState + j];
								}
							}

							//jacobian.
							Jacobian(AuxNeuronState_p, jacnum, index);

							for (int z = 0; z < this->neuron_model->N_DifferentialNeuronState; z++){
								for (int t = 0; t < this->neuron_model->N_DifferentialNeuronState; t++){
									J[z*this->neuron_model->N_DifferentialNeuronState + t] = Coeficient[CoeficientSelector[index]][state[index]][0] * bifixedElapsedTimeInNeuronModelScale * jacnum[z*this->neuron_model->N_DifferentialNeuronState + t];
									if (z == t){
										J[z*this->neuron_model->N_DifferentialNeuronState + t] -= 1;
									}
								}
							}

							this->invermat(J, inv_J);

							for (int z = 0; z < this->neuron_model->N_DifferentialNeuronState; z++){
								float aux = 0.0;
								for (int t = 0; t < this->neuron_model->N_DifferentialNeuronState; t++){
									aux += inv_J[z*this->neuron_model->N_DifferentialNeuronState + t] * (AuxNeuronState_p[t] - AuxNeuronState_c[t]);
								}
								AuxNeuronState_p1[z] = aux + AuxNeuronState_p[z];
							}

							//We calculate the difference between both aproximations.
							float aux = 0.0f;
							float aux2 = 0.0f;
							for (int z = 0; z < this->neuron_model->N_DifferentialNeuronState; z++){
								aux = fabs(AuxNeuronState_p1[z] - AuxNeuronState_p[z]);
								if (aux > aux2){
									aux2 = aux;
								}
							}

							memcpy(AuxNeuronState_p, AuxNeuronState_p1, sizeof(float)* this->neuron_model->N_DifferentialNeuronState);

							epsi = aux2;
							k++;
						}

						//We increase the state of the integration method.
						if (state[index] < BDForder){
							state[index]++;
						}


						//We acumulate these new values for the next step.
						for (int j = 0; j < this->neuron_model->N_DifferentialNeuronState; j++){

							for (int i = (state[index] - 1); i > 0; i--){
								D[i][index*this->neuron_model->N_DifferentialNeuronState + j] = -D[i - 1][index*this->neuron_model->N_DifferentialNeuronState + j];
							}
							D[0][index*this->neuron_model->N_DifferentialNeuronState + j] = AuxNeuronState_p[j] - NeuronState[j];
							for (int i = 1; i < state[index]; i++){
								D[i][index*this->neuron_model->N_DifferentialNeuronState + j] += D[i - 1][index*this->neuron_model->N_DifferentialNeuronState + j];
							}
						}

						if (state[index]>1){
							for (int i = state[index] - 2; i > 0; i--){
								memcpy(PreviousNeuronState[i] + (index*this->neuron_model->N_DifferentialNeuronState), PreviousNeuronState[i - 1] + (index*this->neuron_model->N_DifferentialNeuronState), sizeof(float)* this->neuron_model->N_DifferentialNeuronState);
							}

							memcpy(PreviousNeuronState[0] + (index*this->neuron_model->N_DifferentialNeuronState), NeuronState, sizeof(float)* this->neuron_model->N_DifferentialNeuronState);
						}
						memcpy(NeuronState, AuxNeuronState_p, sizeof(float)* this->neuron_model->N_DifferentialNeuronState);



						//Finaly, we evaluate the neural state variables with time dependence.
						this->neuron_model->EvaluateTimeDependentEquation(NeuronState, index, 1);

						//Update the last spike time.
						this->neuron_model->GetVectorNeuronState()->AddElapsedTime(index, bifixedElapsedTimeInSeconds);

						this->neuron_model->EvaluateSpikeCondition(previous_V, NeuronState, index, bifixedElapsedTimeInNeuronModelScale);

						//Acumulate the membrane potential in a variable
						this->IncrementValidIntegrationVariable(NeuronState[0]);
						
						//Set the CoeficientSelector to 1.
						CoeficientSelector[index] = 1;
						PreviousIntegrationStep[index] = bifixedElapsedTimeInNeuronModelScale;

						if (NeuronState[0] > startVoltageThreshold && integrationMethodState[index] == 1){
							integrationMethodState[index] = 2;
						}
						else if (NeuronState[0] < endVoltageThreshold && integrationMethodState[index] == 2){
							integrationMethodState[index] = 3;
							integrationMethodCounter[index] = N_postBiFixedSteps;
						}
						if (integrationMethodCounter[index]>0 && integrationMethodState[index] == 3){
							integrationMethodCounter[index]--;
							if (integrationMethodCounter[index] == 0){
								integrationMethodState[index] = 0;
								CoeficientSelector[index] = 2;
							}
						}
					}
					if (integrationMethodState[index] == 1){
						integrationMethodState[index] = 0;
						CoeficientSelector[index] = 2;
					}

				}
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
			
			float AuxNeuronState[MAX_VARIABLES];
			float AuxNeuronState1[MAX_VARIABLES];
			float AuxNeuronState_p[MAX_VARIABLES];
			float AuxNeuronState_p1[MAX_VARIABLES];
			float AuxNeuronState_c[MAX_VARIABLES];
			float jacnum[MAX_VARIABLES*MAX_VARIABLES];
			float J[MAX_VARIABLES*MAX_VARIABLES];
			float inv_J[MAX_VARIABLES*MAX_VARIABLES];

			for (int index = 0; index < this->neuron_model->State->GetSizeState(); index++){
				if (integration_required[index] == true){
					float * NeuronState = this->neuron_model->State->GetStateVariableAt(index);
					if (integrationMethodState[index] == 0){
						//If the state of the cell is 0, we use a Euler method to calculate an aproximation of the solution.
						if (state[index] == 0){
							this->neuron_model->EvaluateDifferentialEquation(NeuronState, AuxNeuronState, index, elapsedTimeInNeuronModelScale);
							for (int j = 0; j < this->neuron_model->N_DifferentialNeuronState; j++){
								AuxNeuronState_p[j] = NeuronState[j] + elapsedTimeInNeuronModelScale*AuxNeuronState[j];
							}
						}
						//In this case we use the value of previous states to calculate an aproximation of the solution.
						else{
							float inverseIntegrationStepFactor = PreviousIntegrationStep[index] / elapsedTimeInNeuronModelScale;
							for (int j = 0; j < this->neuron_model->N_DifferentialNeuronState; j++){
								AuxNeuronState_p[j] = NeuronState[j];
								for (int i = 0; i < state[index]; i++){
									AuxNeuronState_p[j] += D[i][index*this->neuron_model->N_DifferentialNeuronState + j]/*/inverseIntegrationStepFactor*/;
								}
							}
						}

						for (int i = this->neuron_model->N_DifferentialNeuronState; i < this->neuron_model->N_NeuronStateVariables; i++){
							AuxNeuronState_p[i] = NeuronState[i];
						}


						this->neuron_model->EvaluateTimeDependentEquation(AuxNeuronState_p, index, 0);



						float epsi = 1.0f;
						int k = 0;

						this->neuron_model->EvaluateDifferentialEquation(NeuronState, AuxNeuronState1, index, elapsedTimeInNeuronModelScale);

						//This integration method is an implicit method. We use this loop to iteratively calculate the implicit value.
						//epsi is the difference between two consecutive aproximation of the implicit method. 
						while (epsi > 1e-16 && k < 5){
							this->neuron_model->EvaluateDifferentialEquation(AuxNeuronState_p, AuxNeuronState, index, elapsedTimeInNeuronModelScale);

							for (int j = 0; j < this->neuron_model->N_DifferentialNeuronState; j++){
								AuxNeuronState_c[j] = Coeficient[CoeficientSelector[index]][state[index]][0] * elapsedTimeInNeuronModelScale*AuxNeuronState[j] + Coeficient[CoeficientSelector[index]][state[index]][1] * elapsedTimeInNeuronModelScale*AuxNeuronState1[j] + Coeficient[CoeficientSelector[index]][state[index]][2] * NeuronState[j];
								for (int i = 1; i < state[index]; i++){
									AuxNeuronState_c[j] += Coeficient[CoeficientSelector[index]][state[index]][i + 2] * PreviousNeuronState[i - 1][index*this->neuron_model->N_DifferentialNeuronState + j];
								}
							}

							//jacobian.
							Jacobian(AuxNeuronState_p, jacnum, index);

							for (int z = 0; z < this->neuron_model->N_DifferentialNeuronState; z++){
								for (int t = 0; t < this->neuron_model->N_DifferentialNeuronState; t++){
									J[z*this->neuron_model->N_DifferentialNeuronState + t] = Coeficient[CoeficientSelector[index]][state[index]][0] * elapsedTimeInNeuronModelScale * jacnum[z*this->neuron_model->N_DifferentialNeuronState + t];
									if (z == t){
										J[z*this->neuron_model->N_DifferentialNeuronState + t] -= 1;
									}
								}
							}

							this->invermat(J, inv_J);

							for (int z = 0; z < this->neuron_model->N_DifferentialNeuronState; z++){
								float aux = 0.0;
								for (int t = 0; t < this->neuron_model->N_DifferentialNeuronState; t++){
									aux += inv_J[z*this->neuron_model->N_DifferentialNeuronState + t] * (AuxNeuronState_p[t] - AuxNeuronState_c[t]);
								}
								AuxNeuronState_p1[z] = aux + AuxNeuronState_p[z];
							}

							//We calculate the difference between both aproximations.
							float aux = 0.0f;
							float aux2 = 0.0f;
							for (int z = 0; z < this->neuron_model->N_DifferentialNeuronState; z++){
								aux = fabs(AuxNeuronState_p1[z] - AuxNeuronState_p[z]);
								if (aux > aux2){
									aux2 = aux;
								}
							}

							memcpy(AuxNeuronState_p, AuxNeuronState_p1, sizeof(float)* this->neuron_model->N_DifferentialNeuronState);

							epsi = aux2;
							k++;
						}

						if (NeuronState[0] > startVoltageThreshold){
							integrationMethodState[index] = 1;

							//Restore the neuron model state to a previous state.
							this->neuron_model->GetVectorNeuronState()->AddElapsedTime(index, -elapsedTimeInSeconds);

							//Comes form a small step and goes to a small step
							if (CoeficientSelector[index] == 2){
								CoeficientSelector[index] = 1;
							}
							else{//goes to a smaller step.
								CoeficientSelector[index] = 0;
							}
						}
						else{
							//We increase the state of the integration method.
							if (state[index] < BDForder){
								state[index]++;
							}

							//We acumulate these new values for the next step.
							for (int j = 0; j < this->neuron_model->N_DifferentialNeuronState; j++){

								for (int i = (state[index] - 1); i > 0; i--){
									D[i][index*this->neuron_model->N_DifferentialNeuronState + j] = -D[i - 1][index*this->neuron_model->N_DifferentialNeuronState + j];
								}
								D[0][index*this->neuron_model->N_DifferentialNeuronState + j] = AuxNeuronState_p[j] - NeuronState[j];
								for (int i = 1; i < state[index]; i++){
									D[i][index*this->neuron_model->N_DifferentialNeuronState + j] += D[i - 1][index*this->neuron_model->N_DifferentialNeuronState + j];
								}
							}

							if (state[index]>1){
								for (int i = state[index] - 2; i > 0; i--){
									memcpy(PreviousNeuronState[i] + (index*this->neuron_model->N_DifferentialNeuronState), PreviousNeuronState[i - 1] + (index*this->neuron_model->N_DifferentialNeuronState), sizeof(float)* this->neuron_model->N_DifferentialNeuronState);
								}

								memcpy(PreviousNeuronState[0] + (index*this->neuron_model->N_DifferentialNeuronState), NeuronState, sizeof(float)* this->neuron_model->N_DifferentialNeuronState);
							}
							memcpy(NeuronState, AuxNeuronState_p, sizeof(float)* this->neuron_model->N_DifferentialNeuronState);



							//Finaly, we evaluate the neural state variables with time dependence.
							this->neuron_model->EvaluateTimeDependentEquation(NeuronState, index, 0);

							//update the integration step size.
							PreviousIntegrationStep[index] = elapsedTimeInNeuronModelScale;

							//Set the coeficient selector to 1 for the next iteration.
							CoeficientSelector[index] = 1;

							//Update the last spike time.
							this->neuron_model->GetVectorNeuronState()->AddElapsedTime(index, elapsedTimeInSeconds);

							//Acumulate the membrane potential in a variable
							this->IncrementValidIntegrationVariable(NeuronState[0]);
						}
					}

					if (integrationMethodState[index] > 0){

						for (int iteration = 0; iteration < ratioLargerSmallerSteps; iteration++){
							float previous_V = NeuronState[0];

							//If the state of the cell is 0, we use a Euler method to calculate an aproximation of the solution.
							if (state[index] == 0){
								this->neuron_model->EvaluateDifferentialEquation(NeuronState, AuxNeuronState, index, bifixedElapsedTimeInNeuronModelScale);
								for (int j = 0; j < this->neuron_model->N_DifferentialNeuronState; j++){
									AuxNeuronState_p[j] = NeuronState[j] + bifixedElapsedTimeInNeuronModelScale*AuxNeuronState[j];
								}
							}
							//In this case we use the value of previous states to calculate an aproximation of the solution.
							else{
								float inverseIntegrationStepFactor = PreviousIntegrationStep[index] / bifixedElapsedTimeInNeuronModelScale;
								for (int j = 0; j < this->neuron_model->N_DifferentialNeuronState; j++){
									AuxNeuronState_p[j] = NeuronState[j];
									for (int i = 0; i < state[index]; i++){
										AuxNeuronState_p[j] += D[i][index*this->neuron_model->N_DifferentialNeuronState + j]/*/inverseIntegrationStepFactor*/;
									}
								}
							}

							for (int i = this->neuron_model->N_DifferentialNeuronState; i < this->neuron_model->N_NeuronStateVariables; i++){
								AuxNeuronState_p[i] = NeuronState[i];
							}

							this->neuron_model->EvaluateTimeDependentEquation(AuxNeuronState_p, index, 1);

							float epsi = 1.0f;
							int k = 0;


							this->neuron_model->EvaluateDifferentialEquation(NeuronState, AuxNeuronState1, index, bifixedElapsedTimeInNeuronModelScale);

							//This integration method is an implicit method. We use this loop to iteratively calculate the implicit value.
							//epsi is the difference between two consecutive aproximation of the implicit method. 
							while (epsi > 1e-16 && k < 5){
								this->neuron_model->EvaluateDifferentialEquation(AuxNeuronState_p, AuxNeuronState, index, bifixedElapsedTimeInNeuronModelScale);

								for (int j = 0; j < this->neuron_model->N_DifferentialNeuronState; j++){
									AuxNeuronState_c[j] = Coeficient[CoeficientSelector[index]][state[index]][0] * bifixedElapsedTimeInNeuronModelScale*AuxNeuronState[j] + Coeficient[CoeficientSelector[index]][state[index]][1] * bifixedElapsedTimeInNeuronModelScale*AuxNeuronState1[j] + Coeficient[CoeficientSelector[index]][state[index]][2] * NeuronState[j];
									for (int i = 1; i < state[index]; i++){
										AuxNeuronState_c[j] += Coeficient[CoeficientSelector[index]][state[index]][i + 2] * PreviousNeuronState[i - 1][index*this->neuron_model->N_DifferentialNeuronState + j];
									}
								}

								//jacobian.
								Jacobian(AuxNeuronState_p, jacnum, index);

								for (int z = 0; z < this->neuron_model->N_DifferentialNeuronState; z++){
									for (int t = 0; t < this->neuron_model->N_DifferentialNeuronState; t++){
										J[z*this->neuron_model->N_DifferentialNeuronState + t] = Coeficient[CoeficientSelector[index]][state[index]][0] * bifixedElapsedTimeInNeuronModelScale * jacnum[z*this->neuron_model->N_DifferentialNeuronState + t];
										if (z == t){
											J[z*this->neuron_model->N_DifferentialNeuronState + t] -= 1;
										}
									}
								}

								this->invermat(J, inv_J);

								for (int z = 0; z < this->neuron_model->N_DifferentialNeuronState; z++){
									float aux = 0.0;
									for (int t = 0; t < this->neuron_model->N_DifferentialNeuronState; t++){
										aux += inv_J[z*this->neuron_model->N_DifferentialNeuronState + t] * (AuxNeuronState_p[t] - AuxNeuronState_c[t]);
									}
									AuxNeuronState_p1[z] = aux + AuxNeuronState_p[z];
								}

								//We calculate the difference between both aproximations.
								float aux = 0.0f;
								float aux2 = 0.0f;
								for (int z = 0; z < this->neuron_model->N_DifferentialNeuronState; z++){
									aux = fabs(AuxNeuronState_p1[z] - AuxNeuronState_p[z]);
									if (aux > aux2){
										aux2 = aux;
									}
								}

								memcpy(AuxNeuronState_p, AuxNeuronState_p1, sizeof(float)* this->neuron_model->N_DifferentialNeuronState);

								epsi = aux2;
								k++;
							}

							//We increase the state of the integration method.
							if (state[index] < BDForder){
								state[index]++;
							}


							//We acumulate these new values for the next step.
							for (int j = 0; j < this->neuron_model->N_DifferentialNeuronState; j++){

								for (int i = (state[index] - 1); i > 0; i--){
									D[i][index*this->neuron_model->N_DifferentialNeuronState + j] = -D[i - 1][index*this->neuron_model->N_DifferentialNeuronState + j];
								}
								D[0][index*this->neuron_model->N_DifferentialNeuronState + j] = AuxNeuronState_p[j] - NeuronState[j];
								for (int i = 1; i < state[index]; i++){
									D[i][index*this->neuron_model->N_DifferentialNeuronState + j] += D[i - 1][index*this->neuron_model->N_DifferentialNeuronState + j];
								}
							}

							if (state[index]>1){
								for (int i = state[index] - 2; i > 0; i--){
									memcpy(PreviousNeuronState[i] + (index*this->neuron_model->N_DifferentialNeuronState), PreviousNeuronState[i - 1] + (index*this->neuron_model->N_DifferentialNeuronState), sizeof(float)* this->neuron_model->N_DifferentialNeuronState);
								}

								memcpy(PreviousNeuronState[0] + (index*this->neuron_model->N_DifferentialNeuronState), NeuronState, sizeof(float)* this->neuron_model->N_DifferentialNeuronState);
							}
							memcpy(NeuronState, AuxNeuronState_p, sizeof(float)* this->neuron_model->N_DifferentialNeuronState);



							//Finaly, we evaluate the neural state variables with time dependence.
							this->neuron_model->EvaluateTimeDependentEquation(NeuronState, index, 1);

							//Update the last spike time.
							this->neuron_model->GetVectorNeuronState()->AddElapsedTime(index, bifixedElapsedTimeInSeconds);

							this->neuron_model->EvaluateSpikeCondition(previous_V, NeuronState, index, bifixedElapsedTimeInNeuronModelScale);

							//Acumulate the membrane potential in a variable
							this->IncrementValidIntegrationVariable(NeuronState[0]);
							
							//Set the CoeficientSelector to 1.
							CoeficientSelector[index] = 1;
							PreviousIntegrationStep[index] = bifixedElapsedTimeInNeuronModelScale;

							if (NeuronState[0] > startVoltageThreshold && integrationMethodState[index] == 1){
								integrationMethodState[index] = 2;
							}
							else if (NeuronState[0] < endVoltageThreshold && integrationMethodState[index] == 2){
								integrationMethodState[index] = 3;
								integrationMethodCounter[index] = N_postBiFixedSteps;
							}
							if (integrationMethodCounter[index]>0 && integrationMethodState[index] == 3){
								integrationMethodCounter[index]--;
								if (integrationMethodCounter[index] == 0){
									integrationMethodState[index] = 0;
									CoeficientSelector[index] = 2;
								}
							}
						}
						if (integrationMethodState[index] == 1){
							integrationMethodState[index] = 0;
							CoeficientSelector[index] = 2;
						}

					}

					//Set last update time for the analytic resolution of the differential equations in lethargic models 
					this->neuron_model->State->SetLastUpdateTime(index, CurrentTime);
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
		void InitializeStates(int N_neurons, float * initialization){
			if (BDForder>1){
				PreviousNeuronState = (float **)new float*[BDForder - 1];
				for (int i = 0; i<(BDForder - 1); i++){
					PreviousNeuronState[i] = new float[N_neurons*this->neuron_model->N_DifferentialNeuronState];
				}
			}
			D = (float **)new float*[BDForder];
			for (int i = 0; i<BDForder; i++){
				D[i] = new float[N_neurons*this->neuron_model->N_DifferentialNeuronState];
			}

			state = new int[N_neurons]();


			PreviousIntegrationStep = new float[N_neurons]();
			CoeficientSelector = new int[N_neurons];

			for (int i = 0; i<N_neurons; i++){
				CoeficientSelector[i] = 1;
			}


			integrationMethodCounter = new int[N_neurons]();
			integrationMethodState = new int[N_neurons]();



			float decrement = bifixedElapsedTimeInNeuronModelScale / elapsedTimeInNeuronModelScale;
			float increment = elapsedTimeInNeuronModelScale / bifixedElapsedTimeInNeuronModelScale;

			//decrement of h
			Coeficient[0][0][0] = 1.0f;
			Coeficient[0][0][1] = 0.0f;
			Coeficient[0][0][2] = 1.0f;
			Coeficient[0][0][3] = 0.0f;
			Coeficient[0][1][0] = 1.0f;
			Coeficient[0][1][1] = 0.0f;
			Coeficient[0][1][2] = 1.0f;
			Coeficient[0][1][3] = 0.0f;
			Coeficient[0][2][0] = 2.0f / 3.0f; //beta 0
			Coeficient[0][2][1] = (decrement - 1) / 3.0f; //beta 1
			Coeficient[0][2][2] = 1 + decrement*decrement / 3.0f;//alpha 0
			Coeficient[0][2][3] = -decrement*decrement / 3.0f;//alpha 1

			//Equal h;
			Coeficient[1][0][0] = 1.0f;
			Coeficient[1][0][1] = 0.0f;
			Coeficient[1][0][2] = 1.0f;
			Coeficient[1][0][3] = 0.0f;
			Coeficient[1][1][0] = 1.0f;
			Coeficient[1][1][1] = 0.0f;
			Coeficient[1][1][2] = 1.0f;
			Coeficient[1][1][3] = 0.0f;
			Coeficient[1][2][0] = 2.0f / 3.0f;
			Coeficient[1][2][1] = 0.0f;
			Coeficient[1][2][2] = 4.0f / 3.0f;
			Coeficient[1][2][3] = -1.0f / 3.0f;

			//increment of h
			Coeficient[2][0][0] = 1.0f;
			Coeficient[2][0][1] = 0.0f;
			Coeficient[2][0][2] = 1.0f;
			Coeficient[2][0][3] = 0.0f;
			Coeficient[2][1][0] = 1.0f;
			Coeficient[2][1][1] = 0.0f;
			Coeficient[2][1][2] = 1.0f;
			Coeficient[2][1][3] = 0.0f;
			Coeficient[2][2][0] = 2.0f / 3.0f;
			Coeficient[2][2][1] = (increment - 1) / 3.0f;
			Coeficient[2][2][2] = 1 + increment*increment / 3.0f;
			Coeficient[2][2][3] = -increment*increment / 3.0f;
		}



		/*!
		 * \brief It reset the state of the integration method for method with memory (e.g. BDF1ad, BDF2, BDF3, etc.).
		 *
		 * It reset the state of the integration method for method with memory (e.g. BDF1ad, BDF2, BDF3, etc.).
		 *
		 * \param index indicate which neuron must be reseted.
		 */
		void resetState(int index){
			state[index] = 0;
		}

		/*!
		 * \brief It calculates the conductance exponential values for time driven neuron models.
		 *
		 * It calculates the conductance exponential values for time driven neuron models.
		 */
		void Calculate_conductance_exp_values(){
			this->neuron_model->Initialize_conductance_exp_values(this->neuron_model->N_TimeDependentNeuronState, 2);
			//index 0
			this->neuron_model->Calculate_conductance_exp_values(0, elapsedTimeInNeuronModelScale);
			//index 1
			this->neuron_model->Calculate_conductance_exp_values(1, bifixedElapsedTimeInNeuronModelScale);
		}

		 /*!
		 * \brief It calculate numerically the Jacobian .
		 *
		 * It calculate numerically the Jacobian.
		 *
		 * \param NeuronState neuron state variables of one neuron.
		 * \param jancum vector where is stored the Jacobian.
		 * \param index neuron state index.
		 */
		void Jacobian(float * NeuronState, float * jacnum, int index){
			float epsi = elapsedTimeInNeuronModelScale * 0.1f;
			float inv_epsi = 1.0f / epsi;
			float JacAuxNeuronState[MAX_VARIABLES];
			float JacAuxNeuronState_pos[MAX_VARIABLES];
			float JacAuxNeuronState_neg[MAX_VARIABLES];

			memcpy(JacAuxNeuronState, NeuronState, sizeof(float)*this->neuron_model->N_NeuronStateVariables);
			this->neuron_model->EvaluateDifferentialEquation(JacAuxNeuronState, JacAuxNeuronState_pos, index, elapsedTimeInNeuronModelScale);

			for (int j = 0; j<this->neuron_model->N_DifferentialNeuronState; j++){
				memcpy(JacAuxNeuronState, NeuronState, sizeof(float)*this->neuron_model->N_NeuronStateVariables);

				JacAuxNeuronState[j] -= epsi;
				this->neuron_model->EvaluateDifferentialEquation(JacAuxNeuronState, JacAuxNeuronState_neg, index, elapsedTimeInNeuronModelScale);

				for (int z = 0; z<this->neuron_model->N_DifferentialNeuronState; z++){
					jacnum[z*this->neuron_model->N_DifferentialNeuronState + j] = (JacAuxNeuronState_pos[z] - JacAuxNeuronState_neg[z])*inv_epsi;
				}
			}
		}

		 /*!
		 * \brief It calculate the inverse of a square matrix using Gauss-Jordan Method.
		 *
		 * It calculate the inverse of a square matrix using Gauss-Jordan Method.
		 *
		 * \param a pointer to the square matrix.
		 * \param ainv pointer where inverse of the square matrix will be stored.
		 */
		void invermat(float *a, float *ainv){
			if (this->neuron_model->N_DifferentialNeuronState == 1){
				ainv[0] = 1.0f / a[0];
			}
			else{
				float coef, element, inv_element;
				int i, j, s;

				float local_a[MAX_VARIABLES*MAX_VARIABLES];
				float local_ainv[MAX_VARIABLES*MAX_VARIABLES] = {};

				memcpy(local_a, a, sizeof(float)*this->neuron_model->N_DifferentialNeuronState*this->neuron_model->N_DifferentialNeuronState);
				for (i = 0; i<this->neuron_model->N_DifferentialNeuronState; i++){
					local_ainv[i*this->neuron_model->N_DifferentialNeuronState + i] = 1.0f;
				}

				//Iteraciones
				for (s = 0; s<this->neuron_model->N_DifferentialNeuronState; s++)
				{
					element = local_a[s*this->neuron_model->N_DifferentialNeuronState + s];

					if (element == 0){
						for (int n = s + 1; n<this->neuron_model->N_DifferentialNeuronState; n++){
							element = local_a[n*this->neuron_model->N_DifferentialNeuronState + s];
							if (element != 0){
								for (int m = 0; m<this->neuron_model->N_DifferentialNeuronState; m++){
									float value = local_a[n*this->neuron_model->N_DifferentialNeuronState + m];
									local_a[n*this->neuron_model->N_DifferentialNeuronState + m] = local_a[s*this->neuron_model->N_DifferentialNeuronState + m];
									local_a[s*this->neuron_model->N_DifferentialNeuronState + m] = value;

									value = local_ainv[n*this->neuron_model->N_DifferentialNeuronState + m];
									local_ainv[n*this->neuron_model->N_DifferentialNeuronState + m] = local_ainv[s*this->neuron_model->N_DifferentialNeuronState + m];
									local_ainv[s*this->neuron_model->N_DifferentialNeuronState + m] = value;
								}
								break;
							}
							if (n == (this->neuron_model->N_DifferentialNeuronState - 1)){
								printf("This matrix is not invertible\n");
								exit(0);
							}

						}
					}

					inv_element = 1.0f / element;
					for (j = 0; j<this->neuron_model->N_DifferentialNeuronState; j++){
						local_a[s*this->neuron_model->N_DifferentialNeuronState + j] *= inv_element;
						local_ainv[s*this->neuron_model->N_DifferentialNeuronState + j] *= inv_element;
					}

					for (i = 0; i<this->neuron_model->N_DifferentialNeuronState; i++)
					{
						if (i != s){
							coef = -local_a[i*this->neuron_model->N_DifferentialNeuronState + s];
							for (j = 0; j<this->neuron_model->N_DifferentialNeuronState; j++){
								local_a[i*this->neuron_model->N_DifferentialNeuronState + j] += local_a[s*this->neuron_model->N_DifferentialNeuronState + j] * coef;
								local_ainv[i*this->neuron_model->N_DifferentialNeuronState + j] += local_ainv[s*this->neuron_model->N_DifferentialNeuronState + j] * coef;
							}
						}
					}
				}
				memcpy(ainv, local_ainv, sizeof(float)*this->neuron_model->N_DifferentialNeuronState*this->neuron_model->N_DifferentialNeuronState);
			}
		}




};

#endif /* Bifixed_BDFn_H_ */
