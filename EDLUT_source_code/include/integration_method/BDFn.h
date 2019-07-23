/***************************************************************************
 *                           BDFn.h                                        *
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

#ifndef BDFn_H_
#define BDFn_H_

/*!
 * \file BDFn.h
 *
 * \author Francisco Naveros
 * \date May 2013
 *
 * This file declares a class which implement six BDF (Backward Differentiation Formulas) integration methods (from 
 * first to sixth order BDF integration method). This method implements a progressive implementation of the
 * higher order integration method using the lower order integration mehtod (BDF1->BDF2->...->BDF6). This class 
 * implement a fixed step integration method.
 */

#include "./FixedStep.h"

class TimeDrivenNeuronModel;

//const float BDFn::BDF_Coeficients[7][7] = { { 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
//{ 1.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f },
//{ 2.0f / 3.0f, 4.0f / 3.0f, -1.0f / 3.0f, 0.0f, 0.0f, 0.0f, 0.0f },
//{ 6.0f / 11.0f, 18.0f / 11.0f, -9.0f / 11.0f, 2.0f / 11.0f, 0.0f, 0.0f, 0.0f },
//{ 12.0f / 25.0f, 48.0f / 25.0f, -36.0f / 25.0f, 16.0f / 25.0f, -3.0f / 25.0f, 0.0f, 0.0f },
//{ 60.0f / 137.0f, 300.0f / 137.0f, -300.0f / 137.0f, 200.0f / 137.0f, -75.0f / 137.0f, 12.0f / 137.0f, 0.0f },
//{ 60.0f / 147.0f, 360.0f / 147.0f, -450.0f / 147.0f, 400.0f / 147.0f, -225.0f / 147.0f, 72.0f / 147.0f, -10.0f / 147.0f } };

/*!
 * \class BDFn
 *
 * \brief BDFn integration methods in CPU
 *
 * This class abstracts the behavior of BDF1,...,BDF6 integration methods for neurons in a 
 * time-driven spiking neural network.
 * It includes internal model functions which define the behavior of integration methods
 * (initialization, calculate next value, ...).
 *
 * \author Francisco Naveros
 * \date May 2013
 */
template <class Neuron_Model>

class BDFn : public FixedStep {
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
		 * \brief This constant matrix stores the coefficients of each BDF order.
		*/
		float BDF_Coeficients[7][7];

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
		 * \param BDForder BDF order (1, 2, ..., 6).
		 */
		BDFn(Neuron_Model * NewModel, int BDForder) :FixedStep("BDFn"), BDForder(BDForder), neuron_model(NewModel){
		}

		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		~BDFn(){
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

			float previous_V = NeuronState[0];

			float AuxNeuronState[MAX_VARIABLES];
			float AuxNeuronState_p[MAX_VARIABLES];
			float AuxNeuronState_p1[MAX_VARIABLES];
			float AuxNeuronState_c[MAX_VARIABLES];
			float jacnum[MAX_VARIABLES*MAX_VARIABLES];
			float J[MAX_VARIABLES*MAX_VARIABLES];
			float inv_J[MAX_VARIABLES*MAX_VARIABLES];



			//If the state of the cell is 0, we use a Euler method to calculate an aproximation of the solution.
			if (state[index] == 0){
				this->neuron_model->EvaluateDifferentialEquation(NeuronState, AuxNeuronState, index, elapsedTimeInNeuronModelScale);
				for (int j = 0; j<this->neuron_model->N_DifferentialNeuronState; j++){
					AuxNeuronState_p[j] = NeuronState[j] + elapsedTimeInNeuronModelScale*AuxNeuronState[j];
				}
			}
			//In this case we use the value of previous states to calculate an aproximation of the solution.
			else{
				for (int j = 0; j<this->neuron_model->N_DifferentialNeuronState; j++){
					AuxNeuronState_p[j] = NeuronState[j];
					for (int i = 0; i<state[index]; i++){
						AuxNeuronState_p[j] += D[i][index*this->neuron_model->N_DifferentialNeuronState + j];
					}
				}
			}

			for (int i = this->neuron_model->N_DifferentialNeuronState; i<this->neuron_model->N_NeuronStateVariables; i++){
				AuxNeuronState_p[i] = NeuronState[i];
			}


			this->neuron_model->EvaluateTimeDependentEquation(AuxNeuronState_p, index, 0);



			float epsi = 1.0f;
			int k = 0;

			//This integration method is an implicit method. We use this loop to iteratively calculate the implicit value.
			//epsi is the difference between two consecutive aproximation of the implicit method. 
			while (epsi>1e-16 && k<5){
				this->neuron_model->EvaluateDifferentialEquation(AuxNeuronState_p, AuxNeuronState, index, elapsedTimeInNeuronModelScale);

				for (int j = 0; j<this->neuron_model->N_DifferentialNeuronState; j++){
					AuxNeuronState_c[j] = BDF_Coeficients[state[index]][0] * elapsedTimeInNeuronModelScale*AuxNeuronState[j] + BDF_Coeficients[state[index]][1] * NeuronState[j];
					for (int i = 1; i<state[index]; i++){
						AuxNeuronState_c[j] += BDF_Coeficients[state[index]][i + 1] * PreviousNeuronState[i - 1][index*this->neuron_model->N_DifferentialNeuronState + j];
					}
				}

				//jacobian.
				Jacobian(AuxNeuronState_p, jacnum, index);

				for (int z = 0; z<this->neuron_model->N_DifferentialNeuronState; z++){
					for (int t = 0; t<this->neuron_model->N_DifferentialNeuronState; t++){
						J[z*this->neuron_model->N_DifferentialNeuronState + t] = BDF_Coeficients[state[index]][0] * elapsedTimeInNeuronModelScale * jacnum[z*this->neuron_model->N_DifferentialNeuronState + t];
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
			this->neuron_model->EvaluateTimeDependentEquation(NeuronState, index, 0);


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

			float AuxNeuronState[MAX_VARIABLES];
			float AuxNeuronState_p[MAX_VARIABLES];
			float AuxNeuronState_p1[MAX_VARIABLES];
			float AuxNeuronState_c[MAX_VARIABLES];
			float jacnum[MAX_VARIABLES*MAX_VARIABLES];
			float J[MAX_VARIABLES*MAX_VARIABLES];
			float inv_J[MAX_VARIABLES*MAX_VARIABLES];

			for (int index = 0; index < this->neuron_model->State->GetSizeState(); index++){
				float * NeuronState = this->neuron_model->State->GetStateVariableAt(index);
				float previous_V = NeuronState[0];

				//If the state of the cell is 0, we use a Euler method to calculate an aproximation of the solution.
				if (state[index] == 0){
					this->neuron_model->EvaluateDifferentialEquation(NeuronState, AuxNeuronState, index, elapsedTimeInNeuronModelScale);
					for (int j = 0; j < this->neuron_model->N_DifferentialNeuronState; j++){
						AuxNeuronState_p[j] = NeuronState[j] + elapsedTimeInNeuronModelScale*AuxNeuronState[j];
					}
				}
				//In this case we use the value of previous states to calculate an aproximation of the solution.
				else{
					for (int j = 0; j < this->neuron_model->N_DifferentialNeuronState; j++){
						AuxNeuronState_p[j] = NeuronState[j];
						for (int i = 0; i < state[index]; i++){
							AuxNeuronState_p[j] += D[i][index*this->neuron_model->N_DifferentialNeuronState + j];
						}
					}
				}

				for (int i = this->neuron_model->N_DifferentialNeuronState; i < this->neuron_model->N_NeuronStateVariables; i++){
					AuxNeuronState_p[i] = NeuronState[i];
				}


				this->neuron_model->EvaluateTimeDependentEquation(AuxNeuronState_p, index, 0);



				float epsi = 1.0f;
				int k = 0;

				//This integration method is an implicit method. We use this loop to iteratively calculate the implicit value.
				//epsi is the difference between two consecutive aproximation of the implicit method. 
				while (epsi > 1e-16 && k < 5){
					this->neuron_model->EvaluateDifferentialEquation(AuxNeuronState_p, AuxNeuronState, index, elapsedTimeInNeuronModelScale);

					for (int j = 0; j < this->neuron_model->N_DifferentialNeuronState; j++){
						AuxNeuronState_c[j] = BDF_Coeficients[state[index]][0] * elapsedTimeInNeuronModelScale*AuxNeuronState[j] + BDF_Coeficients[state[index]][1] * NeuronState[j];
						for (int i = 1; i < state[index]; i++){
							AuxNeuronState_c[j] += BDF_Coeficients[state[index]][i + 1] * PreviousNeuronState[i - 1][index*this->neuron_model->N_DifferentialNeuronState + j];
						}
					}

					//jacobian.
					Jacobian(AuxNeuronState_p, jacnum, index);

					for (int z = 0; z < this->neuron_model->N_DifferentialNeuronState; z++){
						for (int t = 0; t < this->neuron_model->N_DifferentialNeuronState; t++){
							J[z*this->neuron_model->N_DifferentialNeuronState + t] = BDF_Coeficients[state[index]][0] * elapsedTimeInNeuronModelScale * jacnum[z*this->neuron_model->N_DifferentialNeuronState + t];
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
				this->neuron_model->EvaluateTimeDependentEquation(NeuronState, index, 0);


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

			float AuxNeuronState[MAX_VARIABLES];
			float AuxNeuronState_p[MAX_VARIABLES];
			float AuxNeuronState_p1[MAX_VARIABLES];
			float AuxNeuronState_c[MAX_VARIABLES];
			float jacnum[MAX_VARIABLES*MAX_VARIABLES];
			float J[MAX_VARIABLES*MAX_VARIABLES];
			float inv_J[MAX_VARIABLES*MAX_VARIABLES];

			for (int index = 0; index < this->neuron_model->State->GetSizeState(); index++){
				if (integration_required[index] == true){
					float * NeuronState = this->neuron_model->State->GetStateVariableAt(index);
					float previous_V = NeuronState[0];

					//If the state of the cell is 0, we use a Euler method to calculate an aproximation of the solution.
					if (state[index] == 0){
						this->neuron_model->EvaluateDifferentialEquation(NeuronState, AuxNeuronState, index, elapsedTimeInNeuronModelScale);
						for (int j = 0; j < this->neuron_model->N_DifferentialNeuronState; j++){
							AuxNeuronState_p[j] = NeuronState[j] + elapsedTimeInNeuronModelScale*AuxNeuronState[j];
						}
					}
					//In this case we use the value of previous states to calculate an aproximation of the solution.
					else{
						for (int j = 0; j < this->neuron_model->N_DifferentialNeuronState; j++){
							AuxNeuronState_p[j] = NeuronState[j];
							for (int i = 0; i < state[index]; i++){
								AuxNeuronState_p[j] += D[i][index*this->neuron_model->N_DifferentialNeuronState + j];
							}
						}
					}

					for (int i = this->neuron_model->N_DifferentialNeuronState; i < this->neuron_model->N_NeuronStateVariables; i++){
						AuxNeuronState_p[i] = NeuronState[i];
					}


					this->neuron_model->EvaluateTimeDependentEquation(AuxNeuronState_p, index, 0);



					float epsi = 1.0f;
					int k = 0;

					//This integration method is an implicit method. We use this loop to iteratively calculate the implicit value.
					//epsi is the difference between two consecutive aproximation of the implicit method. 
					while (epsi > 1e-16 && k < 5){
						this->neuron_model->EvaluateDifferentialEquation(AuxNeuronState_p, AuxNeuronState, index, elapsedTimeInNeuronModelScale);

						for (int j = 0; j < this->neuron_model->N_DifferentialNeuronState; j++){
							AuxNeuronState_c[j] = BDF_Coeficients[state[index]][0] * elapsedTimeInNeuronModelScale*AuxNeuronState[j] + BDF_Coeficients[state[index]][1] * NeuronState[j];
							for (int i = 1; i < state[index]; i++){
								AuxNeuronState_c[j] += BDF_Coeficients[state[index]][i + 1] * PreviousNeuronState[i - 1][index*this->neuron_model->N_DifferentialNeuronState + j];
							}
						}

						//jacobian.
						Jacobian(AuxNeuronState_p, jacnum, index);

						for (int z = 0; z < this->neuron_model->N_DifferentialNeuronState; z++){
							for (int t = 0; t < this->neuron_model->N_DifferentialNeuronState; t++){
								J[z*this->neuron_model->N_DifferentialNeuronState + t] = BDF_Coeficients[state[index]][0] * elapsedTimeInNeuronModelScale * jacnum[z*this->neuron_model->N_DifferentialNeuronState + t];
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
					this->neuron_model->EvaluateTimeDependentEquation(NeuronState, index, 0);


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



			BDF_Coeficients[0][0] = 1.0f;
			BDF_Coeficients[0][1] = 1.0f;
			BDF_Coeficients[0][2] = 0.0f;
			BDF_Coeficients[0][3] = 0.0f;
			BDF_Coeficients[0][4] = 0.0f;
			BDF_Coeficients[0][5] = 0.0f;
			BDF_Coeficients[0][6] = 0.0f;
			BDF_Coeficients[1][0] = 1.0f;
			BDF_Coeficients[1][1] = 1.0f;
			BDF_Coeficients[1][2] = 0.0f;
			BDF_Coeficients[1][3] = 0.0f;
			BDF_Coeficients[1][4] = 0.0f;
			BDF_Coeficients[1][5] = 0.0f;
			BDF_Coeficients[1][6] = 0.0f;
			BDF_Coeficients[2][0] = 2.0f / 3.0f;
			BDF_Coeficients[2][1] = 4.0f / 3.0f;
			BDF_Coeficients[2][2] = -1.0f / 3.0f;
			BDF_Coeficients[2][3] = 0.0f;
			BDF_Coeficients[2][4] = 0.0f;
			BDF_Coeficients[2][5] = 0.0f;
			BDF_Coeficients[2][6] = 0.0f;
			BDF_Coeficients[3][0] = 6.0f / 11.0f;
			BDF_Coeficients[3][1] = 18.0f / 11.0f;
			BDF_Coeficients[3][2] = -9.0f / 11.0f;
			BDF_Coeficients[3][3] = 2.0f / 11.0f;
			BDF_Coeficients[3][4] = 0.0f;
			BDF_Coeficients[3][5] = 0.0f;
			BDF_Coeficients[3][6] = 0.0f;
			BDF_Coeficients[4][0] = 12.0f / 25.0f;
			BDF_Coeficients[4][1] = 48.0f / 25.0f;
			BDF_Coeficients[4][2] = -36.0f / 25.0f;
			BDF_Coeficients[4][3] = 16.0f / 25.0f;
			BDF_Coeficients[4][4] = -3.0f / 25.0f;
			BDF_Coeficients[4][5] = 0.0f;
			BDF_Coeficients[4][6] = 0.0f;
			BDF_Coeficients[5][0] = 60.0f / 137.0f;
			BDF_Coeficients[5][1] = 300.0f / 137.0f;
			BDF_Coeficients[5][2] = -300.0f / 137.0f;
			BDF_Coeficients[5][3] = 200.0f / 137.0f;
			BDF_Coeficients[5][4] = -75.0f / 137.0f;
			BDF_Coeficients[5][5] = 12.0f / 137.0f;
			BDF_Coeficients[5][6] = 0.0f;
			BDF_Coeficients[6][0] = 60.0f / 147.0f;
			BDF_Coeficients[6][1] = 360.0f / 147.0f;
			BDF_Coeficients[6][2] = -450.0f / 147.0f;
			BDF_Coeficients[6][3] = 400.0f / 147.0f;
			BDF_Coeficients[6][4] = -225.0f / 147.0f;
			BDF_Coeficients[6][5] = 72.0f / 147.0f;
			BDF_Coeficients[6][6] = -10.0f / 147.0f;
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
			this->neuron_model->Initialize_conductance_exp_values(this->neuron_model->N_TimeDependentNeuronState, 1);
			//index 0
			this->neuron_model->Calculate_conductance_exp_values(0, elapsedTimeInNeuronModelScale);
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


#endif /* BDFn_H_ */
