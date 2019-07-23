/***************************************************************************
 *                           BDFn_GPU2.h                                   *
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

#ifndef BDF_GPU2_H_
#define BDF_GPU2_H_

/*!
 * \file BDFn_GPU2.h
 *
 * \author Francisco Naveros
 * \date May 2013
 *
 * This file declares a class which implement six fixed step  BDF (Backward Differentiation Formulas) integration methods (From 
 * first order to sixth order BDF integration method) in GPU. This method implements a progressive implementation of the
 * higher order integration method using the lower order integration mehtod (BDF1->BDF2->...->BDF6).  
 */

#include "./FixedStep_GPU2.h"

#include "../../include/neuron_model/TimeDrivenNeuronModel_GPU2.h"


//Library for CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/*!
 * \class BDFn_GPU2
 *
 * \brief BDF1, BDF2, ..., BDF6 integration methods in GPU.
 * 
 * This class abstracts the behavior of a Euler integration method for neurons in a 
 * time-driven spiking neural network.
 * It includes internal model functions which define the behavior of integration methods
 * (initialization, calculate next value, ...).
 *
 * \author Francisco Naveros
 * \date May 2012
 */
template <class Neuron_Model_GPU2>

class BDFn_GPU2 : public FixedStep_GPU2 {
	public:

		/*
		* Time driven neuron model
		*/
		Neuron_Model_GPU2 * neuron_model;

		/*!
		 * \brief These vectors are used as auxiliar vectors.
		*/
		float * AuxNeuronState;
		float * AuxNeuronState_p;
		float * AuxNeuronState_p1;
		float * AuxNeuronState_c;
		float * jacnum;
		float * J;
		float * inv_J;
		//For Jacobian
		float * AuxNeuronState2;
		float * AuxNeuronState_pos;
		float * AuxNeuronState_neg;

		/*!
		 * \brief This constant matrix contains the coefficients of each order for the BDF integration mehtod.
		*/
		float * Coeficient;
		//								{{1.0,1.0,0.0,0.0,0.0,0.0,0.0},
		//								{1.0,1.0,0.0,0.0,0.0,0.0,0.0},
		//								{2.0/3.0,4.0/3.0,-1.0/3.0,0.0,0.0,0.0,0.0},
		//								{6.0/11.0,18.0/11.0,-9.0/11.0,2.0/11.0,0.0,0.0,0.0},
		//								{12.0/25.0,48.0/25.0,-36.0/25.0,16.0/25.0,-3.0/25.0,0.0,0.0},
		//								{60.0/137.0,300.0/137.0,-300.0/137.0,200.0/137.0,-75.0/137.0,12.0/137.0,0.0},
		//								{60.0/147.0,360.0/147.0,-450.0/147.0,400.0/147.0,-225.0/147.0,72.0/147.0,-10.0/147.0}};

		/*!
		 * \brief This vector stores previous neuron state variable for all neuron. This one is used as a memory.
		*/
		float * PreviousNeuronState;

		/*!
		 * \brief This vector stores previous neuron state variable for all neuron. This one is used as a memory.
		*/
		float * D;

		/*!
		 * \brief This vector contains the state of each neuron (BDF order).
		*/
		int * state;

		/*!
		 * \brief This value stores the order of the integration method.
		*/
		int BDForder;


		/*!
		* \brief Constructor of the class with 2 parameter.
		*
		* It generates a new fourth order Runge-Kutta Integration Method objectin GPU memory.
		*
		* \param TimeDrivenNeuronModel pointer to the time driven neuron model
		* \param Buffer_GPU integration method parameters
		* \param BDForder BDF order (1, 2, ..., 6).
		*/
		__device__ BDFn_GPU2(Neuron_Model_GPU2* NewModel, void ** Buffer_GPU, int BDFOrder) :FixedStep_GPU2(Buffer_GPU, NewModel->timeScale), BDForder(BDFOrder), neuron_model(NewModel){
			AuxNeuronState = ((float*)Buffer_GPU[1]);
			AuxNeuronState_p = ((float*)Buffer_GPU[2]);
			AuxNeuronState_p1 = ((float*)Buffer_GPU[3]);
			AuxNeuronState_c = ((float*)Buffer_GPU[4]);
			jacnum = ((float*)Buffer_GPU[5]);
			J = ((float*)Buffer_GPU[6]);
			inv_J = ((float*)Buffer_GPU[7]);

			Coeficient=((float *)Buffer_GPU[8]);

			PreviousNeuronState = ((float *)Buffer_GPU[9]);
			D = ((float *)Buffer_GPU[10]);
			state = ((int *)Buffer_GPU[11]);

			AuxNeuronState2 = ((float*)Buffer_GPU[12]);
			AuxNeuronState_pos = ((float*)Buffer_GPU[13]);
			AuxNeuronState_neg = ((float*)Buffer_GPU[14]);
		}


		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		__device__ ~BDFn_GPU2(){
		}

		
		/*!
		 * \brief It calculate the next neural state varaibles of the model.
		 *
		 * It calculate the next neural state varaibles of the model.
		 *
		 * \param index Index of the cell inside the neuron model for method with memory (e.g. BDF).
		 * \param SizeStates Number of neurons
		 * \param Model The NeuronModel.
		 * \param NeuronState Vector of neuron state variables for all neurons.
		 */
		__device__ void NextDifferentialEquationValues(int index, int SizeStates, float * NeuronState){
			float previous_V=NeuronState[index];
			int offset1=gridDim.x * blockDim.x;
			int offset2=blockDim.x*blockIdx.x + threadIdx.x;

			//If the state of the cell is 0, we use a Euler method to calculate an aproximation of the solution.
			if(state[index]==0){
				neuron_model->EvaluateDifferentialEquation(index, SizeStates, NeuronState, AuxNeuronState, this->elapsedTimeInNeuronModelScale);
				for (int j=0; j<this->neuron_model->N_DifferentialNeuronState; j++){
					AuxNeuronState_p[j*offset1 + offset2]= NeuronState[j*SizeStates + index] + elapsedTimeInNeuronModelScale*AuxNeuronState[j*offset1 + offset2];
				}
			}
			//In this case we use the value of previous states to calculate an aproximation of the solution.
			else{
				for (int j=0; j<this->neuron_model->N_DifferentialNeuronState; j++){
					AuxNeuronState_p[j*offset1 + offset2]= NeuronState[j*SizeStates + index];
					for (int i=0; i<state[index]; i++){
						AuxNeuronState_p[j*offset1 + offset2]+=D[i*SizeStates*this->neuron_model->N_DifferentialNeuronState + j*SizeStates + index];
					}
				}
			}
			
			for(int i=this->neuron_model->N_DifferentialNeuronState; i<this->neuron_model->N_NeuronStateVariables; i++){
				AuxNeuronState_p[i*offset1 + offset2]=NeuronState[i*SizeStates + index];
			}
				
			neuron_model->EvaluateTimeDependentEquation(offset2, offset1, AuxNeuronState_p, index, 0);

			float epsi=1.0f;
			int k=0;

			//This integration method is an implicit method. We use this loop to iteratively calculate the implicit value.
			//epsi is the difference between two consecutive aproximation of the implicit method.
			while (epsi>1e-16 && k<5){
				neuron_model->EvaluateDifferentialEquation(offset2, offset1, AuxNeuronState_p, AuxNeuronState, elapsedTimeInNeuronModelScale);
				for (int j=0; j<this->neuron_model->N_DifferentialNeuronState; j++){
					AuxNeuronState_c[j*offset1 + offset2]=Coeficient[state[index]*7 + 0]*elapsedTimeInNeuronModelScale*AuxNeuronState[j*offset1 + offset2] + Coeficient[state[index]*7 + 1]*NeuronState[j*SizeStates + index];
					for (int i=1; i<state[index]; i++){
						AuxNeuronState_c[j*offset1 + offset2]+=Coeficient[state[index]*7 + i+1]*PreviousNeuronState[(i-1)*SizeStates*this->neuron_model->N_DifferentialNeuronState + j*SizeStates + index];
					}
				}

				//jacobian.
				Jacobian(offset2, offset1, AuxNeuronState_p, jacnum);

				for(int z=0; z<this->neuron_model->N_DifferentialNeuronState; z++){
					for(int t=0; t<this->neuron_model->N_DifferentialNeuronState; t++){
						J[z*offset1*this->neuron_model->N_DifferentialNeuronState+ t*offset1 + offset2] = Coeficient[state[index]*7 + 0] * elapsedTimeInNeuronModelScale * jacnum[z*offset1*this->neuron_model->N_DifferentialNeuronState+ t*offset1 + offset2];
						if(z==t){
							J[z*offset1*this->neuron_model->N_DifferentialNeuronState+ t*offset1 + offset2]-=1;
						}
					}
				}
				this->invermat(offset2, offset1, J, inv_J);

				for(int z=0; z<this->neuron_model->N_DifferentialNeuronState; z++){
					float aux=0.0;
					for (int t=0; t<this->neuron_model->N_DifferentialNeuronState; t++){
						aux+=inv_J[z*offset1*this->neuron_model->N_DifferentialNeuronState+ t*offset1 + offset2]*(AuxNeuronState_p[t*offset1 + offset2]-AuxNeuronState_c[t*offset1 + offset2]);
					}
					AuxNeuronState_p1[z*offset1 + offset2]=aux + AuxNeuronState_p[z*offset1 + offset2];
				}

				//We calculate the difference between both aproximations.
				float aux=0.0;
				float aux2=0.0;
				for(int z=0; z<this->neuron_model->N_DifferentialNeuronState; z++){
					aux=fabs(AuxNeuronState_p1[z*offset1 + offset2]-AuxNeuronState_p[z*offset1 + offset2]);
					AuxNeuronState_p[z*offset1 + offset2]=AuxNeuronState_p1[z*offset1 + offset2];
					if(aux>aux2){
						aux2=aux;
					}
				}
				epsi=aux2;
				k++;

			}

			//We increase the state of the integration method.
			if(state[index]<BDForder){
				state[index]++;
			}

			//We acumulate these new values for the next step.
			for (int j=0; j<this->neuron_model->N_DifferentialNeuronState; j++){
				for(int i=(state[index]-1); i>0; i--){ 
					D[i*SizeStates*this->neuron_model->N_DifferentialNeuronState + j*SizeStates + index]=-D[(i-1)*SizeStates*this->neuron_model->N_DifferentialNeuronState + j*SizeStates + index];
				}
				D[0*SizeStates*this->neuron_model->N_DifferentialNeuronState + j*SizeStates + index]=AuxNeuronState_p[j*offset1 + offset2]-NeuronState[j*SizeStates + index];
				for(int i=1; i<state[index]; i++){ 
					D[i*SizeStates*this->neuron_model->N_DifferentialNeuronState + j*SizeStates + index]+=D[(i-1)*SizeStates*this->neuron_model->N_DifferentialNeuronState + j*SizeStates + index];
				}
			}

			if(state[index]>1){
				for(int i=state[index]-2; i>0; i--){
					for(int j=0; j<this->neuron_model->N_DifferentialNeuronState; j++){
						PreviousNeuronState[i*SizeStates*this->neuron_model->N_DifferentialNeuronState + j*SizeStates + index] = PreviousNeuronState[(i-1)*SizeStates*this->neuron_model->N_DifferentialNeuronState + j*SizeStates + index];
					}
				}
				for(int j=0; j<this->neuron_model->N_DifferentialNeuronState; j++){
					PreviousNeuronState[0*SizeStates*this->neuron_model->N_DifferentialNeuronState + j*SizeStates + index] = NeuronState[j*SizeStates + index];
				}
			}
			for(int z=0; z<this->neuron_model->N_DifferentialNeuronState; z++){
				NeuronState[z*SizeStates + index] = AuxNeuronState_p[z*offset1 + offset2];
			}

			//Finaly, we evaluate the neural state variables with time dependence.
			neuron_model->EvaluateTimeDependentEquation(index, SizeStates, NeuronState, index, 0);
			
			//Update the last spike time.
			this->neuron_model->vectorNeuronState_GPU2->LastSpikeTimeGPU[index]+=this->elapsedTimeInSeconds;

			neuron_model->EvaluateSpikeCondition(previous_V, NeuronState, index, this->elapsedTimeInNeuronModelScale);
		}



		/*!
		* \brief It calculate the next neural state variables of the model.
		*
		* It calculate the next neural state varaibles of the model.
		*
		* \param SizeStates Number of neurons
		* \param NeuronState Vector of neuron state variables for all neurons.
		*/
		__device__ virtual void NextDifferentialEquationValues(int SizeStates, float * NeuronState) {
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			while (index < SizeStates){
				float previous_V = NeuronState[index];
				int offset1 = gridDim.x * blockDim.x;
				int offset2 = blockDim.x*blockIdx.x + threadIdx.x;

				//If the state of the cell is 0, we use a Euler method to calculate an aproximation of the solution.
				if (state[index] == 0){
					neuron_model->EvaluateDifferentialEquation(index, SizeStates, NeuronState, AuxNeuronState, this->elapsedTimeInNeuronModelScale);
					for (int j = 0; j < this->neuron_model->N_DifferentialNeuronState; j++){
						AuxNeuronState_p[j*offset1 + offset2] = NeuronState[j*SizeStates + index] + elapsedTimeInNeuronModelScale*AuxNeuronState[j*offset1 + offset2];
					}
				}
				//In this case we use the value of previous states to calculate an aproximation of the solution.
				else{
					for (int j = 0; j < this->neuron_model->N_DifferentialNeuronState; j++){
						AuxNeuronState_p[j*offset1 + offset2] = NeuronState[j*SizeStates + index];
						for (int i = 0; i < state[index]; i++){
							AuxNeuronState_p[j*offset1 + offset2] += D[i*SizeStates*this->neuron_model->N_DifferentialNeuronState + j*SizeStates + index];
						}
					}
				}

				for (int i = this->neuron_model->N_DifferentialNeuronState; i < this->neuron_model->N_NeuronStateVariables; i++){
					AuxNeuronState_p[i*offset1 + offset2] = NeuronState[i*SizeStates + index];
				}

				neuron_model->EvaluateTimeDependentEquation(offset2, offset1, AuxNeuronState_p, index, 0);

				float epsi = 1.0f;
				int k = 0;

				//This integration method is an implicit method. We use this loop to iteratively calculate the implicit value.
				//epsi is the difference between two consecutive aproximation of the implicit method.
				while (epsi > 1e-16 && k < 5){
					neuron_model->EvaluateDifferentialEquation(offset2, offset1, AuxNeuronState_p, AuxNeuronState, elapsedTimeInNeuronModelScale);
					for (int j = 0; j < this->neuron_model->N_DifferentialNeuronState; j++){
						AuxNeuronState_c[j*offset1 + offset2] = Coeficient[state[index] * 7 + 0] * elapsedTimeInNeuronModelScale*AuxNeuronState[j*offset1 + offset2] + Coeficient[state[index] * 7 + 1] * NeuronState[j*SizeStates + index];
						for (int i = 1; i < state[index]; i++){
							AuxNeuronState_c[j*offset1 + offset2] += Coeficient[state[index] * 7 + i + 1] * PreviousNeuronState[(i - 1)*SizeStates*this->neuron_model->N_DifferentialNeuronState + j*SizeStates + index];
						}
					}

					//jacobian.
					Jacobian(offset2, offset1, AuxNeuronState_p, jacnum);

					for (int z = 0; z < this->neuron_model->N_DifferentialNeuronState; z++){
						for (int t = 0; t < this->neuron_model->N_DifferentialNeuronState; t++){
							J[z*offset1*this->neuron_model->N_DifferentialNeuronState + t*offset1 + offset2] = Coeficient[state[index] * 7 + 0] * elapsedTimeInNeuronModelScale * jacnum[z*offset1*this->neuron_model->N_DifferentialNeuronState + t*offset1 + offset2];
							if (z == t){
								J[z*offset1*this->neuron_model->N_DifferentialNeuronState + t*offset1 + offset2] -= 1;
							}
						}
					}
					this->invermat(offset2, offset1, J, inv_J);

					for (int z = 0; z < this->neuron_model->N_DifferentialNeuronState; z++){
						float aux = 0.0;
						for (int t = 0; t < this->neuron_model->N_DifferentialNeuronState; t++){
							aux += inv_J[z*offset1*this->neuron_model->N_DifferentialNeuronState + t*offset1 + offset2] * (AuxNeuronState_p[t*offset1 + offset2] - AuxNeuronState_c[t*offset1 + offset2]);
						}
						AuxNeuronState_p1[z*offset1 + offset2] = aux + AuxNeuronState_p[z*offset1 + offset2];
					}

					//We calculate the difference between both aproximations.
					float aux = 0.0;
					float aux2 = 0.0;
					for (int z = 0; z < this->neuron_model->N_DifferentialNeuronState; z++){
						aux = fabs(AuxNeuronState_p1[z*offset1 + offset2] - AuxNeuronState_p[z*offset1 + offset2]);
						AuxNeuronState_p[z*offset1 + offset2] = AuxNeuronState_p1[z*offset1 + offset2];
						if (aux > aux2){
							aux2 = aux;
						}
					}
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
						D[i*SizeStates*this->neuron_model->N_DifferentialNeuronState + j*SizeStates + index] = -D[(i - 1)*SizeStates*this->neuron_model->N_DifferentialNeuronState + j*SizeStates + index];
					}
					D[0 * SizeStates*this->neuron_model->N_DifferentialNeuronState + j*SizeStates + index] = AuxNeuronState_p[j*offset1 + offset2] - NeuronState[j*SizeStates + index];
					for (int i = 1; i < state[index]; i++){
						D[i*SizeStates*this->neuron_model->N_DifferentialNeuronState + j*SizeStates + index] += D[(i - 1)*SizeStates*this->neuron_model->N_DifferentialNeuronState + j*SizeStates + index];
					}
				}

				if (state[index]>1){
					for (int i = state[index] - 2; i > 0; i--){
						for (int j = 0; j < this->neuron_model->N_DifferentialNeuronState; j++){
							PreviousNeuronState[i*SizeStates*this->neuron_model->N_DifferentialNeuronState + j*SizeStates + index] = PreviousNeuronState[(i - 1)*SizeStates*this->neuron_model->N_DifferentialNeuronState + j*SizeStates + index];
						}
					}
					for (int j = 0; j < this->neuron_model->N_DifferentialNeuronState; j++){
						PreviousNeuronState[0 * SizeStates*this->neuron_model->N_DifferentialNeuronState + j*SizeStates + index] = NeuronState[j*SizeStates + index];
					}
				}
				for (int z = 0; z < this->neuron_model->N_DifferentialNeuronState; z++){
					NeuronState[z*SizeStates + index] = AuxNeuronState_p[z*offset1 + offset2];
				}

				//Finaly, we evaluate the neural state variables with time dependence.
				neuron_model->EvaluateTimeDependentEquation(index, SizeStates, NeuronState, index, 0);

				//Update the last spike time.
				this->neuron_model->vectorNeuronState_GPU2->LastSpikeTimeGPU[index] += this->elapsedTimeInSeconds;

				neuron_model->EvaluateSpikeCondition(previous_V, NeuronState, index, this->elapsedTimeInNeuronModelScale);

				neuron_model->CheckValidIntegration(index);
				index += blockDim.x*gridDim.x;
			}
			
		}

		/*!
		 * \brief It calculate numerically the Jacobian.
		 *
		 * It calculate numerically the Jacobian.
		 *
		 * \param index Index of the cell inside the neuron model.
		 * \param SizeStates Number of neuron.
		 * \param NeuronState Vector of neuron state variables for all neurons.
		 * \param jancum vector where is stored the Jacobian.
		 */
		__device__ void Jacobian(int index, int SizeStates, float * NeuronState, float * jacnum){
			float epsi=9.5367431640625e-7;
	
			for (int j=0; j<this->neuron_model->N_DifferentialNeuronState; j++){
				for (int i=0; i<this->neuron_model->N_NeuronStateVariables; i++){
					AuxNeuronState2[i*SizeStates + index]=NeuronState[i*SizeStates + index];
				}
				AuxNeuronState2[j*SizeStates + index]+=epsi;
				neuron_model->EvaluateDifferentialEquation(index, SizeStates, AuxNeuronState2, AuxNeuronState_pos, elapsedTimeInNeuronModelScale);

				AuxNeuronState2[j*SizeStates + index]-=2*epsi;
				neuron_model->EvaluateDifferentialEquation(index, SizeStates, AuxNeuronState2, AuxNeuronState_neg, elapsedTimeInNeuronModelScale);

				for(int z=0; z<this->neuron_model->N_DifferentialNeuronState; z++){
					jacnum[z*SizeStates*this->neuron_model->N_DifferentialNeuronState+ j*SizeStates + index]=(AuxNeuronState_pos[z*SizeStates + index]-AuxNeuronState_neg[z*SizeStates + index])/(2*epsi);
				}

			} 
		}

		/*!
		 * \brief It calculate the inverse of a square matrix using Gauss-Jordan Method.
		 *
		 * It calculate the inverse of a square matrix using Gauss-Jordan Method.
		 *
		 * \param index Index of the cell inside the neuron model.
 		 * \param SizeStates Number of neuron.
		 * \param a pointer to the square matrixs.
		 * \param ainv pointer to the inverse of the square matrixs.
		 */
		__device__ void invermat(int index, int SizeStates, float *a, float *ainv) {
			
			if(this->neuron_model->N_DifferentialNeuronState==1){
				ainv[0]=1/a[0];
			}else{
				float coef, inv_elemento;
				int i,j, s;

				for (i=0;i<this->neuron_model->N_DifferentialNeuronState;i++){
					for(j=0;j<this->neuron_model->N_DifferentialNeuronState;j++){
						if(i==j)
							ainv[i*SizeStates*this->neuron_model->N_DifferentialNeuronState+ j*SizeStates + index]=1.0;
						else
							ainv[i*SizeStates*this->neuron_model->N_DifferentialNeuronState+ j*SizeStates + index]=0.0;
					}
				}
				//Iteraciones
				for (s=0;s<this->neuron_model->N_DifferentialNeuronState;s++)
				{
					inv_elemento=1.0/a[s*SizeStates*this->neuron_model->N_DifferentialNeuronState+ s*SizeStates + index];
					for (j=0;j<this->neuron_model->N_DifferentialNeuronState;j++){
						a[s*SizeStates*this->neuron_model->N_DifferentialNeuronState+ j*SizeStates + index]*=inv_elemento;
						ainv[s*SizeStates*this->neuron_model->N_DifferentialNeuronState+ j*SizeStates + index]*=inv_elemento;
					}

					for(i=0;i<this->neuron_model->N_DifferentialNeuronState;i++)
					{
						if (i==s)
							;
						else 
						{
							coef=a[i*SizeStates*this->neuron_model->N_DifferentialNeuronState+ s*SizeStates + index];
							for(j=0;j<this->neuron_model->N_DifferentialNeuronState;j++){
								a[i*SizeStates*this->neuron_model->N_DifferentialNeuronState+ j*SizeStates + index]-=a[s*SizeStates*this->neuron_model->N_DifferentialNeuronState+ j*SizeStates + index]*coef;
								ainv[i*SizeStates*this->neuron_model->N_DifferentialNeuronState+ j*SizeStates + index]-=ainv[s*SizeStates*this->neuron_model->N_DifferentialNeuronState+ j*SizeStates + index]*coef;
							}

						}
					}
				}
			}
		}


		/*!
		 * \brief It reset the state of the integration method for method with memory (e.g. BDF).
		 *
		 * It reset the state of the integration method for method with memory (e.g. BDF).
		 *
		 * \param index indicate which neuron must be reseted.
		 *
		 */
		__device__ void resetState(int index){
			state[index]=0;
		}


		/*!
		 * \brief It calculates the conductance exponential values for time driven neuron models.
		 *
		 * It calculates the conductance exponential values for time driven neuron models.
		 */
		__device__ virtual void Calculate_conductance_exp_values(){
			this->neuron_model->Initialize_conductance_exp_values(this->neuron_model->N_TimeDependentNeuronState,1);
			//index 0
			this->neuron_model->Calculate_conductance_exp_values(0, elapsedTimeInNeuronModelScale);
		}
};

#endif /* BDFN_GPU2_H_ */
