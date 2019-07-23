/***************************************************************************
 *                           Euler_GPU2.h                                  *
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

#ifndef EULER_GPU2_H_
#define EULER_GPU2_H_

/*!
 * \file Euler_GPU2.h
 *
 * \author Francisco Naveros
 * \date May 2013
 *
 * This file declares a class which implement a fixed step Euler integration method in GPU (this class is stored
 * in GPU memory and executed in GPU. 
 */

#include "./FixedStep_GPU2.h"

#include "../../include/neuron_model/TimeDrivenNeuronModel_GPU2.h"


//Library for CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/*!
 * \class Euler_GPU2
 *
 * \brief Euler integration method in GPU.
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

class Euler_GPU2 : public FixedStep_GPU2 {
	public:

		/*
		* Time driven neuron model
		*/
		Neuron_Model_GPU2 * neuron_model;

		/*!
		 * \brief This vector is used as an auxiliar vector. 
		*/
		float * AuxNeuronState;


		/*!
		* \brief Constructor of the class with 2 parameter.
		*
		* It generates a new Euler Integration Method objectin GPU memory.
		*
		* \param TimeDrivenNeuronModel pointer to the time driven neuron model
		* \param Buffer_GPU integration method parameters
		*/
		__device__ Euler_GPU2(Neuron_Model_GPU2* NewModel, void ** Buffer_GPU) :FixedStep_GPU2(Buffer_GPU, NewModel->timeScale), neuron_model(NewModel){
			AuxNeuronState=((float*)Buffer_GPU[1]);
		}

		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		__device__ ~Euler_GPU2(){
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

			neuron_model->EvaluateDifferentialEquation(index, SizeStates, NeuronState, AuxNeuronState, this->elapsedTimeInNeuronModelScale);

			int offset1=gridDim.x*blockDim.x;
			int offset2=blockDim.x*blockIdx.x + threadIdx.x;
			for (int j=0; j<this->neuron_model->N_DifferentialNeuronState; j++){
				NeuronState[j*SizeStates + index]+=this->elapsedTimeInNeuronModelScale*AuxNeuronState[j*offset1 + offset2];
			}

			//Finaly, we evaluate the neural state variables with time dependence.
			neuron_model->EvaluateTimeDependentEquation(index, SizeStates, NeuronState, this->elapsedTimeInNeuronModelScale, 0);

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

				neuron_model->EvaluateDifferentialEquation(index, SizeStates, NeuronState, AuxNeuronState, this->elapsedTimeInNeuronModelScale);

				int offset1 = gridDim.x*blockDim.x;
				int offset2 = blockDim.x*blockIdx.x + threadIdx.x;
				for (int j = 0; j < this->neuron_model->N_DifferentialNeuronState; j++){
					NeuronState[j*SizeStates + index] += this->elapsedTimeInNeuronModelScale*AuxNeuronState[j*offset1 + offset2];
				}

				//Finaly, we evaluate the neural state variables with time dependence.
				neuron_model->EvaluateTimeDependentEquation(index, SizeStates, NeuronState, this->elapsedTimeInNeuronModelScale, 0);

				//Update the last spike time.
				this->neuron_model->vectorNeuronState_GPU2->LastSpikeTimeGPU[index] += this->elapsedTimeInSeconds;

				neuron_model->EvaluateSpikeCondition(previous_V, NeuronState, index, this->elapsedTimeInNeuronModelScale);

				neuron_model->CheckValidIntegration(index);
				index += blockDim.x*gridDim.x;
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



#endif /* EULER_GPU2_H_ */
