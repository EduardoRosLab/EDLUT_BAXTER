/***************************************************************************
 *                           FixedStep_GPU2.h                              *
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

#ifndef FIXEDSTEP_GPU2_H_
#define FIXEDSTEP_GPU2_H_

/*!
 * \file FixedStep_GPU2.h
 *
 * \author Francisco Naveros
 * \date May 2015
 *
 * This file declares a class which abstract all fixed step integration method in GPU (this class is stored
 * in GPU memory and executed in GPU. 
 */

#include "./IntegrationMethod_GPU2.h"


//Library for CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/*!
 * \class FixedStep_GPU2
 *
 * \brief fixed step integration method in GPU.
 * 
 * This class abstracts the behavior of all fixed step integration method for neurons in a 
 * time-driven spiking neural network.
 * It includes internal model functions which define the behavior of integration methods
 * (initialization, calculate next value, ...).
 *
 * \author Francisco Naveros
 * \date May 2015
 */
class FixedStep_GPU2 : public IntegrationMethod_GPU2 {
	public:


		/*!
		* \brief Constructor of the class with 2 parameter.
		*
		* It generates a new fixed step Integration Method objectin GPU memory.
		*
		* \param TimeDrivenNeuronModel pointer to the time driven neuron model
		* \param Buffer_GPU integration method parameters
		*/
		__device__ FixedStep_GPU2(void ** Buffer_GPU, int time_scale):IntegrationMethod_GPU2(Buffer_GPU, time_scale){
		}

		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		__device__ virtual ~FixedStep_GPU2(){
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
		__device__ virtual void NextDifferentialEquationValues(int index, int SizeStates, float * NeuronState){
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
		}


		/*!
		 * \brief It reset the state of the integration method for method with memory (e.g. BDF).
		 *
		 * It reset the state of the integration method for method with memory (e.g. BDF).
		 *
		 * \param index indicate which neuron must be reseted.
		 *
		 */
		__device__ virtual void resetState(int index){
		}

		/*!
		 * \brief It calculates the conductance exponential values for time driven neuron models.
		 *
		 * It calculates the conductance exponential values for time driven neuron models.
		 */
		__device__ virtual void Calculate_conductance_exp_values(){}


};



#endif /* FIXEDSTEP_GPU2_H_ */
