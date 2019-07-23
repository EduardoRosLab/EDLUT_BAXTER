/***************************************************************************
 *                           IntegrationMethod_GPU2.h                      *
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

#ifndef INTEGRATIONMETHOD_GPU2_H_
#define INTEGRATIONMETHOD_GPU2_H_

/*!
 * \file IntegrationMethod_GPU.h
 *
 * \author Francisco Naveros
 * \date May 2013
 *
 * This file declares a class which abstracts all integration methods in GPU (this class is stored
 * in GPU memory and executed in GPU). This methods can be fixed-step or bi-fixed-step.
 */


//Library for CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"




/*!
 * \class IntegrationMethod_GPU2
 *
 * \brief Integration method in a GPU.
 *
 * This class abstracts the behavior of all integration methods for neurons in GPU in a 
 * time-driven spiking neural network.
 * It includes internal model functions which define the behavior of integration methods
 * (initialization, calculate next value, ...).
 * This is only a virtual function (an interface) which defines the functions of the
 * inherited classes.
 *
 * \author Francisco Naveros
 * \date May 2013
 */

class IntegrationMethod_GPU2 {
	public:

		/*!
		 * \brief Integration step size in seconds (the time scale of the simulator).
		*/
		float elapsedTimeInSeconds;

		/*!
		 * \brief Integration step size in seconds or miliseconds, depending on the neuron model that is going to be integrated.
		*/
		float elapsedTimeInNeuronModelScale;



		/*!
		 * \brief Constructor of the class with 2 parameter.
		 *
		 * It generates a new IntegrationMethod objectin GPU memory.
		 *
		 * \param TimeDrivenNeuronModel pointer to the time driven neuron model
		 * \param Buffer_GPU integration method parameters
		 */
		__device__ IntegrationMethod_GPU2(void ** Buffer_GPU, int time_scale){
			float * integration_method_parameters=((float*)Buffer_GPU[0]);
			elapsedTimeInSeconds=integration_method_parameters[0];
			elapsedTimeInNeuronModelScale=elapsedTimeInSeconds*time_scale;
		}


		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		__device__ virtual ~IntegrationMethod_GPU2(){
		}

		
		/*!
		 * \brief It calculate the next neural state variables of the model.
		 *
		 * It calculate the next neural state varaibles of the model.
		 *
		 * \param index Index of the cell inside the neuron model for method with memory (e.g. BDF).
		 * \param SizeStates Number of neurons
		 * \param NeuronState Vector of neuron state variables for all neurons.
		 */
		__device__ virtual void NextDifferentialEquationValues(int index, int SizeStates, float * NeuronState) {
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

#endif /* INTEGRATIONMETHOD_GPU2_H_ */
