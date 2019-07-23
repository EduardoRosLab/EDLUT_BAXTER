/***************************************************************************
 *                           TimeDrivenNeuronModel_GPU2.h                  *
 *                           -------------------                           *
 * copyright            : (C) 2012 by Francisco Naveros                    *
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

#ifndef TIMEDRIVENNEURONMODEL_GPU2_H_
#define TIMEDRIVENNEURONMODEL_GPU2_H_

/*!
 * \file TimeDrivenNeuronModel_GPU.h
 *
 * \author Francisco Naveros
 * \date November 2012
 *
 * This file declares a class which abstracts an time-driven neuron model in a GPU.
 */

#include <stdio.h>

#include "../integration_method/IntegrationMethod_GPU2.h"
#include "../integration_method/Euler_GPU2.h"
#include "../integration_method/RK2_GPU2.h"
#include "../integration_method/RK4_GPU2.h"
#include "../integration_method/BDFn_GPU2.h"

#include "../integration_method/Bifixed_Euler_GPU2.h"
#include "../integration_method/Bifixed_RK2_GPU2.h"
#include "../integration_method/Bifixed_RK4_GPU2.h"

#include "./VectorNeuronState_GPU2.h"

enum TimeScale_GPU {SecondScale_GPU=1, MilisecondScale_GPU=1000};

//Library for CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


/*!
 * \class TimeDrivenNeuronModel_GPU
 *
 * \brief Time-Driven Spiking neuron model in a GPU
 *
 * This class abstracts the behavior of a neuron in a time-driven spiking neural network.
 * It includes internal model functions which define the behavior of the model
 * (initialization, update of the state, synapses effect, next firing prediction...).
 * This is only a virtual function (an interface) which defines the functions of the
 * inherited classes.
 *
 * \author Francisco Naveros
 * \date November 2012
 */

class TimeDrivenNeuronModel_GPU2{
	public:

		/*!
		 * \brief This variable indicate if the neuron model has a time scale of seconds or miliseconds.
		*/
		TimeScale_GPU timeScale;

		/*!
		 * \brief integration method.
		*/
		IntegrationMethod_GPU2 * integrationMethod_GPU2;


		/*!
		 * \brief Vector neuron state in GPU.
		*/
		VectorNeuronState_GPU2 * vectorNeuronState_GPU2;

		/*!
		 * \brief Auxiliar array for time dependente variables.
		*/
		float * conductance_exp_values;

		/*!
		 * \brief Auxiliar variable for time dependente variables.
		*/
		int N_conductances;


		/*!
		 * \brief Default constructor with parameters.
		 *
		 * It generates a new neuron model object without being initialized.
		 */
		__device__ TimeDrivenNeuronModel_GPU2(TimeScale_GPU timeScale):timeScale(timeScale){
		}


		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		__device__ virtual ~TimeDrivenNeuronModel_GPU2(){
			//delete integrationMethod_GPU2;
			delete vectorNeuronState_GPU2;

			if(N_conductances!=0){
				delete conductance_exp_values;
			}
		}


		/*!
		 * \brief Update the neuron state variables.
		 *
		 * It updates the neuron state variables.
		 *
		 * \param index The cell index inside the StateGPU. 
		 * \param AuxStateGPU Auxiliary incremental conductance vector.
		 * \param StateGPU Neural state variables.
		 * \param LastUpdateGPU Last update time
		 * \param LastSpikeTimeGPU Last spike time
		 * \param InternalSpikeGPU In this vector is stored if a neuron must generate an output spike.
		 * \param SizeStates Number of neurons
		 * \param CurrentTime Current time.
		 *
		 * \return True if an output spike have been fired. False in other case.
		 */
		__device__ virtual void UpdateState(double CurrentTime){
		}


		/*!
		 * \brief It evaluates if a neuron must spike.
		 *
		 * It evaluates if a neuron must spike.
		 *
		 * \param previous_V previous membrane potential
		 * \param NeuronState neuron state variables.
		 * \param index Neuron index inside the neuron model.
 		 * \param elapsedTimeInNeuronModelScale integration method step.
		 * \return It returns if a neuron must spike.
		 */
		//__device__ virtual void EvaluateSpikeCondition(float previous_V, float * NeuronState, int index, float elapsedTimeInNeuronModelScale){
		//}

		/*!
		 * \brief It evaluates the differential equation in NeuronState and it stores the results in AuxNeuronState.
		 *
		 * It evaluates the differential equation in NeuronState and it stores the results in AuxNeuronState.
		 *
		 * \param index index inside the NeuronState vector.
		 * \param SizeStates number of element in NeuronState vector.
		 * \param NeuronState value of the neuron state variables where differential equations are evaluated.
		 * \param AuxNeuronState results of the differential equations evaluation.
		 */
		//__device__ virtual void EvaluateDifferentialEquation(int index, int SizeStates, float * NeuronState, float * AuxNeuronState, float elapsed_time){
		//}

		/*!
		 * \brief It evaluates the time depedendent Equation in NeuronState for elapsed_time and it stores the results in NeuronState.
		 *
		 * It evaluates the time depedendent Equation in NeuronState for elapsed_time and it stores the results in NeuronState.
		 *
		 * \param index index inside the NeuronState vector.
		 * \param SizeStates number of element in NeuronState vector.
		 * \param NeuronState value of the neuron state variables where time dependent equations are evaluated.
		 * \param elapsed_time integration time step.
		 * \param elapsed_time_index index inside the conductance_exp_values array.
		 */
		//__device__ virtual void EvaluateTimeDependentEquation(int index, int SizeStates, float * NeuronState, float elapsed_time, int elapsed_time_index){
		//}


		/*!
		 * \brief It creates an vector neuron state object in GPU and store the vector allocated in the GPU by the CPU. 
		 *
		 * It creates an vector neuron state object in GPU and store the vector allocated in the GPU by the CPU.
		 *
		 * \param AuxStateGPU.
		 * \param VectorNeuronStates_GPU.
		 * \param LastUpdateGPU.
		 * \param LastSpikeTimeGPU.
		 * \param InternalSpikeGPU.
		 * \param SizeStates.
		 */
		__device__ void InitializeVectorNeuronState_GPU2(int NumberOfVariables, float * InitialStateGPU, float * AuxStateGPU, float * VectorNeuronStates_GPU, double * LastUpdateGPU, double * LastSpikeTimeGPU, bool * InternalSpikeGPU, int SizeStates){
			vectorNeuronState_GPU2=new VectorNeuronState_GPU2(NumberOfVariables, InitialStateGPU, AuxStateGPU, VectorNeuronStates_GPU, LastUpdateGPU, LastSpikeTimeGPU, InternalSpikeGPU, SizeStates);
		}


		/*!
		 * \brief It Checks if the integrations has work properly.
		 *
		 * It Checks if the integrations has work properly.
		 *
		 * \param index neuron model index.
		 */

		__device__ void CheckValidIntegration(int index){
			if(vectorNeuronState_GPU2->VectorNeuronStates_GPU[index]!=vectorNeuronState_GPU2->VectorNeuronStates_GPU[index]){
				printf("Integration error in GPU neuron model: neuron %d\n",index);
				vectorNeuronState_GPU2->ResetState(index);
			}
		}

		/*!
		 * \brief It initializes an auxiliar array for time dependente variables.
		 *
		 * It initializes an auxiliar array for time dependente variables.
		 *
		 * \param N_conductances .
		 * \param N_elapsed_times .
		 */
		__device__ void Initialize_conductance_exp_values(int N_conductances, int N_elapsed_times){
			conductance_exp_values = new float[N_conductances *  N_elapsed_times]();
			this->N_conductances=N_conductances;
		}

		/*!
		 * \brief It calculates the conductace exponential value for an elapsed time.
		 *
		 * It calculates the conductace exponential value for an elapsed time.
		 *
		 * \param index elapsed time index .
		 * \param elapses_time elapsed time.
		 */
		__device__ virtual void Calculate_conductance_exp_values(int index, float elapsed_time){}

		/*!
		 * \brief It sets the conductace exponential value for an elapsed time and a specific conductance.
		 *
		 * It sets the conductace exponential value for an elapsed time and a specific conductance.
		 *
		 * \param elapses_time_index elapsed time index.
		 * \param conductance_index conductance index.
		 * \param value.
		 */
		__device__ void Set_conductance_exp_values(int elapsed_time_index, int conductance_index, float value){
			conductance_exp_values[elapsed_time_index*N_conductances+conductance_index]=value;
		}
		/*!
		 * \brief It gets the conductace exponential value for an elapsed time and a specific conductance.
		 *
		 * It gets the conductace exponential value for an elapsed time and a specific conductance.
		 *
		 * \param elapses_time_index elapsed time index.
		 * \param conductance_index conductance index.
		 *
		 * \return A conductance exponential values.
		 */
		__device__ float Get_conductance_exponential_values(int elapsed_time_index, int conductance_index){
			return conductance_exp_values[elapsed_time_index*N_conductances+conductance_index];
		}
		/*!
		 * \brief It gets the conductace exponential value for an elapsed time.
		 *
		 * It gets the conductace exponential value for an elapsed time .
		 *
		 * \param elapses_time_index elapsed time index.
		 * 
		 * \return A pointer to a set of conductance exponential values.
		 */
		__device__ float * Get_conductance_exponential_values(int elapsed_time_index){
			return conductance_exp_values + elapsed_time_index*N_conductances;
		}

		/*!
		* \brief It gets the neuron model time scale.
		*
		* It gets the neuron model time scale.
		*
		* \return new_timeScale time scale.
		*/
		__device__ float GetTimeScale(){
			return this->timeScale;
		}


		__device__ int cmp4(char const* c1, char const* c2, int size){
			for (int j = 0; j<size; j++){
				if ((int)c1[j] >(int)c2[j]){
					return 1;
				}
				else if ((int)c1[j] < (int)c2[j]){
					return -1;
				}
			}
			return 0;
		}

		__device__ int atoiGPU(char const* data, int position){
			return (((int)data[position]) - 48);
		}
};


#endif /* TIMEDRIVENNEURONMODEL_GPU2_H_ */
