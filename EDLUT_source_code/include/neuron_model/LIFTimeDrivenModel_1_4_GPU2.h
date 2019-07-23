/***************************************************************************
 *                           LIFTimeDrivenModel_1_4_GPU2.h                 *
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

#ifndef LIFTIMEDRIVENMODEL_1_4_GPU2_H_
#define LIFTIMEDRIVENMODEL_1_4_GPU2_H_

/*!
 * \file LIFTimeDrivenModel_GPU.h
 *
 * \author Francisco Naveros
 * \date November 2012
 *
 * This file declares a class which abstracts a Leaky Integrate-And-Fire neuron model with one 
 * differential equation and four time dependent equations (conductances). This model is
 * implemented in GPU.
 */

#include "./TimeDrivenNeuronModel_GPU2.h"
#include "../../include/integration_method/IntegrationMethod_GPU2.h"


//Library for CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

/*!
 * \class LIFTimeDrivenModel_GPU
 *
 * \brief Leaky Integrate-And-Fire Time-Driven neuron model
 *
 * This class abstracts the behavior of a neuron in a time-driven spiking neural network.
 * It includes internal model functions which define the behavior of the model
 * (initialization, update of the state, synapses effect, next firing prediction...).
 * This is only a virtual function (an interface) which defines the functions of the
 * inherited classes.
 *
 * \author Francisco Naveros
 * \date January 2012
 */

class LIFTimeDrivenModel_1_4_GPU2 : public TimeDrivenNeuronModel_GPU2 {
	public:
		/*!
		* \brief Excitatory reversal potential in V units
		*/
		const float eexc;

		/*!
		* \brief Inhibitory reversal potential in V units
		*/
		const float einh;

		/*!
		* \brief Resting potential units in V units
		*/
		const float erest;

		/*!
		* \brief Firing threshold units in V units
		*/
		const float vthr;

		/*!
		* \brief Membrane capacitance units in F units
		*/
		const float cm;
		const float inv_cm;

		/*!
		* \brief AMPA receptor time constant units in s units
		*/
		const float tampa;
		const float inv_tampa;

		/*!
		* \brief NMDA receptor time constant units in s units
		*/
		const float tnmda;
		const float inv_tnmda;

		/*!
		* \brief GABA receptor time constant units in s units
		*/
		const float tinh;
		const float inv_tinh;

		/*!
		* \brief Gap Junction time constant units in s units
		*/
		const float tgj;
		const float inv_tgj;

		/*!
		* \brief Refractory period units in s units
		*/
		const float tref;

		/*!
		* \brief Resting conductance units in nS units
		*/
		const float grest;

		/*!
		* \brief Gap junction factor units in V/nS units
		*/
		const float fgj;

		/*!
		 * \brief Number of state variables for each cell.
		*/
		static const int N_NeuronStateVariables=5;

		/*!
		 * \brief Number of state variables which are calculate with a differential equation for each cell.
		*/
		static const int N_DifferentialNeuronState=1;

		/*!
		 * \brief Number of state variables which are calculate with a time dependent equation for each cell.
		*/
		static const int N_TimeDependentNeuronState=4;


		/*!
		 * \brief constructor with parameters.
		 *
		 * It generates a new neuron model object.
		 *
		 * \param Eexc eexc.
		 * \param Einh einh.
		 * \param Erest erest.
		 * \param Vthr vthr.
		 * \param Cm cm.
		 * \param Tampa tampa.
		 * \param Tnmda tnmda.
		 * \param Tinh tinh.
		 * \param Tgj tgj.
		 * \param Tref tref.
		 * \param Grest grest.
		 * \param Fgj fgj.
		 * \param integrationName integration method type.
		 * \param N_neurons number of neurons.
		 * \param Total_N_thread total number of CUDA thread.
		 * \param Buffer_GPU Gpu auxiliar memory.
		 *
		 */
		__device__ LIFTimeDrivenModel_1_4_GPU2(float Eexc,float Einh,float Erest,float Vthr,float Cm,float Tampa,
			float Tnmda,float Tinh,float Tgj,float Tref,float Grest,float Fgj, char const* integrationName, int N_neurons,
			void ** Buffer_GPU):TimeDrivenNeuronModel_GPU2(SecondScale_GPU), eexc(Eexc),einh(Einh),erest(Erest),vthr(Vthr),cm(Cm),tampa(Tampa),
			tnmda(Tnmda),tinh(Tinh),tgj(Tgj), tref(Tref),grest(Grest),fgj(Fgj),inv_tampa(1.0f/tampa),inv_tnmda(1.0f/tnmda),inv_tinh(1.0f/tinh),
			inv_tgj(1.0f/tgj),inv_cm(1.0f/cm){
			loadIntegrationMethod_GPU2(integrationName, Buffer_GPU);

			integrationMethod_GPU2->Calculate_conductance_exp_values();	
		}

		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		__device__ virtual ~LIFTimeDrivenModel_1_4_GPU2(){
			delete integrationMethod_GPU2;
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
		//__device__ void UpdateState(double CurrentTime)
		//{
		//	int index = blockIdx.x * blockDim.x + threadIdx.x;
		//	while (index<vectorNeuronState_GPU2->SizeStates){
		//		vectorNeuronState_GPU2->VectorNeuronStates_GPU[1*vectorNeuronState_GPU2->SizeStates + index]+=vectorNeuronState_GPU2->AuxStateGPU[index];                                            //gAMPA
		//		vectorNeuronState_GPU2->VectorNeuronStates_GPU[2*vectorNeuronState_GPU2->SizeStates + index]+=vectorNeuronState_GPU2->AuxStateGPU[vectorNeuronState_GPU2->SizeStates + index];       //gNMDA
		//		vectorNeuronState_GPU2->VectorNeuronStates_GPU[3*vectorNeuronState_GPU2->SizeStates + index]+=vectorNeuronState_GPU2->AuxStateGPU[2*vectorNeuronState_GPU2->SizeStates + index];     //gGABA
		//		vectorNeuronState_GPU2->VectorNeuronStates_GPU[4*vectorNeuronState_GPU2->SizeStates + index]+=vectorNeuronState_GPU2->AuxStateGPU[3*vectorNeuronState_GPU2->SizeStates + index];     //gGJ

		//		this->integrationMethod_GPU2->NextDifferentialEquationValues(index, vectorNeuronState_GPU2->SizeStates, vectorNeuronState_GPU2->VectorNeuronStates_GPU);
		//		
		//		this->CheckValidIntegration(index);

		//		index+=blockDim.x*gridDim.x;
		//	}
		//} 

		__device__ void UpdateState(double CurrentTime)
		{
			int index = blockIdx.x * blockDim.x + threadIdx.x;
			while (index<vectorNeuronState_GPU2->SizeStates){
				vectorNeuronState_GPU2->VectorNeuronStates_GPU[1 * vectorNeuronState_GPU2->SizeStates + index] += vectorNeuronState_GPU2->AuxStateGPU[index];                                            //gAMPA
				vectorNeuronState_GPU2->VectorNeuronStates_GPU[2 * vectorNeuronState_GPU2->SizeStates + index] += vectorNeuronState_GPU2->AuxStateGPU[vectorNeuronState_GPU2->SizeStates + index];       //gNMDA
				vectorNeuronState_GPU2->VectorNeuronStates_GPU[3 * vectorNeuronState_GPU2->SizeStates + index] += vectorNeuronState_GPU2->AuxStateGPU[2 * vectorNeuronState_GPU2->SizeStates + index];     //gGABA
				vectorNeuronState_GPU2->VectorNeuronStates_GPU[4 * vectorNeuronState_GPU2->SizeStates + index] += vectorNeuronState_GPU2->AuxStateGPU[3 * vectorNeuronState_GPU2->SizeStates + index];     //gGJ

				index += blockDim.x*gridDim.x;
			}

			this->integrationMethod_GPU2->NextDifferentialEquationValues(vectorNeuronState_GPU2->SizeStates, vectorNeuronState_GPU2->VectorNeuronStates_GPU);
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
		__device__ void EvaluateSpikeCondition(float previous_V, float * NeuronState, int index, float elapsedTimeInNeuronModelScale){
			float vm_cou = NeuronState[index] + this->fgj * NeuronState[4*vectorNeuronState_GPU2->SizeStates + index];
			if (vm_cou > this->vthr){		
				NeuronState[index] = this->erest;
				vectorNeuronState_GPU2->LastSpikeTimeGPU[index]=0.0;
				this->integrationMethod_GPU2->resetState(index);
				vectorNeuronState_GPU2->InternalSpikeGPU[index] = true;
			}
		}

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
		__device__ void EvaluateDifferentialEquation(int index, int SizeStates, float * NeuronState, float * AuxNeuronState, float elapsed_time){
			if(vectorNeuronState_GPU2->LastSpikeTimeGPU[index] * this->GetTimeScale() > this->tref){
				float iampa = NeuronState[SizeStates + index]*(this->eexc-NeuronState[index]);
				float gnmdainf = 1.0f/(1.0f + __expf(-62.0f*NeuronState[index])*(1.2f/3.57f));
				float inmda = NeuronState[2*SizeStates + index]*gnmdainf*(this->eexc-NeuronState[index]);
				float iinh = NeuronState[3*SizeStates + index]*(this->einh-NeuronState[index]);
				AuxNeuronState[blockDim.x*blockIdx.x + threadIdx.x]=(iampa + inmda + iinh + this->grest* (this->erest-NeuronState[index]))*1.e-9f*this->inv_cm;
			}else if ((vectorNeuronState_GPU2->LastSpikeTimeGPU[index] * this->GetTimeScale() + elapsed_time)>this->tref){
				float fraction = (this->vectorNeuronState_GPU2->LastSpikeTimeGPU[index] * this->GetTimeScale() + elapsed_time - this->tref) / elapsed_time;
				float iampa = NeuronState[SizeStates + index]*(this->eexc-NeuronState[index]);
				float gnmdainf = 1.0f/(1.0f + __expf(-62.0f*NeuronState[index])*(1.2f/3.57f));
				float inmda = NeuronState[2*SizeStates + index]*gnmdainf*(this->eexc-NeuronState[index]);
				float iinh = NeuronState[3*SizeStates + index]*(this->einh-NeuronState[index]);
				AuxNeuronState[blockDim.x*blockIdx.x + threadIdx.x]=fraction*(iampa + inmda + iinh + this->grest* (this->erest-NeuronState[index]))*1.e-9f*this->inv_cm;
			}else{
				AuxNeuronState[blockDim.x*blockIdx.x + threadIdx.x]=0.0f;
			}
		}


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
		__device__ void EvaluateTimeDependentEquation(int index, int SizeStates, float * NeuronState, float elapsed_time, int elapsed_time_index){
			float limit=1e-9;

			float * Conductance_values=this->Get_conductance_exponential_values(elapsed_time_index);

			if(NeuronState[this->N_DifferentialNeuronState*SizeStates + index]<limit){
				NeuronState[this->N_DifferentialNeuronState*SizeStates + index]=0.0f;
			}else{
				NeuronState[this->N_DifferentialNeuronState*SizeStates + index]*=  Conductance_values[0];
			}
			if(NeuronState[(this->N_DifferentialNeuronState+1)*SizeStates + index]<limit){
				NeuronState[(this->N_DifferentialNeuronState+1)*SizeStates + index]=0.0f;
			}else{
				NeuronState[(this->N_DifferentialNeuronState+1)*SizeStates + index]*=  Conductance_values[1];
			}

			if(NeuronState[(this->N_DifferentialNeuronState+2)*SizeStates + index]<limit){
				NeuronState[(this->N_DifferentialNeuronState+2)*SizeStates + index]=0.0f;
			}else{
				NeuronState[(this->N_DifferentialNeuronState+2)*SizeStates + index]*=  Conductance_values[2];
			}
			if(NeuronState[(this->N_DifferentialNeuronState+3)*SizeStates + index]<limit){
				NeuronState[(this->N_DifferentialNeuronState+3)*SizeStates + index]=0.0f;
			}else{
				NeuronState[(this->N_DifferentialNeuronState+3)*SizeStates + index]*=  Conductance_values[3];
			}
		}


		/*!
		 * \brief It calculates the conductace exponential value for an elapsed time.
		 *
		 * It calculates the conductace exponential value for an elapsed time.
		 *
		 * \param index elapsed time index .
		 * \param elapses_time elapsed time.
		 */
		__device__ void Calculate_conductance_exp_values(int index, float elapsed_time){
			//ampa synapse.
			Set_conductance_exp_values(index, 0, __expf(-elapsed_time*this->inv_tampa));
			//nmda synapse.
			Set_conductance_exp_values(index, 1, __expf(-elapsed_time*this->inv_tnmda));
			//inhibitory synapse.
			Set_conductance_exp_values(index, 2, __expf(-elapsed_time*this->inv_tinh));
			//gap junction synapse.
			Set_conductance_exp_values(index, 3, __expf(-elapsed_time*this->inv_tgj));

		}



		/*!
		* \brief It loads the integration method from the neuron model configuration file
		*
		* It loads the integration method from the neuron model configuration file
		*
		* \param integrationName integration method name
		* \param Buffer_GPU integration method parameters
		*/
		__device__ void loadIntegrationMethod_GPU2(char const* integrationName, void ** Buffer_GPU){

			//DEFINE HERE NEW INTEGRATION METHOD
			if (cmp4(integrationName, "Euler", 5) == 0){
				this->integrationMethod_GPU2 = (Euler_GPU2<LIFTimeDrivenModel_1_4_GPU2> *) new Euler_GPU2<LIFTimeDrivenModel_1_4_GPU2>(this, Buffer_GPU);
			}
			else if (cmp4(integrationName, "RK2", 3) == 0){
				this->integrationMethod_GPU2 = (RK2_GPU2<LIFTimeDrivenModel_1_4_GPU2> *) new RK2_GPU2<LIFTimeDrivenModel_1_4_GPU2>(this, Buffer_GPU);
			}
			else if (cmp4(integrationName, "RK4", 3) == 0){
				this->integrationMethod_GPU2 = (RK4_GPU2<LIFTimeDrivenModel_1_4_GPU2> *) new RK4_GPU2<LIFTimeDrivenModel_1_4_GPU2>(this, Buffer_GPU);
			}
			else if (cmp4(integrationName, "BDF", 3) == 0 && atoiGPU(integrationName, 3)>0 && atoiGPU(integrationName, 3)<7){
				this->integrationMethod_GPU2 = (BDFn_GPU2<LIFTimeDrivenModel_1_4_GPU2> *) new BDFn_GPU2<LIFTimeDrivenModel_1_4_GPU2>(this, Buffer_GPU, atoiGPU(integrationName, 3));
			}
			else if (cmp4(integrationName, "Bifixed_Euler", 13) == 0){
				this->integrationMethod_GPU2 = (Bifixed_Euler_GPU2<LIFTimeDrivenModel_1_4_GPU2> *) new Bifixed_Euler_GPU2<LIFTimeDrivenModel_1_4_GPU2>(this, Buffer_GPU);
			}
			else if (cmp4(integrationName, "Bifixed_RK2", 11) == 0){
				this->integrationMethod_GPU2 = (Bifixed_RK2_GPU2<LIFTimeDrivenModel_1_4_GPU2> *) new Bifixed_RK2_GPU2<LIFTimeDrivenModel_1_4_GPU2>(this, Buffer_GPU);
			}
			else if (cmp4(integrationName, "Bifixed_RK4", 11) == 0){
				this->integrationMethod_GPU2 = (Bifixed_RK4_GPU2<LIFTimeDrivenModel_1_4_GPU2> *) new Bifixed_RK4_GPU2<LIFTimeDrivenModel_1_4_GPU2>(this, Buffer_GPU);
			}
			else{
				printf("There was an error loading the integration methods of the GPU.\n");

			}
		}

};


#endif /* LIFTIMEDRIVENMODEL_1_4_GPU2_H_ */
