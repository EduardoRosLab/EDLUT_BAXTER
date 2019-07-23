/***************************************************************************
 *                           VectorNeuronState_GPU.h                       *
 *                           -------------------                           *
 * copyright            : (C) 2012 by Jesus Garrido and Francisco Naveros  *
 * email                : jgarrido@atc.ugr.es, fnaveros@ugr.es             *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef VECTORNEURONSTATE_GPU2_H_
#define VECTORNEURONSTATE_GPU2_H_

/*!
 * \file VectorNeuronState_GPU.h
 *
 * \author Jesus Garrido
 * \author Francisco Naveros
 * \date February 2012
 *
 * This file declares a class which abstracts the current state of a cell vector in a GPU.
 * It also difines auxiliar vector to comunicate CPU and GPU.
 *
 * \note: This class is a modification of previous NeuronState_GPU class. In this new class,
 * it is generated a only object for a neuron model cell vector instead of a object for
 * each cell.
 */


/*!
 * \class VectorNeuronState
 *
 * \brief Spiking neuron current state.
 *
 * This class abstracts the state of a cell vector and defines the state variables of
 * that cell vector in a GPU. It is only for time-driven methods.
 *
 * \author Francisco Naveros
 * \date February 2012
 */

//Library for CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

class VectorNeuronState_GPU2 {

	public:

		/*!
		 * \brief Number of state variables for each cells.
		 */
		unsigned int NumberOfVariables;

		/*!
		 * brief It is used to store initial state of the neuron model. 
		 */
		float * InitialStateGPU;

		/*!
		 * \brief Auxiliary conductance incremental vector in GPU
		 */
		float * AuxStateGPU;

		/*!
	   	 * \brief Neuron state variables for all neuron model cell vector in GPU.
	   	 */
		float * VectorNeuronStates_GPU;

	   	/*!
	   	 * \brief Last update time for all neuron model cell vector in GPU.
	   	 */
		double * LastUpdateGPU;
	   	
		/*!
		 * \brief Time since last spike fired for all neuron model cell vector in GPU.
		 */
		double * LastSpikeTimeGPU;


		/*!
		 * \brief Time-driven methods in GPU use this vector to indicate which neurons have to
		 * generate a internal spike after a update event.
		 */
		bool * InternalSpikeGPU;

	   	/*!
		 * \brief The cell number inside the vector.
		 */
		int SizeStates;
		

		/*!
		 * \brief Default constructor with parameters.
		 *
		 * It generates a new state of a cell vector.
		 *
		 * \param NumVariables Number of the state variables this model needs.
		 */
		__device__ VectorNeuronState_GPU2(int NewNumberOfVariables ,float * NewInitialStateGPU, float * NewAuxStateGPU, float * NewVectorNeuronStates_GPU, double * NewLastUpdateGPU, double * NewLastSpikeTimeGPU, bool * NewInternalSpikeGPU, int NewSizeStates):
			NumberOfVariables(NewNumberOfVariables),InitialStateGPU(NewInitialStateGPU),AuxStateGPU(NewAuxStateGPU), VectorNeuronStates_GPU(NewVectorNeuronStates_GPU), 
			LastUpdateGPU(NewLastUpdateGPU),LastSpikeTimeGPU(NewLastSpikeTimeGPU),InternalSpikeGPU(NewInternalSpikeGPU), SizeStates(NewSizeStates){
		
		}

		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		__device__ ~VectorNeuronState_GPU2(){}


		__device__ void ResetState(int index){
			for(int i=0; i<NumberOfVariables; i++){
				VectorNeuronStates_GPU[i*this->SizeStates+index]=this->InitialStateGPU[i];
			}
		}
};

#endif /* VECTORNEURONSTATE_GPU_H_ */

