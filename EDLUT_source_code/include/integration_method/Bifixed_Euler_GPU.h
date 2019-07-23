/***************************************************************************
 *                           Bifixed_Euler_GPU.h                                 *
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

#ifndef Bifixed_EULER_GPU_H_
#define Bifixed_EULER_GPU_H_

/*!
 * \file Bifixed_Euler_GPU.h
 *
 * \author Francisco Naveros
 * \date May 2015
 *
 * This file declares a class which implements a multi step integration method in GPU (this class is stored
 * in CPU memory and controles the allocation and deleting of GPU auxiliar memory). 
 */

#include "./BiFixedStep_GPU.h"


/*!
 * \class Bifixed_Euler_GPU
 *
 * \brief Euler integration method in CPU for GPU.
 *
 * This class abstracts the initializacion in CPU of an Euler integration methods for GPU. This CPU class
 * controles the reservation and freeing of GPU auxiliar memory.
 *
 * \author Francisco Naveros
 * \date May 2013
 */


class Bifixed_Euler_GPU: public BiFixedStep_GPU{
	public:

		/*!
		 * \brief This vector is used as auxiliar vector.
		*/
		float * AuxNeuronState;


		/*!
		 * \brief Constructor of the class with 3 parameter.
		 *
		 * It generates a new Euler object.
		 *
		 * \param N_neuronStateVariables number of state variables for each cell.
		 * \param N_differentialNeuronState number of state variables which are calculate with a differential equation for each cell.
		 * \param N_timeDependentNeuronState number of state variables which ara calculate with a time dependent equation for each cell.
		 */
		Bifixed_Euler_GPU(TimeDrivenNeuronModel_GPU * NewModel, int N_neuronStateVariables, int N_differentialNeuronState, int N_timeDependentNeuronState);


		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		virtual ~Bifixed_Euler_GPU();


		/*!
		 * \brief This method reserves all the necesary GPU memory (this memory could be reserved directly in the GPU, but this 
		 * suppose some restriction in the amount of memory which can be reserved).
		 *
		 * This method reserves all the necesary GPU memory (this memory could be reserved directly in the GPU, but this 
		 * suppose some restriction in the amount of memory which can be reserved).
		 *
		 * \param N_neurons Number of neurons.
		 * \param Total_N_thread Number of thread in GPU.
		 */
		void InitializeMemoryGPU(int N_neurons, int Total_N_thread);
};

#endif /* Bifixed_EULER_GPU_H_ */
