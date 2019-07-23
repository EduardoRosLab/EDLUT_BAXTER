/***************************************************************************
 *                           BiFixedStep_GPU.h                               *
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

#ifndef BIFIXEDSTEP_GPU_H_
#define BIFIXEDSTEP_GPU_H_

/*!
 * \file BiFixedStep_GPU.h
 *
 * \author Francisco Naveros
 * \date May 2015
 *
 * This file declares a class which abstracts all multi step integration methods in GPU (this class is stored
 * in CPU memory and controles the allocation and deleting of GPU auxiliar memory). All integration
 * methods in GPU are fixed step due to the parallel architecture of this one.
 */

#include <string>
#include "../../include/integration_method/IntegrationMethod_GPU.h"



using namespace std;

class TimeDrivenNeuronModel_GPU;


/*!
 * \class IntegrationMethod_GPU
 *
 * \brief Integration method in CPU for GPU.
 *
 * This class abstracts the initializacion in CPU of multi step integration methods for GPU. This CPU class
 * controles the allocation and deleting of GPU auxiliar memory.
 *
 * \author Francisco Naveros
 * \date May 2013
 */
class BiFixedStep_GPU : public IntegrationMethod_GPU  {
	public:

		/*!
		 * \brief Number of multi step in the adapatative zone.
		*/
		int N_BiFixedSteps;

		/*!
		 * \brief Elapsed time in neuron model scale of the adaptative zone.
		*/
		float bifixedElapsedTimeInNeuronModelScale;

		/*!
		 * \brief Elapsed time in second of the adaptative zone.
		*/
		float bifixedElapsedTimeInSeconds;

		/*!
		 * \brief Constructor of the class with 4 parameter.
		 *
		 * It generates a new IntegrationMethod_GPU object.
		 *
		 * \param integrationMethodType Integration method type.
		 * \param N_neuronStateVariables number of state variables for each cell.
		 * \param N_differentialNeuronState number of state variables which are calculate with a differential equation for each cell.
		 * \param N_timeDependentNeuronState number of state variables which ara calculate with a time dependent equation for each cell.
		 */
		BiFixedStep_GPU(TimeDrivenNeuronModel_GPU * NewModel, char * integrationMethodType, int N_neuronStateVariables, int N_differentialNeuronState, int N_timeDependentNeuronState);


		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		virtual ~BiFixedStep_GPU();
		

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
		virtual void InitializeMemoryGPU(int N_neurons, int Total_N_thread)=0;

	
		/*!
		 * \brief It loads the integration method parameters.
		 *
		 * It loads the integration method parameters from the file that define the parameter of the neuron model.
		 *
		 * \param Pointer to a neuron description file (*.cfg). At the end of this file must be included 
		 *  the integration method type and its parameters.
		 * \param Currentline line inside the neuron description file where start the description of the integration method parameter. 
		 * \param fileName file name.
		 *
		 * \throw EDLUTFileException If something wrong has happened in the file load.
		 */
		void loadParameter(FILE *fh, long * Currentline, string fileName) throw (EDLUTFileException);
};

#endif /* BIFIXEDSTEP_GPU_H_ */
