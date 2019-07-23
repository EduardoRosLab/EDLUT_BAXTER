/***************************************************************************
 *                           IntegrationMethod_GPU.h                       *
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

#ifndef INTEGRATIONMETHOD_GPU_H_
#define INTEGRATIONMETHOD_GPU_H_

/*!
 * \file IntegrationMethod_GPU.h
 *
 * \author Francisco Naveros
 * \date May 2013
 *
 * This file declares a class which abstracts all integration methods in GPU (this class is stored
 * in CPU memory and controles the allocation and deleting of GPU auxiliar memory). This methods can
 * be fixed-step or bi-fixed-step.
 */

#include <string>
#include <string.h>
#include "../../include/integration_method/IntegrationMethod_GPU2.h"

#include "../../include/simulation/Utils.h"
#include "../../include/spike/EDLUTFileException.h"



using namespace std;

class TimeDrivenNeuronModel_GPU;


/*!
 * \class IntegrationMethod_GPU
 *
 * \brief Integration method in CPU for GPU.
 *
 * This class abstracts the initializacion in CPU of integration methods for GPU. This CPU class
 * controles the allocation and deleting of GPU auxiliar memory.
 *
 * \author Francisco Naveros
 * \date May 2013
 */
class IntegrationMethod_GPU {
	public:

		/*!
		 * \brief Time driven neuron model associated to this integration method.
		*/
		TimeDrivenNeuronModel_GPU * model;


		/*!
		 * \brief This vector contains all the necesary GPU memory which have been reserved in the CPU (this memory
		 * could be reserved directly in the GPU, but this suppose some restriction in the amount of memory which can be reserved).
		*/
		void ** Buffer_GPU;

		/*!
		 * \brief Number of state variables for each cell.
		*/
		int N_NeuronStateVariables;

		/*!
		 * \brief Number of state variables which are calculate with a differential equation for each cell.
		*/
		int N_DifferentialNeuronState;

		/*!
		 * \brief Number of state variables which are calculate with a time dependent equation for each cell.
		*/
		int N_TimeDependentNeuronState;

		/*!
		 * \brief Integration method type.
		*/
		char * IntegrationMethodType;

		/*!
		 * \brief Integration step size in seconds (the time scale of the simulator).
		*/
		float elapsedTimeInSeconds;

		/*!
		 * \brief Integration step size in seconds or miliseconds, depending on the neuron model that is going to be integrated.
		*/
		float elapsedTimeInNeuronModelScale;


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
		IntegrationMethod_GPU(TimeDrivenNeuronModel_GPU * NewModel, char * integrationMethodType, int N_neuronStateVariables, int N_differentialNeuronState, int N_timeDependentNeuronState);


		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		virtual ~IntegrationMethod_GPU();


		/*!
		 * \brief It gets the integration method type.
		 *
		 * It gets the integration method type.
		 *
		 * \return The integration method type.
		 */
		char * GetType();


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
		 * \brief It prints the integration method info.
		 *
		 * It prints the current integration method characteristics.
		 *
		 * \param out The stream where it prints the information.
		 *
		 * \return The stream after the printer.
		 */
		ostream & PrintInfo(ostream & out);


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
		virtual void loadParameter(FILE *fh, long * Currentline, string fileName) throw (EDLUTFileException) = 0 ;
};

#endif /* INTEGRATIONMETHOD_GPU_H_ */
