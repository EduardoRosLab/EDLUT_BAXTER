/***************************************************************************
 *                           BDFn_GPU.h                                    *
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

#ifndef BDFN_GPU_H_
#define BDFN_GPU_H_

/*!
 * \file BDFn_GPU.h
 *
 * \author Francisco Naveros
 * \date May 2013
 *
 * This file declares a class which implements six fixed step BDF (Backward Differentiation Formulas) integration methods (from 
 * first order to sixth order BDF integration method) in GPU (this class is stored in CPU memory and controles the
 * reservation and freeing of GPU auxiliar memory). This method implements a progressive implementation of the
 * higher order integration method using the lower order integration mehtod (BDF1->BDF2->...->BDF6).
 */

#include "./FixedStep_GPU.h"


/*!
 * \class BDFn
 *
 * \brief BDFn integration method in CPU for GPU
 *
 * This class abstracts the initializacion in CPU of six BDF integration methods (BDF1, BDF2, ..., BDF6) for GPU. 
 * This CPU class controles the reservation and freeing of GPU auxiliar memory.
 *
 * \author Francisco Naveros
 * \date May 2013
 */
class BDFn_GPU: public FixedStep_GPU{
	public:

		/*!
		 * \brief These vectors are used as auxiliar vectors in GPU memory.
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
		//For Coeficient
		float * Coeficient;


		/*!
		 * \brief This vector stores previous neuron state variable for all neurons. This one is used as a memory.
		*/
		float * PreviousNeuronState;


		/*!
		 * \brief This vector stores the difference between previous neuron state variable for all neurons. This 
		 * one is used as a memory.
		*/
		float * D;

		/*!
		 * \brief This vector contains the state of each neuron (BDF order). When the integration method is reseted (the values of the neuron model variables are
		 * changed outside the integration method, for instance when a neuron spikes and the membrane potential is reseted to the resting potential), the values
		 * store in PreviousNeuronState and D are no longer valid. In this case the order it is set to 0 and must grow in each integration step until it is reache
		 * the target order.
		*/
		int * state;

		/*!
		 * \brief This value stores the order of the integration method.
		*/
		int BDForder;

		/*!
		 * \brief This constant matrix stores the coefficients of each BDF order.
		*/
		const static float Coeficient_CPU [7*7];

	
		/*!
		 * \brief Constructor of the class with 5 parameter.
		 *
		 * It generates a new BDF object indicating the order of the method.
		 *
		 * \param N_neuronStateVariables number of state variables for each cell.
		 * \param N_differentialNeuronState number of state variables which are calculate with a differential equation for each cell.
		 * \param N_timeDependentNeuronState number of state variables which ara calculate with a time dependent equation for each cell.
		 * \param BDForder BDF order (1, 2, ..., 6).
		 * \param intergrationMethod name of the integration method (BDF1, BDF2, ..., BDF6)
		 */
		BDFn_GPU(TimeDrivenNeuronModel_GPU * NewModel, int N_neuronStateVariables, int N_differentialNeuronState, int N_timeDependentNeuronState, int BDForder, char * intergrationMethod);


		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		virtual ~BDFn_GPU();


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

#endif /* BDFN_GPU_H_ */
