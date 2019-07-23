/***************************************************************************
 *                           LoadIntegrationMethod_GPU.h                   *
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

#ifndef LOADINTEGRATIONMETHOD_GPU_H_
#define LOADINTEGRATIONMETHOD_GPU_H_

/*!
 * \file LoadIntegrationMethod_GPU.h
 *
 * \author Francisco Naveros
 * \date November 2013
 *
 * This file declares a class which load all integration methods in CPU for GPU.
 */

#include <string>
#include <string.h>
using namespace std;

#include "../../include/neuron_model/TimeDrivenNeuronModel_GPU.h"

#include "./IntegrationMethod_GPU.h"
#include "./FixedStep_GPU.h"
#include "./Euler_GPU.h"
#include "./RK2_GPU.h"
#include "./RK4_GPU.h"
#include "./BDFn_GPU.h"

#include "./BiFixedStep_GPU.h"
#include "./Bifixed_Euler_GPU.h"
#include "./Bifixed_RK2_GPU.h"
#include "./Bifixed_RK4_GPU.h"


#include "../../include/simulation/Utils.h"
#include "../../include/simulation/Configuration.h"




/*!
 * \class LoadIntegrationMethod_GPU
 *
 * \brief Load Integration methods in CPU for GPU
 *
 * \author Francisco Naveros
 * \date May 2013
 */
class LoadIntegrationMethod_GPU {
	protected:

	public:

		static IntegrationMethod_GPU * loadIntegrationMethod_GPU(TimeDrivenNeuronModel_GPU* model, string fileName, FILE *fh, long * Currentline, int N_NeuronStateVariables, int N_DifferentialNeuronState, int N_TimeDependentNeuronState)throw (EDLUTFileException){
			IntegrationMethod_GPU * Method;
			char ident_type[MAXIDSIZE+1];

			skip_comments(fh,*Currentline);
			if(fscanf(fh,"%s",ident_type)==1){
				skip_comments(fh,*Currentline);

				//DEFINE HERE NEW INTEGRATION METHOD
				if(strncmp(ident_type,"Euler",5)==0){
					Method=(Euler_GPU *) new Euler_GPU(model, N_NeuronStateVariables, N_DifferentialNeuronState, N_TimeDependentNeuronState);
				}else if(strncmp(ident_type,"RK2",3)==0){
					Method=(RK2_GPU *) new RK2_GPU(model, N_NeuronStateVariables, N_DifferentialNeuronState, N_TimeDependentNeuronState);
				}else if(strncmp(ident_type,"RK4",3)==0){
					Method=(RK4_GPU *) new RK4_GPU(model, N_NeuronStateVariables, N_DifferentialNeuronState, N_TimeDependentNeuronState);
				}else if(strncmp(ident_type,"BDF",3)==0 && atoi(&ident_type[3])>0 && atoi(&ident_type[3])<7){
					Method=(BDFn_GPU *) new BDFn_GPU(model, N_NeuronStateVariables, N_DifferentialNeuronState, N_TimeDependentNeuronState, atoi(&ident_type[3]), ident_type);
				}else if(strncmp(ident_type,"Bifixed_Euler",13)==0){
					Method=(Bifixed_Euler_GPU *) new Bifixed_Euler_GPU(model, N_NeuronStateVariables, N_DifferentialNeuronState, N_TimeDependentNeuronState);
				}else if(strncmp(ident_type,"Bifixed_RK2",11)==0){
					Method=(Bifixed_RK2_GPU *) new Bifixed_RK2_GPU(model, N_NeuronStateVariables, N_DifferentialNeuronState, N_TimeDependentNeuronState);
				}else if(strncmp(ident_type,"Bifixed_RK4",11)==0){
					Method=(Bifixed_RK4_GPU *) new Bifixed_RK4_GPU(model, N_NeuronStateVariables, N_DifferentialNeuronState, N_TimeDependentNeuronState);
				}else{
					throw EDLUTFileException(TASK_INTEGRATION_METHOD_TYPE, ERROR_INTEGRATION_METHOD_TYPE, REPAIR_INTEGRATION_METHOD_TYPE, *Currentline, fileName.c_str(), true);
				}

			}else{
				throw EDLUTFileException(TASK_INTEGRATION_METHOD_TYPE, ERROR_INTEGRATION_METHOD_READ, REPAIR_INTEGRATION_METHOD_READ, *Currentline, fileName.c_str(), true);
			}

			//We load the integration method parameter.
			Method->loadParameter(fh,Currentline,fileName.c_str());

			return Method;
		}
};




#endif /* LOADINTEGRATIONMETHOD_GPU_H_ */
