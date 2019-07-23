/***************************************************************************
 *                           AdditiveKernelChange.cpp                      *
 *                           -------------------                           *
 * copyright            : (C) 2010 by Jesus Garrido                        *
 * email                : jgarrido@atc.ugr.es                              *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#include "../../include/learning_rules/AdditiveKernelChange.h"

#include "../../include/spike/Interconnection.h"
#include "../../include/spike/Neuron.h"

#include "../../include/simulation/Utils.h"

#include "../../include/openmp/openmp.h"

#include <cmath>

AdditiveKernelChange::AdditiveKernelChange(int NewLearningRuleIndex):WithoutPostSynaptic(NewLearningRuleIndex){
}

AdditiveKernelChange::~AdditiveKernelChange(){
}


int AdditiveKernelChange::GetNumberOfVar() const{
	return 2;
}

void AdditiveKernelChange::LoadLearningRule(FILE * fh, long & Currentline, string fileName) throw (EDLUTFileException){
	skip_comments(fh,Currentline);

	if(fscanf(fh,"%f",&this->maxpos)==1 && fscanf(fh,"%f",&this->a1pre)==1 && fscanf(fh,"%f",&this->a2prepre)==1){
		if (this->maxpos <= 0){
			throw EDLUTFileException(TASK_LEARNING_RULE_LOAD, ERROR_ADDITIVE_KERNEL_CHANGE_VALUES, REPAIR_LEARNING_RULE_VALUES, Currentline, fileName.c_str(), true);
		}
	}else{
		throw EDLUTFileException(TASK_LEARNING_RULE_LOAD, ERROR_LEARNING_RULE_LOAD, REPAIR_ADDITIVE_KERNEL_CHANGE_LOAD, Currentline, fileName.c_str(), true);
	}
}



void AdditiveKernelChange::ApplyPreSynapticSpike(Interconnection * Connection,double SpikeTime){
	

	if(Connection->GetTriggerConnection()==false){
		int LearningRuleIndex = Connection->GetLearningRuleIndex_withoutPost();

		// Second case: the weight change is linked to this connection
		Connection->IncrementWeight(this->a1pre);

		// Update the presynaptic activity
		State->SetNewUpdateTime(LearningRuleIndex, SpikeTime, false);

		// Add the presynaptic spike influence
		State->ApplyPresynapticSpike(LearningRuleIndex);
	
	}else{
		Neuron * TargetNeuron=Connection->GetTarget();

		for (int i = 0; i<TargetNeuron->GetInputNumberWithoutPostSynapticLearning(); ++i){
			Interconnection * interi=TargetNeuron->GetInputConnectionWithoutPostSynapticLearningAt(i);

			if(interi->GetTriggerConnection()==false){
				// Apply sinaptic plasticity driven by teaching signal
				int LearningRuleIndex = interi->GetLearningRuleIndex_withoutPost();

				// Update the presynaptic activity
				State->SetNewUpdateTime(LearningRuleIndex, SpikeTime, false);
				// Update synaptic weight
				interi->IncrementWeight(this->a2prepre*State->GetPresynapticActivity(LearningRuleIndex));
			}
		}
	}
}



ostream & AdditiveKernelChange::PrintInfo(ostream & out){

	out << "- Additive Kernel Learning Rule: \t" << this->maxpos << "\t" << this->a1pre << "\t" << this->a2prepre << endl;


	return out;
}


