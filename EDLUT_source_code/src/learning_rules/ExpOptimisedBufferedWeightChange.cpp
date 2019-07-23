/***************************************************************************
 *                           ExpOptimisedBufferedWeightChange.cpp          *
 *                           -------------------                           *
 * copyright            : (C) 2019 by Francisco Naveros                    *
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


#include "../../include/learning_rules/ExpOptimisedBufferedWeightChange.h"
#include "../../include/learning_rules/BufferedActivityTimes.h"

#include "../../include/spike/Interconnection.h"
#include "../../include/spike/Neuron.h"

#include "../../include/simulation/Utils.h"

#include "../../include/openmp/openmp.h"

ExpOptimisedBufferedWeightChange::ExpOptimisedBufferedWeightChange(int NewLearningRuleIndex):AdditiveKernelChange(NewLearningRuleIndex), initpos(0),
bufferedActivityTimesNoTrigger(0), kernelLookupTable(0){
}

ExpOptimisedBufferedWeightChange::~ExpOptimisedBufferedWeightChange(){
	if(bufferedActivityTimesNoTrigger!=0){
		delete bufferedActivityTimesNoTrigger;
	}
	if(kernelLookupTable!=0){
		delete kernelLookupTable;
	}
}


void ExpOptimisedBufferedWeightChange::InitializeConnectionState(unsigned int NumberOfSynapses, unsigned int NumberOfNeurons){
	if (this->maxpos <= this->initpos){
		this->maxpos = this->initpos + 1e-6;
	}

	double step_size = 0.0001;

	this->maxTimeMeasured = this->maxpos;
	while (1){
		this->maxTimeMeasured += step_size;
		if ((1.0/(this->maxpos-this->initpos))*this->maxTimeMeasured*exp(-(this->maxTimeMeasured/(this->maxpos-this->initpos))+1)< 1e-2){
			break;
		}
	}
	this->maxTimeMeasured += this->initpos;


	this->N_elements = this->maxTimeMeasured / step_size + 1;
	kernelLookupTable = new float[this->N_elements];


	this->inv_maxTimeMeasured = 1.0f / this->maxTimeMeasured;
	//Precompute the kernel in the look-up table.
	for (int i = 0; i<N_elements; i++){
		double time = maxTimeMeasured*i / N_elements;
		if(time < this->initpos){
			kernelLookupTable[i] = 0.0;
		}else{
			double time_ = time - this->initpos;
		  kernelLookupTable[i] = (1.0/(this->maxpos-this->initpos))*time_*exp(-(time_/(this->maxpos-this->initpos))+1);
		}
	}

	//Inicitialize de buffer of activity
	bufferedActivityTimesNoTrigger = new BufferedActivityTimes(NumberOfNeurons);
}

int ExpOptimisedBufferedWeightChange::GetNumberOfVar() const{
	return 2;
}


void ExpOptimisedBufferedWeightChange::LoadLearningRule(FILE * fh, long & Currentline, string fileName) throw (EDLUTFileException){
	AdditiveKernelChange::LoadLearningRule(fh,Currentline,fileName);

	if((fscanf(fh,"%f",&this->initpos)==1)){
		if (this->initpos<0.0 and this->initpos < this->maxpos){
			throw EDLUTFileException(TASK_LEARNING_RULE_LOAD, ERROR_SIN_WEIGHT_CHANGE_EXPONENT, REPAIR_LEARNING_RULE_VALUES, Currentline, fileName.c_str(), true);
		}
	}else{
		throw EDLUTFileException(TASK_LEARNING_RULE_LOAD, ERROR_LEARNING_RULE_LOAD, REPAIR_EXP_OPTIMISED_BUFFERED_WEIGHT_CHANGE_LOAD, Currentline, fileName.c_str(), true);
	}
}



void ExpOptimisedBufferedWeightChange::ApplyPreSynapticSpike(Interconnection * Connection, double SpikeTime){

	if (Connection->GetTriggerConnection() == false){
		Connection->IncrementWeight(this->a1pre);
		int neuron_index = Connection->GetTarget()->GetIndex();
		int synapse_index = Connection->LearningRuleIndex_withoutPost_insideTargetNeuron;
		this->bufferedActivityTimesNoTrigger->InsertElement(neuron_index, SpikeTime, SpikeTime - this->maxTimeMeasured, synapse_index);

	}
	else{
		Neuron * TargetNeuron = Connection->GetTarget();
		int neuron_index = TargetNeuron->GetIndex();

		int N_elements = bufferedActivityTimesNoTrigger->ProcessElements(neuron_index, SpikeTime - this->maxTimeMeasured);
		SpikeData * spike_data = bufferedActivityTimesNoTrigger->GetOutputSpikeData();


		for (int i = 0; i < N_elements; i++){
			Interconnection * interi = TargetNeuron->GetInputConnectionWithoutPostSynapticLearningAt(spike_data[i].synapse_index);

			double ElapsedTime = SpikeTime - spike_data[i].time;
			int tableIndex = ElapsedTime*this->N_elements*this->inv_maxTimeMeasured;
			float value = this->kernelLookupTable[tableIndex];

			// Update synaptic weight
			interi->IncrementWeight(this->a2prepre*value);
		}
	}
}
