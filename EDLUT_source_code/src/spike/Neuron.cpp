/***************************************************************************
 *                           Neuron.cpp                                    *
 *                           -------------------                           *
 * copyright            : (C) 2009 by Jesus Garrido and Richard Carrillo   *
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

#include "../../include/spike/Neuron.h"

#include "../../include/spike/InternalSpike.h"
#include "../../include/spike/PropagatedSpike.h"
#include "../../include/spike/Interconnection.h"

#include "../../include/neuron_model/NeuronModel.h"
#include "../../include/neuron_model/VectorNeuronState.h"

#include "../../include/simulation/EventQueue.h"

#include "../../include/learning_rules/WithoutPostSynaptic.h"

#include "../../include/openmp/openmp.h"

#include "../../include/spike/NeuronPropagationDelayStructure.h"
#include "../../include/spike/NeuronModelPropagationDelayStructure.h"

#include <string>

using namespace std;

Neuron::Neuron():N_TriggerConnection(0){
}

Neuron::~Neuron(){
	//state is deleted en Neuron Model.
	if (this->OutputConnections!=0){
		for(int i=0; i<NumberOfOpenMPQueues; i++){
			delete [] OutputConnections[i];
		}
		delete [] this->OutputConnections;
	}

	if (this->InputLearningConnectionsWithPostSynapticLearning!=0){
		delete [] this->InputLearningConnectionsWithPostSynapticLearning;
	}

	if (this->InputLearningConnectionsWithoutPostSynapticLearning!=0){
		delete [] this->InputLearningConnectionsWithoutPostSynapticLearning;
	}

	if(N_TriggerConnection>0){
		delete [] this->TriggerConnection;
	}

	delete PropagationStructure;

	if (IndexInputLearningConnections[0]){
		delete IndexInputLearningConnections[0];
	}
	if (IndexInputLearningConnections[1]){
		delete IndexInputLearningConnections[1];
	}
	delete IndexInputLearningConnections;
}

void Neuron::InitNeuron(int NewIndex, int index_VectorNeuronState, NeuronModel * Type, bool Monitored, bool IsOutput, int blockIndex){

	this->type = Type;

	this->state = type->InitializeState();

	this->OutputConnections = 0;

	this->OutputConNumber = 0;

	this->InputLearningConnectionsWithPostSynapticLearning = 0;

	this->InputLearningConnectionsWithoutPostSynapticLearning = 0;

	this->InputConLearningNumberWithPostSynaptic = 0;

	this->InputConLearningNumberWithoutPostSynaptic = 0;

	this->index = NewIndex;

	this->index_VectorNeuronState = index_VectorNeuronState;

	this->monitored=Monitored;

	this->spikeCounter = 0; // For LSAM

	this->isOutput = IsOutput;

	this->OpenMP_queue_index=blockIndex;

	this->IndexInputLearningConnections = (int **) new int*[2]();
}

//long int Neuron::GetIndex() const{
//	return this->index;	
//}

//VectorNeuronState * Neuron::GetVectorNeuronState() const{
//	return this->state;
//}
   		
unsigned int Neuron::GetInputNumberWithPostSynapticLearning() const{
	return this->InputConLearningNumberWithPostSynaptic;
}

unsigned int Neuron::GetInputNumberWithoutPostSynapticLearning() const{
	return this->InputConLearningNumberWithoutPostSynaptic;
}

   		
//unsigned int Neuron::GetOutputNumber(int index) const{
//	return this->OutputConNumber[index];
//}

Interconnection * Neuron::GetInputConnectionWithPostSynapticLearningAt(unsigned int index) const{
	return *(this->InputLearningConnectionsWithPostSynapticLearning+index);
}

Interconnection * Neuron::GetInputConnectionWithoutPostSynapticLearningAt(unsigned int index) const{
	return *(this->InputLearningConnectionsWithoutPostSynapticLearning+index);
}
   		
void Neuron::SetInputConnectionsWithPostSynapticLearning(Interconnection ** Connections, unsigned int NumberOfConnections){
	if (this->InputLearningConnectionsWithPostSynapticLearning!=0){
		delete [] this->InputLearningConnectionsWithPostSynapticLearning;
	}
	
	this->InputLearningConnectionsWithPostSynapticLearning = Connections;

	this->InputConLearningNumberWithPostSynaptic = NumberOfConnections;
}

void Neuron::SetInputConnectionsWithoutPostSynapticLearning(Interconnection ** Connections, unsigned int NumberOfConnections){
	if (this->InputLearningConnectionsWithoutPostSynapticLearning!=0){
		delete [] this->InputLearningConnectionsWithoutPostSynapticLearning;
	}
	
	this->InputLearningConnectionsWithoutPostSynapticLearning = Connections;

	this->InputConLearningNumberWithoutPostSynaptic = NumberOfConnections;



	if(NumberOfConnections>0){
		Interconnection ** aux= (Interconnection **) new Interconnection * [NumberOfConnections];
		for(int i=0; i<NumberOfConnections; i++){
			if(Connections[i]->GetTriggerConnection()){
				aux[N_TriggerConnection]=Connections[i];
				N_TriggerConnection++;
			}
		}

		if(N_TriggerConnection>0){
			TriggerConnection=(Interconnection**)new Interconnection*[N_TriggerConnection];
			for(int i=0; i<N_TriggerConnection; i++){
				TriggerConnection[i]=aux[i];
			}
		}
		delete [] aux;
	}
}



   		
//Interconnection * Neuron::GetOutputConnectionAt(unsigned int index1, unsigned int index2) const{
//	return OutputConnections[index1][index2];
//}
   		
void Neuron::SetOutputConnections(Interconnection *** Connections, unsigned long * NumberOfConnections){
	if (this->OutputConnections!=0){
		delete [] this->OutputConnections;
	}
	
	this->OutputConnections = Connections;

	this->OutputConNumber = NumberOfConnections;
}

bool Neuron::IsOutputConnected(int index) const{
	return this->OutputConNumber[index]!=0;
}
   		
//int Neuron::IsMonitored() const{
//	return this->monitored;	
//}

//int Neuron::IsOutput() const{
//	return this->isOutput;	
//}

//NeuronModel * Neuron::GetNeuronModel() const{
//	return this->type;
//}

ostream & Neuron::PrintInfo(ostream & out) {
	out << "- Neuron: " << this->index << endl;

	out << "\tType: " << this->type->GetModelID() << endl;

	out << "\tInput Connections With Learning: " << this->InputConLearningNumberWithPostSynaptic + this->InputConLearningNumberWithoutPostSynaptic << endl;

	out << "\tOutput Connections: " << this->OutputConNumber << endl;

   	if (this->monitored) out << "\tMonitored" << endl;
   	else out << "\tNon-monitored" << endl;

   	if (this->isOutput) out << "\tOutput" << endl;
   	else out << "\tNon-output" << endl;

	return out;
}

/* For LSAM */
void Neuron::SetSpikeCounter(long n) {
  this->spikeCounter = n;
}

/* For LSAM */
long Neuron::GetSpikeCounter() {
  return this->spikeCounter;
}

void Neuron::SetIndex_VectorNeuronState(long int index){
	index_VectorNeuronState=index;
}

//long Neuron::GetIndex_VectorNeuronState(){
//	return index_VectorNeuronState;
//}

void Neuron::set_OpenMP_queue_index(int index){
	this->OpenMP_queue_index=index;
}

void Neuron::CalculateOutputDelayStructure(){
	PropagationStructure=new NeuronPropagationDelayStructure(this);

	for(int i=0; i<NumberOfOpenMPQueues; i++){
		for(int j=0;j<PropagationStructure->NDifferentDelays[i];j++){ 
			this->GetNeuronModel()->GetNeuronModelPropagationDelayStructure()->IncludeNewDelay(i, PropagationStructure->SynapseDelay[i][j]);
		}
	}
}


void Neuron::CalculateOutputDelayIndex(){
	PropagationStructure->CalculateOutputDelayIndex(this->GetNeuronModel()->GetNeuronModelPropagationDelayStructure());
}


void Neuron::initializeLearningRuleIndex(){
	//Calculate learning rule index for learning rules with postsynaptic learning
	if (this->GetInputNumberWithPostSynapticLearning()){
		this->IndexInputLearningConnections[0] = new int[this->GetInputNumberWithPostSynapticLearning()]();
		for (int i = 0; i < this->GetInputNumberWithPostSynapticLearning(); i++){
			if (this->GetInputConnectionWithPostSynapticLearningAt(i)->GetTriggerConnection()){
				IndexInputLearningConnections[0][i] = -1;
			}
			else{
				IndexInputLearningConnections[0][i] = this->GetInputConnectionWithPostSynapticLearningAt(i)->GetLearningRuleIndex_withPost();
			}
			this->GetInputConnectionWithPostSynapticLearningAt(i)->LearningRuleIndex_withPost_insideTargetNeuron = i;
		}
	}

	//Calculate learning rule index for learning rules without postsynaptic learning (trigger learning)
	if (this->GetInputNumberWithoutPostSynapticLearning()){
		this->IndexInputLearningConnections[1] = new int[this->GetInputNumberWithoutPostSynapticLearning()]();
		for (int i = 0; i < this->GetInputNumberWithoutPostSynapticLearning(); i++){
			if (this->GetInputConnectionWithoutPostSynapticLearningAt(i)->GetTriggerConnection()){
				IndexInputLearningConnections[1][i] = -1;
			}
			else{
				IndexInputLearningConnections[1][i] = this->GetInputConnectionWithoutPostSynapticLearningAt(i)->GetLearningRuleIndex_withoutPost();
			}
			this->GetInputConnectionWithoutPostSynapticLearningAt(i)->LearningRuleIndex_withoutPost_insideTargetNeuron = i;
		}
	}
}

