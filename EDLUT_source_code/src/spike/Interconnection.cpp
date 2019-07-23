/***************************************************************************
 *                           Interconnection.cpp                           *
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

#include "../../include/spike/Interconnection.h"

#include <cmath>

#include "../../include/spike/Neuron.h"
#include "../../include/neuron_model/NeuronModel.h"

#include "../../include/learning_rules/LearningRule.h"
#include "../../include/learning_rules/ConnectionState.h"

Interconnection::Interconnection() : source(0), target(0), index(0), delay(0), type(0), subindex_type(0), weight(0), maxweight(0), wchange_withPost(0), LearningRuleIndex_withPost(-1), wchange_withoutPost(0), LearningRuleIndex_withoutPost(-1), TriggerConnection(false), LearningRuleIndex_withPost_insideTargetNeuron(-1), LearningRuleIndex_withoutPost_insideTargetNeuron(-1), targetNeuronModel(0), targetNeuronModelIndex(1){

}

Interconnection::Interconnection(int NewIndex, Neuron * NewSource, Neuron * NewTarget, float NewDelay, int NewType, float NewWeight, float NewMaxWeight, LearningRule* NewWeightChange_withPost, unsigned int NewLearningRuleIndex_withPost, LearningRule* NewWeightChange_withoutPost, unsigned int NewLearningRuleIndex_withoutPost):
source(NewSource), target(NewTarget), index(NewIndex), delay(NewDelay), type(NewType), subindex_type(0), weight(NewWeight), maxweight(NewMaxWeight), wchange_withPost(NewWeightChange_withPost), LearningRuleIndex_withPost(NewLearningRuleIndex_withPost), wchange_withoutPost(NewWeightChange_withoutPost), LearningRuleIndex_withoutPost(NewLearningRuleIndex_withoutPost),
	LearningRuleIndex_withPost_insideTargetNeuron(-1), LearningRuleIndex_withoutPost_insideTargetNeuron(-1){
}

Interconnection::~Interconnection(){

}

long int Interconnection::GetIndex() const{
	return this->index;
}
		
void Interconnection::SetIndex(long int NewIndex){
	this->index = NewIndex;
}
		
//Neuron * Interconnection::GetSource() const{
//	return this->source;	
//}
		
void Interconnection::SetSource(Neuron * NewSource){
	this->source = NewSource;	
}
		
//Neuron * Interconnection::GetTarget() const{
//	return this->target;
//}
		
void Interconnection::SetTarget(Neuron * NewTarget){
	this->target = NewTarget;
}
		
//double Interconnection::GetDelay() const{
//	return this->delay;
//}
		
void Interconnection::SetDelay(double NewDelay){
	this->delay = NewDelay;
}
		
//int Interconnection::GetType() const{
//	return this->type;
//}
		
void Interconnection::SetType(int NewType){
	this->type = NewType;
}

//int Interconnection::GetSubindexType() const{
//	return this->subindex_type;
//}

void Interconnection::SetSubindexType(int index){
	this->subindex_type = index;
}
		
//float Interconnection::GetWeight() const{
//	return this->weight;
//}
		
//void Interconnection::SetWeight(float NewWeight){
//	this->weight = NewWeight;
//}
		
//float Interconnection::GetMaxWeight() const{
//	return this->maxweight;
//}
		
void Interconnection::SetMaxWeight(float NewMaxWeight){
	this->maxweight = NewMaxWeight;
}
		
//LearningRule * Interconnection::GetWeightChange_withPost() const{
//	return this->wchange_withPost;
//}
		
void Interconnection::SetWeightChange_withPost(LearningRule * NewWeightChange_withPost){
	this->wchange_withPost=NewWeightChange_withPost;
}

//LearningRule * Interconnection::GetWeightChange_withoutPost() const{
//	return this->wchange_withoutPost;
//}
		
void Interconnection::SetWeightChange_withoutPost(LearningRule * NewWeightChange_withoutPost){
	this->wchange_withoutPost=NewWeightChange_withoutPost;
}


void Interconnection::SetLearningRuleIndex_withPost(int NewIndex){
	this->LearningRuleIndex_withPost=NewIndex;
}

//int Interconnection::GetLearningRuleIndex_withPost() const{
//	return this->LearningRuleIndex_withPost;
//}

void Interconnection::SetLearningRuleIndex_withoutPost(int NewIndex){
	this->LearningRuleIndex_withoutPost=NewIndex;
}

//int Interconnection::GetLearningRuleIndex_withoutPost() const{
//	return this->LearningRuleIndex_withoutPost;
//}

//void Interconnection::IncrementWeight(float Increment){
//	this->weight += Increment;
//	if(this->weight > this->GetMaxWeight()){
//		this->weight = this->GetMaxWeight();
//	}else if(this->weight < 0.0f){
//		this->weight = 0.0f;
//	}
//}


ostream & Interconnection::PrintInfo(ostream & out) {
	out << "- Interconnection: " << this->index << endl;

	out << "\tSource: " << this->source->GetIndex() << endl;

	out << "\tTarget: " << this->target->GetIndex() << endl;

   	out << "\tDelay: " << this->GetDelay() << "s" << endl;

   	out << "\tConnection Type: " << this->GetType() << endl;

   	out << "\tCurrent Weight: " << this->GetWeight() << endl;

   	out << "\tMaximum Weight: " << this->GetMaxWeight() << endl;

   	out << "\tWeight Change: ";

   	if (this->GetWeightChange_withPost()!=0) this->GetWeightChange_withPost()->PrintInfo(out);
   	else out << "None learning rule with postsynaptic learning" << endl;

	if (this->GetWeightChange_withoutPost()!=0) this->GetWeightChange_withoutPost()->PrintInfo(out);
   	else out << "None learning rule without postsynaptic learning" << endl;

   	return out;
}

void Interconnection::SetTriggerConnection(){
	this->TriggerConnection=true;
}

bool Interconnection::GetTriggerConnection(){
	return this->TriggerConnection;
}

