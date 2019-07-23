/***************************************************************************
 *                           VectorBufferedState.cpp                       *
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

#include "../../include/neuron_model/VectorBufferedState.h"

VectorBufferedState::VectorBufferedState(unsigned int NumVariables, unsigned int NumBuffers, bool isTimeDriven):
	VectorNeuronState(NumVariables, isTimeDriven), FirstElement(0), LastElement(0), BufferAmplitude(0), NumberOfElements(0), NumberOfBuffers(NumBuffers) {
}

VectorBufferedState::VectorBufferedState(const VectorBufferedState & OldState): VectorNeuronState(OldState), FirstElement(0), LastElement(0),
		BufferAmplitude(0), NumberOfElements(0), NumberOfBuffers(OldState.NumberOfBuffers) {

	FirstElement = (ActivityNode ***) new ActivityNode ** [OldState.SizeStates];
	for (int j=0; j<OldState.SizeStates; j++){
		FirstElement[j] = (ActivityNode **) new ActivityNode * [NumberOfBuffers];
		for (unsigned int i=0; i<NumberOfBuffers; ++i){
			this->FirstElement[i] = 0;
		}
	}

	LastElement = (ActivityNode ***) new ActivityNode ** [OldState.SizeStates];
	for (int j=0; j<OldState.SizeStates; j++){	
		LastElement[j] = (ActivityNode **) new ActivityNode * [NumberOfBuffers];
		for (unsigned int i=0; i<NumberOfBuffers; ++i){
			this->LastElement[i] = 0;
		}
	}
	
	BufferAmplitude = (float **) new float * [OldState.SizeStates];
	for (int j=0; j<OldState.SizeStates; j++){
		BufferAmplitude[j] = (float *) new float [NumberOfBuffers];
		for (unsigned int i=0; i<NumberOfBuffers; ++i){
			this->BufferAmplitude[i] = 0;
		}
	}
	
	NumberOfElements = (unsigned int **) new unsigned int * [OldState.SizeStates];
	for (int j=0; j<OldState.SizeStates; j++){
		NumberOfElements[j] = (unsigned int *) new unsigned int [NumberOfBuffers];
		for (unsigned int i=0; i<NumberOfBuffers; ++i){
			this->NumberOfElements[i] = 0;
		}
	}
		
	for (int j=0; j<OldState.SizeStates; j++){
		for (unsigned int i=0; i<this->NumberOfBuffers; ++i){
			this->BufferAmplitude[j][i] = OldState.BufferAmplitude[j][i];

			ActivityNode * Iterator = OldState.FirstElement[j][i];

			while (Iterator!=0){
				ActivityNode * NewElement = (ActivityNode *) new ActivityNode;

				NewElement->Spike = Iterator->Spike;
				NewElement->NextNode = 0;

				if (this->FirstElement[j][i]==0){
					// This is the first element of the list
					this->FirstElement[j][i] = NewElement;
					this->LastElement[j][i] = NewElement;
				} else {
					// Add the element after the last element
					this->LastElement[j][i]->NextNode = NewElement;
					this->LastElement[j][i] = NewElement;
				}

				this->NumberOfElements[j][i] ++;

				Iterator = Iterator->NextNode;
			}
		}
	}
}

void VectorBufferedState::SetBufferAmplitude(int index, unsigned int NumBuffer, float BufferAmpl){
	this->BufferAmplitude[index][NumBuffer] = BufferAmpl;
}

VectorBufferedState::~VectorBufferedState() {
	// TODO Auto-generated destructor stub
	for (int j=0; j<GetSizeState(); j++){
		for (unsigned int i=0; i<this->NumberOfBuffers; ++i){
			ActivityNode * Iterator = this->FirstElement[j][i];

			while (Iterator!=0){
				ActivityNode * NextElement = Iterator->NextNode;

				delete Iterator;

				Iterator = NextElement;
			}

		}

		delete [] this->NumberOfElements[j];
		delete [] this->BufferAmplitude[j];
		delete [] this->FirstElement[j];
		delete [] this->LastElement[j];
	}

	delete [] this->NumberOfElements;
	this->NumberOfElements = 0;
	delete [] this->BufferAmplitude;
	this->BufferAmplitude = 0;
	delete [] this->FirstElement;
	this->FirstElement = 0;
	delete [] this->LastElement;
	this->LastElement = 0;
}

void VectorBufferedState::AddActivity(int index, Interconnection * InputConnection){
	ActivityNode * NewElement = (ActivityNode *) new ActivityNode;

	unsigned int NumBuffer = (unsigned int) InputConnection->GetType();

	NewElement->Spike.first = 0;
	NewElement->Spike.second = InputConnection;
	NewElement->NextNode = 0;

	if (this->FirstElement[index][NumBuffer]==0){
		// This is the first element of the list
		this->FirstElement[index][NumBuffer] = NewElement;
		this->LastElement[index][NumBuffer] = NewElement;
	} else {
		// Add the element after the last element
		this->LastElement[index][NumBuffer]->NextNode = NewElement;
		this->LastElement[index][NumBuffer] = NewElement;
	}

	this->NumberOfElements[index][NumBuffer] ++;
}

void VectorBufferedState::CheckActivity(int index){
	for (unsigned int i=0; i<this->NumberOfBuffers; ++i){
		// If the first element is older than we accept, remove it.
		ActivityNode * Iterator = this->FirstElement[index][i];
		while (Iterator!=0 && Iterator->Spike.first>this->BufferAmplitude[index][i]){
			ActivityNode * Next = Iterator->NextNode;
			delete Iterator;
			this->FirstElement[index][i] = Next;
			if (Next==0){
				// Empty buffer
				this->LastElement[index][i] = 0;
			}
			Iterator = Next;
			this->NumberOfElements[index][i] --;
		}
	}
}

void VectorBufferedState::AddElapsedTime(int index, double ElapsedTime){
	VectorNeuronState::AddElapsedTime(index, ElapsedTime);

	for (unsigned int i=0; i<this->NumberOfBuffers; ++i){
		ActivityNode * Iterator = this->FirstElement[index][i];
		while (Iterator!=0){
			Iterator->Spike.first += ElapsedTime;
			Iterator = Iterator->NextNode;
		}
	}

	this->CheckActivity(index);
}

unsigned int VectorBufferedState::GetNumberOfSpikes(int index, unsigned int NumBuffer){
	return this->NumberOfElements[index][NumBuffer];
}

double VectorBufferedState::GetSpikeTimeAt(int index, unsigned int Position, unsigned int NumBuffer){
	ActivityNode * Iterator = this->FirstElement[index][NumBuffer];
	for (unsigned int i = 0; i<Position && Iterator!=0; ++i, Iterator=Iterator->NextNode){
	}
	return (Iterator==0)?-1:Iterator->Spike.first;
}

Interconnection * VectorBufferedState::GetInterconnectionAt(int index, unsigned int Position, unsigned int NumBuffer){
	ActivityNode * Iterator = this->FirstElement[index][NumBuffer];
	for (unsigned int i = 0; i<Position && Iterator!=0; ++i, Iterator=Iterator->NextNode){
	}
	return (Iterator==0)?0:Iterator->Spike.second;
}

VectorBufferedState::Iterator VectorBufferedState::Begin(int index, unsigned int NumBuffer){
	return Iterator(this->FirstElement[index][NumBuffer]);
}

VectorBufferedState::Iterator VectorBufferedState::End(){
	return Iterator();
}

VectorBufferedState::Iterator::Iterator():element(0){}

VectorBufferedState::Iterator::Iterator(const VectorBufferedState::Iterator & ItAux){
	this->element = ItAux.element;
}

VectorBufferedState::Iterator::Iterator(ActivityNode * ElemAux){
	this->element=ElemAux;
}

VectorBufferedState::Iterator & VectorBufferedState::Iterator::operator++(){
	this->element = this->element->NextNode;

	return *this;
}

bool VectorBufferedState::Iterator::operator==(VectorBufferedState::Iterator Aux){
	return this->element==Aux.element;
}

bool VectorBufferedState::Iterator::operator!=(VectorBufferedState::Iterator Aux){
	return this->element!=Aux.element;
}

double VectorBufferedState::Iterator::GetSpikeTime(){
	return this->element->Spike.first;
}

Interconnection * VectorBufferedState::Iterator::GetConnection(){
	return this->element->Spike.second;
}


void VectorBufferedState::InitializeBufferedStates(int size, float * initialization){
	InitializeStates(size, initialization);
	
	FirstElement = (ActivityNode ***) new ActivityNode ** [size];
	for (int j=0; j<size; j++){
		FirstElement[j] = (ActivityNode **) new ActivityNode * [NumberOfBuffers];
		for (unsigned int i=0; i<NumberOfBuffers; ++i){
			this->FirstElement[j][i] = 0;
		}
	}

	LastElement = (ActivityNode ***) new ActivityNode ** [size];
	for (int j=0; j<size; j++){	
		LastElement[j] = (ActivityNode **) new ActivityNode * [NumberOfBuffers];
		for (unsigned int i=0; i<NumberOfBuffers; ++i){
			this->LastElement[j][i] = 0;
		}
	}
	
	BufferAmplitude = (float **) new float * [size];
	for (int j=0; j<size; j++){
		BufferAmplitude[j] = (float *) new float [NumberOfBuffers];
		for (unsigned int i=0; i<NumberOfBuffers; ++i){
			this->BufferAmplitude[j][i] = 0;
		}
	}
	
	NumberOfElements = (unsigned int **) new unsigned int * [size];
	for (int j=0; j<size; j++){
		NumberOfElements[j] = (unsigned int *) new unsigned int [NumberOfBuffers];
		for (unsigned int i=0; i<NumberOfBuffers; ++i){
			this->NumberOfElements[j][i] = 0;
		}
	}
}