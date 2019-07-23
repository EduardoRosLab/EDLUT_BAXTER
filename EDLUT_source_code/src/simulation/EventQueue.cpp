/***************************************************************************
 *                           EventQueue.cpp                                *
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

#include "../../include/simulation/EventQueue.h"

#include "../../include/simulation/Event.h"
#include "../../include/spike/Neuron.h"
#include "../../include/spike/InputSpike.h"

#include "../../include/openmp/openmp.h"

EventQueue::EventQueue(int numberOfQueues) : NumberOfQueues(numberOfQueues){
	//Queues for each OpenMP thread that manage a queue.
	NumberOfElements=new unsigned int[NumberOfQueues]();
	AllocatedSize=new unsigned int[NumberOfQueues]();

	Events=(EventForQueue **)new EventForQueue *[NumberOfQueues];
	for(int i=0; i<NumberOfQueues; i++){
		// Allocate memory for a MIN_SIZE sized array
		this->Events[i] = (EventForQueue *) new EventForQueue [MIN_SIZE];

		this->AllocatedSize[i] = MIN_SIZE;

		// The first element in the array is discard
		NumberOfElements[i] = 1;
	}

	//Queue for events that require synchronization between OpenMP threads.
	EventsWithSynchronization=(EventForQueue *) new EventForQueue [MIN_SIZE];
	NumberOfElementsWithSynchronization=1;
	AllocatedSizeWithSynchronization=MIN_SIZE;


	//Memory for buffers
	AllocatedBuffers=(int**)new int * [NumberOfQueues];
	SizeBuffers=(int**)new int * [NumberOfQueues];
	Buffers=(Event ****)new Event ***[NumberOfQueues];
	for(int i=0; i<NumberOfQueues; i++){
		AllocatedBuffers[i]=(int*)new int [NumberOfQueues];
		SizeBuffers[i]=(int*)new int [NumberOfQueues]();
		Buffers[i]=(Event ***)new Event **[NumberOfQueues];
		for(int j=0; j<NumberOfQueues; j++){
			AllocatedBuffers[i][j]=MIN_SIZE;
			Buffers[i][j]=(Event **) new Event * [MIN_SIZE];
		}
	}

}
   		
EventQueue::~EventQueue(){
	for(int j=0; j<NumberOfQueues; j++){
		for (unsigned int i=1; i<this->NumberOfElements[j]; ++i){
			delete (this->Events[j]+i)->EventPtr;
		}

		delete this->Events[j];
	}
	delete this->Events;
	delete NumberOfElements;
	delete AllocatedSize;

	
	delete EventsWithSynchronization;


	//Memory for buffers
		for(int i=0; i<NumberOfQueues; i++){
			for(int j=0; j<NumberOfQueues; j++){
				for(int k=0; k<this->SizeBuffers[i][j]; k++){
					delete Buffers[i][j][k];
				}
				delete [] Buffers[i][j];
			}
			delete [] Buffers[i];
			delete [] SizeBuffers[i];
			delete [] AllocatedBuffers[i];
		}
		delete [] Buffers;
		delete [] SizeBuffers;
		delete [] AllocatedBuffers;
}

void EventQueue::SwapEvents(int index, unsigned int c1, unsigned int c2){
	EventForQueue exchange;
	exchange=*(this->Events[index]+c2);
	*(this->Events[index]+c2)=*(this->Events[index]+c1);
	*(this->Events[index]+c1)=exchange;
}


void EventQueue::Resize(int index, unsigned int NewSize){
	EventForQueue * Temp = this->Events[index];

	// Allocate the new array
	this->Events[index] = (EventForQueue *) new EventForQueue [NewSize];

	this->AllocatedSize[index] = NewSize;

	// Copy all the elements from the original array
	for (unsigned int i = 0; i<this->NumberOfElements[index]; ++i){
		*(this->Events[index]+i) = *(Temp+i);
	}
	
	// Release old memory
	delete [] Temp;
}
   		
void EventQueue::InsertEvent(Event * event){
	int index = event->GetQueueIndex();

	if (this->NumberOfElements[index] == this->AllocatedSize[index]){
		this->Resize(index, this->AllocatedSize[index]*RESIZE_FACTOR);
	}
	
	(this->Events[index]+this->NumberOfElements[index])->EventPtr = event;
	(this->Events[index]+this->NumberOfElements[index])->Time = event->GetTime();
	
	this->NumberOfElements[index]++;


	for(unsigned int c=this->Size(index);c>1 && (((this->Events[index]+c/2)->Time > (this->Events[index]+c)->Time) ||(((this->Events[index]+c/2)->Time == (this->Events[index]+c)->Time) && ((this->Events[index]+c/2)->EventPtr->ProcessingPriority() < (this->Events[index]+c)->EventPtr->ProcessingPriority()) )); c/=2){
    	SwapEvents(index,c, c/2);
  	}
   
    return;
}

void EventQueue::InsertEvent(int index, Event * event){
	if (this->NumberOfElements[index] == this->AllocatedSize[index]){
		this->Resize(index, this->AllocatedSize[index]*RESIZE_FACTOR);
	}
	
	(this->Events[index]+this->NumberOfElements[index])->EventPtr = event;
	(this->Events[index]+this->NumberOfElements[index])->Time = event->GetTime();
	
	this->NumberOfElements[index]++;


	for(unsigned int c=this->Size(index);c>1 && (((this->Events[index]+c/2)->Time > (this->Events[index]+c)->Time) ||(((this->Events[index]+c/2)->Time == (this->Events[index]+c)->Time) && ((this->Events[index]+c/2)->EventPtr->ProcessingPriority() < (this->Events[index]+c)->EventPtr->ProcessingPriority()) )); c/=2){
    	SwapEvents(index,c, c/2);
  	}
   
    return;
}

unsigned int EventQueue::Size(int index) const{
	return this->NumberOfElements[index]-1;
}
   		
Event * EventQueue::RemoveEvent(int index){
	unsigned int c,p;

   	Event * first = 0;


	if(this->NumberOfElements[index]>2){
		first=(this->Events[index]+1)->EventPtr;

		*(this->Events[index]+1)=*(this->Events[index]+this->Size(index));
		this->NumberOfElements[index]--;

		if (this->NumberOfElements[index]>MIN_SIZE && this->NumberOfElements[index]<this->AllocatedSize[index]/(RESIZE_FACTOR*2)){
			this->Resize(index, this->AllocatedSize[index]/RESIZE_FACTOR);
		}
      	
      	p=1;
		for(c=p*2;c<this->Size(index);p=c,c=p*2){
			if(((this->Events[index]+c)->Time > (this->Events[index]+c+1)->Time) || (((this->Events[index]+c)->Time == (this->Events[index]+c+1)->Time) && ((this->Events[index]+c)->EventPtr->ProcessingPriority() < (this->Events[index]+c+1)->EventPtr->ProcessingPriority())))
      			c++;
      		
			if(((this->Events[index]+c)->Time < (this->Events[index]+p)->Time) || (((this->Events[index]+c)->Time == (this->Events[index]+p)->Time) && ((this->Events[index]+c)->EventPtr->ProcessingPriority() > (this->Events[index]+p)->EventPtr->ProcessingPriority()) ))
            	SwapEvents(index,p, c);
         	else
            	break;
        }
		if(c==this->Size(index) && (((this->Events[index]+c)->Time < (this->Events[index]+p)->Time) || (((this->Events[index]+c)->Time == (this->Events[index]+p)->Time) && ((this->Events[index]+c)->EventPtr->ProcessingPriority() > (this->Events[index]+p)->EventPtr->ProcessingPriority()))))
        	SwapEvents(index,p, c);
	} else if (this->NumberOfElements[index]==2){
		first = (this->Events[index]+1)->EventPtr;

		this->NumberOfElements[index]--;
	}
 
    return(first);
}


   		
double EventQueue::FirstEventTime(int index) const{
	double ti;
	
	if(this->NumberOfElements[index]>1)
		ti=(this->Events[index]+1)->Time;
   	else
    	ti=-1.0;
   
   	return(ti);		
}

void EventQueue::RemoveSpikes(int index){
	unsigned int OldNumberOfElements=this->NumberOfElements[index];
	Event * TmpEvent;

	this->NumberOfElements[index]=1; // Initially resize occupied size of the heap so that all the events are out
	// Reinsert in the heap only the events which are spikes 
	for (unsigned int i = 1; i<OldNumberOfElements; ++i){
		TmpEvent=(this->Events[index]+i)->EventPtr;
		if(!TmpEvent->IsSpikeOrCurrent()){
			InsertEvent(index, TmpEvent);
		}else{
			delete TmpEvent;
		}		
	}
}



void EventQueue::InsertInputSpikeEvent(double time, Neuron * neuron){
	InputSpike * newEvent=new InputSpike(time, neuron->get_OpenMP_queue_index(), neuron);
	InsertEvent(newEvent->GetQueueIndex(), newEvent);
}




void EventQueue::SwapEventsWithSynchronization(unsigned int c1, unsigned int c2){
	EventForQueue exchange;
	exchange=*(this->EventsWithSynchronization+c2);
	*(this->EventsWithSynchronization+c2)=*(this->EventsWithSynchronization+c1);
	*(this->EventsWithSynchronization+c1)=exchange;
}


void EventQueue::ResizeWithSynchronization(unsigned int NewSize){
	EventForQueue * Temp = this->EventsWithSynchronization;

	// Allocate the new array
	this->EventsWithSynchronization = (EventForQueue *) new EventForQueue [NewSize];

	this->AllocatedSizeWithSynchronization = NewSize;

	// Copy all the elements from the original array
	for (unsigned int i = 0; i<this->NumberOfElementsWithSynchronization; ++i){
		*(this->EventsWithSynchronization+i) = *(Temp+i);
	}
	
	// Release old memory
	delete [] Temp;
}
   		
void EventQueue::InsertEventWithSynchronization(Event * event){

	if (this->NumberOfElementsWithSynchronization == this->AllocatedSizeWithSynchronization){
		this->ResizeWithSynchronization(this->AllocatedSizeWithSynchronization*RESIZE_FACTOR);
	}
	
	(this->EventsWithSynchronization+this->NumberOfElementsWithSynchronization)->EventPtr = event;
	(this->EventsWithSynchronization+this->NumberOfElementsWithSynchronization)->Time = event->GetTime();
	
	this->NumberOfElementsWithSynchronization++;


	for(unsigned int c=this->SizeWithSynchronization();c>1 && (((this->EventsWithSynchronization+c/2)->Time > (this->EventsWithSynchronization+c)->Time) ||(((this->EventsWithSynchronization+c/2)->Time == (this->EventsWithSynchronization+c)->Time) && ((this->EventsWithSynchronization+c/2)->EventPtr->ProcessingPriority() < (this->EventsWithSynchronization+c)->EventPtr->ProcessingPriority()) )); c/=2){
    	SwapEventsWithSynchronization(c, c/2);
  	}
   
    return;
}


unsigned int EventQueue::SizeWithSynchronization() const{
	return this->NumberOfElementsWithSynchronization-1;
}

   		
Event * EventQueue::RemoveEventWithSynchronization(){

	unsigned int c,p;

   	Event * first = 0;


	if(this->NumberOfElementsWithSynchronization>2){
		first=(this->EventsWithSynchronization+1)->EventPtr;

		*(this->EventsWithSynchronization+1)=*(this->EventsWithSynchronization+this->SizeWithSynchronization());
		this->NumberOfElementsWithSynchronization--;

		if (this->NumberOfElementsWithSynchronization>MIN_SIZE && this->NumberOfElementsWithSynchronization<this->AllocatedSizeWithSynchronization/(RESIZE_FACTOR*2)){
			this->ResizeWithSynchronization(this->AllocatedSizeWithSynchronization/RESIZE_FACTOR);
		}
      	
      	p=1;
		for(c=p*2;c<this->SizeWithSynchronization();p=c,c=p*2){
			if(((this->EventsWithSynchronization+c)->Time > (this->EventsWithSynchronization+c+1)->Time) || (((this->EventsWithSynchronization+c)->Time == (this->EventsWithSynchronization+c+1)->Time) && ((this->EventsWithSynchronization+c)->EventPtr->ProcessingPriority() < (this->EventsWithSynchronization+c+1)->EventPtr->ProcessingPriority())))
      			c++;
      		
			if(((this->EventsWithSynchronization+c)->Time < (this->EventsWithSynchronization+p)->Time) || (((this->EventsWithSynchronization+c)->Time == (this->EventsWithSynchronization+p)->Time) && ((this->EventsWithSynchronization+c)->EventPtr->ProcessingPriority() > (this->EventsWithSynchronization+p)->EventPtr->ProcessingPriority()) ))
            	SwapEventsWithSynchronization(p, c);
         	else
            	break;
        }

		if(c==this->SizeWithSynchronization() && (((this->EventsWithSynchronization+c)->Time < (this->EventsWithSynchronization+p)->Time) || (((this->EventsWithSynchronization+c)->Time == (this->EventsWithSynchronization+p)->Time) && ((this->EventsWithSynchronization+c)->EventPtr->ProcessingPriority() > (this->EventsWithSynchronization+p)->EventPtr->ProcessingPriority()))))
        	SwapEventsWithSynchronization(p, c);
		} else if (this->NumberOfElementsWithSynchronization==2){
			first = (this->EventsWithSynchronization+1)->EventPtr;
 		this->NumberOfElementsWithSynchronization--;
	}

    return(first);
}


   		
double EventQueue::FirstEventTimeWithSynchronization() const{
	double ti;
	
	if(this->NumberOfElementsWithSynchronization>1)
		ti=(this->EventsWithSynchronization+1)->Time;
   	else
    	ti=-1.0;
   
   	return(ti);		
}





void EventQueue::InsertEventInBuffer(int index1, int index2, Event * NewEvent){
	if((this->SizeBuffers[index1][index2])==this->AllocatedBuffers[index1][index2]){
		ResizeBuffer(index1, index2);
	}

	this->Buffers[index1][index2][this->SizeBuffers[index1][index2]]=NewEvent;
	this->IncrementSizeBuffer(index1, index2);
}

void EventQueue::InsertBufferInQueue(int index){

	for(int i=0; i<NumberOfQueues; i++){
		for(int j=0; j<this->SizeBuffers[i][index]; j++){
			this->InsertEvent(index,this->Buffers[i][index][j]);
		}
		this->ResetSizeBuffer(i,index);
	}
}

void EventQueue::ResetBuffer(int index){

	for(int i=0; i<NumberOfQueues; i++){
		for(int j=0; j<this->SizeBuffers[i][index]; j++){
			delete this->Buffers[i][index][j];
		}
		this->ResetSizeBuffer(i,index);
	}
}




void EventQueue::ResizeBuffer(int index1, int index2){
	Event ** aux=this->Buffers[index1][index2];

	this->Buffers[index1][index2]=(Event **)new Event * [this->AllocatedBuffers[index1][index2]*RESIZE_FACTOR];

	for(int i=0; i<this->AllocatedBuffers[index1][index2]; i++){
		Buffers[index1][index2][i]=aux[i];
	}

	delete [] aux;
	this->AllocatedBuffers[index1][index2]*=RESIZE_FACTOR;

}

int EventQueue::GetAllocatedBuffer(int index1, int index2){
	return this->AllocatedBuffers[index1][index2];
}

int EventQueue::GetSizeBuffer( int index1, int index2){
	return this->SizeBuffers[index1][index2];
}

void EventQueue::IncrementSizeBuffer(int index1, int index2){
	this->SizeBuffers[index1][index2]++;
}

void EventQueue::ResetSizeBuffer(int index1, int index2){
	this->SizeBuffers[index1][index2]=0;
}