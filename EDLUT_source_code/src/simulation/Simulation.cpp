/***************************************************************************
 *                           Simulation.cpp                                *
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

#if (defined (_WIN32) || defined(_WIN64))
   #include "windows.h"
#else 
   #include <unistd.h>
#endif



#include "../../include/simulation/Simulation.h"
#include "../../include/simulation/StopSimulationEvent.h"
#include "../../include/simulation/TimeEventUpdateNeurons.h"

#include "../../include/spike/Network.h"
#include "../../include/spike/Spike.h"
#include "../../include/spike/Neuron.h"

#include "../../include/simulation/EventQueue.h"
#include "../../include/simulation/EndSimulationEvent.h"
#include "../../include/simulation/SaveWeightsEvent.h"
#include "../../include/simulation/CommunicationEvent.h"
#include "../../include/simulation/SynchronizeActivityEvent.h"
#include "../../include/simulation/RealTimeRestriction.h"

#include "../../include/communication/OutputSpikeDriver.h"
#include "../../include/communication/InputSpikeDriver.h"
#include "../../include/communication/InputCurrentDriver.h"
#include "../../include/communication/OutputWeightDriver.h"

#include "../../include/neuron_model/NeuronModel.h"
#include "../../include/neuron_model/TimeDrivenNeuronModel.h"

#include "../../include/openmp/openmp.h"



Simulation::Simulation(const char * NetworkFile, const char * WeightsFile, double SimulationTime, double NewSimulationStep, int NewNumberOfQueues) throw (EDLUTException) : Net(0), Queue(0), InputSpike(), InputCurrent(), OutputSpike(), OutputWeight(), Totsimtime(SimulationTime), SimulationStep(NewSimulationStep), SaveWeightStep(0), RealTimeRestrictionObject(0), monitore_neurons(false){

	//Fixe the number of OpenMP threads.
	Set_Number_of_openmp_threads(NewNumberOfQueues);
	NumberOfQueues=NumberOfOpenMPQueues;
	
	CurrentSimulationTime=new double[NumberOfQueues]();
	EndOfSimulation=new bool[NumberOfQueues]();
	StopOfSimulation=new bool[NumberOfQueues]();
	SynchronizeThread=new bool[NumberOfQueues]();
	Updates=new long long[NumberOfQueues]();
	Heapoc=new long long[NumberOfQueues]();
	TotalSpikeCounter=new long long[NumberOfQueues]();
	TotalPropagateCounter=new long long[NumberOfQueues]();
	TotalPropagateEventCounter=new long long[NumberOfQueues]();


	//Initialize one eventqueue object that manage one queue for each OpenMP thread.
	Queue=new EventQueue(NumberOfQueues);

	Net = new Network(NetworkFile, WeightsFile, this->Queue, NumberOfQueues);

	this->monitore_neurons = Net->monitore_neurons;

	MinInterpropagationTime=Net->GetMinInterpropagationTime();

	//Create the real time restriction object with default parameter. This object need its own thread to work properly.
	RealTimeRestrictionObject=new RealTimeRestriction();
}

Simulation::Simulation(const Simulation & ant) :Net(ant.Net), Queue(ant.Queue), InputSpike(ant.InputSpike), InputCurrent(ant.InputCurrent), OutputSpike(ant.OutputSpike), OutputWeight(ant.OutputWeight), Totsimtime(ant.Totsimtime), SaveWeightStep(ant.SaveWeightStep), EndOfSimulation(ant.EndOfSimulation), Updates(ant.Updates), Heapoc(ant.Heapoc), NumberOfQueues(ant.NumberOfQueues){
}

Simulation::~Simulation(){
	if (this->Net){
		delete this->Net;
		this->Net=NULL;
	}

	if (this->Queue){
		delete Queue;
		this->Queue=NULL;
	}

	if(this->CurrentSimulationTime){
		delete this->CurrentSimulationTime;
		this->CurrentSimulationTime=NULL;
	}

	if(this->EndOfSimulation){
		delete this->EndOfSimulation;
		this->EndOfSimulation=NULL;
	}

	if(this->StopOfSimulation){
		delete this->StopOfSimulation;
		this->StopOfSimulation=NULL;
	}

	if(this->SynchronizeThread){
		delete this->SynchronizeThread;
		this->SynchronizeThread=NULL;
	}

	if(this->Updates){
		delete this->Updates;
		this->Updates=NULL;
	}

	if(this->Heapoc){
		delete this->Heapoc;
		this->Heapoc=NULL;
	}

	if(this->TotalSpikeCounter){
		delete this->TotalSpikeCounter;
		this->TotalSpikeCounter=NULL;
	}

	
	if(this->TotalPropagateCounter){
		delete this->TotalPropagateCounter;
		this->TotalPropagateCounter=NULL;
	}

	if(this->TotalPropagateEventCounter){
		delete this->TotalPropagateEventCounter;
		this->TotalPropagateEventCounter=NULL;
	}

	if(this->RealTimeRestrictionObject){
		delete this->RealTimeRestrictionObject;
		this->RealTimeRestrictionObject=NULL;
	}
}

void Simulation::EndSimulation(int indexThread){
	this->EndOfSimulation[indexThread] = true;
}

void Simulation::StopSimulation(int indexThread){
	this->StopOfSimulation[indexThread] = true;
}


void Simulation::SetSynchronizeSimulationEvent(int indexThread){
	this->SynchronizeThread[indexThread] = true;
}

void Simulation::ResetSynchronizeSimulationEvent(int indexThread){
	this->SynchronizeThread[indexThread] = false;
}

bool Simulation::GetSynchronizeSimulationEvent(int indexThread){
	return this->SynchronizeThread[indexThread];
}



void Simulation::SetSaveStep(float NewSaveStep){
	this->SaveWeightStep = NewSaveStep;	
}

double Simulation::GetSaveStep(){
	return this->SaveWeightStep;	
}

void Simulation::SetSimulationStep(double NewSimulationStep){
	this->SimulationStep = NewSimulationStep;		
}
		
double Simulation::GetSimulationStep(){
	return this->SimulationStep;	
}



void Simulation::InitSimulation() throw (EDLUTException){
	// Get the external initial inputs
	this->GetInput();
	
	for(int i=0; i<this->NumberOfQueues; i++){
		this->CurrentSimulationTime[i] = 0.0;
		
		// Add a final simulation event
		if (this->Totsimtime >= 0){
			this->Queue->InsertEvent(i, new EndSimulationEvent(this->Totsimtime, i));
		}
		
		// Add the CPU time-driven simulation events
		int ** N_TimeDrivenNeuron=this->GetNetwork()->GetTimeDrivenNeuronNumber();
		for(int z=0; z<this->GetNetwork()->GetNneutypes(); z++){
			if(N_TimeDrivenNeuron[z][i]>0){
				TimeDrivenModel * model=(TimeDrivenModel *) this->GetNetwork()->GetNeuronModelAt(z,i);

				this->Queue->InsertEvent(i, new TimeEventUpdateNeurons(model->GetTimeDrivenStepSize(), i, model,this->GetNetwork()->GetTimeDrivenNeuronAt(z,i)));
			}
		}


		SetTotalSpikeCounter(i,0);
	}

	// Add the first synchronization event between OpenMP queues.
	if(this->MinInterpropagationTime>0.0){
		this->Queue->InsertEventWithSynchronization(new SynchronizeActivityEvent(this->MinInterpropagationTime, this));
	}

	// Add the first save weight event
	if (this->SaveWeightStep>0.0F){
		this->Queue->InsertEventWithSynchronization(new SaveWeightsEvent(this->SaveWeightStep, this));	
	}
	
	// Add the first communication event
	if (this->SimulationStep>0.0F){
		this->Queue->InsertEventWithSynchronization(new CommunicationEvent(this->SimulationStep, this));		
	}	
}


void Simulation::SynchronizeThreads(){
	#pragma omp barrier

	#pragma omp single
	{
		if (!StopOfSimulation[0] && !EndOfSimulation[0]){
			Event * newEvent = this->GetQueue()->RemoveEventWithSynchronization();
			newEvent->ProcessEvent(this);
			delete newEvent;
		}
	}
}


void Simulation::RunSimulation()  throw (EDLUTException){

	Event * NewEvent;
	int openMP_index;

	#pragma omp parallel num_threads(NumberOfQueues) if(NumberOfQueues>1) private(NewEvent, openMP_index)
	{
		openMP_index = omp_get_thread_num();


		this->CurrentSimulationTime[openMP_index] = 0.0; 

		while (!this->EndOfSimulation[openMP_index]){ 
				
			if (this->GetSynchronizeSimulationEvent(openMP_index)){
				ResetSynchronizeSimulationEvent(openMP_index);
				SynchronizeThreads();
			}
				
			NewEvent = this->Queue->RemoveEvent(openMP_index);
				
			Updates[openMP_index]++;
			Heapoc[openMP_index] += Queue->Size(openMP_index);
					
			if (NewEvent->GetTime() < this->CurrentSimulationTime[openMP_index]){ 
				cerr << "Thread " << openMP_index << "--> Internal error: Bad spike time. Spike: " << NewEvent->GetTime() << " Current: " << this->CurrentSimulationTime[openMP_index] << endl; /*asdfgf*/
				NewEvent->PrintType();
			} else {
				this->CurrentSimulationTime[openMP_index] = NewEvent->GetTime();   
				NewEvent->ProcessEvent(this);
			}				

			delete NewEvent;
		}
	}
}

void Simulation::RunSimulationSlot(double preempt_time)  throw (EDLUTException){
	Event * NewEvent;
	int openMP_index;

	#pragma omp parallel num_threads(NumberOfQueues) if(NumberOfQueues>1) private(NewEvent, openMP_index)
	{
		openMP_index = omp_get_thread_num();


		this->StopOfSimulation[openMP_index] = false;

		// Add a stop simulation event in preempt_time
		this->Queue->InsertEvent(openMP_index, new StopSimulationEvent(preempt_time, openMP_index));

		while (!this->EndOfSimulation[openMP_index] && !this->StopOfSimulation[openMP_index]){

			if (this->GetSynchronizeSimulationEvent(openMP_index)){
				ResetSynchronizeSimulationEvent(openMP_index);
				SynchronizeThreads();
			}

			NewEvent = this->Queue->RemoveEvent(openMP_index);

			Updates[openMP_index]++;
			Heapoc[openMP_index] += Queue->Size(openMP_index);

			bool SkipEvent = false;

			if (NewEvent->GetTime() < this->CurrentSimulationTime[openMP_index]){
				cerr << "Internal error: Bad spike time. Spike: " << NewEvent->GetTime() << " Current: " << this->CurrentSimulationTime[openMP_index] << endl;
				NewEvent->PrintType();
				SkipEvent = true;
			} else {
				this->CurrentSimulationTime[openMP_index] = NewEvent->GetTime(); // only for checking 	
			}

			while (this->RealTimeRestrictionObject->RestrictionLevel == SIMULATION_TOO_FAST){
				#if (defined (_WIN32) || defined(_WIN64))
					Sleep(1);
				#else
					usleep(1*1000);
				#endif
			}

			if (!SkipEvent) {
				NewEvent->ProcessEvent(this, this->RealTimeRestrictionObject->RestrictionLevel);	
			}				

			delete NewEvent;
		}
	}
}


void Simulation::WriteSpike(const Spike * spike){
	Neuron * neuron=spike->GetSource();  // source of the spike
    
    if(neuron->IsMonitored()){
		for (list<OutputSpikeDriver *>::iterator it=this->MonitorSpike.begin(); it!=this->MonitorSpike.end(); ++it){
			if (!(*it)->IsWritePotentialCapable()){
				(*it)->WriteSpike(spike);
			}
		}
	}
	   	
	if(neuron->IsOutput()){
		for (list<OutputSpikeDriver *>::iterator it=this->OutputSpike.begin(); it!=this->OutputSpike.end(); ++it){
			(*it)->WriteSpike(spike);
		}
	}		
}




void Simulation::WriteState(float time, Neuron * neuron){
	if (neuron->IsMonitored()){
		for (list<OutputSpikeDriver *>::iterator it=this->MonitorSpike.begin(); it!=this->MonitorSpike.end(); ++it){
			if ((*it)->IsWritePotentialCapable()){
				(*it)->WriteState(time, neuron);
			}
		}
	}
}

void Simulation::SaveWeights(){
	cout << "Saving weights in time " << this->CurrentSimulationTime[0] << endl; 
	for (list<OutputWeightDriver *>::iterator it=this->OutputWeight.begin(); it!=this->OutputWeight.end(); ++it){
		(*it)->WriteWeights(this->Net,float(this->CurrentSimulationTime[0]));
	}
	
}

void Simulation::SendOutput(){
//	cout << "Sending outputs in time " << this->CurrentSimulationTime[0] << endl; 
	for (list<OutputSpikeDriver *>::iterator it=this->OutputSpike.begin(); it!=this->OutputSpike.end(); ++it){
		if ((*it)->IsBuffered()){
			(*it)->FlushBuffers();	
		}
	}
}

void Simulation::GetInput(){
	//Input spike driver
	for (list<InputSpikeDriver *>::iterator it=this->InputSpike.begin(); it!=this->InputSpike.end(); ++it){
		if (!(*it)->IsFinished()){
			(*it)->LoadInputs(this->Queue, this->Net);
		}
	}

	//Input current driver
	for (list<InputCurrentDriver *>::iterator it = this->InputCurrent.begin(); it != this->InputCurrent.end(); ++it){
		if (!(*it)->IsFinished()){
			(*it)->LoadInputs(this->Queue, this->Net);
		}
	}
}

long long Simulation::GetTotalSpikeCounter(int indexThread){
	return this->TotalSpikeCounter[indexThread];
}

long long Simulation::GetTotalSpikeCounter(){
	long long counter=0;
	for(int i=0; i<NumberOfQueues; i++){
		counter+=this->TotalSpikeCounter[i];
	}
	return counter;
}

void Simulation::SetTotalSpikeCounter(int indexThread, long long value) {
	this->TotalSpikeCounter[indexThread] = value;
}

void Simulation::IncrementTotalSpikeCounter(int indexThread) {
	this->TotalSpikeCounter[indexThread]++;
}


long long Simulation::GetTotalPropagateCounter(int indexThread){
	return this->TotalPropagateCounter[indexThread];
}

long long Simulation::GetTotalPropagateCounter(){
	long long counter=0;
	for(int i=0; i<NumberOfQueues; i++){
		counter+=this->TotalPropagateCounter[i];
	}
	return counter;
}

long long Simulation::GetTotalPropagateEventCounter(int indexThread){
	return this->TotalPropagateEventCounter[indexThread];
}

long long Simulation::GetTotalPropagateEventCounter(){
	long long counter=0;
	for(int i=0; i<NumberOfQueues; i++){
		counter+=this->TotalPropagateEventCounter[i];
	}
	return counter;
}

void Simulation::SetTotalPropagateCounter(int indexThread, long long value) {
	this->TotalPropagateCounter[indexThread] = value;
}

void Simulation::SetTotalPropagateEventCounter(int indexThread, long long value) {
	this->TotalPropagateEventCounter[indexThread] = value;
}

void Simulation::IncrementTotalPropagateCounter(int indexThread) {
	this->TotalPropagateCounter[indexThread]++;
}

void Simulation::IncrementTotalPropagateEventCounter(int indexThread) {
	this->TotalPropagateEventCounter[indexThread]++;
}

void Simulation::IncrementTotalPropagateCounter(int indexThread, int increment) {
	this->TotalPropagateCounter[indexThread]+=increment;
}

Network * Simulation::GetNetwork() const{
	return this->Net;	
}

EventQueue * Simulation::GetQueue() const{
	return this->Queue;
}
		
double Simulation::GetTotalSimulationTime() const{
	return this->Totsimtime;	
}
		
long long Simulation::GetSimulationUpdates(int indexThread) const{
	return this->Updates[indexThread];	
}

long long Simulation::GetSimulationUpdates() const{
	long long counter=0;
	for(int i=0; i<NumberOfQueues; i++){
		counter += this->Updates[i];	
	}
	return counter;
}
		
long long Simulation::GetHeapAcumSize(int indexThread) const{
	return this->Heapoc[indexThread];
}		

long long Simulation::GetHeapAcumSize() const{
	long long counter=0;
	for(int i=0; i<NumberOfQueues; i++){
		counter+= this->Heapoc[i];
	}
	counter/=NumberOfQueues;
	return counter;
}

void Simulation::AddInputSpikeDriver(InputSpikeDriver * NewInput){
	this->InputSpike.push_back(NewInput);	
}

InputSpikeDriver *Simulation::GetInputSpikeDriver(unsigned int ElementPosition){
    InputSpikeDriver *list_element;
    if(this->InputSpike.size() > ElementPosition){
        list<InputSpikeDriver *>::iterator list_it = this->InputSpike.begin();
        advance(list_it, ElementPosition);
        list_element=*list_it;
    } else
        list_element=NULL;
    return list_element;
}

void Simulation::RemoveInputSpikeDriver(InputSpikeDriver * NewInput){
	this->InputSpike.remove(NewInput);	
}

void Simulation::AddInputCurrentDriver(InputCurrentDriver * NewInput){
	this->InputCurrent.push_back(NewInput);
}

InputCurrentDriver *Simulation::GetInputCurrentDriver(unsigned int ElementPosition){
	InputCurrentDriver *list_element;
	if (this->InputCurrent.size() > ElementPosition){
		list<InputCurrentDriver *>::iterator list_it = this->InputCurrent.begin();
		advance(list_it, ElementPosition);
		list_element = *list_it;
	}
	else
		list_element = NULL;
	return list_element;
}

void Simulation::RemoveInputCurrentDriver(InputCurrentDriver * NewInput){
	this->InputCurrent.remove(NewInput);
}
		 
void Simulation::AddOutputSpikeDriver(OutputSpikeDriver * NewOutput){
	this->OutputSpike.push_back(NewOutput);	
}

OutputSpikeDriver *Simulation::GetOutputSpikeDriver(unsigned int ElementPosition){
    OutputSpikeDriver *list_element;
    if(this->OutputSpike.size() > ElementPosition){
        list<OutputSpikeDriver *>::iterator list_it = this->OutputSpike.begin();
        advance(list_it, ElementPosition);
        list_element=*list_it;
    } else
        list_element=NULL;
    return list_element;
}

void Simulation::RemoveOutputSpikeDriver(OutputSpikeDriver * NewOutput){
	this->OutputSpike.remove(NewOutput);
}

void Simulation::AddMonitorActivityDriver(OutputSpikeDriver * NewMonitor){
	this->MonitorSpike.push_back(NewMonitor);
}

OutputSpikeDriver *Simulation::GetMonitorActivityDriver(unsigned int ElementPosition){
    OutputSpikeDriver *list_element;
    if(this->MonitorSpike.size() > ElementPosition){
        list<OutputSpikeDriver *>::iterator list_it = this->MonitorSpike.begin();
        advance(list_it, ElementPosition);
        list_element=*list_it;
    } else
        list_element=NULL;
    return list_element;
}
		
void Simulation::RemoveMonitorActivityDriver(OutputSpikeDriver * NewMonitor){
	this->MonitorSpike.remove(NewMonitor);
}

void Simulation::AddOutputWeightDriver(OutputWeightDriver * NewOutput){
	this->OutputWeight.push_back(NewOutput);	
}

OutputWeightDriver *Simulation::GetOutputWeightDriver(unsigned int ElementPosition){
    OutputWeightDriver *list_element;
    if(this->OutputWeight.size() > ElementPosition){
        list<OutputWeightDriver *>::iterator list_it = this->OutputWeight.begin();
        advance(list_it, ElementPosition);
        list_element=*list_it;
    } else
        list_element=NULL;
    return list_element;
}

void Simulation::RemoveOutputWeightDriver(OutputWeightDriver * NewOutput){
	this->OutputWeight.remove(NewOutput);
}

ostream & Simulation::PrintInfo(ostream & out) {
	out << "- Simulation:" << endl;

	out << "  * End simulation time: " << this->GetTotalSimulationTime() << " s."<< endl;

	out << "  * Saving weight step time: " << this->GetSaveStep() << " s." << endl;

	out << "  * Communication step time: " << this->GetSimulationStep() << " s." << endl;

	out << "  * Total simulation time: " << this->GetTotalSimulationTime() << " s." << endl;

	this->GetNetwork()->PrintInfo(out);

	out << "  * Input spike channels: " << this->InputSpike.size() << endl;

	for (list<InputSpikeDriver *>::iterator it=this->InputSpike.begin(); it!=this->InputSpike.end(); ++it){
		(*it)->PrintInfo(out);
	}

	out << "  * Input current channels: " << this->InputCurrent.size() << endl;

	for (list<InputCurrentDriver *>::iterator it = this->InputCurrent.begin(); it != this->InputCurrent.end(); ++it){
		(*it)->PrintInfo(out);
	}

	out << "  * Output spike channels: " << this->OutputSpike.size() << endl;

	for (list<OutputSpikeDriver *>::iterator it=this->OutputSpike.begin(); it!=this->OutputSpike.end(); ++it){
		(*it)->PrintInfo(out);
	}

	out << "  * Monitor spike channels: " << this->MonitorSpike.size() << endl;

	for (list<OutputSpikeDriver *>::iterator it=this->MonitorSpike.begin(); it!=this->MonitorSpike.end(); ++it){
		(*it)->PrintInfo(out);
	}

	out << "  * Saving weight channels: " << this->OutputWeight.size() << endl;

	for (list<OutputWeightDriver *>::iterator it=this->OutputWeight.begin(); it!=this->OutputWeight.end(); ++it){
		(*it)->PrintInfo(out);
	}

	out << "  * Network description:" << endl;
	this->GetNetwork()->PrintInfo(out);

	return out;
}


int Simulation::GetNumberOfQueues(){
	return NumberOfQueues;
}


double Simulation::GetMinInterpropagationTime(){
	return MinInterpropagationTime;
}

void Simulation::SelectGPU(int OpenMP_index){
}