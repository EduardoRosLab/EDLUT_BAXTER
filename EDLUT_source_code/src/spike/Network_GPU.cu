/***************************************************************************
 *                           Network_GPU.cpp                               *
 *                           -------------------                           *
 * copyright            : (C) 2012 by Jesus Garrido, Richard Carrillo and  *
 *						: Francisco Naveros                                *
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

/*
 * \note: this file Network_GPU.cpp must be used instead of file Network.cpp to 
 * implement a CPU-GPU hybrid architecture.
*/


#include "../../include/spike/Network.h"
#include "../../include/spike/Interconnection.h"
#include "../../include/spike/Neuron.h"
#include "../../include/spike/InternalSpike.h"


#include "../../include/learning_rules/ExpOptimisedBufferedWeightChange.h"
#include "../../include/learning_rules/STDPWeightChange.h"

#include "../../include/neuron_model/InputSpikeNeuronModel.h"
#include "../../include/neuron_model/InputCurrentNeuronModel.h"

#include "../../include/neuron_model/NeuronModel.h"
#include "../../include/neuron_model/LIFTimeDrivenModel_1_1.h"
#include "../../include/neuron_model/LIFTimeDrivenModel_1_2.h"
#include "../../include/neuron_model/LIFTimeDrivenModel_1_4.h"
#include "../../include/neuron_model/TimeDrivenNeuronModel.h"
#include "../../include/neuron_model/EventDrivenNeuronModel.h"
#include "../../include/neuron_model/CompressTableBasedModel.h"
#include "../../include/neuron_model/CompressSynchronousTableBasedModel.h"
#include "../../include/neuron_model/TableBasedModel.h"
#include "../../include/neuron_model/SynchronousTableBasedModel.h"
#include "../../include/neuron_model/VectorNeuronState.h"
#include "../../include/neuron_model/TimeDrivenSinCurrentGenerator.h"

#include "../../include/neuron_model/TimeDrivenNeuronModel_GPU.h"
#include "../../include/neuron_model/LIFTimeDrivenModel_1_1_GPU.h"
#include "../../include/neuron_model/LIFTimeDrivenModel_1_2_GPU.h"
#include "../../include/neuron_model/LIFTimeDrivenModel_1_4_GPU.h"

#include "../../include/simulation/EventQueue.h"
#include "../../include/simulation/Utils.h"
#include "../../include/simulation/Configuration.h"
#include "../../include/simulation/RandomGenerator.h"

#include "../../include/cudaError.h"

#include "../../include/openmp/openmp.h"

#include <iostream>
#include <string>
using namespace std;


int qsort_inters(const void *e1, const void *e2){
	int ord;
	double ordf;
	
	//Calculate source index
	ord=((Interconnection *)e1)->GetTarget()->get_OpenMP_queue_index() - ((Interconnection *)e2)->GetTarget()->get_OpenMP_queue_index();
	if(!ord){
		//the same source index-> calculate target OpenMP index
		ord=((Interconnection *)e1)->GetSource()->GetIndex() - ((Interconnection *)e2)->GetSource()->GetIndex();
		if(!ord){
			ordf=((Interconnection *)e1)->GetDelay() - ((Interconnection *)e2)->GetDelay();
			if(ordf<0.0){
				ord=-1;
			}else if(ordf>0.0){
				ord=1;
			}//The same propagation time-> calculate targe index
			else if(ordf==0){
				ord=((Interconnection *)e1)->GetTarget()->GetIndex() - ((Interconnection *)e2)->GetTarget()->GetIndex();
			}
		}
	}
   
	return(ord);
}

void Network::FindOutConnections(){
	// Change the ordenation
   	qsort(inters,ninters,sizeof(Interconnection),qsort_inters);
	if(ninters>0){
		// Calculate the number of input connections with learning for each cell
		unsigned long ** NumberOfOutputs = (unsigned long **) new unsigned long *[this->nneurons];
		unsigned long ** OutputsLeft = (unsigned long **) new unsigned long *[this->nneurons];
		for (unsigned long i=0; i<this->nneurons; i++){
			NumberOfOutputs[i]=(unsigned long *) new unsigned long [this->GetNumberOfQueues()]();
			OutputsLeft[i]=(unsigned long *) new unsigned long [this->GetNumberOfQueues()];
		}
		


		for (unsigned long con= 0; con<this->ninters; ++con){
			NumberOfOutputs[this->inters[con].GetSource()->GetIndex()][this->inters[con].GetTarget()->get_OpenMP_queue_index()]++;
		}

		for (unsigned long neu = 0; neu<this->nneurons; ++neu){
			for(int i=0; i<this->GetNumberOfQueues(); i++){
				OutputsLeft[neu][i] = NumberOfOutputs[neu][i];
			}
		}

		Interconnection **** OutputConnections = (Interconnection ****) new Interconnection *** [this->nneurons];
		for (unsigned long neu = 0; neu<this->nneurons; ++neu){
			OutputConnections[neu] = (Interconnection ***) new Interconnection ** [this->GetNumberOfQueues()];
		}

		for (unsigned long neu = 0; neu<this->nneurons; ++neu){
			for(int i=0; i<this->GetNumberOfQueues(); i++){
				if (NumberOfOutputs[neu][i]>0){
					OutputConnections[neu][i] = (Interconnection **) new Interconnection * [NumberOfOutputs[neu][i]];
				} else {
					OutputConnections[neu][i] = 0;
				}
			}
		}

		for (unsigned long con= this->ninters-1; con<this->ninters; --con){
			unsigned long SourceCell = this->inters[con].GetSource()->GetIndex();
			int OpenMP_index=this->inters[con].GetTarget()->get_OpenMP_queue_index();
			OutputConnections[SourceCell][OpenMP_index][--OutputsLeft[SourceCell][OpenMP_index]] = this->inters+con;
		}

		for (unsigned long neu = 0; neu<this->nneurons; ++neu){
			this->neurons[neu].SetOutputConnections(OutputConnections[neu],NumberOfOutputs[neu]);

//			for(int i=0; i<this->GetNumberOfQueues(); i++){	
//				for (unsigned long aux = 0; aux < NumberOfOutputs[neu][i]; aux++){
//					if(OutputConnections[neu][i][aux]->GetWeightChange_withoutPost()!=0){
//						OutputConnections[neu][i][aux]->SetLearningRuleIndex_withoutPost(OutputConnections[neu][i][aux]->GetWeightChange_withoutPost()->counter);
//						OutputConnections[neu][i][aux]->GetWeightChange_withoutPost()->counter++;
//					}
//				}
//			}
		}

		delete [] OutputConnections;
		delete [] NumberOfOutputs;
		delete [] OutputsLeft;
	}
}

void Network::SetWeightOrdination(){
	if (ninters>0){
		for (int ninter=0; ninter<ninters;ninter++){
			int index = inters[ninter].GetIndex();
			this->wordination[index] = &(this->inters[ninter]);
		}
	}	
}

void Network::FindInConnections(){
	if(this->ninters>0){

		// Calculate the number of input connections with learning for each cell
		unsigned long * NumberOfInputsWithPostSynapticLearning = (unsigned long *) new unsigned long [this->nneurons]();
		unsigned long * InputsLeftWithPostSynapticLearning = (unsigned long *) new unsigned long [this->nneurons];

		unsigned long * NumberOfInputsWithoutPostSynapticLearning = (unsigned long *) new unsigned long [this->nneurons]();
		unsigned long * InputsLeftWithoutPostSynapticLearning = (unsigned long *) new unsigned long [this->nneurons];

		for (unsigned long con= 0; con<this->ninters; ++con){
			if(this->inters[con].GetWeightChange_withPost()!=0){
				NumberOfInputsWithPostSynapticLearning[this->inters[con].GetTarget()->GetIndex()]++;
			}
			if(this->inters[con].GetWeightChange_withoutPost()!=0){
				NumberOfInputsWithoutPostSynapticLearning[this->inters[con].GetTarget()->GetIndex()]++;
			}
		}

		for (unsigned long neu = 0; neu<this->nneurons; ++neu){
			InputsLeftWithPostSynapticLearning[neu] = NumberOfInputsWithPostSynapticLearning[neu];
			InputsLeftWithoutPostSynapticLearning[neu] = NumberOfInputsWithoutPostSynapticLearning[neu];
		}

		Interconnection *** InputConnectionsWithPostSynapticLearning = (Interconnection ***) new Interconnection ** [this->nneurons];
		Interconnection *** InputConnectionsWithoutPostSynapticLearning = (Interconnection ***) new Interconnection ** [this->nneurons];

		for (unsigned long neu = 0; neu<this->nneurons; ++neu){
			InputConnectionsWithPostSynapticLearning[neu] = 0;
			InputConnectionsWithoutPostSynapticLearning[neu] = 0;

			if (NumberOfInputsWithPostSynapticLearning[neu]>0){
				InputConnectionsWithPostSynapticLearning[neu] = (Interconnection **) new Interconnection * [NumberOfInputsWithPostSynapticLearning[neu]];
			} 
			if (NumberOfInputsWithoutPostSynapticLearning[neu]>0){
				InputConnectionsWithoutPostSynapticLearning[neu] = (Interconnection **) new Interconnection * [NumberOfInputsWithoutPostSynapticLearning[neu]];
			} 
		}

		for (unsigned long con= this->ninters-1; con<this->ninters; --con){
			if (this->inters[con].GetWeightChange_withPost()!=0){
				unsigned long TargetCell = this->inters[con].GetTarget()->GetIndex();
				InputConnectionsWithPostSynapticLearning[TargetCell][--InputsLeftWithPostSynapticLearning[TargetCell]] = this->inters+con;
			}
			if (this->inters[con].GetWeightChange_withoutPost()!=0){
				unsigned long TargetCell = this->inters[con].GetTarget()->GetIndex();
				InputConnectionsWithoutPostSynapticLearning[TargetCell][--InputsLeftWithoutPostSynapticLearning[TargetCell]] = this->inters+con;
			}
		}

		for (unsigned long neu = 0; neu<this->nneurons; ++neu){
			this->neurons[neu].SetInputConnectionsWithPostSynapticLearning(InputConnectionsWithPostSynapticLearning[neu],NumberOfInputsWithPostSynapticLearning[neu]);
			this->neurons[neu].SetInputConnectionsWithoutPostSynapticLearning(InputConnectionsWithoutPostSynapticLearning[neu],NumberOfInputsWithoutPostSynapticLearning[neu]);

			//learning rule index are ordered by input synapses index in order to improve trigger learning.
			for (unsigned long aux = 0; aux < NumberOfInputsWithoutPostSynapticLearning[neu]; aux++){
				InputConnectionsWithoutPostSynapticLearning[neu][aux]->SetLearningRuleIndex_withoutPost(InputConnectionsWithoutPostSynapticLearning[neu][aux]->GetWeightChange_withoutPost()->counter);
				InputConnectionsWithoutPostSynapticLearning[neu][aux]->GetWeightChange_withoutPost()->counter++;
			}
			//learning rule index are ordered by input synapses index in order to improve postsynaptic learning.
		
			for (unsigned long aux = 0; aux < NumberOfInputsWithPostSynapticLearning[neu]; aux++){
				InputConnectionsWithPostSynapticLearning[neu][aux]->SetLearningRuleIndex_withPost(InputConnectionsWithPostSynapticLearning[neu][aux]->GetWeightChange_withPost()->counter);
				InputConnectionsWithPostSynapticLearning[neu][aux]->GetWeightChange_withPost()->counter++;
			}

			//Initialize the learning rule index in aech neuron in order to improve cache friendly.
			this->neurons[neu].initializeLearningRuleIndex();

		}

		delete [] InputConnectionsWithPostSynapticLearning;
		delete [] NumberOfInputsWithPostSynapticLearning;
		delete [] InputsLeftWithPostSynapticLearning;

		delete [] InputConnectionsWithoutPostSynapticLearning;
		delete [] NumberOfInputsWithoutPostSynapticLearning;
		delete [] InputsLeftWithoutPostSynapticLearning;
	}
}



NeuronModel ** Network::LoadNetTypes(string ident_type, string neutype, int & ni, long Currentline, const char * netfile) throw (EDLUTException){
	NeuronModel ** type;

   	for(ni=0;ni<nneutypes && neutypes[ni][0]!=0 && ( neutypes[ni][0]->GetModelID()==neutype && neutypes[ni][0]->GetTypeID()!=ident_type || neutypes[ni][0]->GetModelID()!=neutype);++ni);
	
	if (ni<nneutypes && neutypes[ni][0]==0){
		for(int n=0; n<this->GetNumberOfQueues(); n++){
			if (ident_type == "InputSpikeNeuronModel" || ident_type == "InputNeuronModel"){
				neutypes[ni][n] = (InputSpikeNeuronModel *) new InputSpikeNeuronModel(ident_type, neutype);
			}else if (ident_type == "InputCurrentNeuronModel"){
				neutypes[ni][n] = (InputCurrentNeuronModel *) new InputCurrentNeuronModel(ident_type, neutype);
			}else if (ident_type=="LIFTimeDrivenModel_1_4"){
				neutypes[ni][n] = (LIFTimeDrivenModel_1_4 *) new LIFTimeDrivenModel_1_4(ident_type, neutype);
			}else if (ident_type == "LIFTimeDrivenModel_1_2"){
				neutypes[ni][n] = (LIFTimeDrivenModel_1_2 *) new LIFTimeDrivenModel_1_2(ident_type, neutype);
			}else if (ident_type == "LIFTimeDrivenModel_1_1"){
				neutypes[ni][n] = (LIFTimeDrivenModel_1_1 *) new LIFTimeDrivenModel_1_1(ident_type, neutype);
			}else if (ident_type=="CompressSynchronousTableBasedModel"){
   				neutypes[ni][n] = (CompressTableBasedModel *) new CompressSynchronousTableBasedModel(ident_type, neutype);
			}else if (ident_type=="CompressTableBasedModel"){
   				neutypes[ni][n] = (CompressTableBasedModel *) new CompressTableBasedModel(ident_type, neutype);
			}else if (ident_type=="SynchronousTableBasedModel"){
   				neutypes[ni][n] = (SynchronousTableBasedModel *) new SynchronousTableBasedModel(ident_type, neutype);
			}else if (ident_type=="TableBasedModel"){
   				neutypes[ni][n] = (TableBasedModel *) new TableBasedModel(ident_type, neutype);
			}else if (ident_type == "TimeDrivenSinCurrentGenerator"){
				neutypes[ni][n] = (TimeDrivenSinCurrentGenerator *) new TimeDrivenSinCurrentGenerator(ident_type, neutype);
			}else if (ident_type == "LIFTimeDrivenModel_1_1_GPU"){
				if (NumberOfGPUs>0){
					neutypes[ni][n] = (LIFTimeDrivenModel_1_1_GPU *) new LIFTimeDrivenModel_1_1_GPU(ident_type, neutype);
				}else{
					printf("WARNING: CUDA capable GPU not available. Implementing CPU model intead of GPU model\n");
					neutypes[ni][n] = (LIFTimeDrivenModel_1_1 *) new LIFTimeDrivenModel_1_1(ident_type, neutype);
				}
			}else if (ident_type=="LIFTimeDrivenModel_1_2_GPU"){
				if(NumberOfGPUs>0){
					neutypes[ni][n] = (LIFTimeDrivenModel_1_2_GPU *) new LIFTimeDrivenModel_1_2_GPU(ident_type, neutype);
				}else{
					printf("WARNING: CUDA capable GPU not available. Implementing CPU model intead of GPU model\n");
					neutypes[ni][n] = (LIFTimeDrivenModel_1_2 *) new LIFTimeDrivenModel_1_2(ident_type, neutype);
				}
			}else if (ident_type=="LIFTimeDrivenModel_1_4_GPU"){
				if(NumberOfGPUs>0){
					neutypes[ni][n] = (LIFTimeDrivenModel_1_4_GPU *) new LIFTimeDrivenModel_1_4_GPU(ident_type, neutype);
				}else{
					printf("WARNING: CUDA capable GPU not available. Implementing CPU model intead of GPU model\n");
					neutypes[ni][n] = (LIFTimeDrivenModel_1_4 *) new LIFTimeDrivenModel_1_4(ident_type, neutype);
				}
			}else {
				throw EDLUTFileException(TASK_NETWORK_LOAD_NEURON_MODELS, ERROR_NETWORK_NEURON_MODEL_TYPE, REPAIR_NETWORK_NEURON_MODEL_TYPE, Currentline, netfile, true);
			}
			neutypes[ni][n]->LoadNeuronModel();
		}
		type=neutypes[ni];
	} else if (ni<nneutypes) {
		type = neutypes[ni];
	} else {
		throw EDLUTFileException(TASK_NETWORK_LOAD_NEURON_MODELS, ERROR_NETWORK_NEURON_MODEL_NUMBER, REPAIR_NETWORK_NEURON_MODEL_NUMBER, Currentline, netfile, true);
	}

	return(type);
}


void Network::InitializeStates(int ** N_neurons){
	for( int z=0; z< this->nneutypes; z++){
		for(int j=0; j<this->GetNumberOfQueues(); j++){
			if(N_neurons[z][j]>0){
				neutypes[z][j]->InitializeStates(N_neurons[z][j], j);
			}else{
				neutypes[z][j]->InitializeStates(1,j);
			}
		}
	}
}



void Network::InitNetPredictions(EventQueue * Queue){
	int nneu;
	for(nneu=0;nneu<nneurons;nneu++){
		if (neurons[nneu].GetNeuronModel()->GetModelSimulationMethod()==EVENT_DRIVEN_MODEL && neurons[nneu].GetNeuronModel()->GetModelType()==NEURAL_LAYER){
			EventDrivenNeuronModel * Model = (EventDrivenNeuronModel *) neurons[nneu].GetNeuronModel();
			InternalSpike * spike = Model->GenerateInitialActivity(neurons+nneu);
			if (spike!=0){
				Queue->InsertEvent(spike->GetSource()->get_OpenMP_queue_index(),spike);
			}
		}
	}

}

Network::Network(const char * netfile, const char * wfile, EventQueue * Queue, int numberOfQueues) throw (EDLUTException) : inters(0), ninters(0), neutypes(0), nneutypes(0), neurons(0), nneurons(0), timedrivenneurons(0), ntimedrivenneurons(0), wchanges(0), nwchanges(0), wordination(0), NumberOfQueues(numberOfQueues), minpropagationdelay(0.0001), invminpropagationdelay(1.0 / 0.0001), monitore_neurons(false){
	this->LoadNet(netfile);	
	this->LoadWeights(wfile);
	this->InitNetPredictions(Queue);
	this->CalculaElectricalCouplingDepedencies();
}
   		
Network::~Network(){
	if (inters!=0) {
		delete [] inters;
	}

	if (neutypes!=0) {
		for (int i=0; i<this->nneutypes; ++i){
			if (this->neutypes[i]!=0){
				for( int j=0; j<this->GetNumberOfQueues(); j++){
					if (this->neutypes[i][j]!=0){
						if(ntimedrivenneurons_GPU[i][0]>0){
							HANDLE_ERROR(cudaSetDevice(GPUsIndex[j % NumberOfGPUs]));
						}
						delete this->neutypes[i][j];
					}
				}
				delete [] this->neutypes[i];
			}
		}
		delete [] neutypes;
	}

	
	if (neurons!=0) {
		delete [] neurons;
	}

	if (timedrivenneurons!=0) {
		for (int z=0; z<this->nneutypes; z++){
			if(timedrivenneurons[z]!=0){
				for(int j=0; j<this->GetNumberOfQueues(); j++){
					delete [] timedrivenneurons[z][j];
				}
				delete [] timedrivenneurons[z];
			}
		}
		delete [] timedrivenneurons;
	}

	if(ntimedrivenneurons!=0){
		for(int i=0; i<this->nneutypes; i++){
			if(ntimedrivenneurons[i]!=0){
				delete [] ntimedrivenneurons[i];
			}
		}
		delete [] ntimedrivenneurons;
	}

	if (timedrivenneurons_GPU!=0) {
		for (int z=0; z<this->nneutypes; z++){
			if(timedrivenneurons_GPU[z]!=0){
				for(int j=0; j<this->GetNumberOfQueues(); j++){
					delete [] timedrivenneurons_GPU[z][j];
				}
				delete [] timedrivenneurons_GPU[z];
			}
		}
		delete [] timedrivenneurons_GPU;
	}

	if(ntimedrivenneurons_GPU!=0){
		for(int i=0; i<this->nneutypes; i++){
			if(ntimedrivenneurons_GPU[i]!=0){
				delete [] ntimedrivenneurons_GPU[i];
			}
		}
		delete [] ntimedrivenneurons_GPU;
	}

	if (wchanges!=0) {
		for (int i=0; i<this->nwchanges; ++i){
			delete this->wchanges[i];
		}
		delete [] wchanges;
	}
	if (wordination!=0) delete [] wordination;
}
   		
Neuron * Network::GetNeuronAt(int index) const{
	return &(this->neurons[index]);
}
   		
int Network::GetNeuronNumber() const{
	return this->nneurons;	
}

Neuron ** Network::GetTimeDrivenNeuronAt(int index0, int index1) const{
	return this->timedrivenneurons[index0][index1];
}

Neuron * Network::GetTimeDrivenNeuronAt(int index0,int index1, int index2) const{
	return this->timedrivenneurons[index0][index1][index2];
}

Neuron ** Network::GetTimeDrivenNeuronGPUAt(int index0, int index1) const{
	return this->timedrivenneurons_GPU[index0][index1];
}
Neuron * Network::GetTimeDrivenNeuronGPUAt(int index0,int index1, int index2) const{
	return this->timedrivenneurons_GPU[index0][index1][index2];
}
   		
int ** Network::GetTimeDrivenNeuronNumber() const{
	return this->ntimedrivenneurons;
}

int Network::GetNneutypes() const{
	return this->nneutypes;
}
int ** Network::GetTimeDrivenNeuronNumberGPU() const{
	return this->ntimedrivenneurons_GPU;
}

NeuronModel ** Network::GetNeuronModelAt(int index) const{
	return this->neutypes[index];
}

NeuronModel * Network::GetNeuronModelAt(int index1, int index2) const{
	return this->neutypes[index1][index2];
}

LearningRule * Network::GetLearningRuleAt(int index) const{
	return this->wchanges[index];
}

int Network::GetLearningRuleNumber() const{
	return this->nwchanges;
}

void Network::LoadNet(const char *netfile) throw (EDLUTException, EDLUTFileException){
	FILE *fh;
	long savedcurrentline;
	long Currentline=1L;
	fh=fopen(netfile,"rt");
	if(fh){
		skip_comments(fh, Currentline);
		if (fscanf(fh, "%i", &(this->nneutypes)) == 1 && this->nneutypes > 0){
			this->neutypes=(NeuronModel ***) new NeuronModel ** [this->nneutypes];
			if(this->neutypes){
				for(int ni=0;ni<this->nneutypes;ni++){
					this->neutypes[ni]=(NeuronModel **) new NeuronModel * [this->GetNumberOfQueues()];
					for(int n=0; n<this->GetNumberOfQueues(); n++){
						this->neutypes[ni][n]=0;
					}
				}
            	skip_comments(fh, Currentline);
				if (fscanf(fh, "%i", &(this->nneurons)) == 1 && this->nneurons > 0){
            		int tind,nind,nn,outn,monit;
            		NeuronModel ** type;
            		char ident[MAXIDSIZE+1];
            		char ident_type[MAXIDSIZE+1];
            		this->neurons=(Neuron *) new Neuron [this->nneurons];

					ntimedrivenneurons= (int**) new int* [this->nneutypes]();
					int *** time_driven_index = (int ***) new int **[this->nneutypes];
							
					ntimedrivenneurons_GPU= (int **) new int* [this->nneutypes]();
					int *** time_driven_index_GPU=(int ***) new int **[this->nneutypes];

					int ** N_neurons= (int **) new int *[this->nneutypes]();

					for (int z=0; z<this->nneutypes; z++){
						ntimedrivenneurons[z]=new int [this->GetNumberOfQueues()]();
						ntimedrivenneurons_GPU[z]=new int [this->GetNumberOfQueues()]();

						time_driven_index[z]=(int**)new int* [this->GetNumberOfQueues()];
						time_driven_index_GPU[z]=(int**)new int* [this->GetNumberOfQueues()];
						for(int j=0; j<this->GetNumberOfQueues(); j++){
							time_driven_index[z][j]=new int [this->nneurons]();
							time_driven_index_GPU[z][j]=new int [this->nneurons]();
						}

						N_neurons[z]=new int [this->GetNumberOfQueues()]();
					}


            		if(this->neurons){
            			for(tind=0;tind<this->nneurons;tind+=nn){
                     		skip_comments(fh,Currentline);
                     		if(fscanf(fh,"%i",&nn)==1 && nn>0 && fscanf(fh," %"MAXIDSIZEC"[^ ]%*[^ ]",ident_type)==1 && fscanf(fh," %"MAXIDSIZEC"[^ ]%*[^ ]",ident)==1 && fscanf(fh,"%i",&outn)==1 && fscanf(fh,"%i",&monit)==1){
                     			if(tind+nn>this->nneurons){
									throw EDLUTFileException(TASK_NETWORK_LOAD, ERROR_NETWORK_NUMBER_OF_NEURONS, REPAIR_NETWORK_NUMBER_OF_NEURONS, Currentline, netfile, true);
                     			}
								int ni;                    
                        		savedcurrentline=Currentline;
								type = LoadNetTypes(ident_type, ident, ni, Currentline, netfile);
                        		Currentline=savedcurrentline;

								int blockSize=(nn + NumberOfQueues - 1) / NumberOfQueues;
								int blockIndex;
	                    		for(nind=0;nind<nn;nind++){
									blockIndex=nind/blockSize;

									neurons[nind + tind].InitNeuron(nind + tind, N_neurons[ni][blockIndex], type[blockIndex], ((bool)monit), ((bool)outn), blockIndex);
									N_neurons[ni][blockIndex]++;

									//If some neuron is monitored.
									if(monit){
										for(int n=0; n<this->GetNumberOfQueues(); n++){
											type[n]->GetVectorNeuronState()->Set_Is_Monitored(true);
										}
										this->monitore_neurons = true;
									}
									
									if (type[0]->GetModelSimulationMethod() == TIME_DRIVEN_MODEL_CPU){
										time_driven_index[ni][blockIndex][this->ntimedrivenneurons[ni][blockIndex]] = nind+tind;
										this->ntimedrivenneurons[ni][blockIndex]++;
									}
											
									if (type[0]->GetModelSimulationMethod()==TIME_DRIVEN_MODEL_GPU){
										time_driven_index_GPU[ni][blockIndex][this->ntimedrivenneurons_GPU[ni][blockIndex]]=nind+tind;
										this->ntimedrivenneurons_GPU[ni][blockIndex]++;
									}
                        		}
                        	}else{
								throw EDLUTFileException(TASK_NETWORK_LOAD, ERROR_NETWORK_NEURON_PARAMETERS, REPAIR_NETWORK_NEURON_PARAMETERS, Currentline, netfile, true);
                        	}
                     	}

						// Create the time-driven cell array
						timedrivenneurons=(Neuron ****) new Neuron *** [this->nneutypes]();
						for (int z=0; z<this->nneutypes; z++){ 
							if (this->ntimedrivenneurons[z][0]>0){
								this->timedrivenneurons[z]=(Neuron ***) new Neuron ** [this->GetNumberOfQueues()];
								for(int j=0; j<this->GetNumberOfQueues(); j++){
									this->timedrivenneurons[z][j]=(Neuron **) new Neuron * [this->ntimedrivenneurons[z][j]];
									for (int i=0; i<this->ntimedrivenneurons[z][j]; ++i){
										this->timedrivenneurons[z][j][i] = &(this->neurons[time_driven_index[z][j][i]]);
									}	
								}
							}
						}

						// Create the time-driven cell array for GPU
						timedrivenneurons_GPU=(Neuron ****) new Neuron *** [this->nneutypes]();
						for (int z=0; z<this->nneutypes; z++){ 
							if (this->ntimedrivenneurons_GPU[z][0]>0){
								this->timedrivenneurons_GPU[z]=(Neuron ***) new Neuron ** [this->GetNumberOfQueues()];
								for(int j=0; j<this->GetNumberOfQueues(); j++){
									this->timedrivenneurons_GPU[z][j]=(Neuron **) new Neuron * [this->ntimedrivenneurons_GPU[z][j]];
									for (int i=0; i<this->ntimedrivenneurons_GPU[z][j]; ++i){
										this->timedrivenneurons_GPU[z][j][i] = &(this->neurons[time_driven_index_GPU[z][j][i]]);
									}	
								}
							}
						}

						// Initialize states. 
						InitializeStates(N_neurons);

            		}else{
						throw EDLUTFileException(TASK_NETWORK_LOAD, ERROR_NETWORK_ALLOCATE, REPAIR_NETWORK_ALLOCATE, Currentline, netfile, true);
            		}

					

					for (int z=0; z<this->nneutypes; z++){
						for(int j=0; j<this->GetNumberOfQueues(); j++){
							delete [] time_driven_index[z][j];
							delete [] time_driven_index_GPU[z][j];
						}
						delete [] time_driven_index[z];
						delete [] time_driven_index_GPU[z];
						delete [] N_neurons[z];
					} 
					delete [] time_driven_index;
					delete [] time_driven_index_GPU;
					delete [] N_neurons;

            		/////////////////////////////////////////////////////////
            		// Check the number of neuron types
					int ni;
            		for(ni=0;ni<this->nneutypes && this->neutypes[ni]!=0;ni++);

            		if (ni!=this->nneutypes){
						throw EDLUTFileException(TASK_NETWORK_LOAD_NEURON_MODELS, ERROR_NETWORK_NEURON_MODEL_NUMBER, REPAIR_NETWORK_NEURON_MODEL_NUMBER, Currentline, netfile, true);
            		}
            	}else{
					throw EDLUTFileException(TASK_NETWORK_LOAD, ERROR_NETWORK_READ_NUMBER_OF_NEURONS, REPAIR_NETWORK_READ_NUMBER_OF_NEURONS, Currentline, netfile, true);
            	}
            	
            	
            	skip_comments(fh,Currentline);
            	int * N_ConectionWithLearning;
				if (fscanf(fh, "%i", &(this->nwchanges)) == 1 && this->nwchanges >= 0){
        			int wcind;
        			this->wchanges=new LearningRule * [this->nwchanges];
        			N_ConectionWithLearning=new int [this->nwchanges](); 
        			if(this->wchanges){
        				for(wcind=0;wcind<this->nwchanges;wcind++){
        					char ident_type[MAXIDSIZE+1];
        					skip_comments(fh,Currentline);
        					string LearningModel;
        					if(fscanf(fh," %"MAXIDSIZEC"[^ \n]%*[^ ]",ident_type)==1){
        						if (string(ident_type)==string("ExpOptimisedBufferedAdditiveKernel")){
        							this->wchanges[wcind] = new ExpOptimisedBufferedWeightChange(wcind);
        						} else if (string(ident_type)==string("STDP")){
        							this->wchanges[wcind] = new STDPWeightChange(wcind);
        						} else {
									throw EDLUTFileException(TASK_NETWORK_LOAD_LEARNING_RULES, ERROR_NETWORK_LEARNING_RULE_TYPE, REPAIR_NETWORK_LEARNING_RULE_TYPE, Currentline, netfile, true);
        						}

        						this->wchanges[wcind]->LoadLearningRule(fh,Currentline,netfile);

                       		}else{
								throw EDLUTFileException(TASK_NETWORK_LOAD_LEARNING_RULES, ERROR_NETWORK_LEARNING_RULE_LOAD, REPAIR_NETWORK_LEARNING_RULE_LOAD, Currentline, netfile, true);
                       		}
        				}
        			}else{
						throw EDLUTFileException(TASK_NETWORK_LOAD_LEARNING_RULES, ERROR_NETWORK_ALLOCATE, REPAIR_NETWORK_ALLOCATE, Currentline, netfile, true);
        			}
        		}else{
					throw EDLUTFileException(TASK_NETWORK_LOAD_LEARNING_RULES, ERROR_NETWORK_LEARNING_RULE_NUMBER, REPAIR_NETWORK_LEARNING_RULE_NUMBER, Currentline, netfile, true);
        		}

        		skip_comments(fh,Currentline);
				if (fscanf(fh, "%li", &(this->ninters)) == 1 && this->ninters > 0){
        			int source,nsources,target,ntargets,nreps;
					int intwchange1, intwchange2;
					bool trigger1, trigger2;

        			float delay,delayinc,maxweight;
        			int type;
        			int iind,sind,tind,rind,posc;
        			this->inters=(Interconnection *) new Interconnection [this->ninters];
        			this->wordination=(Interconnection **) new Interconnection * [this->ninters];
        			if(this->inters && this->wordination){
        				for(iind=0;iind<this->ninters;iind+=nsources*ntargets*nreps){
        					skip_comments(fh,Currentline);
        					if(fscanf(fh,"%i",&source)==1 && fscanf(fh,"%i",&nsources)==1 && fscanf(fh,"%i",&target)==1 && fscanf(fh,"%i",&ntargets)==1 && fscanf(fh,"%i",&nreps)==1 && fscanf(fh,"%f",&delay)==1 && fscanf(fh,"%f",&delayinc)==1 && fscanf(fh,"%i",&type)==1 && fscanf(fh,"%f",&maxweight)==1){

								//Load the first learning rule index
								trigger1 = false;
								//Check if the synapse implement a trigger learning rule
								if (fscanf(fh, " t%d", &intwchange1) == 1){
									trigger1 = true;
								}
								//Check if the synapse implement a non trigger learning rule
								else {
									if (fscanf(fh, "%d", &intwchange1) == 1){
										trigger1 = false;
									}else{
										throw EDLUTFileException(TASK_NETWORK_LOAD_SYNAPSES, ERROR_NETWORK_SYNAPSES_FIRST_LEARNING_RULE_LOAD, REPAIR_NETWORK_SYNAPSES_LEARNING_RULE_INDEX, Currentline, netfile, true);
									}
								}

								if (intwchange1 >= this->nwchanges){
									throw EDLUTFileException(TASK_NETWORK_LOAD_SYNAPSES, ERROR_NETWORK_SYNAPSES_FIRST_LEARNING_RULE_INDEX, REPAIR_NETWORK_SYNAPSES_LEARNING_RULE_INDEX, Currentline, netfile, true);
								}


								//Load the second learning rule index
								intwchange2 = -1;
								trigger2 = false;
								if (intwchange1 >= 0){
									//Check if there is a second learning rule defined for this synapse.
									if (is_end_line(fh, Currentline) == false){
										//Check if the synapse implement a trigger learning rule
										if (fscanf(fh, " t%d", &intwchange2) == 1){
											trigger2 = true;
										}
										//Check if the synapse implement a non trigger learning rule
										else {
											if (fscanf(fh, "%d", &intwchange2) == 1){
												trigger2 = false;
											}else{
												throw EDLUTFileException(TASK_NETWORK_LOAD_SYNAPSES, ERROR_NETWORK_SYNAPSES_SECOND_LEARNING_RULE_LOAD, REPAIR_NETWORK_SYNAPSES_LEARNING_RULE_INDEX, Currentline, netfile, true);
											}
										}

										if(intwchange2>= this->nwchanges){
											throw EDLUTFileException(TASK_NETWORK_LOAD_SYNAPSES, ERROR_NETWORK_SYNAPSES_SECOND_LEARNING_RULE_INDEX, REPAIR_NETWORK_SYNAPSES_LEARNING_RULE_INDEX, Currentline, netfile, true);
										}
									}
								}


								//Check if the number of synapses do not exceed the total number of synapses.
								if(iind+nsources*ntargets*nreps>this->ninters){
									throw EDLUTFileException(TASK_NETWORK_LOAD_SYNAPSES, ERROR_NETWORK_SYNAPSES_NUMBER, REPAIR_NETWORK_SYNAPSES_NUMBER, Currentline, netfile, true);
        						}else{
									//Check if the source and target neuron indexs do not exceed the total number of neurons.
        							if(source+nreps*nsources>this->nneurons || target+nreps*ntargets>this->nneurons){
										throw EDLUTFileException(TASK_NETWORK_LOAD_SYNAPSES, ERROR_NETWORK_SYNAPSES_NEURON_INDEX, REPAIR_NETWORK_SYNAPSES_NEURON_INDEX, Currentline, netfile, true);
  									}
        						}

								if(trigger1==false && (intwchange2>=0 && trigger2==false)){
									throw EDLUTFileException(TASK_NETWORK_LOAD_SYNAPSES, ERROR_NETWORK_SYNAPSES_LEARNING_RULE_NON_TRIGGER, REPAIR_NETWORK_SYNAPSES_LEARNING_RULE, Currentline, netfile, true);
								}
								if(trigger1==true && trigger2==true){
									throw EDLUTFileException(TASK_NETWORK_LOAD_SYNAPSES, ERROR_NETWORK_SYNAPSES_LEARNING_RULE_TRIGGER, REPAIR_NETWORK_SYNAPSES_LEARNING_RULE, Currentline, netfile, true);
								}

        						for(rind=0;rind<nreps;rind++){
        							for(sind=0;sind<nsources;sind++){
        								for(tind=0;tind<ntargets;tind++){
        									posc=iind+rind*nsources*ntargets+sind*ntargets+tind;
        									this->inters[posc].SetIndex(posc);
        									this->inters[posc].SetSource(&(this->neurons[source+rind*nsources+sind]));
        									this->inters[posc].SetTarget(&(this->neurons[target+rind*ntargets+tind]));
										this->inters[posc].SetTargetNeuronModel(this->neurons[target + rind*ntargets + tind].GetNeuronModel());
										this->inters[posc].SetTargetNeuronModelIndex(this->neurons[target + rind*ntargets + tind].GetIndex_VectorNeuronState());

        									this->inters[posc].SetDelay(RoundPropagationDelay(delay+delayinc*tind));
											this->inters[posc].SetType(type);
											if (this->inters[posc].GetTarget()->GetNeuronModel()->CheckSynapseType(&this->inters[posc]) == false){
												throw EDLUTFileException(TASK_NETWORK_LOAD_SYNAPSES, ERROR_NETWORK_SYNAPSES_TYPE, REPAIR_NETWORK_SYNAPSES_TYPE, Currentline, netfile, true);
											}
											
        									this->inters[posc].SetWeight(maxweight);   //TODO: Use max interconnection conductance
   									
											this->inters[posc].SetMaxWeight(maxweight);

											this->inters[posc].SetWeightChange_withPost(0);
											this->inters[posc].SetWeightChange_withoutPost(0);
											if(intwchange1 >= 0){
												//Set the new learning rule
												if(wchanges[intwchange1]->ImplementPostSynaptic()==true){
													this->inters[posc].SetWeightChange_withPost(this->wchanges[intwchange1]);
												}else{
													this->inters[posc].SetWeightChange_withoutPost(this->wchanges[intwchange1]);
													if(trigger1==true){
														this->inters[posc].SetTriggerConnection();
													}
												}
												N_ConectionWithLearning[intwchange1]++;
											}

											if(intwchange2 >= 0){
												//Set the new learning rule
												if(wchanges[intwchange2]->ImplementPostSynaptic()==true){
													this->inters[posc].SetWeightChange_withPost(this->wchanges[intwchange2]);
												}else{
													this->inters[posc].SetWeightChange_withoutPost(this->wchanges[intwchange2]);
													if(trigger2==true){
														this->inters[posc].SetTriggerConnection();
													}
												}
												N_ConectionWithLearning[intwchange2]++;
											}
                                		}
        							}
        						}
        					}else{
								throw EDLUTFileException(TASK_NETWORK_LOAD_SYNAPSES, ERROR_NETWORK_SYNAPSES_LOAD, REPAIR_NETWORK_SYNAPSES_LOAD, Currentline, netfile, true);
        					}
        				}
						for(int t=0; t<this->nwchanges; t++){
							if(N_ConectionWithLearning[t]>0){
								this->wchanges[t]->InitializeConnectionState(N_ConectionWithLearning[t], this->GetNeuronNumber());
							}
						}
						if(this->nwchanges>0){
						delete [] N_ConectionWithLearning;
						}
        				
						FindOutConnections();
                    	SetWeightOrdination(); // must be before find_in_c() and after find_out_c()
                    	FindInConnections();

						//Calculate the output delay structure of each neuron. This structure is used by PropagatepSpikeGroup event to group several
						//PropagatedSpike events in just one.
						for(int i=0; i<this->GetNeuronNumber(); i++){
							this->GetNeuronAt(i)->CalculateOutputDelayStructure();
						}
						for(int i=0; i<this->GetNeuronNumber(); i++){
							this->GetNeuronAt(i)->CalculateOutputDelayIndex();
						}


						//Initialize the Input Current Sypases Structure in each neuron model if it implements this kind of input synapses.
						for (int z = 0; z < this->nneutypes; z++){
							for (int j = 0; j < this->GetNumberOfQueues(); j++){
								neutypes[z][j]->InitializeInputCurrentSynapseStructure();
							}
						}
                    }else{
						throw EDLUTException(TASK_NETWORK_LOAD_SYNAPSES, ERROR_NETWORK_ALLOCATE, REPAIR_NETWORK_ALLOCATE);
        			}
        		}else{
					throw EDLUTFileException(TASK_NETWORK_LOAD_SYNAPSES, ERROR_NETWORK_SYNAPSES_LOAD_NUMBER, REPAIR_NETWORK_SYNAPSES_LOAD_NUMBER, Currentline, netfile, true);
        		}
            }else{
				throw EDLUTException(TASK_NETWORK_LOAD, ERROR_NETWORK_ALLOCATE, REPAIR_NETWORK_ALLOCATE);
			}
		}else{
			throw EDLUTFileException(TASK_NETWORK_LOAD, ERROR_NETWORK_NEURON_MODEL_LOAD_NUMBER, REPAIR_NETWORK_NEURON_MODEL_LOAD_NUMBER, Currentline, netfile, true);
		}
		
		fclose(fh);
	}else{
		throw EDLUTFileException(TASK_NETWORK_LOAD, ERROR_NETWORK_OPEN, REPAIR_NETWORK_OPEN, Currentline, netfile, true);
	}
	
	return;
}

void Network::LoadWeights(const char *wfile) throw (EDLUTFileException){
	FILE *fh;
	int connind;
	long Currentline=1L;
	fh=fopen(wfile,"rt");
	if(fh){
		int nweights,weind;
		float weight;
		skip_comments(fh,Currentline);
		for(connind=0;connind<this->ninters;connind+=nweights){
			skip_comments(fh, Currentline);
			if(fscanf(fh,"%i",&nweights)==1 && fscanf(fh,"%f",&weight)==1){
				if(nweights < 0 || nweights + connind > this->ninters){
					throw EDLUTFileException(TASK_WEIGHTS_LOAD, ERROR_WEIGHTS_NUMBER, REPARI_WEIGHTS_NUMBER, Currentline, wfile, true);
				}
				
				if(nweights == 0){
					nweights=this->ninters-connind;
				}
				
				for(weind=0;weind<nweights;weind++){
					Interconnection * Connection = this->wordination[connind+weind];
					Connection->SetWeight(((weight < 0.0)?RandomGenerator::frand()*Connection->GetMaxWeight():((weight > Connection->GetMaxWeight())?Connection->GetMaxWeight():weight)));
				}
			}else{
				throw EDLUTFileException(TASK_WEIGHTS_LOAD, ERROR_WEIGHTS_READ, REPAIR_WEIGHTS_READ, Currentline, wfile, true);
			}
		}
		fclose(fh);
	}else{
		throw EDLUTFileException(TASK_WEIGHTS_LOAD, ERROR_WEIGHTS_OPEN, REPAIR_WEIGHTS_OPEN, Currentline, wfile, true);
	}
	
}

void Network::SaveWeights(const char *wfile) throw (EDLUTException){
	FILE *fh;
	int connind;
	fh=fopen(wfile,"wt");
	if(fh){
		float weight,antweight;
		int nantw;
		nantw=0;
		antweight=0.0;
		weight=0.0; // just to avoid compiler warning messages
		
		// Write the number of weights
		//if(fprintf(fh,"%li\n",this->ninters) <= 0){
		//	throw EDLUTException(12,33,4,0);
		//}
					
		for(connind=0;connind<=this->ninters;connind++){
			if(connind < this->ninters){
				weight=this->wordination[connind]->GetWeight();
			}
			
			if(antweight != weight || connind == this->ninters){
				if(nantw > 0){
					if(fprintf(fh,"%i %g\n",nantw,antweight) <= 0){
						throw EDLUTException(TASK_WEIGHTS_SAVE, ERROR_WEIGHTS_SAVE, REPAIR_WEIGHTS_SAVE);
					}
				}
				
				antweight=weight;
				nantw=1;
			}else{
				nantw++;
			}
		}
		
		// fprintf(fh,"// end of written data\n");
		
		fclose(fh);
	}else{
		throw EDLUTException(TASK_WEIGHTS_SAVE, ERROR_WEIGHTS_SAVE_OPEN, REPAIR_WEIGHTS_SAVE);
	}
	
}

ostream & Network::PrintInfo(ostream & out) {
	int ind;

	out << "- Neuron types:" << endl;

	for(ind=0;ind<this->nneutypes;ind++){
		out << "\tType: " << ind << endl;

		this->neutypes[ind][0]->PrintInfo(out);
	}
   
	out << "- Neurons:" << endl;
   	
   	for(ind=0;ind<this->nneurons;ind++){
		this->neurons[ind].PrintInfo(out);
	}

   	out << "- Weight change types:" << endl;

	for(ind=0;ind<this->nwchanges;ind++){
		out << "\tChange: " << ind << endl;
		this->wchanges[ind]->PrintInfo(out);
	}

	out << "- Interconnections:" << endl;

	for(ind=0; ind<this->ninters; ind++){
		out << "\tConnection: " << ind << endl;
		this->inters[ind].PrintInfo(out);
	}
	
	return out;
}

int Network::GetNumberOfQueues(){
	return NumberOfQueues;
}


double Network::GetMinInterpropagationTime(){
	double time = 10000000;
	for(int i=0; i<this->ninters; i++){
		if(inters[i].GetSource()->get_OpenMP_queue_index()!=inters[i].GetTarget()->get_OpenMP_queue_index() && inters[i].GetDelay()<time){
			time=inters[i].GetDelay();
		}
	}

	if(time == 10000000){
		time=0;
	}
	return time;
}


double Network::RoundPropagationDelay(double time){
	double result=floor(time*invminpropagationdelay + minpropagationdelay*0.5)*minpropagationdelay;
	if(result<=0){
		return minpropagationdelay;
	} 
	return result;
}




void Network::CalculaElectricalCouplingDepedencies(){
	//Calculate the electrical coupling dependencies for each neuron model
	for (int i = 0; i < this->ninters; i++){
		(this->inters + i)->GetTarget()->GetNeuronModel()->CalculateElectricalCouplingSynapseNumber((this->inters + i));
	}

	for (int i = 0; i < this->GetNneutypes(); i++){
		for (int j = 0; j < this->GetNumberOfQueues(); j++){
			this->GetNeuronModelAt(i, j)->InitializeElectricalCouplingSynapseDependencies();
		}
	}
	for (int i = 0; i < this->ninters; i++){
		(this->inters + i)->GetTarget()->GetNeuronModel()->CalculateElectricalCouplingSynapseDependencies((this->inters + i));
	}
}



