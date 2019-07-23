/***************************************************************************
 *                           TableBasedModel.cpp                           *
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

#include "../../include/neuron_model/TableBasedModel.h"
#include "../../include/neuron_model/NeuronModelTable.h"
#include "../../include/neuron_model/VectorNeuronState.h"
#include "../../include/neuron_model/NeuronModel.h"

#include "../../include/spike/InternalSpike.h"
#include "../../include/spike/EndRefractoryPeriodEvent.h"
#include "../../include/spike/Neuron.h"
#include "../../include/spike/Interconnection.h"
#include "../../include/spike/PropagatedSpike.h"

#include "../../include/simulation/Utils.h"

#include "../../include/openmp/openmp.h"

#include <cstring>

void TableBasedModel::LoadNeuronModel(string ConfigFile) throw (EDLUTFileException){
	FILE *fh;
	long Currentline = 0L;
	fh=fopen(ConfigFile.c_str(),"rt");
	if(fh){
		Currentline=1L;
		skip_comments(fh,Currentline);
		if(fscanf(fh,"%i",&this->NumStateVar)==1){
			unsigned int nv;

			// Initialize all vectors.
			this->StateVarTable = (NeuronModelTable **) new NeuronModelTable * [this->NumStateVar];
			this->StateVarOrder = (unsigned int *) new unsigned int [this->NumStateVar];

			// Auxiliary table index
			unsigned int * TablesIndex = (unsigned int *) new unsigned int [this->NumStateVar];

			skip_comments(fh,Currentline);
			for(nv=0;nv<this->NumStateVar;nv++){
				if(fscanf(fh,"%i",&TablesIndex[nv])!=1){
					throw EDLUTFileException(TASK_TABLE_BASED_MODEL_LOAD, ERROR_TABLE_BASED_MODEL_INDEX, REPAIR_NEURON_MODEL_TABLE_TABLES_STRUCTURE, Currentline, ConfigFile.c_str(), true);
					delete [] TablesIndex;
				}
			}

			skip_comments(fh,Currentline);

			float InitValue;
			InitValues=new float[NumStateVar+1]();

			// Create a new initial state
       		this->State = (VectorNeuronState *) new VectorNeuronState(this->NumStateVar+1, false);
//       		this->InitialState->SetLastUpdateTime(0);
//       		this->InitialState->SetNextPredictedSpikeTime(NO_SPIKE_PREDICTED);
//      		this->InitialState->SetStateVariableAt(0,0);

       		for(nv=0;nv<this->NumStateVar;nv++){
       			if(fscanf(fh,"%f",&InitValue)!=1){
					throw EDLUTFileException(TASK_TABLE_BASED_MODEL_LOAD, ERROR_TABLE_BASED_MODEL_INITIAL_VALUES, REPAIR_NEURON_MODEL_TABLE_TABLES_STRUCTURE, Currentline, ConfigFile.c_str(), true);
					delete [] TablesIndex;
       			} else {
					InitValues[nv+1]=InitValue;
//       			this->InitialState->SetStateVariableAt(nv+1,InitValue);
       			}
       		}

			// Allocate temporal state vars

   			skip_comments(fh,Currentline);
   			unsigned int FiringIndex, FiringEndIndex;
   			if(fscanf(fh,"%i",&FiringIndex)==1){
   				skip_comments(fh,Currentline);
   				if(fscanf(fh,"%i",&FiringEndIndex)==1){
   					skip_comments(fh,Currentline);
   					if(fscanf(fh,"%i",&this->NumSynapticVar)==1){
               			skip_comments(fh,Currentline);

               			this->SynapticVar = (unsigned int *) new unsigned int [this->NumSynapticVar];
               			for(nv=0;nv<this->NumSynapticVar;nv++){
                  			if(fscanf(fh,"%i",&this->SynapticVar[nv])!=1){
								throw EDLUTFileException(TASK_TABLE_BASED_MODEL_LOAD, ERROR_TABLE_BASED_MODEL_SYNAPSE_INDEXS, REPAIR_NEURON_MODEL_TABLE_TABLES_STRUCTURE, Currentline, ConfigFile.c_str(), true);
								delete [] TablesIndex;
                  			}
                  		}

              			skip_comments(fh,Currentline);
              			if(fscanf(fh,"%i",&this->NumTables)==1){
              				unsigned int nt;
              				int tdeptables[MAXSTATEVARS];
              				int tstatevarpos,ntstatevarpos;

              				this->Tables = (NeuronModelTable *) new NeuronModelTable [this->NumTables];

              				// Update table links
              				for(nv=0;nv<this->NumStateVar;nv++){
								this->StateVarTable[nv] = this->Tables+TablesIndex[nv];
								this->StateVarTable[nv]->SetOutputStateVariableIndex(TablesIndex[nv]);
							}
              				this->FiringTable = this->Tables+FiringIndex;
              				this->EndFiringTable = this->Tables+FiringEndIndex;

              				for(nt=0;nt<this->NumTables;nt++){
								this->Tables[nt].LoadTableDescription(ConfigFile, fh, Currentline);
								this->Tables[nt].CalculateOutputTableDimensionIndex();
                   			}

              				this->NumTimeDependentStateVar = 0;
                 			for(nt=0;nt<this->NumStateVar;nt++){
         						for(nv=0;nv<this->StateVarTable[nt]->GetDimensionNumber() && this->StateVarTable[nt]->GetDimensionAt(nv)->statevar != 0;nv++);
            					if(nv<this->StateVarTable[nt]->GetDimensionNumber()){
            						tdeptables[nt]=1;
            						this->NumTimeDependentStateVar++;
            					}else{
               						tdeptables[nt]=0;
            					}
            				}

         					tstatevarpos=0;
         					ntstatevarpos=this->NumTimeDependentStateVar; // we place non-t-depentent variables in the end, so that they are evaluated afterwards
         					for(nt=0;nt<this->NumStateVar;nt++){
            					this->StateVarOrder[(tdeptables[nt])?tstatevarpos++:ntstatevarpos++]=nt;
         					}

							//Load the neuron model time scale.
							skip_comments(fh,Currentline);
							char scale[MAXIDSIZE+1];
							if (fscanf(fh, " %"MAXIDSIZEC"[^ \n]", scale) == 1){
								if (strncmp(scale, "Milisecond", 10) == 0){
									//Milisecond scale								
									this->SetTimeScale(1000);
								}else if (strncmp(scale, "Second", 6) == 0){
									//Second scale
									this->SetTimeScale(1);
								}else{
									throw EDLUTFileException(TASK_TABLE_BASED_MODEL_LOAD, ERROR_TABLE_BASED_MODEL_TIME_SCALE, REPAIR_TABLE_BASED_MODEL_TIME_SCALE, Currentline, ConfigFile.c_str(), true);
								}
							}else{
								throw EDLUTFileException(TASK_TABLE_BASED_MODEL_LOAD, ERROR_TABLE_BASED_MODEL_TIME_SCALE, REPAIR_TABLE_BASED_MODEL_TIME_SCALE, Currentline, ConfigFile.c_str(), true);
							}
            			}else{
							throw EDLUTFileException(TASK_TABLE_BASED_MODEL_LOAD, ERROR_TABLE_BASED_MODEL_NUMBER_OF_TABLES, REPAIR_NEURON_MODEL_TABLE_TABLES_STRUCTURE, Currentline, ConfigFile.c_str(), true);
							delete [] TablesIndex;
      					}
      				}else{
						throw EDLUTFileException(TASK_TABLE_BASED_MODEL_LOAD, ERROR_TABLE_BASED_MODEL_NUMBER_OF_SYNAPSES, REPAIR_NEURON_MODEL_TABLE_TABLES_STRUCTURE, Currentline, ConfigFile.c_str(), true);
						delete [] TablesIndex;
          			}
				}else{
					throw EDLUTFileException(TASK_TABLE_BASED_MODEL_LOAD, ERROR_TABLE_BASED_MODEL_END_FIRING_INDEX, REPAIR_NEURON_MODEL_TABLE_TABLES_STRUCTURE, Currentline, ConfigFile.c_str(), true);
					delete [] TablesIndex;
          		}
			}else{
				throw EDLUTFileException(TASK_TABLE_BASED_MODEL_LOAD, ERROR_TABLE_BASED_MODEL_FIRING_INDEX, REPAIR_NEURON_MODEL_TABLE_TABLES_STRUCTURE, Currentline, ConfigFile.c_str(), true);
				delete [] TablesIndex;
			}

			delete [] TablesIndex;
		}else{
			throw EDLUTFileException(TASK_TABLE_BASED_MODEL_LOAD, ERROR_TABLE_BASED_MODEL_NUMBER_OF_STATE_VARIABLES, REPAIR_NEURON_MODEL_TABLE_TABLES_STRUCTURE, Currentline, ConfigFile.c_str(), true);
		}
	}else{
		throw EDLUTFileException(TASK_TABLE_BASED_MODEL_LOAD, ERROR_TABLE_BASED_MODEL_OPEN, REPAIR_NEURON_MODEL_TABLE_TABLES_STRUCTURE, Currentline, ConfigFile.c_str(), true);
	}
	fclose(fh);

}

void TableBasedModel::LoadTables(string TableFile) throw (EDLUTException){
	FILE *fd;
	unsigned int i;
	NeuronModelTable * tab;
	fd=fopen(TableFile.c_str(),"rb");
	if(fd){
		for(i=0;i<this->NumTables;i++){
			tab=&this->Tables[i];
			tab->LoadTable(fd);
		}
		fclose(fd);
	}else{
		throw EDLUTFileException(TASK_TABLE_BASED_MODEL_LOAD, ERROR_NEURON_MODEL_OPEN, REPAIR_NEURON_MODEL_NAME, 0, TableFile.c_str(), true);
	}
}

TableBasedModel::TableBasedModel(string NeuronTypeID, string NeuronModelID): EventDrivenNeuronModel(NeuronTypeID, NeuronModelID),
		NumStateVar(0), NumTimeDependentStateVar(0), NumSynapticVar(0), SynapticVar(0),
		StateVarOrder(0), StateVarTable(0), FiringTable(0), EndFiringTable(0),
		NumTables(0), Tables(0) {
}

TableBasedModel::~TableBasedModel() {
	
	if (this->StateVarOrder!=0) {
		delete [] this->StateVarOrder;
	}

	if (this->StateVarTable!=0) {
		delete [] this->StateVarTable;
	}

	if (this->SynapticVar!=0) {
		delete [] this->SynapticVar;
	}

	if (this->Tables!=0) {
		delete [] this->Tables;
	}


	if(this->InitValues!=0){
		delete [] InitValues;
	}
}

void TableBasedModel::LoadNeuronModel() throw (EDLUTFileException){

	this->LoadNeuronModel(this->GetModelID()+".cfg");

	this->LoadTables(this->GetModelID()+".dat");

}

VectorNeuronState * TableBasedModel::InitializeState(){
	return State;
}

InternalSpike * TableBasedModel::GenerateInitialActivity(Neuron *  Cell){
	double Predicted = this->NextFiringPrediction(Cell->GetIndex_VectorNeuronState(),Cell->GetVectorNeuronState());

	InternalSpike * spike = 0;

	if(Predicted != NO_SPIKE_PREDICTED){
		Predicted += Cell->GetVectorNeuronState()->GetLastUpdateTime(Cell->GetIndex_VectorNeuronState());
		Predicted*=this->inv_timeScale;//REVISAR
		spike = new InternalSpike(Predicted, Cell->get_OpenMP_queue_index(), Cell);
	}

	this->GetVectorNeuronState()->SetNextPredictedSpikeTime(Cell->GetIndex_VectorNeuronState(),Predicted);

	return spike;
}


void TableBasedModel::UpdateState(int index, VectorNeuronState * State, double CurrentTime){

	unsigned int ivar1,orderedvar1;
	unsigned int ivar2,orderedvar2;
	unsigned int ivar3,orderedvar3;
	float TempStateVars[MAX_VARIABLES];

	State->SetStateVariableAt(index,0,CurrentTime-State->GetLastUpdateTime(index));

	for(ivar1=0;ivar1<this->NumTimeDependentStateVar;ivar1++){
		orderedvar1=this->StateVarOrder[ivar1];
		TempStateVars[orderedvar1]=this->StateVarTable[orderedvar1]->TableAccess(index,State);
	}

	for(ivar2=0;ivar2<this->NumTimeDependentStateVar;ivar2++){
		orderedvar2=this->StateVarOrder[ivar2];
		State->SetStateVariableAt(index,orderedvar2+1,TempStateVars[orderedvar2]);
	}

	for(ivar3=this->NumTimeDependentStateVar;ivar3<this->NumStateVar;ivar3++){
		orderedvar3=this->StateVarOrder[ivar3];
		State->SetStateVariableAt(index,orderedvar3+1,this->StateVarTable[orderedvar3]->TableAccess(index,State));
	}


	State->SetLastUpdateTime(index,CurrentTime);

}


void TableBasedModel::SynapsisEffect(int index, Interconnection * InputConnection){
	this->GetVectorNeuronState()->IncrementStateVariableAtCPU(index, this->SynapticVar[InputConnection->GetType()]+1, InputConnection->GetWeight()*WEIGHTSCALE);
}

double TableBasedModel::NextFiringPrediction(int index, VectorNeuronState * State){
	return this->FiringTable->TableAccessDirect(index, State);
}

double TableBasedModel::EndRefractoryPeriod(int index, VectorNeuronState * State){
	double value = NO_SPIKE_PREDICTED;
	if(this->EndFiringTable!=this->FiringTable){
		value=this->EndFiringTable->TableAccessDirect(index, State);
	}
	return value;
}



InternalSpike * TableBasedModel::ProcessInputSpike(Interconnection * inter, double time){
	int TargetIndex = inter->GetTargetNeuronModelIndex();
	Neuron * target = inter->GetTarget();

	// Update the neuron state until the current time
	if(time*this->timeScale - this->GetVectorNeuronState()->GetLastUpdateTime(TargetIndex)!=0){
		this->UpdateState(TargetIndex,this->GetVectorNeuronState(),time*this->timeScale);
	}
	// Add the effect of the input spike
	this->SynapsisEffect(TargetIndex,inter);

	InternalSpike * GeneratedSpike = 0;
	//check if the neuron is in the refractory period.
	if(time*this->timeScale>this->GetVectorNeuronState()->GetEndRefractoryPeriod(TargetIndex)){
		// Check if an spike will be fired
		double NextSpike = this->NextFiringPrediction(TargetIndex, this->GetVectorNeuronState());

		if (NextSpike != NO_SPIKE_PREDICTED){
			NextSpike += this->GetVectorNeuronState()->GetLastUpdateTime(TargetIndex);
			NextSpike*=this->inv_timeScale;
			if(NextSpike!=this->GetVectorNeuronState()->GetNextPredictedSpikeTime(TargetIndex)){

				GeneratedSpike = new InternalSpike(NextSpike, target->get_OpenMP_queue_index(), target);
			}
		}
		this->GetVectorNeuronState()->SetNextPredictedSpikeTime(TargetIndex,NextSpike);
	}

	return GeneratedSpike;
}

InternalSpike * TableBasedModel::ProcessActivityAndPredictSpike(Neuron * target, double time){
	return NULL;
}


EndRefractoryPeriodEvent * TableBasedModel::ProcessInternalSpike(InternalSpike *  OutputSpike){

	EndRefractoryPeriodEvent * endRefractoryPeriodEvent=0;

	Neuron * SourceCell = OutputSpike->GetSource();

	int SourceIndex=SourceCell->GetIndex_VectorNeuronState();

	VectorNeuronState * CurrentState = SourceCell->GetVectorNeuronState();

	this->UpdateState(SourceIndex,CurrentState,OutputSpike->GetTime()*this->timeScale);

	double EndRefractory = this->EndRefractoryPeriod(SourceIndex,CurrentState);

	if(this->EndFiringTable!=this->FiringTable){
		if(EndRefractory != NO_SPIKE_PREDICTED){
			EndRefractory += OutputSpike->GetTime()*this->timeScale; 
		}else{
			EndRefractory = (OutputSpike->GetTime()+DEF_REF_PERIOD)*this->timeScale;
		#ifdef _DEBUG
			cerr << "Warning: firing table and firing-end table discrepance (using def. ref. period)" << endl;
		#endif
		}
		endRefractoryPeriodEvent = new EndRefractoryPeriodEvent(EndRefractory, SourceCell->get_OpenMP_queue_index(), SourceCell);
	}
	CurrentState->SetEndRefractoryPeriod(SourceIndex,EndRefractory);

	return endRefractoryPeriodEvent;
}

InternalSpike * TableBasedModel::GenerateNextSpike(double time, Neuron * neuron){
	int SourceIndex=neuron->GetIndex_VectorNeuronState();

	VectorNeuronState * CurrentState = neuron->GetVectorNeuronState();

	this->UpdateState(SourceIndex,CurrentState,time*this->timeScale);

	double PredictedSpike = this->NextFiringPrediction(SourceIndex,CurrentState);

	InternalSpike * NextSpike = 0;

	if(PredictedSpike != NO_SPIKE_PREDICTED){
		PredictedSpike += CurrentState->GetEndRefractoryPeriod(SourceIndex);
		PredictedSpike*=this->inv_timeScale;
		if(PredictedSpike!=this->GetVectorNeuronState()->GetNextPredictedSpikeTime(SourceIndex)){
			NextSpike = new InternalSpike(PredictedSpike, neuron->get_OpenMP_queue_index(), neuron);
		}
	}

	CurrentState->SetNextPredictedSpikeTime(SourceIndex,PredictedSpike);

	return NextSpike;
}


bool TableBasedModel::DiscardSpike(InternalSpike *  OutputSpike){
	return (OutputSpike->GetSource()->GetVectorNeuronState()->GetNextPredictedSpikeTime(OutputSpike->GetSource()->GetIndex_VectorNeuronState())!=OutputSpike->GetTime());
}

enum NeuronModelOutputActivityType TableBasedModel::GetModelOutputActivityType(){
	return OUTPUT_SPIKE;
}

enum NeuronModelInputActivityType TableBasedModel::GetModelInputActivityType(){
	return INPUT_SPIKE;
}

ostream & TableBasedModel::PrintInfo(ostream & out) {
	out << "- Table-Based Model: " << this->GetModelID() << endl;

	for(unsigned int itab=0;itab<this->NumTables;itab++){
		out << this->Tables[itab].GetDimensionNumber() << " " << this->Tables[itab].GetInterpolation() << " (" << this->Tables[itab].GetFirstInterpolation() << ")\t";

		for(unsigned int idim=0;idim<this->Tables[itab].GetDimensionNumber();idim++){
			out << this->Tables[itab].GetDimensionAt(idim)->statevar << " " << this->Tables[itab].GetDimensionAt(idim)->interp << " (" << this->Tables[itab].GetDimensionAt(idim)->nextintdim << ")\t";
		}
	}

	out << endl;

	return out;
}


void TableBasedModel::InitializeStates(int N_neurons, int OpenMPQueueIndex){
	State->InitializeStates(N_neurons, InitValues);
}


bool TableBasedModel::CheckSynapseType(Interconnection * connection){
	int Type = connection->GetType();
	if (Type<NumSynapticVar && Type >= 0){
		NeuronModel * model = connection->GetSource()->GetNeuronModel();
		//Synapse types that process input spikes 
		if (Type < NumSynapticVar && model->GetModelOutputActivityType() == OUTPUT_SPIKE)
			return true;
		else{
			cout << "Synapses type " << Type << " of neuron model " << this->GetTypeID() << ", " << this->GetModelID() << " must receive spikes. The source model generates currents." << endl;
			return false;
		}
		//Synapse types that process input current 
	}
	else{
		cout << "Neuron model " << this->GetTypeID() << ", " << this->GetModelID() << " does not support input synapses of type " << Type << ". Just defined " << NumSynapticVar << " synapses types." << endl;
		return false;
	}
}
