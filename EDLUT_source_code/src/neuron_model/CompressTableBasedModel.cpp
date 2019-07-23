/***************************************************************************
 *                           CompressTableBasedModel.cpp                   *
 *                           -------------------                           *
 * copyright            : (C) 2015 by Francisco Naveros                    *
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

#include "../../include/neuron_model/CompressTableBasedModel.h"
#include "../../include/neuron_model/CompressNeuronModelTable.h"
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


void CompressTableBasedModel::LoadNeuronModel(string ConfigFile) throw (EDLUTFileException){
	FILE *fh;
	long Currentline = 0L;
	fh=fopen(ConfigFile.c_str(),"rt");
	if(fh){
		Currentline=1L;
		skip_comments(fh,Currentline);
		if(fscanf(fh,"%i",&this->NumStateVar)==1){
			unsigned int nv;

			// Initialize all vectors.
			this->StateVarTable = (CompressNeuronModelTable **) new CompressNeuronModelTable * [this->NumStateVar];
			this->StateVarOrderOriginalIndex = (unsigned int *) new unsigned int [this->NumStateVar];

			// Auxiliary table index
			this->TablesIndex = (unsigned int *) new unsigned int [this->NumStateVar];

			skip_comments(fh,Currentline);
			for(nv=0;nv<this->NumStateVar;nv++){
				if(fscanf(fh,"%i",&TablesIndex[nv])!=1){
					throw EDLUTFileException(TASK_TABLE_BASED_MODEL_LOAD, ERROR_TABLE_BASED_MODEL_INDEX, REPAIR_NEURON_MODEL_TABLE_TABLES_STRUCTURE, Currentline, ConfigFile.c_str(), true);
				}
			}

			skip_comments(fh,Currentline);

			float InitValue;
			InitValues=new float[NumStateVar+1]();

			// Create a new initial state
       		this->State = (VectorNeuronState *) new VectorNeuronState(this->NumStateVar+1, false);

       		for(nv=0;nv<this->NumStateVar;nv++){
       			if(fscanf(fh,"%f",&InitValue)!=1){
					throw EDLUTFileException(TASK_TABLE_BASED_MODEL_LOAD, ERROR_TABLE_BASED_MODEL_INITIAL_VALUES, REPAIR_NEURON_MODEL_TABLE_TABLES_STRUCTURE, Currentline, ConfigFile.c_str(), true);
       			} else {
					InitValues[nv+1]=InitValue;
       			}
       		}

			// Allocate temporal state vars

   			skip_comments(fh,Currentline);

   			if(fscanf(fh,"%i",&this->FiringIndex)==1){
   				skip_comments(fh,Currentline);
   				if(fscanf(fh,"%i",&this->EndFiringIndex)==1){
   					skip_comments(fh,Currentline);
   					if(fscanf(fh,"%i",&this->NumSynapticVar)==1){
               			skip_comments(fh,Currentline);

               			this->SynapticVar = (unsigned int *) new unsigned int [this->NumSynapticVar];
               			for(nv=0;nv<this->NumSynapticVar;nv++){
                  			if(fscanf(fh,"%i",&this->SynapticVar[nv])!=1){
								throw EDLUTFileException(TASK_TABLE_BASED_MODEL_LOAD, ERROR_TABLE_BASED_MODEL_SYNAPSE_INDEXS, REPAIR_NEURON_MODEL_TABLE_TABLES_STRUCTURE, Currentline, ConfigFile.c_str(), true);
                  			}
                  		}

              			skip_comments(fh,Currentline);
              			if(fscanf(fh,"%i",&this->NumTables)==1){
              				unsigned int nt;
              				int tdeptables[MAXSTATEVARS];
              				int tstatevarpos,ntstatevarpos;

              				this->Tables = (CompressNeuronModelTable *) new CompressNeuronModelTable [this->NumTables];

              				// Update table links
              				for(nv=0;nv<this->NumStateVar;nv++){
								this->StateVarTable[nv] = this->Tables+TablesIndex[nv];
								this->StateVarTable[nv]->SetOutputStateVariableIndex(TablesIndex[nv]);
							}
              				this->FiringTable = this->Tables+FiringIndex;
              				this->EndFiringTable = this->Tables+EndFiringIndex;

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
            					this->StateVarOrderOriginalIndex[(tdeptables[nt])?tstatevarpos++:ntstatevarpos++]=nt;
         					}

							//Load the neuron model time scale.
							skip_comments(fh,Currentline);
							char scale[MAXIDSIZE+1];
							if (fscanf(fh, " %"MAXIDSIZEC"[^ \n]", scale) == 1){
								if (strncmp(scale, "Milisecond", 10) == 0){
									//Milisecond scale								
									this->SetTimeScale(1000);
								}
								else if (strncmp(scale, "Second", 6) == 0){
									//Second scale
									this->SetTimeScale(1);
								}
								else{
									throw EDLUTFileException(TASK_TABLE_BASED_MODEL_LOAD, ERROR_TABLE_BASED_MODEL_TIME_SCALE, REPAIR_TABLE_BASED_MODEL_TIME_SCALE, Currentline, ConfigFile.c_str(), true);
								}
							}
							else{
								throw EDLUTFileException(TASK_TABLE_BASED_MODEL_LOAD, ERROR_TABLE_BASED_MODEL_TIME_SCALE, REPAIR_TABLE_BASED_MODEL_TIME_SCALE, Currentline, ConfigFile.c_str(), true);
							}

              			}else{
							throw EDLUTFileException(TASK_TABLE_BASED_MODEL_LOAD, ERROR_TABLE_BASED_MODEL_NUMBER_OF_TABLES, REPAIR_NEURON_MODEL_TABLE_TABLES_STRUCTURE, Currentline, ConfigFile.c_str(), true);
      					}
      				}else{
						throw EDLUTFileException(TASK_TABLE_BASED_MODEL_LOAD, ERROR_TABLE_BASED_MODEL_NUMBER_OF_SYNAPSES, REPAIR_NEURON_MODEL_TABLE_TABLES_STRUCTURE, Currentline, ConfigFile.c_str(), true);
          			}
				}else{
					throw EDLUTFileException(TASK_TABLE_BASED_MODEL_LOAD, ERROR_TABLE_BASED_MODEL_END_FIRING_INDEX, REPAIR_NEURON_MODEL_TABLE_TABLES_STRUCTURE, Currentline, ConfigFile.c_str(), true);
          		}
			}else{
				throw EDLUTFileException(TASK_TABLE_BASED_MODEL_LOAD, ERROR_TABLE_BASED_MODEL_FIRING_INDEX, REPAIR_NEURON_MODEL_TABLE_TABLES_STRUCTURE, Currentline, ConfigFile.c_str(), true);
			}
		}else{
		throw EDLUTFileException(TASK_TABLE_BASED_MODEL_LOAD, ERROR_TABLE_BASED_MODEL_NUMBER_OF_STATE_VARIABLES, REPAIR_NEURON_MODEL_TABLE_TABLES_STRUCTURE, Currentline, ConfigFile.c_str(), true);
		}
	}else{
	throw EDLUTFileException(TASK_TABLE_BASED_MODEL_LOAD, ERROR_TABLE_BASED_MODEL_OPEN, REPAIR_NEURON_MODEL_TABLE_TABLES_STRUCTURE, Currentline, ConfigFile.c_str(), true);
	}
	fclose(fh);

}

void CompressTableBasedModel::LoadTables(string TableFile) throw (EDLUTException){
	FILE *fd;
	unsigned int i;
	CompressNeuronModelTable * tab;
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

CompressTableBasedModel::CompressTableBasedModel(string NeuronTypeID, string NeuronModelID): EventDrivenNeuronModel(NeuronTypeID, NeuronModelID),
		NumStateVar(0), NumTimeDependentStateVar(0), NumSynapticVar(0), SynapticVar(0),
		StateVarOrderOriginalIndex(0),StateVarOrderIndex(0), StateVarOrderSubIndex(0), TablesIndex(0), StateVarTable(0), FiringTable(0),FiringIndex(0), FiringSubIndex(0), 
		EndFiringTable(0),EndFiringIndex(0), EndFiringSubIndex(0), NumTables(0), NumCompresedTables(0), Tables(0), NumVariablesPerCompressedTable(0) {
}

CompressTableBasedModel::~CompressTableBasedModel() {
	
	
	if (this->StateVarOrderOriginalIndex!=0) {
		delete [] this->StateVarOrderOriginalIndex;
	}

	if (this->StateVarOrderIndex!=0) {
		delete [] this->StateVarOrderIndex;
	}

	if (this->StateVarOrderSubIndex!=0) {
		delete [] this->StateVarOrderSubIndex;
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

	if(this->TablesIndex!=0){
		delete [] TablesIndex;
	}

	if(this->NumVariablesPerCompressedTable!=0){
		delete [] NumVariablesPerCompressedTable;
	}
}

void CompressTableBasedModel::LoadNeuronModel() throw (EDLUTFileException){

	this->LoadNeuronModel(this->GetModelID()+".cfg");

	this->LoadTables(this->GetModelID()+".dat");

	this->CompactTables();
}

void CompressTableBasedModel::CompactTables(){
	int * compactTableIndex=new int[this->NumTables]();
	int * compactTableSubIndex=new int[this->NumTables]();
	CompressNeuronModelTable * compactTables;

	int index=0;
	int subindex=0;
	int compactNumTables=1;
	compactTableIndex[0]=index;
	compactTableSubIndex[0]=subindex;



	for(int i=1; i<this->NumTables; i++){
		bool equal=CompareNeuronModelTableIndex(Tables+i-1, Tables+i);
		if(equal){
			subindex++;
		}else{
			index++;
			subindex=0;
			compactNumTables++;
		}
		compactTableIndex[i]=index;
		compactTableSubIndex[i]=subindex;
	}

	compactTables = (CompressNeuronModelTable *) new CompressNeuronModelTable [compactNumTables];

	int N_tables=1;
	int offset=0;
	int i;
	for(i=1; i<this->NumTables; i++){
		if(compactTableSubIndex[i]!=0){
			N_tables++;
		}else{
			(compactTables+offset)->CopyTables(Tables+i-N_tables,N_tables);
			offset++;
			N_tables=1;
		}
	}

	(compactTables+offset)->CopyTables(Tables+i-N_tables,N_tables);


	//Compute how many tables are in each compressedTable.
	NumVariablesPerCompressedTable=new int[compactNumTables]();
	NumVariablesPerCompressedTable[0]=1;
	int r,s=0;

	for(r=1; r<this->NumTables; r++){
		if(compactTableIndex[r-1]!=compactTableIndex[r]){
			s++;
		}
		NumVariablesPerCompressedTable[s]++;
	}



	//delete de original tables.
	delete [] this->Tables;
	this->Tables=compactTables;


	this->NumCompresedTables=compactNumTables;

	//Calculate new index
	CompressNeuronModelTable ** AuxStateVarTable = (CompressNeuronModelTable **) new CompressNeuronModelTable * [this->NumStateVar];
	unsigned int * AuxStateVarOrderIndex = (unsigned int *) new unsigned int [this->NumStateVar];
	unsigned int * AuxStateVarOrderSubIndex = (unsigned int *) new unsigned int [this->NumStateVar];
	for(int i=0; i<this->NumStateVar; i++){
		AuxStateVarTable[i]=compactTables+compactTableIndex[TablesIndex[i]];
		AuxStateVarOrderIndex[i]=compactTableIndex[StateVarOrderOriginalIndex[i]];
		AuxStateVarOrderSubIndex[i]=compactTableSubIndex[StateVarOrderOriginalIndex[i]];
	}

	delete StateVarTable;
	
	StateVarTable=AuxStateVarTable;
	StateVarOrderIndex=AuxStateVarOrderIndex;
	StateVarOrderSubIndex=AuxStateVarOrderSubIndex ;


	this->FiringSubIndex=compactTableSubIndex[FiringIndex];
	this->FiringIndex=compactTableIndex[FiringIndex];
	this->FiringTable = this->Tables+FiringIndex;
    
	this->EndFiringSubIndex=compactTableSubIndex[EndFiringIndex];
	this->EndFiringIndex=compactTableIndex[EndFiringIndex];
	this->EndFiringTable = this->Tables+EndFiringIndex;
}
bool CompressTableBasedModel::CompareNeuronModelTableIndex(CompressNeuronModelTable * table1, CompressNeuronModelTable * table2){
	bool different;
	different=(table1->GetElementsNumber()!=table2->GetElementsNumber() || table1->GetDimensionNumber()!=table2->GetDimensionNumber());

	if(different){
		return false;
	}

	for(unsigned long idim=0;idim < table1->GetDimensionNumber();idim++){
		different=(different || table1->GetDimensionAt(idim)->size!=table2->GetDimensionAt(idim)->size || 
					table1->GetDimensionAt(idim)->statevar!=table2->GetDimensionAt(idim)->statevar || 
					table1->GetDimensionAt(idim)->interp!=table2->GetDimensionAt(idim)->interp);
		if(different){
			return false;
		}
		for(unsigned long coord=0; coord<table1->GetDimensionAt(idim)->size; coord++){
			different=(different || table1->GetDimensionAt(idim)->coord[coord]!=table2->GetDimensionAt(idim)->coord[coord]);
		} 
		if(different){
			return false;
		}
	}
	return true;
}

VectorNeuronState * CompressTableBasedModel::InitializeState(){
	return State;
}

InternalSpike * CompressTableBasedModel::GenerateInitialActivity(Neuron *  Cell){
	double Predicted = this->NextFiringPrediction(Cell->GetIndex_VectorNeuronState(),Cell->GetVectorNeuronState());

	InternalSpike * spike = 0;

	if(Predicted != NO_SPIKE_PREDICTED){
		Predicted += Cell->GetVectorNeuronState()->GetLastUpdateTime(Cell->GetIndex_VectorNeuronState());
		Predicted*=this->inv_timeScale;//REVISAR
		spike = new InternalSpike(Predicted, Cell->get_OpenMP_queue_index(),Cell);
	}

	this->GetVectorNeuronState()->SetNextPredictedSpikeTime(Cell->GetIndex_VectorNeuronState(),Predicted);

	return spike;
}



void CompressTableBasedModel::UpdateState(int index, VectorNeuronState * State, double CurrentTime){

	unsigned int ivar1,orderedvar1;
	unsigned int ivar2,orderedvar2;
	unsigned int ivar3,orderedvar3;

	float TempStateVars[MAX_VARIABLES];
	unsigned int AuxIndex[MAX_VARIABLES];
	float OutputValues[MAX_VARIABLES];

	State->SetStateVariableAt(index,0,CurrentTime-State->GetLastUpdateTime(index));

	ivar1=0;
	while(ivar1<this->NumTimeDependentStateVar){
		int N_variables=NumVariablesPerCompressedTable[StateVarOrderIndex[ivar1]];
		if(N_variables==1){
			orderedvar1=this->StateVarOrderSubIndex[ivar1];
			TempStateVars[this->StateVarOrderOriginalIndex[ivar1]]=this->StateVarTable[this->StateVarOrderOriginalIndex[ivar1]]->TableAccess(index,State);
		}else{
			for(int i=0; i<N_variables;i++){
				AuxIndex[i]=this->StateVarOrderSubIndex[ivar1+i];
				OutputValues[i]=0;
			}
			this->StateVarTable[this->StateVarOrderOriginalIndex[ivar1]]->TableAccess(index,State,AuxIndex,OutputValues,N_variables);

			for(int i=0; i<N_variables;i++){
				TempStateVars[this->StateVarOrderOriginalIndex[ivar1+i]]=OutputValues[i];
			}
		}
		ivar1+=N_variables;
	}

	for(ivar2=0;ivar2<this->NumTimeDependentStateVar;ivar2++){
		orderedvar2=this->StateVarOrderOriginalIndex[ivar2];
		State->SetStateVariableAt(index,orderedvar2+1,TempStateVars[orderedvar2]);
	}


	ivar3=this->NumTimeDependentStateVar;
	while(ivar3<this->NumStateVar){
		int N_variables=NumVariablesPerCompressedTable[StateVarOrderIndex[ivar3]];
		for(int i=0; i<N_variables;i++){
			AuxIndex[i]=this->StateVarOrderSubIndex[ivar3+i];
		}

		this->StateVarTable[AuxIndex[0]]->TableAccess(index,State,AuxIndex,OutputValues,N_variables);
		
		for(int i=0; i<N_variables;i++){
			State->SetStateVariableAt(index,AuxIndex[i]+1,OutputValues[i]);
		}
		ivar3+=N_variables;
	}

	State->SetLastUpdateTime(index,CurrentTime);
}


void CompressTableBasedModel::SynapsisEffect(int index, Interconnection * InputConnection){
	this->GetVectorNeuronState()->IncrementStateVariableAtCPU(index, this->SynapticVar[InputConnection->GetType()]+1, InputConnection->GetWeight()*WEIGHTSCALE);
}

double CompressTableBasedModel::NextFiringPrediction(int index, VectorNeuronState * State){
	float resultado;
	if(this->FiringSubIndex!=this->EndFiringSubIndex){
		this->FiringTable->CompressTableAccessDirect(index, State,&this->FiringSubIndex,&resultado,1);
	}else{
		resultado=this->FiringTable->TableAccessDirect(index, State);
	}
	return resultado;
}

double CompressTableBasedModel::EndRefractoryPeriod(int index, VectorNeuronState * State){
	float resultado = NO_SPIKE_PREDICTED;
	if(this->FiringSubIndex!=this->EndFiringSubIndex){
		this->EndFiringTable->CompressTableAccessDirect(index, State, &this->EndFiringSubIndex,&resultado,1);
	}
	return resultado;
}



InternalSpike * CompressTableBasedModel::ProcessInputSpike(Interconnection * inter, double time){

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
				GeneratedSpike = new InternalSpike(NextSpike, target->get_OpenMP_queue_index(),target);
			}
		}

		this->GetVectorNeuronState()->SetNextPredictedSpikeTime(TargetIndex,NextSpike);
	}

	return GeneratedSpike;
}


InternalSpike * CompressTableBasedModel::ProcessActivityAndPredictSpike(Neuron * target, double time){
	return NULL;
}


EndRefractoryPeriodEvent * CompressTableBasedModel::ProcessInternalSpike(InternalSpike *  OutputSpike){

	EndRefractoryPeriodEvent * endRefractoryPeriodEvent=0;

	Neuron * SourceCell = OutputSpike->GetSource();

	int SourceIndex=SourceCell->GetIndex_VectorNeuronState();

	VectorNeuronState * CurrentState = SourceCell->GetVectorNeuronState();

	this->UpdateState(SourceIndex,CurrentState,OutputSpike->GetTime()*this->timeScale);

	double EndRefractory = this->EndRefractoryPeriod(SourceIndex,CurrentState);

	if(this->FiringSubIndex!=this->EndFiringSubIndex){
		if(EndRefractory != NO_SPIKE_PREDICTED){
			EndRefractory += OutputSpike->GetTime()*this->timeScale; 
		}else{
			EndRefractory = (OutputSpike->GetTime()+DEF_REF_PERIOD)*this->timeScale;
		#ifdef _DEBUG
			cerr << "Warning: firing table and firing-end table discrepance (using def. ref. period)" << endl;
		#endif
		}
		endRefractoryPeriodEvent = new EndRefractoryPeriodEvent(EndRefractory, SourceCell->get_OpenMP_queue_index(),SourceCell);
	}
	CurrentState->SetEndRefractoryPeriod(SourceIndex,EndRefractory);

	return endRefractoryPeriodEvent;
}

InternalSpike * CompressTableBasedModel::GenerateNextSpike(double time, Neuron * neuron){
	int SourceIndex=neuron->GetIndex_VectorNeuronState();

	VectorNeuronState * CurrentState = neuron->GetVectorNeuronState();

	this->UpdateState(SourceIndex,CurrentState,time*this->timeScale);

	double PredictedSpike = this->NextFiringPrediction(SourceIndex,CurrentState);

	InternalSpike * NextSpike = 0;

	if(PredictedSpike != NO_SPIKE_PREDICTED){
		PredictedSpike += CurrentState->GetEndRefractoryPeriod(SourceIndex);
		PredictedSpike*=this->inv_timeScale;
		if(PredictedSpike!=this->GetVectorNeuronState()->GetNextPredictedSpikeTime(SourceIndex)){
			NextSpike = new InternalSpike(PredictedSpike, neuron->get_OpenMP_queue_index(),neuron);
		}
	}

	CurrentState->SetNextPredictedSpikeTime(SourceIndex,PredictedSpike);

	return NextSpike;
}


bool CompressTableBasedModel::DiscardSpike(InternalSpike *  OutputSpike){
	return (OutputSpike->GetSource()->GetVectorNeuronState()->GetNextPredictedSpikeTime(OutputSpike->GetSource()->GetIndex_VectorNeuronState())!=OutputSpike->GetTime());
}

enum NeuronModelOutputActivityType CompressTableBasedModel::GetModelOutputActivityType(){
	return OUTPUT_SPIKE;
}

enum NeuronModelInputActivityType CompressTableBasedModel::GetModelInputActivityType(){
	return INPUT_SPIKE;
}

ostream & CompressTableBasedModel::PrintInfo(ostream & out) {
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


void CompressTableBasedModel::InitializeStates(int N_neurons, int OpenMPQueueIndex){
	State->InitializeStates(N_neurons, InitValues);
}


bool CompressTableBasedModel::CheckSynapseType(Interconnection * connection){
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
