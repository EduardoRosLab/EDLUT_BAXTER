/***************************************************************************
 *                           LIFTimeDrivenModel_1_4.cpp                    *
 *                           -------------------                           *
 * copyright            : (C) 2013 by Jesus Garrido and Francisco Naveros  *
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

#include "../../include/neuron_model/LIFTimeDrivenModel_1_4.h"
#include "../../include/neuron_model/VectorNeuronState.h"

#include <iostream>
#include <cmath>
#include <string>

#include "../../include/openmp/openmp.h"


#include "../../include/spike/EDLUTFileException.h"
#include "../../include/spike/Neuron.h"
#include "../../include/spike/InternalSpike.h"
#include "../../include/spike/PropagatedSpike.h"
#include "../../include/spike/Interconnection.h"

#include "../../include/simulation/Utils.h"





void LIFTimeDrivenModel_1_4::LoadNeuronModel(string ConfigFile) throw (EDLUTFileException){
	FILE *fh;
	long Currentline = 0L;
	fh=fopen(ConfigFile.c_str(),"rt");
	if(fh){
		Currentline=1L;
		skip_comments(fh,Currentline);
		if(fscanf(fh,"%f",&this->eexc)==1){
			skip_comments(fh,Currentline);

			if (fscanf(fh,"%f",&this->einh)==1){
				skip_comments(fh,Currentline);

				if(fscanf(fh,"%f",&this->erest)==1){
					skip_comments(fh,Currentline);

					if(fscanf(fh,"%f",&this->vthr)==1){
						skip_comments(fh,Currentline);

						if (fscanf(fh, "%f", &this->cm) == 1 && this->cm > 0.0f){
							inv_cm = 1.0f / cm;
							skip_comments(fh,Currentline);

							if (fscanf(fh, "%f", &this->tampa) == 1 && this->tampa > 0.0f){
								inv_tampa=1.0f/tampa;
								skip_comments(fh,Currentline);

								if (fscanf(fh, "%f", &this->tnmda) == 1 && this->tnmda > 0.0f){
									inv_tnmda=1.0f/tnmda;
									skip_comments(fh,Currentline);
									
									if (fscanf(fh, "%f", &this->tinh) == 1 && this->tinh > 0.0f){
										inv_tinh=1.0f/tinh;
										skip_comments(fh,Currentline);

										if (fscanf(fh, "%f", &this->tgj) == 1 && this->tgj > 0.0f){
											inv_tgj=1.0f/tgj;
											skip_comments(fh,Currentline);

											if (fscanf(fh, "%f", &this->tref) == 1 && this->tref > 0.0f){
												skip_comments(fh,Currentline);

												if (fscanf(fh, "%f", &this->grest) == 1 && this->grest > 0.0f){
													skip_comments(fh,Currentline);

													if(fscanf(fh,"%f",&this->fgj)==1){
														skip_comments(fh,Currentline);

														this->State = (VectorNeuronState *) new VectorNeuronState(N_NeuronStateVariables, true);
													} else {
														throw EDLUTFileException(TASK_LIF_TIME_DRIVEN_MODEL_1_4_LOAD, ERROR_LIF_TIME_DRIVEN_MODEL_1_4_FGJ, REPAIR_NEURON_MODEL_VALUES, Currentline, ConfigFile.c_str(), true);
													}
												} else {
													throw EDLUTFileException(TASK_LIF_TIME_DRIVEN_MODEL_1_4_LOAD, ERROR_LIF_TIME_DRIVEN_MODEL_1_4_GREST, REPAIR_NEURON_MODEL_VALUES, Currentline, ConfigFile.c_str(), true);
												}
											} else {
												throw EDLUTFileException(TASK_LIF_TIME_DRIVEN_MODEL_1_4_LOAD, ERROR_LIF_TIME_DRIVEN_MODEL_1_4_TREF, REPAIR_NEURON_MODEL_VALUES, Currentline, ConfigFile.c_str(), true);
											}
										} else {
											throw EDLUTFileException(TASK_LIF_TIME_DRIVEN_MODEL_1_4_LOAD, ERROR_LIF_TIME_DRIVEN_MODEL_1_4_GAP, REPAIR_NEURON_MODEL_VALUES, Currentline, ConfigFile.c_str(), true);
										}
									} else {
										throw EDLUTFileException(TASK_LIF_TIME_DRIVEN_MODEL_1_4_LOAD, ERROR_LIF_TIME_DRIVEN_MODEL_1_4_TINH, REPAIR_NEURON_MODEL_VALUES, Currentline, ConfigFile.c_str(), true);
									}
								} else {
									throw EDLUTFileException(TASK_LIF_TIME_DRIVEN_MODEL_1_4_LOAD, ERROR_LIF_TIME_DRIVEN_MODEL_1_4_TNMDA, REPAIR_NEURON_MODEL_VALUES, Currentline, ConfigFile.c_str(), true);
								}
							} else {
								throw EDLUTFileException(TASK_LIF_TIME_DRIVEN_MODEL_1_4_LOAD, ERROR_LIF_TIME_DRIVEN_MODEL_1_4_TAMPA, REPAIR_NEURON_MODEL_VALUES, Currentline, ConfigFile.c_str(), true);
							}
						} else {
							throw EDLUTFileException(TASK_LIF_TIME_DRIVEN_MODEL_1_4_LOAD, ERROR_LIF_TIME_DRIVEN_MODEL_1_4_CM, REPAIR_NEURON_MODEL_VALUES, Currentline, ConfigFile.c_str(), true);
						}
					} else {
						throw EDLUTFileException(TASK_LIF_TIME_DRIVEN_MODEL_1_4_LOAD, ERROR_LIF_TIME_DRIVEN_MODEL_1_4_VTHR, REPAIR_NEURON_MODEL_VALUES, Currentline, ConfigFile.c_str(), true);
					}
				} else {
					throw EDLUTFileException(TASK_LIF_TIME_DRIVEN_MODEL_1_4_LOAD, ERROR_LIF_TIME_DRIVEN_MODEL_1_4_EREST, REPAIR_NEURON_MODEL_VALUES, Currentline, ConfigFile.c_str(), true);
				}
			} else {
				throw EDLUTFileException(TASK_LIF_TIME_DRIVEN_MODEL_1_4_LOAD, ERROR_LIF_TIME_DRIVEN_MODEL_1_4_EINH, REPAIR_NEURON_MODEL_VALUES, Currentline, ConfigFile.c_str(), true);
			}
		} else {
			throw EDLUTFileException(TASK_LIF_TIME_DRIVEN_MODEL_1_4_LOAD, ERROR_LIF_TIME_DRIVEN_MODEL_1_4_EEXC, REPAIR_NEURON_MODEL_VALUES, Currentline, ConfigFile.c_str(), true);
		}

		// gnmdainf table
		this->Max_V = this->eexc;
		this->Min_V = this->einh;
		this->TableSize = (this->Max_V - this->Min_V) * 100000 + 1; //0.00001 s resultion
		this->aux = (this->TableSize - 1) / (this->Max_V - this->Min_V);
		this->channel_values = Generate_channel_values();


		//INTEGRATION METHOD
		loadIntegrationMethod(this->GetModelID(), fh, &Currentline);

		this->integrationMethod->SetBiFixedStepParameters((erest+vthr)/2,(2*erest+vthr)/3,0);

		this->integrationMethod->Calculate_conductance_exp_values();

		//SET TIME-DRIVEN STEP SIZE
		this->SetTimeDrivenStepSize(this->integrationMethod->elapsedTimeInSeconds);
	}else{
		throw EDLUTFileException(TASK_LIF_TIME_DRIVEN_MODEL_1_4_LOAD, ERROR_NEURON_MODEL_OPEN, REPAIR_NEURON_MODEL_NAME, Currentline, ConfigFile.c_str(), true);
	}
	fclose(fh);
}

//this neuron model is implemented in a second scale.
LIFTimeDrivenModel_1_4::LIFTimeDrivenModel_1_4(string NeuronTypeID, string NeuronModelID): TimeDrivenNeuronModel(NeuronTypeID, NeuronModelID, SecondScale), eexc(0), einh(0), erest(0), vthr(0), cm(0), tampa(0), tnmda(0), tinh(0), tgj(0),
		tref(0), grest(0){
}

LIFTimeDrivenModel_1_4::~LIFTimeDrivenModel_1_4(void)
{
	if(this->channel_values){
		delete this->channel_values;
		this->channel_values=0;
	}
}

void LIFTimeDrivenModel_1_4::LoadNeuronModel() throw (EDLUTFileException){
	this->LoadNeuronModel(this->GetModelID()+".cfg");
}

VectorNeuronState * LIFTimeDrivenModel_1_4::InitializeState(){
	return this->GetVectorNeuronState();
}


InternalSpike * LIFTimeDrivenModel_1_4::ProcessInputSpike(Interconnection * inter, double time){
	// Add the effect of the input spike
	this->GetVectorNeuronState()->IncrementStateVariableAtCPU(inter->GetTargetNeuronModelIndex(), N_DifferentialNeuronState + inter->GetType(), inter->GetWeight());

	return 0;
}


bool LIFTimeDrivenModel_1_4::UpdateState(int index, double CurrentTime){
	//NeuronState[0] --> vm 
	//NeuronState[1] --> gampa 
	//NeuronState[2] --> gnmda 
	//NeuronState[3] --> ginh 
	//NeuronState[4] --> ggj 

	//Reset the number of internal spikes in this update period
	this->State->NInternalSpikeIndexs = 0;


	this->integrationMethod->NextDifferentialEquationValues();

	this->CheckValidIntegration(CurrentTime, this->integrationMethod->GetValidIntegrationVariable());

	return false;
}


enum NeuronModelOutputActivityType LIFTimeDrivenModel_1_4::GetModelOutputActivityType(){
	return OUTPUT_SPIKE;
}

enum NeuronModelInputActivityType LIFTimeDrivenModel_1_4::GetModelInputActivityType(){
	return INPUT_SPIKE;
}


ostream & LIFTimeDrivenModel_1_4::PrintInfo(ostream & out){
	out << "- Leaky Time-Driven Model 1_4: " << this->GetModelID() << endl;

	out << "\tExc. Reversal Potential: " << this->eexc << "V\tInh. Reversal Potential: " << this->einh << "V\tResting potential: " << this->erest << "V" << endl;

	out << "\tFiring threshold: " << this->vthr << "V\tMembrane capacitance: " << this->cm << "nS\tAMPA Time Constant: " << this->tampa << "sNMDA Time Constant: " << this->tnmda << "s" << endl;

	out << "\tInhibitory time constant: " << this->tinh << "s\tGap junction time constant: " << this->tgj << "s\tRefractory Period: " << this->tref << "s\tResting Conductance: " << this->grest << "nS" << endl;

	return out;
}	



void LIFTimeDrivenModel_1_4::InitializeStates(int N_neurons, int OpenMPQueueIndex){
	//Initialize neural state variables.
	float initialization[] = {erest,0.0f,0.0f,0.0f,0.0f};
	State->InitializeStates(N_neurons, initialization);

	//Initialize integration method state variables.
	this->integrationMethod->InitializeStates(N_neurons, initialization);
}


void LIFTimeDrivenModel_1_4::EvaluateSpikeCondition(float previous_V, float * NeuronState, int index, float elapsedTimeInNeuronModelScale){
	float vm_cou = NeuronState[0] + this->fgj * NeuronState[4];
	if (vm_cou > this->vthr  && this->GetVectorNeuronState()->GetLastSpikeTime(index)>this->tref){
		NeuronState[0] = this->erest;
		State->NewFiredSpike(index);
		this->integrationMethod->resetState(index);
		this->State->InternalSpikeIndexs[this->State->NInternalSpikeIndexs] = index;
		this->State->NInternalSpikeIndexs++;
	}
}


void LIFTimeDrivenModel_1_4::EvaluateDifferentialEquation(float * NeuronState, float * AuxNeuronState, int index, float elapsed_time){
	if (this->GetVectorNeuronState()->GetLastSpikeTime(index)>this->tref){
		//float gnmdainf = 1.0f/(1.0f + ExponentialTable::GetResult(-62.0f*NeuronState[0])*(1.2f/3.57f));
		float gnmdainf = Get_channel_values(NeuronState[0]);
		float iampa = NeuronState[1] * (this->eexc - NeuronState[0]);
		float iinh = NeuronState[3] * (this->einh - NeuronState[0]);
		float inmda = NeuronState[2] * gnmdainf*(this->eexc - NeuronState[0]);

		AuxNeuronState[0] = (iampa + inmda + iinh + this->grest* (this->erest - NeuronState[0]))*1.e-9f * this->inv_cm;
	}
	else if ((this->GetVectorNeuronState()->GetLastSpikeTime(index) + elapsed_time)>this->tref){
		//float gnmdainf = 1.0f/(1.0f + ExponentialTable::GetResult(-62.0f*NeuronState[0])*(1.2f/3.57f));
		float gnmdainf = Get_channel_values(NeuronState[0]);
		float fraction = (this->GetVectorNeuronState()->GetLastSpikeTime(index) + elapsed_time - this->tref) / elapsed_time;
		float iampa = NeuronState[1] * (this->eexc - NeuronState[0]);
		float iinh = NeuronState[3] * (this->einh - NeuronState[0]);
		float inmda = NeuronState[2] * gnmdainf*(this->eexc - NeuronState[0]);
		AuxNeuronState[0] = fraction*(iampa + inmda + iinh + this->grest* (this->erest - NeuronState[0]))*1.e-9f * this->inv_cm;
	}
	else{
		AuxNeuronState[0] = 0;
	}
}

void LIFTimeDrivenModel_1_4::EvaluateTimeDependentEquation(float * NeuronState, int index, int elapsed_time_index){
	float limit=1e-9;
	float * Conductance_values=this->Get_conductance_exponential_values(elapsed_time_index);
	
	if(NeuronState[N_DifferentialNeuronState]<limit){
		NeuronState[N_DifferentialNeuronState]=0.0f;
	}else{
		NeuronState[N_DifferentialNeuronState]*= Conductance_values[0];
	}
	if(NeuronState[N_DifferentialNeuronState+1]<limit){
		NeuronState[N_DifferentialNeuronState+1]=0.0f;
	}else{
		NeuronState[N_DifferentialNeuronState+1]*= Conductance_values[1];
	}
	if(NeuronState[N_DifferentialNeuronState+2]<limit){
		NeuronState[N_DifferentialNeuronState+2]=0.0f;
	}else{
		NeuronState[N_DifferentialNeuronState+2]*= Conductance_values[2];
	}
	if(NeuronState[N_DifferentialNeuronState+3]<limit){
		NeuronState[N_DifferentialNeuronState+3]=0.0f;
	}else{
		NeuronState[N_DifferentialNeuronState+3]*= Conductance_values[3];
	}

}

void LIFTimeDrivenModel_1_4::Calculate_conductance_exp_values(int index, float elapsed_time){
	//ampa synapse.
	Set_conductance_exp_values(index, 0, exp(-elapsed_time*this->inv_tampa));
	//nmda synapse.
	Set_conductance_exp_values(index, 1, exp(-elapsed_time*this->inv_tnmda));
	//inhibitory synapse.
	Set_conductance_exp_values(index, 2, exp(-elapsed_time*this->inv_tinh));
	//gap junction synapse.
	Set_conductance_exp_values(index, 3, exp(-elapsed_time*this->inv_tgj));
}


bool LIFTimeDrivenModel_1_4::CheckSynapseType(Interconnection * connection){
	int Type = connection->GetType();
	if (Type<N_TimeDependentNeuronState && Type >= 0){
		NeuronModel * model = connection->GetSource()->GetNeuronModel();
		//Synapse types that process input spikes 
		if (Type < N_TimeDependentNeuronState && model->GetModelOutputActivityType() == OUTPUT_SPIKE)
			return true;
		else{
			cout << "Synapses type " << Type << " of neuron model " << this->GetTypeID() << ", " << this->GetModelID() << " must receive spikes. The source model generates currents." << endl;
			return false;
		}
		//Synapse types that process input current 
	}
	else{
		cout << "Neuron model " << this->GetTypeID() << ", " << this->GetModelID() << " does not support input synapses of type " << Type << ". Just defined " << N_TimeDependentNeuronState << " synapses types." << endl;
		return false;
	}
}



void LIFTimeDrivenModel_1_4::loadIntegrationMethod(string fileName, FILE *fh, long * Currentline)throw (EDLUTFileException){
	char ident_type[MAXIDSIZE + 1];

	//We load the integration method type.
	skip_comments(fh, *Currentline);
	if (fscanf(fh, "%s", ident_type) == 1){
		skip_comments(fh, *Currentline);
		//DEFINE HERE NEW INTEGRATION METHOD
		if (strncmp(ident_type, "Euler", 5) == 0){
			integrationMethod = (Euler<LIFTimeDrivenModel_1_4> *) new Euler<LIFTimeDrivenModel_1_4>(this);
		}
		else if (strncmp(ident_type, "RK2", 3) == 0){
			integrationMethod = (RK2<LIFTimeDrivenModel_1_4> *) new RK2<LIFTimeDrivenModel_1_4>(this);
		}
		else if (strncmp(ident_type, "RK4", 3) == 0){
			integrationMethod = (RK4<LIFTimeDrivenModel_1_4> *) new RK4<LIFTimeDrivenModel_1_4>(this);
		}
		else if (strncmp(ident_type, "BDF", 3) == 0 && atoi(&ident_type[3])>0 && atoi(&ident_type[3])<7){
			integrationMethod = (BDFn<LIFTimeDrivenModel_1_4> *) new BDFn<LIFTimeDrivenModel_1_4>(this, atoi(&ident_type[3]));
		}
		else if (strncmp(ident_type, "Bifixed_Euler", 13) == 0){
			integrationMethod = (Bifixed_Euler<LIFTimeDrivenModel_1_4> *) new Bifixed_Euler<LIFTimeDrivenModel_1_4>(this);
		}
		else if (strncmp(ident_type, "Bifixed_RK2", 11) == 0){
			integrationMethod = (Bifixed_RK2<LIFTimeDrivenModel_1_4> *) new Bifixed_RK2<LIFTimeDrivenModel_1_4>(this);
		}
		else if (strncmp(ident_type, "Bifixed_RK4", 11) == 0){
			integrationMethod = (Bifixed_RK4<LIFTimeDrivenModel_1_4> *) new Bifixed_RK4<LIFTimeDrivenModel_1_4>(this);
		}
		else if (strncmp(ident_type, "Bifixed_BDF", 11) == 0 && atoi(&ident_type[11]) == 2){
			integrationMethod = (Bifixed_BDFn<LIFTimeDrivenModel_1_4> *) new Bifixed_BDFn<LIFTimeDrivenModel_1_4>(this, atoi(&ident_type[11]));
		}
		else{
			throw EDLUTFileException(TASK_INTEGRATION_METHOD_TYPE, ERROR_INTEGRATION_METHOD_TYPE, REPAIR_INTEGRATION_METHOD_TYPE, *Currentline, fileName.c_str(), true);
		}
	}
	else{
		throw EDLUTFileException(TASK_INTEGRATION_METHOD_TYPE, ERROR_INTEGRATION_METHOD_READ, REPAIR_INTEGRATION_METHOD_READ, *Currentline, fileName.c_str(), true);
	}

	//We load the integration method parameter.
	integrationMethod->loadParameter((TimeDrivenNeuronModel *)this, fh, Currentline, fileName.c_str());
}