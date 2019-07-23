/***************************************************************************
 *                           TimeDrivenInputDevice.cpp                     *
 *                           -------------------                           *
 * copyright            : (C) 2018 by Francisco Naveros                    *
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

#include "../../include/neuron_model/TimeDrivenInputDevice.h"
#include "../../include/neuron_model/TimeDrivenModel.h"
#include "../../include/neuron_model/VectorNeuronState.h"

#include "../../include/spike/EDLUTFileException.h"

#include "../../include/openmp/openmp.h"

TimeDrivenInputDevice::TimeDrivenInputDevice(string NeuronTypeID, string NeuronModelID, TimeScale timeScale) : TimeDrivenModel(NeuronTypeID, NeuronModelID, timeScale){
	// TODO Auto-generated constructor stub

}

TimeDrivenInputDevice::~TimeDrivenInputDevice() {
}

double TimeDrivenInputDevice::LoadTimeDrivenStepSize(string fileName, FILE *fh, long Initialline) throw (EDLUTFileException){
	long Currentline = Initialline;
	skip_comments(fh, Currentline);
	double step_size = 0;;
	if (fscanf(fh, "%lf", &step_size) == 0 || step_size <=0.0){
		throw EDLUTFileException(TASK_TIME_DRIVEN_INPUT_DEVICE_LOAD, ERROR_TIME_DRIVEN_INPUT_DEVICE_STEP_SIZE, REPAIR_NEURON_MODEL_VALUES, Currentline, fileName.c_str(), true);
	}
	return step_size;
}


enum NeuronModelSimulationMethod TimeDrivenInputDevice::GetModelSimulationMethod(){
	return TIME_DRIVEN_MODEL_CPU;
}

enum NeuronModelType TimeDrivenInputDevice::GetModelType(){
	return INPUT_DEVICE;
}

enum NeuronModelInputActivityType TimeDrivenInputDevice::GetModelInputActivityType(){
	return NONE_INPUT;
}

bool TimeDrivenInputDevice::CheckSynapseType(Interconnection * connection){
	cout << "Neuron model " << this->GetTypeID() << ", " << this->GetModelID() << " does not support input synapses." << endl;
	return false;
}

