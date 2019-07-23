/***************************************************************************
 *                           EventDrivenInputDevice.cpp                    *
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

#include "../../include/neuron_model/EventDrivenInputDevice.h"
#include "../../include/neuron_model/NeuronModel.h"
#include "../../include/neuron_model/VectorNeuronState.h"

#include "../../include/openmp/openmp.h"

EventDrivenInputDevice::EventDrivenInputDevice(string NeuronTypeID, string NeuronModelID, TimeScale timeScale): NeuronModel(NeuronTypeID, NeuronModelID, timeScale){
	// TODO Auto-generated constructor stub

}

EventDrivenInputDevice::~EventDrivenInputDevice() {

}


enum NeuronModelSimulationMethod EventDrivenInputDevice::GetModelSimulationMethod(){
	return EVENT_DRIVEN_MODEL;
}


enum NeuronModelType EventDrivenInputDevice::GetModelType(){
	return INPUT_DEVICE;
}


enum NeuronModelInputActivityType EventDrivenInputDevice::GetModelInputActivityType(){
	return NONE_INPUT;
}








