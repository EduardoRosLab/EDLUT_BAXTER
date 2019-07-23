/***************************************************************************
 *                           EventDrivenNeuronModel.cpp                    *
 *                           -------------------                           *
 * copyright            : (C) 2011 by Jesus Garrido                        *
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

#include "../../include/neuron_model/EventDrivenNeuronModel.h"

EventDrivenNeuronModel::EventDrivenNeuronModel(string NeuronTypeID, string NeuronModelID): NeuronModel(NeuronTypeID, NeuronModelID) {
	// TODO Auto-generated constructor stub

}

EventDrivenNeuronModel::~EventDrivenNeuronModel() {
	// TODO Auto-generated destructor stub
}

enum NeuronModelSimulationMethod EventDrivenNeuronModel::GetModelSimulationMethod(){
	return EVENT_DRIVEN_MODEL;
}


enum NeuronModelType EventDrivenNeuronModel::GetModelType(){
	return NEURAL_LAYER;
}

