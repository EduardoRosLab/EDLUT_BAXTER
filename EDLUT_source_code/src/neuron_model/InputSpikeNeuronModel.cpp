/***************************************************************************
 *                           InputSpikeNeuronModel.cpp                     *
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

#include "../../include/neuron_model/InputSpikeNeuronModel.h"

InputSpikeNeuronModel::InputSpikeNeuronModel(string NeuronTypeID, string NeuronModelID) : EventDrivenInputDevice(NeuronTypeID, NeuronModelID, MilisecondScale) {

}

InputSpikeNeuronModel::~InputSpikeNeuronModel() {
	// TODO Auto-generated destructor stub
}


enum NeuronModelOutputActivityType InputSpikeNeuronModel::GetModelOutputActivityType(){
	return OUTPUT_SPIKE;
}

ostream & InputSpikeNeuronModel::PrintInfo(ostream & out){
	out << "- Input Neuron Model: " << this->GetModelID() << endl;

	return out;
}	