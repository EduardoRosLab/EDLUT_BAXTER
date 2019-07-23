/***************************************************************************
 *                           TimeDrivenModel.cpp                           *
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

#include "../../include/neuron_model/TimeDrivenModel.h"
#include "../../include/neuron_model/NeuronModel.h"


TimeDrivenModel::TimeDrivenModel(string NeuronTypeID, string NeuronModelID) : NeuronModel(NeuronTypeID, NeuronModelID), time_driven_step_size(0.0){
	// TODO Auto-generated constructor stub
}

TimeDrivenModel::TimeDrivenModel(string NeuronTypeID, string NeuronModelID, TimeScale new_timeScale) : NeuronModel(NeuronTypeID, NeuronModelID, new_timeScale), time_driven_step_size(0.0) {
	// TODO Auto-generated constructor stub
}


TimeDrivenModel::~TimeDrivenModel() {
	// TODO Auto-generated destructor stub
}

void TimeDrivenModel::SetTimeDrivenStepSize(double step){
	this->time_driven_step_size = step;
}


double TimeDrivenModel::GetTimeDrivenStepSize(){
	return this->time_driven_step_size;
}
