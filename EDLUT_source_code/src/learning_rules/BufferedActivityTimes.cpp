/***************************************************************************
 *                           WithPostSynaptic.cpp                          *
 *                           -------------------                           *
 * copyright            : (C) 2013 by Francisco Naveros                    *
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

#include "../../include/learning_rules/BufferedActivityTimes.h"

#include "../../include/openmp/openmp.h"
#include <iostream>
using namespace std;



BufferedActivityTimes::BufferedActivityTimes(int newSize):size_buffer(newSize){
	structure = new BufferedActivityTimesData[newSize]();
	for (int i = 0; i < newSize; i++){
		structure[i].size = 2;
		structure[i].first_element = 0;
		structure[i].last_element = 0;
		structure[i].N_elements = 0;
		structure[i].spike_data = new SpikeData[2];
	}

	size_output_array = new int[NumberOfOpenMPQueues];
	output_spike_data = (SpikeData**) new SpikeData*[NumberOfOpenMPQueues];
	for (int i = 0; i < NumberOfOpenMPQueues; i++){
		size_output_array[i] = 64;
		output_spike_data[i] = new SpikeData[size_output_array[i]];
	}
}

BufferedActivityTimes::~BufferedActivityTimes(){
	for (int i = 0; i < size_buffer; i++){
		delete structure[i].spike_data;
	}
	delete structure;

	for (int i = 0; i < NumberOfOpenMPQueues; i++){
		delete output_spike_data[i];
	}
	delete [] output_spike_data;

	delete size_output_array;
}

