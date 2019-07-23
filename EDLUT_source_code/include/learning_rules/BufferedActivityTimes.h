/***************************************************************************
 *                           BufferedActivityTimes.h                       *
 *                           -------------------                           *
 * copyright            : (C) 2016 by Francisco Naveros                    *
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

#ifndef BUFFEREDACTIVITYTIMES_H_
#define BUFFEREDACTIVITYTIMES_H_

#include <iostream>


using namespace std;

/*!
 * \file BufferedActivityTimes.h
 *
 * \author Francisco Naveros
 * \date May 2016
 *
 * This file declares a class which implements a linked list of input activity times for each input sinapse in a learning rule with kernels precomputed in look-up tables.
 */

/*!
 * \class BufferedActivityTimes
 *
 * \brief Buffer for activity times.
 *
 * This class declares a class which implements a linked list of input activity times for each input sinapse in a learning rule with kernels precomputed in look-up tables.
 *
 * \author Francisco Naveros
 * \date May 2016
 */

#include "../../include/openmp/openmp.h"

struct SpikeData{
	double time;
	int synapse_index;
};

struct BufferedActivityTimesData{
	int size;
	int N_elements;
	int first_element;
	int last_element;
	SpikeData * spike_data;
};


class BufferedActivityTimes {

	public:

		int size_buffer;
		BufferedActivityTimesData * structure;



		/*!
		* Size of the output containers (one for each OpenMP thread)
		*/
		int * size_output_array;

		/*!
		* output containers (one for each OpenMP thread)
		*/
		SpikeData ** output_spike_data;




		BufferedActivityTimes(int newSize);

		~BufferedActivityTimes();

		//void InsertElement(int neuron_index, double time, double ThresholdTime, int synapse_index);
		inline void InsertElement(int neuron_index, double time, double ThresholdTime, int synapse_index){
			DuplicateSize(neuron_index, ThresholdTime);

			//insert the new element
			structure[neuron_index].N_elements++;
			structure[neuron_index].first_element++;
			if (structure[neuron_index].first_element == structure[neuron_index].size){
				structure[neuron_index].first_element = 0;
			}

			structure[neuron_index].spike_data[structure[neuron_index].first_element].time = time;
			structure[neuron_index].spike_data[structure[neuron_index].first_element].synapse_index = synapse_index;
		}


		//int ProcessElements(int neuron_index, double ThresholdTime);
		inline int ProcessElements(int neuron_index, double ThresholdTime){
			//check if the old times are under the threshold time.
			if (structure[neuron_index].N_elements > 0){
				int OpenMPThreadIndex = omp_get_thread_num();

				if (structure[neuron_index].N_elements > this->size_output_array[OpenMPThreadIndex]){
					this->size_output_array[OpenMPThreadIndex] = structure[neuron_index].N_elements;
					delete output_spike_data[OpenMPThreadIndex];
					output_spike_data[OpenMPThreadIndex] = new SpikeData[size_output_array[OpenMPThreadIndex]];
				}

				int i = structure[neuron_index].first_element;//first position
				int counter = 0;
				while (i != structure[neuron_index].last_element){
					if (structure[neuron_index].spike_data[i].time > ThresholdTime){
						output_spike_data[OpenMPThreadIndex][counter].time = structure[neuron_index].spike_data[i].time;
						output_spike_data[OpenMPThreadIndex][counter].synapse_index = structure[neuron_index].spike_data[i].synapse_index;
						counter++;
						if (i == 0){
							i = structure[neuron_index].size;
						}
						i--;
					}
					else{
						structure[neuron_index].last_element = i;
						structure[neuron_index].N_elements = counter;
						break;
					}
				}
			}
			return structure[neuron_index].N_elements;
		}




		//void DuplicateSize(int index, double limit_time);
		inline void DuplicateSize(int index, double ThresholdTime){
			//We check if the vector is full
			if (structure[index].size == (structure[index].N_elements + 1)){
				//we check if the last element is out of the range and some elements can be discard

				int last_element = structure[index].last_element + 1;
				if (last_element == structure[index].size){
					last_element = 0;
				}
				if (structure[index].spike_data[last_element].time < ThresholdTime){
					int i = structure[index].first_element - structure[index].size/2;
					if (i < 0){
						i += structure[index].size;
					}
					int counter = structure[index].size / 2;
					while (i != structure[index].last_element){
						if (structure[index].spike_data[i].time > ThresholdTime){
							counter++;
							if (i == 0){
								i = structure[index].size;
							}
							i--;
						}
						else{
							structure[index].last_element = i;
							structure[index].N_elements = counter;
							break;
						}
					}
				}
				//the vector size must be incremented.
				else{
					SpikeData * aux_data = structure[index].spike_data;

					structure[index].spike_data = new SpikeData[structure[index].size * 2];

					int i = structure[index].last_element;//first position
					int counter = 0;
					do{
						i++;
						if (i == structure[index].size){
							i = 0;
						}
						structure[index].spike_data[counter].time = aux_data[i].time;
						structure[index].spike_data[counter].synapse_index = aux_data[i].synapse_index;
						counter++;

					} while (i != structure[index].first_element);
					structure[index].size *= 2;
					structure[index].first_element = counter - 1;
					structure[index].last_element = structure[index].size - 1;
					delete aux_data;
				}
			}
		}

		//SpikeData * GetOutputSpikeData();
		inline SpikeData * GetOutputSpikeData(){
			return output_spike_data[omp_get_thread_num()];
		}



};






#endif /* BUFFEREDACTIVITYTIMES_H_ */
