/***************************************************************************
 *                           openmp.h                                      *
 *                           -------------------                           *
 * copyright            : (C) 2014 by Francisco Naveros                    *
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

#include <iostream>

using namespace std;

#include "../../include/openmp/openmp.h"


int NumberOfOpenMPQueues;
int NumberOfGPUs;
int * GPUsIndex;

void Set_Number_of_openmp_threads(int NumberOfQueues){
	if(NumberOfQueues>omp_get_max_threads()){
		NumberOfQueues=omp_get_max_threads();
		cout<<"WARNING: queue number higher than CPU cores number. Queue number fixed to "<< NumberOfQueues<<endl;
	}

	#ifdef _OPENMP 
		cout << "OpenMP support is enabled" << endl;
	#else
		cout<<"OpenMP support is disabled"<<endl;
	#endif



	(void)omp_set_num_threads(NumberOfQueues);
	NumberOfOpenMPQueues=NumberOfQueues;
	NumberOfGPUs=0;

}
