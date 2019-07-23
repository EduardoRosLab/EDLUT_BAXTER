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
#include "../../include/cudaError.h"

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


	

	int nDevices=0;
	int nValidDevices=0;
	cudaGetDeviceCount(&nDevices);
	if(nDevices==0){
		printf("WARNING: No CUDA capable GPU available\n");
		NumberOfGPUs=0;
	}else{
		GPUsIndex=new int[nDevices]();
		cudaDeviceProp prop;
		for(int i=0; i<nDevices; i++){
			cudaGetDeviceProperties(&prop, i);
			if(prop.major>=2){
				GPUsIndex[nValidDevices]=i;
				nValidDevices++;
			}
		}
		if(nValidDevices==0){
			printf("WARNING: No CUDA capable GPU available with CUDA architecture 2.0 or higher\n");
		}else{
			printf("%d CUDA capable GPU available with CUDA architecture 2.0 or higher\n", nValidDevices);
		}
	}


	if(nValidDevices>NumberOfOpenMPQueues){
		NumberOfGPUs=NumberOfOpenMPQueues;
	}else{
		NumberOfGPUs=nValidDevices;
	}

	printf("%d GPUs will be used\n",NumberOfGPUs);

}
