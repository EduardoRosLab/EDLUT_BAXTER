/***************************************************************************
 *                           EDLUTMatlab.cpp                               *
 *                           -------------------                           *
 * copyright            : (C) 2010 by Jesus Garrido                        *
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

#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <time.h>
#include <math.h> // TODO: maybe remove this

#include <iostream>

#include "mex.h"

#include "../include/simulation/Simulation.h"

#include "../include/communication/FileInputSpikeDriver.h"
#include "../include/communication/FileOutputSpikeDriver.h"
#include "../include/communication/ArrayInputSpikeDriver.h"
#include "../include/communication/ArrayOutputSpikeDriver.h"

#include "../include/spike/EDLUTFileException.h"
#include "../include/spike/EDLUTException.h"
#include "../include/spike/Network.h"


using namespace std;

/*!
 * 
 * \note Input arguments in mex file:
 * 		1. Simulation time (-time kernel parameter).
 * 		2. Network File (-nf kernel parameter).
 * 		3. Weights File (-wf kernel parameter).
 * 		4. Input File (-if kernel parameter).
 * 		5. Log File (-log kernel parameter).
 *
 * 
 * \note Obligatory parameters:
 * 			-time Simulation_Time(in_seconds) It sets the total simulation time.
 * 			-nf Network_File	It sets the network description file.
 * 			-wf Weights_File	It sets the weights file.
 * 			
 * \note  parameters:
 * 			-info 	It shows the network information.
 * 			-sf File_Name	It saves the final weights in file File_Name.
 * 			-wt Save_Weight_Step	It sets the step time between weights saving.
 * 			-st Step_Time(in_seconds) It sets the step time in simulation.
 * 			-log File_Name	It saves the activity register in file File_Name.
 * 			-if Input_File	It adds the Input_File file in the input sources of the simulation.
 * 			-of Output_File	It adds the Output_File file in the output targets of the simulation.
 * 			-ic IPDirection:Port Server|Client	It adds the connection as a server or a client in the specified address in the input sources of the simulation.
 * 			-oc IPDirection:Port Server|Client	It adds the connection as a server or a client in the specified address in the output targets of the simulation.	 
 * 			-ioc IPDirection:Port Server|Client	It adds the connection as a server or a client in the specified address in the input sources and in the output targets of the simulation.	 
 * 
 */

extern void _main();

const int numInputArgs  = 5;
const int numOutputArgs = 2;

// Function declarations.
// -----------------------------------------------------------------
double  getMatlabScalar(const mxArray* ptr);
char * getMatlabString(const mxArray* ptr);
double& createMatlabScalar(mxArray*& ptr);
double* createMatlabDoubleArray(mxArray*& ptr,int OutputNumber);
long int* createMatlabIntArray(mxArray*& ptr,int OutputNumber);

// Function definitions.
// -----------------------------------------------------------------
void mexFunction (int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {

	// Check to see if we have the correct number of input and output
	// arguments.
	if (nrhs != numInputArgs){
		mexErrMsgTxt("Incorrect number of input arguments");
		mexErrMsgTxt("Use: [SpikeTimes, CellNumbers] = EDLUTKernel(SimulationTime,NetworkFile,WeightFile,InputFile,LogFile)");
	}

	if (nlhs != numOutputArgs){
		mexErrMsgTxt("Incorrect number of output arguments");
		mexErrMsgTxt("[SpikeTimes, CellNumbers] = EDLUTKernel(SimulationTime,NetworkFile,WeightFile,InputFile,LogFile)");
	}

	// Get the inputs.
	double SimulationTime = getMatlabScalar(prhs[0]);
	char * NetworkFile = getMatlabString(prhs[1]);
	char * WeightFile  = getMatlabString(prhs[2]);
	char * InputFile = getMatlabString(prhs[3]);
	char * LogFile = getMatlabString(prhs[4]);

	clock_t startt,endt;

	double * OutputSpikeTimes;
	long int * OutputSpikeCells;


	cout << "Int size: " << sizeof(int) << endl;
	cout << "Long int size: " << sizeof(long int) << endl;

	srand ( time(NULL) );

	try {
		Simulation Simul(NetworkFile, WeightFile, SimulationTime, 0);

		// Create a new input object to add input spikes
		// ArrayInputSpikeDriver * InputDriver = new ArrayInputSpikeDriver();
		//Simul->AddInputSpikeDriver(InputDriver);

		// Create a new output object to get output spikes
		ArrayOutputSpikeDriver * OutputDriver = new ArrayOutputSpikeDriver();

		Simul.AddOutputSpikeDriver(OutputDriver);

		FileInputSpikeDriver * InputDriver = new FileInputSpikeDriver(InputFile);
		Simul.AddInputSpikeDriver(InputDriver);

		FileOutputSpikeDriver * MonitorDriver = new FileOutputSpikeDriver(LogFile,false);
		Simul.AddMonitorActivityDriver(MonitorDriver);

		cout << "Simulating network..." << endl;

		startt=clock();
		Simul.RunSimulation();
		endt=clock();

		// Get outputs and print them
		int OutputNumber = OutputDriver->GetBufferedSpikes(OutputSpikeTimes,OutputSpikeCells);

		// Create the output. It is also a double-precision scalar.
		double* SpikeTimes = createMatlabDoubleArray(plhs[0],OutputNumber);
		long int* SpikeCells = createMatlabIntArray(plhs[1],OutputNumber);

		if (OutputNumber>0){
			memcpy(SpikeTimes, OutputSpikeTimes, OutputNumber*sizeof(double));
			memcpy(SpikeCells, OutputSpikeCells, OutputNumber*sizeof(long int));

			delete [] OutputSpikeTimes;
			delete [] OutputSpikeCells;
		}

		cout << "Oky doky" << endl;

		cout << "Elapsed time: " << (endt-startt)/(float)CLOCKS_PER_SEC << " sec" << endl;
		cout << "Number of updates: " << Simul.GetSimulationUpdates() << endl;
		cout << "Mean number of spikes in heap: " << Simul.GetHeapAcumSize()/(float)Simul.GetSimulationUpdates() << endl;
		cout << "Updates per second: " << Simul.GetSimulationUpdates()/((endt-startt)/(float)CLOCKS_PER_SEC) << endl;


		delete OutputDriver;
		delete InputDriver;
		delete MonitorDriver;
	} catch (EDLUTFileException Exc){
		cerr << Exc << ": " << Exc.GetErrorNum() << endl;
	} catch (EDLUTException Exc){
		cerr << Exc << ": " << Exc.GetErrorNum() << endl;
	}

}

double getMatlabScalar (const mxArray* ptr) {

	// Make sure the input argument is a scalar in double-precision.
	if (!mxIsDouble(ptr) || mxGetNumberOfElements(ptr) != 1)
		mexErrMsgTxt("The input argument must be a double-precision scalar");

	return *mxGetPr(ptr);
}

char * getMatlabString (const mxArray* ptr) {

	// Make sure the input argument is a scalar in double-precision.
	if (!mxIsChar(ptr))
		mexErrMsgTxt("The input argument must be a string");

	return mxArrayToString (ptr);
}

double& createMatlabScalar (mxArray*& ptr) {
	ptr = mxCreateDoubleMatrix(1,1,mxREAL);
	return *mxGetPr(ptr);
}

double* createMatlabDoubleArray(mxArray*& ptr,int OutputNumber){
	ptr = mxCreateDoubleMatrix(OutputNumber,1,mxREAL);
	return mxGetPr(ptr);
}

long int* createMatlabIntArray(mxArray*& ptr,int OutputNumber){
	ptr = mxCreateNumericMatrix(OutputNumber,1,mxINT32_CLASS,mxREAL);
	return (long int *) mxGetData(ptr);
}


