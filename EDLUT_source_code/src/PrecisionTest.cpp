/***************************************************************************
 *                           SpeedTest.cpp                                 *
 *                           -------------------                           *
 * copyright            : (C) 2011 by Jesus Garrido and Richard Carrillo   *
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

#include "../include/simulation/ParamReader.h"
#include "../include/simulation/Simulation.h"

#include "../include/simulation/ParameterException.h"
#include "../include/simulation/RandomGenerator.h"

#include "../include/communication/ConnectionException.h"
#include "../include/communication/InputSpikeDriver.h"
#include "../include/communication/OutputSpikeDriver.h"
#include "../include/communication/FileOutputSpikeDriver.h"
#include "../include/communication/OutputWeightDriver.h"
#include "../include/communication/ArrayInputSpikeDriver.h"

#include "../include/spike/EDLUTFileException.h"
#include "../include/spike/EDLUTException.h"
#include "../include/spike/Network.h"


using namespace std;

void GenerateInput(int InputFrequency,double InitTime, double SimulationTime, Simulation ** Simul, int NumberOfSimulations, ArrayInputSpikeDriver * InputDriver){

	int FirstInputCell = 0;
	int LastInputCell = 349;
	int NumberOfSpikesPerCell = (int)(SimulationTime * InputFrequency);
	int TotalNumberOfSpikes = NumberOfSpikesPerCell * (LastInputCell-FirstInputCell+1);

	double * SpikeTimes = new double [TotalNumberOfSpikes];
	long int * SpikeCells = new long int [TotalNumberOfSpikes];

	int SpikeCounter = 0;

	for (int i=FirstInputCell; i<=LastInputCell; ++i){
		for (int j=0; j<NumberOfSpikesPerCell; ++j){
			double randvalue = RandomGenerator::drand();
			SpikeTimes[SpikeCounter] = randvalue*SimulationTime + InitTime;
			SpikeCells[SpikeCounter++] = i;
		}
	}

	for (int i=0; i<NumberOfSimulations; ++i){
		InputDriver[i].LoadInputs(Simul[i]->GetQueue(), Simul[i]->GetNetwork(), TotalNumberOfSpikes, SpikeTimes, SpikeCells);
	}

	delete [] SpikeTimes;
	delete [] SpikeCells;
}

/*!
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
int main(int ac, char *av[]) {
   
	int InitialFreq = 0;
	int FreqStep = 5;
	int FinalFreq = 100;
	int NumberOfIterations = 3;
	double SimulationTime = 0.5;
	double EmptyTime = 0.1;

	string TimeFile = "times.dat";
	string NetworkFiles[] = {"NetCerebellumTestED.dat", "NetCerebellumTestTD.dat", "NetCerebellumTestTD.dat", "NetCerebellumTestTD.dat", "NetCerebellumTestTD.dat"};
	string WeightFiles[] = {"WeightsCerebellumTest.dat", "WeightsCerebellumTest.dat", "WeightsCerebellumTest.dat", "WeightsCerebellumTest.dat", "WeightsCerebellumTest.dat"};
	double StepTimes[] = {1e-4, 1e-4, 5e-4, 1e-3, 1e-2};
	string OutputFiles[] = {"outputED.dat", "outputTD01.dat", "outputTD05.dat", "outputTD1.dat", "outputTD10.dat"};

	int NumberOfSimulations = 5;


	cout << "Loading tables..." << endl;

	srand ( time(NULL) );

	clock_t startt,endt;
   
	try {
   		ParamReader Reader(ac, av);

		double TotalSimulationTime = (SimulationTime + EmptyTime) * NumberOfIterations * (FinalFreq/FreqStep+1);
			
		Simulation ** Simulations = new Simulation * [NumberOfSimulations];
		ArrayInputSpikeDriver * InputDriver = new ArrayInputSpikeDriver [NumberOfSimulations];
		FileOutputSpikeDriver ** OutputDriver = new FileOutputSpikeDriver * [NumberOfSimulations];

		for (int i=0; i<NumberOfSimulations; ++i){
			Simulations[i] = new Simulation(NetworkFiles[i].c_str(),WeightFiles[i].c_str(),TotalSimulationTime, Reader.GetSimulationStepTime());
			Simulations[i]->AddInputSpikeDriver(&InputDriver[i]);
			OutputDriver[i] = new FileOutputSpikeDriver(OutputFiles[i].c_str(),false);
			Simulations[i]->AddOutputSpikeDriver(OutputDriver[i]);
			if (StepTimes[i]!=-1){
//				Simulations[i]->SetTimeDrivenStep(StepTimes[i]);
			}

			Simulations[i]->InitSimulation();
		}
			
		cout << "Simulating network..." << endl;

		double CurrentSimulationTime = 0;

		ofstream OutputFile(TimeFile.c_str());

		for (int j=InitialFreq; j<=FinalFreq; j+=FreqStep){
		
			for (int i=0; i<NumberOfIterations; ++i){
		

				cout << "Generating input with frequency " << j << endl;
				
				GenerateInput(j,CurrentSimulationTime,SimulationTime,Simulations, NumberOfSimulations, InputDriver);
				
				cout << "Simulating network with frequency " << j << endl;

				OutputFile << j;

				// Add Input
				for (int k=0; k<NumberOfSimulations; ++k){
					cout << "Running simulation on network " << k << endl;

					startt=clock();
					Simulations[k]->RunSimulationSlot(CurrentSimulationTime+SimulationTime);
					endt=clock();

					double SimulTime = (endt-startt)/(float)CLOCKS_PER_SEC;

					OutputFile << "\t" << SimulTime;

					Simulations[k]->RunSimulationSlot(CurrentSimulationTime+SimulationTime+EmptyTime);
				}

				OutputFile << endl;
				
				// Write simulation time
				CurrentSimulationTime += SimulationTime + EmptyTime;
			}
		}

		OutputFile.close();
		
		for (int i=0; i<NumberOfSimulations; ++i){
			delete Simulations[i];
			delete OutputDriver[i];
		}

		// Closing simulation connections
		delete [] OutputDriver;
		delete [] InputDriver;
		delete [] Simulations;
		cout << "Oky doky" << endl;     

	} catch (ParameterException Exc){
		cerr << Exc << endl;
		cerr << av[0] << " -time Simulation_Time -nf Network_File -wf Weights_File";
		cerr << " [-info] [-sf Final_Weights_File] [-wt Save_Weight_Step] [-st Simulation_Step_Time] [-log Activity_Register_File] [-if Input_File] ";
		cerr << "[-ic IPAddress:Port Server|Client] [-of Output_File] [-oc IPAddress:Port Server|Client] [-ioc IPAddress:Port Server|Client]" << endl;	
	} catch (ConnectionException Exc){
		cerr << Exc << endl;
		return 1;
	} catch (EDLUTFileException Exc){
		cerr << Exc << endl;
		return Exc.GetErrorNum();
	} catch (EDLUTException Exc){
		cerr << Exc << endl;
		return Exc.GetErrorNum();
	}
return 0;
}
