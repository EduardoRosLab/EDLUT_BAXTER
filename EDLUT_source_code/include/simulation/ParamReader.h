/***************************************************************************
 *                           ParamReader.h                                 *
 *                           -------------------                           *
 * copyright            : (C) 2009 by Jesus Garrido, Richard Carrillo and  *
 *						: Francisco Naveros                                *
 * email                : jgarrido@atc.ugr.es, fnaveros@ugr.es             *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef PARAMREADER_H_
#define PARAMREADER_H_

/*!
 * \file ParamReader.h
 *
 * \author Jesus Garrido
 * \author Francisco Naveros
 * \date September 2008
 *
 * \note Modified on January 2012 in order to include time-driven simulation support in GPU.
 * New state variables (TimeDrivenStepTimeGPU)
 *
 * This file declares a class which read the input parameters and generates the
 * simulation characteristics (inputs, outputs, simulation time, simulation step...)
 */
 
#include <vector>
#include <string>
#include <sstream>
#include <fstream>

#include "./ParameterException.h"
#include "../communication/ConnectionException.h"
#include "../spike/EDLUTException.h"

using namespace std;

class InputSpikeDriver;
class InputCurrentDriver;
class OutputSpikeDriver;
class OutputWeightDriver;
class Simulation;

 
 /*!
 * \class ParamReader
 *
 * \brief Class for read input parameters. 
 *
 * This class abstract methods for get the input parameters in the simulation.
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
 * 			-log File_Name It saves the activity register in file File_Name.
 *          -logp File_Name It saves all events register in file File_Name.
 * 			-if Input_File	It adds the Input_File file in the input sources of the simulation (SPIKES).
 *          -ifc Input_File	It adds the Input_File file in the input sources of the simulation (CURRENTS).
 * 			-of Output_File	It adds the Output_File file in the output targets of the simulation.
 *          -openmpQ number_of_OpenMP_queues It sets the number of OpenMP queues.
 *          -openmp number_of_OpenMP_threads It sets the number of OpenMP threads.
 * 			-ic IPAddress:Port Server|Client	It adds the connection as a server or a client in the specified direction in the input sources of the simulation.
 * 			-oc IPAddress:Port Server|Client	It adds the connection as a server or a client in the specified direction in the output targets of the simulation.
 * 			-ioc IPAddress:Port Server|Client	It adds the connection as a server or a client in the specified direction in the input sources and in the output targets.	 
 *			-rt   It activates the real time option.
 *			-rtgap Time     Time in second that the simulation can be executed in advance (time between the generation of an output spike and the arrive of a response to this activity)
 *          -rt1 factor1    Value between 0 and 1 that represent a precentage of "rtgap".
 *          -rt2 factor2    Value between factor1 and 1 that represent a precentage of "rtgap".
 *          -rt3 factor3    Value between factor2 and 1 that represent a precentage of "rtgap".
 *
 *
 * \author Jesus Garrido
 * \date September 2008
 */
 class ParamReader {
 	
 	private:
 	
 		/*!
 		 * Total simulation time.
 		 */
 		double SimulationTime;
 		
 		/*!
 		 * Network configuration file.
 		 */
 		char * NetworkFile;
 		 
 		/*!
 		 * Weights configuration file.
 		 */
 		char * WeightsFile;
 		
 		/*!
 		 * Save weights step time
 		 */
 		double WeightTime;
 		
 		/*!
 		 * Network info?
 		 */
 		bool NetworkInfo;
 		
 		/*!
 		 * Simulation step time. 
 		 */
 		double SimulationStepTime;

 		/*!
 		 * Input spike drivers.
 		 */ 	
 		vector<InputSpikeDriver *> InputSpikeDrivers;

		/*!
		* Input current drivers.
		*/
		vector<InputCurrentDriver *> InputCurrentDrivers;
 		
 		/*!
 		 * Output drivers. 
 		 */ 	
 		vector<OutputSpikeDriver *> OutputSpikeDrivers;
 		
 		/*!
 		 * Output monitor. 
 		 */ 	
 		vector<OutputSpikeDriver *> MonitorDrivers;
 		
 		/*!
 		 * Output drivers. 
 		 */ 	
 		vector<OutputWeightDriver *> OutputWeightDrivers; 

  		/*!
 		 * Number of OpenMP queues. 
 		 */
		int NumberOfQueues;

 		/*!
 		 * Variable that indicate if it is a real time simulation (0=no real time, 1=real time).
 		 */
		int RealTimeOption;

		 /*!
 		 * Time in second that the simulation can be executed in advance (time between the generation of an output 
		 * spike and the arrive of a response to this activity).
 		 */
		float rtgap;

		/*!
 		 * Value between 0 and 1 that represent a precentage of "rtgap". This boundary is used to calculate which 
		 * events must be processed in real time and which must be discard in function of the consumed time.
 		 */
        float rt1;
        
		/*!
 		 * Value between rt1 and 1 that represent a precentage of "rtgap". This boundary is used to calculate which 
		 * events must be processed in real time and which must be discard in function of the consumed time.
 		 */
		float rt2;
        
		/*!
 		 * Value between rt2 and 1 that represent a precentage of "rtgap". This boundary is used to calculate which 
		 * events must be processed in real time and which must be discard in function of the consumed time.
 		 */
		float rt3;


 	
 		/*!
 		 * \brief It parses the input arguments.
 		 * 
 		 * It parses the input arguments and gets the simulation configuration.
 		 * 
 		 * \param Number Number of arguments in Arguments.
 		 * \param Arguments Arguments to be parsed. 
 		 * 
 		 * \throw ParameterException When something wrong happens in the parser proccess.
 		 */
 		void ParseArguments(int Number, char ** Arguments) throw (ParameterException, ConnectionException);
 		
 		/*!
 		 * \brief It tests if a file exists.
 		 * 
 		 * It tests if a file exists and if we can read it.
 		 * 
 		 * \param Name The file name.
 		 * 
 		 * \return True if the file exists and we can read it. False in other case.
 		 */
 		bool FileExists(string Name);
 		 
 		
 	public:
 		
 		/*!
 		 * \brief Default constructor.
 		 * 
 		 * It creates a new object with the arguments in the parameters.
 		 * 
 		 * \param ArgNumber Number of arguments in Arg.
 		 * \param Arg Arguments of the simulation.
 		 *
 		 * \throw ParameterException When something wrong happens in the parser proccess.
 		 */
 		ParamReader(int ArgNumber, char ** Arg) throw (ParameterException, ConnectionException);
 		
 		/*!
 		 * \brief It gets the total simulation time.
 		 * 
 		 * It gets the total simulation time. The argument indicator for simulation time
 		 * is -time, so it searchs -time and returns the value as a double.
 		 * 
 		 * \return The total simulation time. -1 if the parameter isn't used.
 		 */
 		double GetSimulationTime();
 		
 		/*!
 		 * \brief It gets the saving weights step time.
 		 * 
 		 * It gets the saving weights step time. The argument indicator for saving weights step time
 		 * is -wt, so it searchs -wt and returns the value as a double.
 		 * 
 		 * \return The saving weights step time. -1 if the parameter isn't used.
 		 */
 		double GetSaveWeightStepTime();
 		
 		/*!
 		 * \brief It gets the network configuration file.
 		 * 
 		 * It gets the network configuration file. The argument indicator for network configuration file
 		 * is -nf, so it searchs -nf and returns the file name.
 		 * 
 		 * \return The network configuration file name. NULL if network file isn't introduced.
 		 */
 		char * GetNetworkFile();
 		
 		/*!
 		 * \brief It gets the weights configuration file.
 		 * 
 		 * It gets the weights configuration file. The argument indicator for weights configuration file
 		 * is -wf, so it searchs -wf and returns the file name.
 		 * 
 		 * \return The weights configuration file name. NULL if weights file isn't introduced.
 		 */
 		char * GetWeightsFile();
 		
 		/*!
 		 * \brief It checks if the -info option is enabled.
 		 * 
 		 * It checks if the -info option is enabled. The argument indicator for network information
 		 * is -info, so it searchs -info.
 		 * 
 		 * \return True if the -info option is enabled. False in other case. 
 		 */ 		
 		bool CheckInfo();
 		
 		/*!
 		 * \brief It gets the simulation step time.
 		 * 
 		 * It gets the simulation step time. The argument indicator for simulation step time
 		 * is -st, so it searchs -st and returns the value as a float.
 		 * 
 		 * \return The simulation step time. -1 if this option isn't enabled. 
 		 */
 		double GetSimulationStepTime();


		/*!
 		 * \brief It gets the total number of OpenMP queues.
 		 * 
 		 * It gets the total number of OpenMP queues. The argument indicator for number of OpenMP queues
 		 * is -openmpQ, so it searchs -openmpQ and returns the value as a int.
 		 * 
 		 * \return The total number of OpenMP queues. 1 if the parameter isn't used.
 		 */
		int GetNumberOfQueues();


        /*!
 		 * \brief It gets the real time option.
 		 * 
 		 * It gets the real time option.
 		 * 
 		 * \return The real time option.
 		 */
		int GetRealTimeOption();

		/*!
 		 * \brief It gets the RtGap value.
 		 * 
 		 * It gets the RtGap value.
 		 * 
 		 * \return The RtGap value.
 		 */
		float GetRtGap();

		/*!
 		 * \brief It gets the Rt1 value.
 		 * 
 		 * It gets the Rt1 value.
 		 * 
 		 * \return The Rt1 value.
 		 */
		float GetRt1();

		/*!
 		 * \brief It gets the Rt2 value.
 		 * 
 		 * It gets the Rt2 value.
 		 * 
 		 * \return The Rt2 value.
 		 */
 		float GetRt2();

		/*!
 		 * \brief It gets the Rt3 value.
 		 * 
 		 * It gets the Rt3 value.
 		 * 
 		 * \return The Rt3 value.
 		 */
		float GetRt3();
 		

 		/*!
 		 * \brief It gets the input spike drivers.
 		 * 
 		 * It gets the input spike drivers of the simulation. The argument indicator for input spike configuration file
 		 * is -if, so it searchs -if and returns the file name.
 		 * 
 		 * \return A vector of the simulation input spike drivers. 
 		 */ 	
 		vector<InputSpikeDriver *> GetInputSpikeDrivers();

		/*!
		* \brief It gets the input current drivers.
		*
		* It gets the input current drivers of the simulation. The argument indicator for input spike configuration file
		* is -ifc, so it searchs -ifc and returns the file name.
		*
		* \return A vector of the simulation input spike drivers.
		*/
		vector<InputCurrentDriver *> GetInputCurrentDrivers();
 		
 		/*!
 		 * \brief It gets the output drivers.
 		 * 
 		 * It gets the output drivers of the simulation. It has two kinds of output drivers:
 		 * -of Output_File, -oc IPAddress:Port Server|Client and -ioc IPAddress:Port Server|Client. It adds all the output drivers.
 		 * 
 		 * \return A vector of the simulation output drivers. 
 		 */ 	
 		vector<OutputSpikeDriver *> GetOutputSpikeDrivers();
 		
 		/*!
 		 * \brief It gets the monitoring drivers.
 		 * 
 		 * It gets the monitoring drivers of the simulation. -log Monitor_File.
 		 * It adds all the monitoring drivers.
 		 * 
 		 * \return A vector of the simulation monitor drivers. 
 		 */ 	
 		vector<OutputSpikeDriver *> GetMonitorDrivers();
 		
 		/*!
 		 * \brief It gets the output weight drivers.
 		 * 
 		 * It gets the output weight drivers of the simulation. The argument indicator for final weights configuration file
 		 * is -sf, so it searchs -sf and returns the file name.
 		 * 
 		 * \return A vector of the simulation output drivers. 
 		 */ 	
 		vector<OutputWeightDriver *> GetOutputWeightDrivers();

 		/*!
		 * \brief It creates and initializes the simulation.
		 *
		 * It creates and initializes the simulation.
		 *
		 * \throw EDLUTException If something wrong happens.
		 * \throw ConnectionException If the connection haven't been able to be stablished.
		 *
		 * \param Reader is the parser of parameters
		 */
		Simulation * CreateAndInitializeSimulation() throw (EDLUTException, ConnectionException);
};

#endif /*PARAMREADER_H_*/
