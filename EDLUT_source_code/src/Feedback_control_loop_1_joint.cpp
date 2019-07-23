/*******************************************************************************
 *                       SimulatedRobotControl3laws.c                           *
 *                       -----------------------                               *
 * copyright            : (C) 2013 by Richard R. Carrillo and Niceto R. Luque  *
 * email                : rcarrillo, nluque at ugr.es                          *
 *******************************************************************************/

/***************************************************************************
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 ***************************************************************************/

/*!
 * \file SimulatedRobotControl.c
 *
 * \author Niceto R. Luque
 * \author Richard R. Carrillo
 * \date 7 of November 2013
 * In this file the main robot-control loop is implemented.
 */

#include <iostream>

using namespace std;


#if defined(_DEBUG) && (defined(_WIN32) || defined(_WIN64))
#   define _CRTDBG_MAP_ALLOC
#   include <crtdbg.h> // To check for memory leaks
#endif

#if defined(__APPLE_CC__)
  // MAC OS X includes
#	define REAL_TIME_OSX
#elif defined (__linux__)
  // Linux includes
#	define REAL_TIME_LINUX
#elif (defined(_WIN32) || defined(_WIN64))
#	define REAL_TIME_WINNT
#else
#	error "Unsupported platform"
#endif

#if defined(REAL_TIME_OSX)
#	include <mach/mach.h>
#	include <mach/mach_time.h>
#	include <CoreServices/CoreServices.h>
#	include <unistd.h>
#elif defined(REAL_TIME_LINUX)
#	include <time.h>
#elif defined(REAL_TIME_WINNT)
#	include <windows.h>
#endif

#include <stdio.h>

#include "../include/interface/C_interface_for_robot_control.h"
//ROBOT//
#include "../include/arm_robot_simulator/ArmRobotSimulation.h"

#include "../include/openmp/openmp.h"


// Neural-network simulation files
#define NET_FILE "NetDistributedPlasticity1DOFClassicSTDP.net"// Neural-network definition file used by EDLUT
#define INPUT_WEIGHT_FILE "Weights_1DOF.net"// Neural-network input weight file used by EDLUT
#define OUTPUT_WEIGHT_FILE "OutputWeight.dat" // Neural-network output weight file used by EDLUT
#define WEIGHT_SAVE_PERIOD 100// The weights will be saved each period (in seconds) (0=weights are not saved periodically)
#define INPUT_ACTIVITY_FILE  "GranularActivity_mf10000seg.dat"//Optional input activity file 
#define OUTPUT_ACTIVITY_FILE "OutputActivity.dat" // Output activity file used to register the neural network activity
#define LOG_FILE "vars.dat"  // Log file used to register the simulation variables
#define REAL_TIME_NEURAL_SIM 0 // EDLUT's simulation mode (0=No real-time neural network simulation 1=For real robot control)


#define FIRST_REAL_TIME_RESTRICTION 0.7
#define SECOND_REAL_TIME_RESTRICTION 0.8
#define THIRD_REAL_TIME_RESTRICTION 0.9

#define MAX_SIMULATION_DELAY 0.05f

#define NUMBER_OF_OPENMP_QUEUES 2


#define ERROR_AMP 1 // Amplitude of the injected error
#define NUM_REP 1 // Number of repetition of the exponential shape along the Trajectory Time
#define TRAJECTORY_TIME 1 // Simulation time in seconds required to execute the desired trajectory once
#define MAX_TRAJ_EXECUTIONS 5000 // Maximum number of trajectories repetitions that will be executed by the robot
#define ERROR_DELAY_TIME 0.1000 // Delay after calculating the error vars
#define ROBOT_JOINT_ERROR_NORM 160 // proportional error {160}

const double ROBOT_GRAVITY[3]={0, 0, 9.81}; // Earth's standard acceleration due to gravity [Gx Gy Gz]
const double ROBOT_EXTERNAL_FORCE[6]={0, 0, 0, 0, 0, 0}; // External force on manipulator tip [Fx Fy Fz MOMENTUMx MOMENTUMy MOMENTUMz]


///////////////////////////// MAIN LOOP //////////////////////////

int main(int ac, char *av[])
{

      
   int errorn;
   long total_output_spks; 
   double cerebellar_output_vars[NUM_OUTPUT_VARS]={0.0, 0.0}; // Corrective cerebellar output torque
   double cerebellar_error_vars[NUM_JOINTS*3]={0.0,0.0,0.0}; // error corrective cerebellar output torque
   double cerebellar_learning_vars[NUM_OUTPUT_VARS]={0.0, 0.0}; // Error-related learning signals
  
   // Error-related vars(contruction of the error-base reference)
   double cerebellar_gaussian_poscenters[NUM_REP*NUM_JOINTS*3];
   double cerebellar_gaussian_negcenters[NUM_REP*NUM_JOINTS*3];
   double cerebellar_gaussian_sigmas[NUM_REP*NUM_JOINTS*3];
   double input_error_vars[NUM_JOINTS*3]={0.0,0.0, 0.0};
   double error_vars[NUM_JOINTS]={0.0}; 
   
   // delayed Error-related learning signals
   double *delayed_cerebellar_vars; 
   double *delayed_cerebellar_vars_normalized;
   double *delayed_cerebellar_learning_vars;
   double *delayed_error_vars;
   
   // Simul variables
   Simulation *neural_sim;
   
	// Robot variables
	int n_robot_joints;
   
	// Time variables
	double sim_time,cur_traject_time;
	float slot_elapsed_time,sim_elapsed_time;
	int n_traj_exec;
   
   // Delays
   struct delay cerebellar_delay;
   struct delay cerebellar_learning_delay;
   struct delay cerebellar_delay_normalized;
   struct delay_joint error_delay;

   // Variable for logging the simulation state variables
   struct log var_log;
 

#if defined(REAL_TIME_WINNT)
	// Variables for consumed-CPU-time measurement
	LARGE_INTEGER startt,endt,freq;

#elif defined(REAL_TIME_OSX)
	uint64_t startt, endt, elapsed;
	static mach_timebase_info_data_t freq;
#elif defined(REAL_TIME_LINUX)
	// Calculate time taken by a request - Link with real-time library -lrt
	struct timespec startt, endt, freq;
#endif

#if defined(_DEBUG) && (defined(_WIN32) || defined(_WIN64))
	//   _CrtMemState state0;
	_CrtSetReportMode(_CRT_WARN, _CRTDBG_MODE_FILE);
	_CrtSetReportFile(_CRT_WARN, _CRTDBG_FILE_STDERR);
#endif

#if defined(REAL_TIME_WINNT)
	if(!QueryPerformanceFrequency(&freq))
		puts("QueryPerformanceFrequency failed");
#elif defined (REAL_TIME_LINUX)
	if(clock_getres(CLOCK_REALTIME, &freq))
		puts("clock_getres failed");
#elif defined (REAL_TIME_OSX)
	// If this is the first time we've run, get the timebase.
	// We can use denom == 0 to indicate that sTimebaseInfo is
	// uninitialised because it makes no sense to have a zero
	// denominator is a fraction.
	if (freq.denom == 0 ) {
		(void) mach_timebase_info(&freq);
	}
#endif

	
	
			// Initialize variable log
			if(!(errorn=create_log(&var_log, MAX_TRAJ_EXECUTIONS, TRAJECTORY_TIME)))
			{
				
				// Initialize EDLUT and load neural network files
				neural_sim=create_neural_simulation(NET_FILE, INPUT_WEIGHT_FILE, INPUT_ACTIVITY_FILE, OUTPUT_WEIGHT_FILE, OUTPUT_ACTIVITY_FILE, WEIGHT_SAVE_PERIOD, NUMBER_OF_OPENMP_QUEUES);
				if(neural_sim)
				{
					total_output_spks=0L;
					puts("Simulating...");
					sim_elapsed_time=0.0;
					errorn=0;
//    _CrtMemCheckpoint(&state0);
					bool real_time_neural_simulation=false;
					if(REAL_TIME_NEURAL_SIM==1){
						#ifdef _OPENMP 
							omp_set_nested(true);
							real_time_neural_simulation=true;
							cout<<"\nFixed REAL TIME SIMULATION option\n"<<endl;
							init_real_time_restriction(neural_sim, SIM_SLOT_LENGTH, MAX_SIMULATION_DELAY, FIRST_REAL_TIME_RESTRICTION, SECOND_REAL_TIME_RESTRICTION, THIRD_REAL_TIME_RESTRICTION);
						#else
							cout<<"\nREAL TIME SIMULATION option is not available due to the openMP support is disabled\n"<<endl;
						#endif
					}
					 
					#pragma omp parallel if(real_time_neural_simulation) num_threads(2) 
					{
						if(omp_get_thread_num()==1){
							start_real_time_restriction(neural_sim);
						}else{
							for(n_traj_exec=0;n_traj_exec<MAX_TRAJ_EXECUTIONS && !errorn;n_traj_exec++){
									         
								init_delay(&cerebellar_learning_delay, ERROR_DELAY_TIME);
				
								//////////////////////////////REVISAR ESTE IF///////////////////////////////////////
								if(INPUT_ACTIVITY_FILE==0){
									reset_neural_simulation(neural_sim); // after each trajectory execution the network simulation state must be reset (pending activity events are discarded)
								} 
					 									 
								cur_traject_time=0.0;
					 					
								if(REAL_TIME_NEURAL_SIM==1){
									reset_real_time_restriction(neural_sim);
								}
								do
								{
									int n_joint;
									if(REAL_TIME_NEURAL_SIM==1){
										next_step_real_time_restriction(neural_sim);
									}
									#if defined(REAL_TIME_WINNT)
        								QueryPerformanceCounter(&startt);
									#elif defined(REAL_TIME_LINUX)
        								clock_gettime(CLOCK_REALTIME, &startt);
									#elif defined(REAL_TIME_OSX)
        								startt = mach_absolute_time();
									#endif

								// control loop iteration starts                  
								sim_time=(double)n_traj_exec*TRAJECTORY_TIME + cur_traject_time; // Calculate absolute simulation time
								calculate_input_error_repetitions(input_error_vars, ERROR_AMP,TRAJECTORY_TIME,cur_traject_time,cerebellar_gaussian_poscenters,cerebellar_gaussian_negcenters, cerebellar_gaussian_sigmas,NUM_REP);		
					 			cerebellar_error_vars[0]=cerebellar_output_vars[0]-cerebellar_output_vars[1];//cerebellar_output_vars[0]-cerebellar_output_vars[1]
					 			calculate_error_signals(input_error_vars,cerebellar_error_vars,error_vars); // Calculated robot's performed error
								calculate_learning_signals(error_vars, cerebellar_output_vars, cerebellar_learning_vars); // Calculate learning signal from the calculated error
								delayed_cerebellar_learning_vars=delay_line(&cerebellar_learning_delay,cerebellar_learning_vars);
								generate_learning_activity(neural_sim, sim_time,delayed_cerebellar_learning_vars);  
					  					
									errorn=run_neural_simulation_slot(neural_sim, sim_time+SIM_SLOT_LENGTH); // Simulation the neural network during a time slot
                     
									total_output_spks+=(long)compute_output_activity(neural_sim, cerebellar_output_vars); // Translates cerebellum output activity into analog output variables (corrective torques)
                     
									// control loop iteration ends

#if defined(REAL_TIME_WINNT)
									QueryPerformanceCounter(&endt); // measures time
									slot_elapsed_time=(endt.QuadPart-startt.QuadPart)/(float)freq.QuadPart; // to be logged
#elif defined(REAL_TIME_LINUX)
									clock_gettime(CLOCK_REALTIME, &endt);
									// Calculate time it took
									slot_elapsed_time = (endt.tv_sec-startt.tv_sec ) + (endt.tv_nsec-startt.tv_nsec )/float(1e9);
#elif defined(REAL_TIME_OSX)
									// Stop the clock.
									endt = mach_absolute_time();
									// Calculate the duration.
									elapsed = endt - startt;
									slot_elapsed_time = 1e-9 * elapsed * freq.numer / freq.denom;
#endif
                    
                     
									sim_elapsed_time+=slot_elapsed_time;
									log_vars_reduced(&var_log, sim_time, input_error_vars,cerebellar_output_vars,cerebellar_output_vars,delayed_cerebellar_learning_vars, error_vars, slot_elapsed_time,get_neural_simulation_spike_counter_for_each_slot_time()); // Store vars into RAM
									//log_vars(&var_log, sim_time, input_traject_vars, robot_state_vars, robot_inv_dyn_torque, cerebellar_output_vars, cerebellar_learning_vars, robot_error_vars, slot_elapsed_time,get_neural_simulation_spike_counter_for_each_slot_time()); // Store vars into RAM
									cur_traject_time+=SIM_SLOT_LENGTH;
					 				}
								while(cur_traject_time<TRAJECTORY_TIME-(SIM_SLOT_LENGTH/2.0) && !errorn); // we add -(SIM_SLOT_LENGTH/2.0) because of floating-point-type codification problems
					
							} 
						}
						if(real_time_neural_simulation){
							stop_real_time_restriction(neural_sim);
						}
					}

//     reset_neural_simulation(neural_sim);
//     _CrtMemDumpAllObjectsSince(&state0);
					if(errorn)
						printf("Error %i performing neural network simulation\n",errorn);
					printf("Total neural-network output spikes: %li\n",total_output_spks);
					printf("Total number of neural updates: %Ld\n",get_neural_simulation_event_counter(neural_sim));
					printf("Mean number of neural-network spikes in heap: %f\n",get_accumulated_heap_occupancy_counter(neural_sim)/(double)get_neural_simulation_event_counter(neural_sim));

					long TotalSpikeCounter=0;
					long TotalPropagateCounter=0;
					for(int i=0; i<neural_sim->GetNumberOfQueues(); i++){
						cout << "Thread "<<i<<"--> Number of updates: " << neural_sim->GetSimulationUpdates(i) << endl; /*asdfgf*/
						cout << "Thread "<<i<<"--> Number of InternalSpike: " << neural_sim->GetTotalSpikeCounter(i) << endl; /*asdfgf*/
						cout << "Thread "<<i<<"--> Number of PropagatedEvent: " << neural_sim->GetTotalPropagateCounter(i) << endl; /*asdfgf*/
						cout << "Thread "<<i<<"--> Mean number of spikes in heap: " << neural_sim->GetHeapAcumSize(i)/(float)neural_sim->GetSimulationUpdates(i) << endl; /*asdfgf*/
						TotalSpikeCounter+=neural_sim->GetTotalSpikeCounter(i);
						TotalPropagateCounter+=neural_sim->GetTotalPropagateCounter(i);
					}
					cout << "Total InternalSpike: " << TotalSpikeCounter<<endl; 
					cout << "Total PropagatedEvent: " << TotalPropagateCounter<<endl;

#if defined(REAL_TIME_WINNT)
					printf("Total elapsed time: %fs (time resolution: %fus)\n",sim_elapsed_time,1.0e6/freq.QuadPart);
#elif defined(REAL_TIME_LINUX)
					printf("Total elapsed time: %fs (time resolution: %fus)\n",sim_elapsed_time,freq.tv_sec*1.0e6+freq.tv_nsec/float(1e3));
#elif defined(REAL_TIME_OSX)
					printf("Total elapsed time: %fs (time resolution: %fus)\n",sim_elapsed_time,1e-3*freq.numer/freq.denom);
#endif

					save_neural_weights(neural_sim);
					finish_neural_simulation(neural_sim);
				}
				else
				{
					errorn=10000;
					printf("Error initializing neural network simulation\n");
				}              
				puts("Saving log file");
				errorn=save_and_finish_log(&var_log, LOG_FILE); // Store logged vars in disk
				if(errorn)
					printf("Error %i while saving log file\n",errorn);
			}
			else
			{
				errorn*=1000;
				printf("Error allocating memory for the log of the simulation variables\n");
			}         
       
	 
		if(!errorn)
			puts("OK");
		else
			printf("Error: %i\n",errorn);
#if defined(_DEBUG) && (defined(_WIN32) || defined(_WIN64))
		_CrtDumpMemoryLeaks();
#endif
	return(errorn);
}
