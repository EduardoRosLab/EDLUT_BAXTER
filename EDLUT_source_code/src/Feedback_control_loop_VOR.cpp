 /*******************************************************************************
 *                       Feedback_control_loop.c                                *
 *                       -----------------------                                *
 * copyright            : (C) 2015 by Niceto R. Luque                           *
 * email                :  nluque at ugr.es                                     *
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
 * \date 1 of November 2015
 * In this file the control loop is implemented.
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
#include "../include/vor_model/Individual_VOR.h"
#include "../include/vor_model/EntrySignal_VOR.h"
#include "../include/vor_model/RK4_VOR.h"
#include "../include/openmp/openmp.h"

#include "../include/vor_model/CommunicationAvatarVOR.h"

// Neural-network simulation files
//#define NET_FILE "Network200dcn_aging_2DOF.net"// Neural-network definition file used by EDLUT
//#define INPUT_WEIGHT_FILE "Weight_Network200dcn_aging_2DOF.net"// Neural-network input weight file used by EDLUT
#define NET_FILE "Network200dcn_aging.net"// Neural-network definition file used by EDLUT
#define INPUT_WEIGHT_FILE "Weight_Network200dcn_aging.net"// Neural-network input weight file used by EDLUT

#define OUTPUT_WEIGHT_FILE "OutputWeight.dat" // Neural-network output weight file used by EDLUT
#define WEIGHT_SAVE_PERIOD 300  // The weights will be saved each period (in seconds) (0=weights are not saved periodically)
#define INPUT_ACTIVITY_FILE  "GranularActivity_Cosine_mf_Cosine_GC_1000seg_amplitude_25_25.dat"//Optional input activity file 
#define OUTPUT_ACTIVITY_FILE "OutputActivity.dat" // Output activity file used to register the neural network activity
#define LOG_FILE "vars.dat"  // Log file used to register the simulation variables

#define REAL_TIME_NEURAL_SIM 0 //1 // EDLUT's simulation mode (0=No real-time neural network simulation, 1 = real-time neural network simulation for real robot control)
#define CONNECTION_WITH_AVATAR 0 //1 // EDLUT's simulation mode (0 = no connection with avatar, 1 = connection with avatar (Real time option must be enabled in this case)).

#define Avatar_IP "192.168.10.2"//"127.0.0.1" //Avatar큦 or robot큦 IP
#define Avatar_port 11000 //Avatar큦 or robot큦 TCP port.

#define MAX_SIMULATION_IN_ADVANCE 0.050 //This value must be inferior to ERROR_DELAY_TIME and greater than ROBOT_PROCESSING_DELAY. 
#define ROBOT_PROCESSING_DELAY1 0.014//14 //0.064 //How much time need head motor to proccess a input command.
#define ROBOT_PROCESSING_DELAY2 0.014//14 //0.064 //How much time need eye motors to proccess a input command.
#define ROBOT_PROCESSING_DELAY ((ROBOT_PROCESSING_DELAY1>ROBOT_PROCESSING_DELAY2)?ROBOT_PROCESSING_DELAY1:ROBOT_PROCESSING_DELAY2)

//Real time boundaries.
#define FIRST_REAL_TIME_RESTRICTION (0.7*(MAX_SIMULATION_IN_ADVANCE-ROBOT_PROCESSING_DELAY)/MAX_SIMULATION_IN_ADVANCE)
#define SECOND_REAL_TIME_RESTRICTION (0.8*(MAX_SIMULATION_IN_ADVANCE-ROBOT_PROCESSING_DELAY)/MAX_SIMULATION_IN_ADVANCE)
#define THIRD_REAL_TIME_RESTRICTION (0.9*(MAX_SIMULATION_IN_ADVANCE-ROBOT_PROCESSING_DELAY)/MAX_SIMULATION_IN_ADVANCE)

//Number of OpenMP threads (in how many sections the neural network will be divided).
#define NUMBER_OF_OPENMP_QUEUES 4


#define ERROR_AMP 150.0/(2*3.141592)//25.0 // Amplitude of the injected error
#define ERROR_AMP_VOR_REVERSAL_PHASE 1.0 // Amplitude of the injected error phase reversal process 10% more
#define NUM_REP 1 // Number of repetition of the exponential/sinusoidal shape along the Trajectory Time
#define TRAJECTORY_TIME 1 // Simulation time in seconds required to execute the desired trajectory once
#define MAX_TRAJ_EXECUTIONS 301 //300//2500 // Maximum number of trajectories repetitions that will be executed by the robot
#define ERROR_DELAY_TIME 0.090//0.10000 // Delay after calculating the error vars
#define ROBOT_JOINT_ERROR_NORM 160 // proportional error {160}
#define N_SWITCH 2 // NUMBER OF CYCLES WHERE THE ERROR SIGNAL IS PRESENTED

// VOR PARAMETERS
#define K  1.0//1.375
#define TC1 15.0
#define TC2 0.005
///////////////////////////// MAIN LOOP //////////////////////////

int main(int ac, char *av[])
{

     
   int errorn,i;
   long total_output_spks; 
   double cerebellar_output_vars[NUM_OUTPUT_VARS] = {0.0};  // Corrective cerebellar output torque
   double cerebellar_error_vars[NUM_JOINTS] = {0.0}; // error corrective cerebellar output torque
   double cerebellar_learning_vars[NUM_OUTPUT_VARS] = {0.0}; // Error-related learning signals
   double cerebellar_aux_vars[NUM_JOINTS*2] = { 0.0 };
   double input_aux_vars[NUM_JOINTS*2] = { 0.0 };

   double outer_loop_input_aux_vars[NUM_JOINTS * 2] = { 0.0 };
   double outer_loop_cerebellar_aux_vars[NUM_JOINTS * 2] = { 0.0 };
   double outer_loop_cerebellar_output_vars[NUM_OUTPUT_VARS] = { 0.0 };  // Corrective cerebellar output torque 
   double outer_loop_cerebellar_learning_vars[NUM_OUTPUT_VARS] = { 0.0 };

   // Error-related vars(contruction of the error-base reference)
   double error_vars[NUM_JOINTS] = { 0.0};
   double outer_loop_error_vars[NUM_JOINTS] = { 0.0 };
   
   // delayed Error-related learning signals
    double *delayed_cerebellar_learning_vars;



	   
   // Simul variables
   Simulation *neural_sim;
   
	// Robot variables
	int n_robot_joints;
   
	// Time variables
	double sim_time,cur_traject_time;
	float slot_elapsed_time,sim_elapsed_time;
	int n_traj_exec;
   
   // Delays
   /*struct delay cerebellar_delay;*/
   struct delay cerebellar_learning_delay;
   /*struct delay cerebellar_delay_normalized;*/
   struct delay_joint input_delay;
   
   // VOR PLAN DEFINITIONS

  	//Entry signal 
   EntrySignalVOR ** signal = (EntrySignalVOR**) new EntrySignalVOR*[NUM_JOINTS];
	
	//VOR.
   Individual ** VOR_PLAN = (Individual**) new Individual*[NUM_JOINTS];
	
	// Integrator procedure 
   RK4_VOR ** Integrator = (RK4_VOR**) new RK4_VOR*[NUM_JOINTS];

	for (int t = 0; t < NUM_JOINTS; t++){
		signal[t]=new EntrySignalVOR();
		VOR_PLAN[t]=new Individual(K, TC1, TC2);
		Integrator[t]=new RK4_VOR();
	}

	


	// Variable for logging the simulation state variables
   struct log_vor var_log_vor;
   
   //Object used for the communication with the avatar.
   CommunicationAvatarVOR * communicationAvatarVOR = new CommunicationAvatarVOR();    


#if defined(REAL_TIME_WINNT)
	// Variables for consumed-CPU-time measurement
   LARGE_INTEGER startt, endt, freq, totalstartt, totalendt;

#elif defined(REAL_TIME_OSX)
   uint64_t startt, endt, elapsed, totalstartt, totalendt;
	static mach_timebase_info_data_t freq;
#elif defined(REAL_TIME_LINUX)
	// Calculate time taken by a request - Link with real-time library -lrt
   struct timespec startt, endt, freq, totalstartt, totalendt;
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

	
	
			// Initialize variable log vor
			if(!(errorn=create_log_vor(&var_log_vor, MAX_TRAJ_EXECUTIONS, TRAJECTORY_TIME)))
			{
				
				// Initialize EDLUT and load neural network files
				neural_sim=create_neural_simulation(NET_FILE, INPUT_WEIGHT_FILE, INPUT_ACTIVITY_FILE, OUTPUT_WEIGHT_FILE, OUTPUT_ACTIVITY_FILE, WEIGHT_SAVE_PERIOD, NUMBER_OF_OPENMP_QUEUES);
				if(neural_sim)
				{
					total_output_spks=0L;
					puts("Simulating...");
					errorn=0;

					//Auxiliar variables for real time, connection with avatar or robot and number of openMP threads
					bool real_time_neural_simulation=false;
					bool avatar_connection = false;
					int N_threads = 1;//Just one thread by default for the simulation of the neural network.

					//Check if the simulation need to implement additional OpenMP thread to compute a real time watch dog or a communication interface.
					if (REAL_TIME_NEURAL_SIM == 1 || CONNECTION_WITH_AVATAR == 1){
						//This options need the OpenMP language to be enabled.
						#ifdef _OPENMP 
							//Enable nested OpenMP threads. We can creat three OpenMP thread: one for real time watchdog, one for a communication interface and one for
							//the neural network. This last thread can at after generate additional OpenMP threads. For that reason we need to enable the nested option.
							omp_set_nested(true);
							//Check if the connection with the avatar or robot must be enabled. In this case the real time option must be enabled by default.
							if(CONNECTION_WITH_AVATAR == 1){
								N_threads += 2; //one thread for real time watchdog, one for a communication interface and one for the neural network.
								avatar_connection= true;//Enable the connection with the avatar or the robot
								cout<<"\nENABLED CONNECTION WITH AVATAR\n"<<endl;

								real_time_neural_simulation=true;//Enable the real time option
								cout<<"\nENABLED STRICT REAL TIME SIMULATION FOR AVATAR COMMUNICATION\n"<<endl;
								//Set the real time parameters.
								enable_real_time_restriction(neural_sim, SIM_SLOT_LENGTH, MAX_SIMULATION_IN_ADVANCE, FIRST_REAL_TIME_RESTRICTION, SECOND_REAL_TIME_RESTRICTION, THIRD_REAL_TIME_RESTRICTION);
							}
							//Check if the real time option must be enabled
							else if(REAL_TIME_NEURAL_SIM == 1){
								N_threads++; //one thread for real time watchdog and one for the neural network.
								real_time_neural_simulation=true;//Enable the real time watchdog.
								cout << "\nENABLED REAL TIME SIMULATION\n" << endl;
								//Set the real time parameters.
								enable_real_time_restriction(neural_sim, SIM_SLOT_LENGTH, MAX_SIMULATION_IN_ADVANCE, FIRST_REAL_TIME_RESTRICTION, SECOND_REAL_TIME_RESTRICTION, THIRD_REAL_TIME_RESTRICTION);
							}

							
						#else
							//The OpenMP support is disabled and this options cannot be performed.
							if (REAL_TIME_NEURAL_SIM == 1){
								cout << "\nREAL TIME SIMULATION option is not available due to the openMP support is disabled\n" << endl;
							}
							if (CONNECTION_WITH_AVATAR == 1){
								cout << "\nCONNECTION WITH AVATAR is not available due to the openMP support is disabled\n" << endl;
							}
						#endif
					}
					
					//Generate a pool of "N_threads" thread. 
					#pragma omp parallel if(N_threads>1) num_threads(N_threads) 
					{
						//Check if the connection with the avatar is enabled
						if (avatar_connection){
							//Just one thread of the pool compute the communication interface (note: the nowait option is enabled to allow the other thread to perform other task at the same time). 
							#pragma omp single nowait
							{
								printf("Waiting for connection with avatar interface\n");
								//Indexs used in the communication interface to know with value must be sended to the avatar taking into account the motor delays.
								int robot_processing_delay1 = ROBOT_PROCESSING_DELAY1 / SIM_SLOT_LENGTH;
								int robot_processing_delay2 = ROBOT_PROCESSING_DELAY2 / SIM_SLOT_LENGTH;
								//Start the communication interface.
								enable_and_start_connection_with_avatar_in_seconday_thread(neural_sim, communicationAvatarVOR, &var_log_vor, Avatar_IP, Avatar_port, robot_processing_delay1, robot_processing_delay2);
							}
							//The other threads wait here until the connection is established.
							while (!communicationAvatarVOR->GetConnectionStarted());
						}

						//Check if the real time option is enabled (note: the nowait option is enabled to allow the other thread to perform other task at the same time).
						if (real_time_neural_simulation){
							//Just one thread of the pool compute the real time watchdog.
							#pragma omp single nowait
							{
								enable_real_time_restriction_in_secondary_thread(neural_sim);
							}
						}

						//The last thread perform the neural network (note: the nowait option is enabled to allow the other thread to perform other task at the same time). 
						#pragma omp single nowait
						{
							//Init the structures to delay the teaching signals in the control loop.
							init_delay(&cerebellar_learning_delay, ERROR_DELAY_TIME);
							init_delay_joint(&input_delay, ERROR_DELAY_TIME);

							//If the real time option is enabled, we reset the realtime controler before start with the task.
							if (real_time_neural_simulation){
								start_real_time_restriction(neural_sim);
							}

							//Get a time stamp to calculate the time spent in compute the whole simulation.
							#if defined(REAL_TIME_WINNT)
								QueryPerformanceCounter(&totalstartt);
							#elif defined(REAL_TIME_LINUX)
								clock_gettime(CLOCK_REALTIME, &totalstartt);
							#elif defined(REAL_TIME_OSX)
								totalstartt = mach_absolute_time();
							#endif

							//JUST INNER LOOP (THE CONNECTION WITH THE AVATAR IS DISABLED)
							if (!avatar_connection){
								//A loop for all the trajectories that must be performed.
								for (n_traj_exec = 0; n_traj_exec < MAX_TRAJ_EXECUTIONS && !errorn; n_traj_exec++){
									//Set the initial time of each trajectory of zero.
									cur_traject_time = 0.0;
									do
									{
										//If the real time option is enabled, this thread must indicate to the watchdog thread the simulation evolution (the next simulation slot
										//of size SIM_SLOT_LENGTH is going to be performed).
										if (real_time_neural_simulation){
											next_step_real_time_restriction(neural_sim);
										}

										//Get a time stamp to calculate the time spent in compute this simulation slot of size SIM_SLOT_LENGTH.
										#if defined(REAL_TIME_WINNT)
											QueryPerformanceCounter(&startt);
										#elif defined(REAL_TIME_LINUX)
											clock_gettime(CLOCK_REALTIME, &startt);
										#elif defined(REAL_TIME_OSX)
											startt = mach_absolute_time();
										#endif

										// control loop iteration starts                  
										sim_time = (double)n_traj_exec*TRAJECTORY_TIME + cur_traject_time; // Calculate absolute simulation time
										//Initializing eye plant

										for (int n_joint = 0; n_joint < NUM_JOINTS; n_joint++){
											delete signal[n_joint];
											//In the first second the robot does not move.
											if (n_traj_exec > 0){
												//NOTE: WE INCLUDE THE DESPHASE 0.25 TO SYNCHRONIZE THIS SIGNAL WITH THE ACTIVITY IN THE GRANULLE CELLS.
												signal[n_joint] = new EntrySignalVOR(0.002, cur_traject_time/* + 0.25*/, ERROR_AMP, NUM_REP, 0, 1);
											}
											else{
												//NOTE: WE INCLUDE THE DESPHASE 0.25 TO SYNCHRONIZE THIS SIGNAL WITH THE ACTIVITY IN THE GRANULLE CELLS.
												signal[n_joint] = new EntrySignalVOR(0.002, 0/* + 0.25*/, ERROR_AMP, NUM_REP, 0, 1);
											}
											cerebellar_error_vars[n_joint] = cerebellar_output_vars[2 * (n_joint)] - cerebellar_output_vars[2 * (n_joint)+1];//agonist-antagonist cerebellar_output_vars
											VOR_PLAN[n_joint]->CalculateOutputVOR(signal[n_joint], Integrator[n_joint], &cerebellar_error_vars[n_joint], &cerebellar_aux_vars[NUM_JOINTS + n_joint]);//VOR output     --> eye velocity
											input_aux_vars[n_joint] = signal[n_joint]->GetSignalVOR(0); //head position
											input_aux_vars[NUM_JOINTS + n_joint] = signal[n_joint]->GetSignalVOR(1); //head velocity
											cerebellar_aux_vars[n_joint] = cerebellar_aux_vars[n_joint] + cerebellar_aux_vars[NUM_JOINTS + n_joint] * SIM_SLOT_LENGTH; //eye position calculated using the integration of the eye velocity
										}


										calculate_error_signals(input_aux_vars, cerebellar_aux_vars, error_vars);// Calculated EYE's performed error

										calculate_learning_signals(error_vars, cerebellar_output_vars, cerebellar_learning_vars); // Calculate learning signal from the calculated error
										delayed_cerebellar_learning_vars = delay_line(&cerebellar_learning_delay, cerebellar_learning_vars);

										generate_learning_activity(neural_sim, sim_time, delayed_cerebellar_learning_vars);

										errorn = run_neural_simulation_slot(neural_sim, sim_time + SIM_SLOT_LENGTH); // Simulation the neural network during a time slot

										total_output_spks += (long)compute_output_activity(neural_sim, cerebellar_output_vars); // Translates cerebellum output activity into analog output variables (corrective torques)

										// control loop iteration ends

#if defined(REAL_TIME_WINNT)
										QueryPerformanceCounter(&endt); // measures time
										slot_elapsed_time = (endt.QuadPart - startt.QuadPart) / (float)freq.QuadPart; // to be logged
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

										// LOGING THE VARIABLES
										/* Inputs
										//&var_log									-- STRUCT STORING ALL THE DATA
										//sim_time									-- SIMULATION TIME
										//input_aux_vars							-- VOR HEAD MOVEMENTS (DIMENSION = NJOINTS)
										//cerebellar_aux_vars						-- VOR OUTPUTS (DIMENSION = NJOINTS)
										//cerebellar_output_vars					-- CEREBELLAR OUTPUTS == VOR INPUTS, AGONIST AND ANTANGONIST VALUES (DIMENSION NJOINTS*2)
										//delayed_cerebellar_learning_vars			-- LEARNING SIGNAL TOWARDS THE CEREBELLUM DELAYED  (DIMENSION NJOINTS*2)
										//error_vars						        -- EYE'S PERFORMANCE ERROR (DIMENSION NJOINTS)
										// slot_elapsed_time						-- slot time
										//get_neural_simulation_spike_counter_for_each_slot_time() -- number of spikes
										*/

										log_vars_vor(&var_log_vor, sim_time, input_aux_vars, cerebellar_aux_vars, cerebellar_output_vars, cerebellar_learning_vars, error_vars, slot_elapsed_time, get_neural_simulation_spike_counter_for_each_slot_time()); // Store vars into RAM

										cur_traject_time += SIM_SLOT_LENGTH;
									} while (cur_traject_time < TRAJECTORY_TIME - (SIM_SLOT_LENGTH / 2.0) && !errorn); // we add -(SIM_SLOT_LENGTH/2.0) because of floating-point-type codification problems
	
									//Calculate the position MAE for the last trajectory
									float MAE = calculate_avatar_MAE(var_log_vor, n_traj_exec * 500, n_traj_exec * 500 + 499);
									printf("Normalized MAE for trajectory %d: %f\n", n_traj_exec, MAE);
								}
							}
							// INNER AND OUTER LOOPS (THE CONNECTION WITH THE AVATAR IS ENABLED)
							else{
								int last_outer_loop_index = 0;//index of first outer loop value that must be processed

								//A loop for all the trajectories that must be performed.
								for (n_traj_exec = 0; n_traj_exec < MAX_TRAJ_EXECUTIONS && !errorn; n_traj_exec++){
									//Set the initial time of each trajectory of zero.
									cur_traject_time = 0.0;
									do
									{
										//If the real time option is enabled, this thread must indicate to the watchdog thread the simulation evolution (the next simulation slot
										//of size SIM_SLOT_LENGTH is going to be performed).
										if (real_time_neural_simulation){
											next_step_real_time_restriction(neural_sim);
										}

										//Get a time stamp to calculate the time spent in compute this simulation slot of size SIM_SLOT_LENGTH.
										#if defined(REAL_TIME_WINNT)
											QueryPerformanceCounter(&startt);
										#elif defined(REAL_TIME_LINUX)
											clock_gettime(CLOCK_REALTIME, &startt);
										#elif defined(REAL_TIME_OSX)
											startt = mach_absolute_time();
										#endif

										// control loop iteration starts                  
										sim_time = (double)n_traj_exec*TRAJECTORY_TIME + cur_traject_time; // Calculate absolute simulation time
										//Initializing eye plant


										for (int n_joint = 0; n_joint < NUM_JOINTS; n_joint++){
											delete signal[n_joint];
											//The first second the robot does not move.
//											if (n_traj_exec > 0){
												//NOTE: WE INCLUDE THE DESPHASE 0.25 TO SYNCHRONIZE THIS SIGNAL WITH THE ACTIVITY IN THE GRANULLE CELLS.
												signal[n_joint] = new EntrySignalVOR(0.002, cur_traject_time/* + 0.25*/, ERROR_AMP, NUM_REP, 0, 1);
//											}
//											else{
												//NOTE: WE INCLUDE THE DESPHASE 0.25 TO SYNCHRONIZE THIS SIGNAL WITH THE ACTIVITY IN THE GRANULLE CELLS.
//												signal[n_joint] = new EntrySignalVOR(0.002, 0/* + 0.25*/, ERROR_AMP, NUM_REP, 0, 1);
//											}
											cerebellar_error_vars[n_joint] = cerebellar_output_vars[2 * (n_joint)] - cerebellar_output_vars[2 * (n_joint)+1];//agonist-antagonist cerebellar_output_vars
											VOR_PLAN[n_joint]->CalculateOutputVOR(signal[n_joint], Integrator[n_joint], &cerebellar_error_vars[n_joint], &cerebellar_aux_vars[NUM_JOINTS + n_joint]);//VOR output     --> eye velocity
											input_aux_vars[n_joint] = signal[n_joint]->GetSignalVOR(0); //head position
											input_aux_vars[NUM_JOINTS + n_joint] = signal[n_joint]->GetSignalVOR(1); //head velocity
											cerebellar_aux_vars[n_joint] = cerebellar_aux_vars[n_joint] + cerebellar_aux_vars[NUM_JOINTS + n_joint] * SIM_SLOT_LENGTH; //eye position calculated using the integration of the eye velocity
										}

										errorn = run_neural_simulation_slot(neural_sim, sim_time + SIM_SLOT_LENGTH); // Simulation the neural network during a time slot

										total_output_spks += (long)compute_output_activity(neural_sim, cerebellar_output_vars); // Translates cerebellum output activity into analog output variables (corrective torques)

										// control loop iteration ends

#if defined(REAL_TIME_WINNT)
										QueryPerformanceCounter(&endt); // measures time
										slot_elapsed_time = (endt.QuadPart - startt.QuadPart) / (float)freq.QuadPart; // to be logged
#elif defined(REAL_TIME_LINUX)
										clock_gettime(CLOCK_REALTIME, &endt);
										// Calculate time it took
										slot_elapsed_time = (endt.tv_sec - startt.tv_sec) + (endt.tv_nsec - startt.tv_nsec) / float(1e9);
#elif defined(REAL_TIME_OSX)
										// Stop the clock.
										endt = mach_absolute_time();
										// Calculate the duration.
										elapsed = endt - startt;
										slot_elapsed_time = 1e-9 * elapsed * freq.numer / freq.denom;
#endif

										// LOGING THE VARIABLES
										/* Inputs
										//&var_log									-- STRUCT STORING ALL THE DATA
										//sim_time									-- SIMULATION TIME
										//input_aux_vars							-- VOR HEAD MOVEMENTS (DIMENSION = NJOINTS)
										//cerebellar_aux_vars						-- VOR OUTPUTS (DIMENSION = NJOINTS)
										//cerebellar_output_vars					-- CEREBELLAR OUTPUTS == VOR INPUTS, AGONIST AND ANTANGONIST VALUES (DIMENSION NJOINTS*2)
										//delayed_cerebellar_learning_vars			-- LEARNING SIGNAL TOWARDS THE CEREBELLUM DELAYED  (DIMENSION NJOINTS*2)
										//error_vars						        -- EYE'S PERFORMANCE ERROR (DIMENSION NJOINTS)
										// slot_elapsed_time						-- slot time
										//get_neural_simulation_spike_counter_for_each_slot_time() -- number of spikes
										*/


										log_vars_vor_first_etage_inner_outer_loop(&var_log_vor, sim_time, input_aux_vars, cerebellar_aux_vars, cerebellar_output_vars, slot_elapsed_time, get_neural_simulation_spike_counter_for_each_slot_time()); // Store vars into RAM


										cur_traject_time += SIM_SLOT_LENGTH;
										//Check the outer values
										while (last_outer_loop_index < var_log_vor.outer_loop_index && last_outer_loop_index  < var_log_vor.nregs){
											int nvar;
											for (nvar = 0; nvar < NUM_JOINTS * 2; nvar++){
												////MIXED
												//if (nvar % 2 == 0){
													outer_loop_input_aux_vars[nvar] = (var_log_vor.regs[last_outer_loop_index].vor_head_vars[nvar] + var_log_vor.regs[last_outer_loop_index].avatar_head_vars[nvar])*0.5f;
													outer_loop_cerebellar_aux_vars[nvar] = (var_log_vor.regs[last_outer_loop_index].vor_output_vars[nvar] + var_log_vor.regs[last_outer_loop_index].avatar_output_vars[nvar])*0.5f;
												//}
												//else{
													////ONLY INNER
													//outer_loop_input_aux_vars[nvar] = var_log_vor.regs[last_outer_loop_index].vor_head_vars[nvar];
													//outer_loop_cerebellar_aux_vars[nvar] = var_log_vor.regs[last_outer_loop_index].vor_output_vars[nvar];
												//}
												//ONLY OUTER
												//outer_loop_input_aux_vars[nvar] = var_log_vor.regs[last_outer_loop_index].avatar_head_vars[nvar];
												//outer_loop_cerebellar_aux_vars[nvar] = var_log_vor.regs[last_outer_loop_index].avatar_output_vars[nvar];


												outer_loop_cerebellar_output_vars[nvar] = var_log_vor.regs[last_outer_loop_index].cereb_output_vars[nvar];
											}

											calculate_error_signals(outer_loop_input_aux_vars, outer_loop_cerebellar_aux_vars, outer_loop_error_vars);// Calculated EYE's performed error

											calculate_learning_signals(outer_loop_error_vars, outer_loop_cerebellar_output_vars, outer_loop_cerebellar_learning_vars); // Calculate learning signal from the calculated error

											if (last_outer_loop_index * SIM_SLOT_LENGTH + ERROR_DELAY_TIME > cur_traject_time){
												generate_learning_activity(neural_sim, last_outer_loop_index * SIM_SLOT_LENGTH + ERROR_DELAY_TIME, outer_loop_cerebellar_learning_vars);
											}

											log_vars_vor_second_etage_inner_outer_loop(&var_log_vor, last_outer_loop_index, outer_loop_cerebellar_learning_vars, outer_loop_error_vars); // Store vars into RAM

											last_outer_loop_index++;

											//Calculate the position MAE for the last trajectory
											if (last_outer_loop_index % 500 == 0){
												float MAE = calculate_avatar_MAE(var_log_vor, last_outer_loop_index - 500, last_outer_loop_index - 1);
												printf("Normalized MAE for trajectory %d: %f\n", last_outer_loop_index / 500 , MAE);
											}
										}


									} while (cur_traject_time < TRAJECTORY_TIME - (SIM_SLOT_LENGTH / 2.0) && !errorn); // we add -(SIM_SLOT_LENGTH/2.0) because of floating-point-type codification problems
								}
							}

#if defined(REAL_TIME_WINNT)
							QueryPerformanceCounter(&totalendt); // measures time
							sim_elapsed_time = (totalendt.QuadPart - totalstartt.QuadPart) / (float)freq.QuadPart; // to be logged
#elif defined(REAL_TIME_LINUX)
							clock_gettime(CLOCK_REALTIME, &totalendt);
							// Calculate time it took
							sim_elapsed_time = (totalendt.tv_sec - totalstartt.tv_sec) + (totalendt.tv_nsec - totalstartt.tv_nsec) / float(1e9);
#elif defined(REAL_TIME_OSX)
							// Stop the clock.
							totalendt = mach_absolute_time();
							// Calculate the duration.
							elapsed = totalendt - totalstartt;
							sim_elapsed_time = 1e-9 * elapsed * freq.numer / freq.denom;
#endif

							if (real_time_neural_simulation){
								stop_real_time_restriction(neural_sim);
							}if (avatar_connection){
								stop_connection_with_avatar(communicationAvatarVOR);
							}
						}
					}

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
				errorn=save_and_finish_log_vor(&var_log_vor, LOG_FILE); // Store logged vars in disk
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
