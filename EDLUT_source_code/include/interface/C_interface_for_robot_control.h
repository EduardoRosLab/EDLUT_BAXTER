/***************************************************************************
 *                    C_interface_for_robot_control.h                      *
 *                    -------------------------------                      *
 * copyright        : (C) 2013 by Richard R. Carrillo and Niceto R. Luque  *
 * email            : rcarrillo at ugr.es                                  *
 ***************************************************************************/

/***************************************************************************
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 ***************************************************************************/

#ifndef _EDLUT_INTERFACE_H_
#define _EDLUT_INTERFACE_H_


//Number of internal spikes in each slot time.
extern long N_one_slot_internal_spikes2;

/*!
 * \file C_interface_for_robot_control.h
 *
 * \author Richard R. Carrillo
 * \author Niceto R. Luque
 * \date 7 of November 2013
 *
 * This file declares the interface functions to access EDLUT's functionality
 * for robot control.
 */
 
#ifndef EXTERN_C
#  ifdef __cplusplus
#    define EXTERN_C extern "C"
#  else
#    define EXTERN_C extern
#  endif
#endif


#ifdef __cplusplus
#include "../../include/simulation/Simulation.h"
class CommunicationAvatarVOR;
//class Simulation;
#else // incomplete typedef for Simulation
typedef struct Simulation_tag Simulation;
#endif

/// \brief Duration of a inner control loop iteration (time slot)
/// A smaller length provides a higher accuracy
#define SIM_SLOT_LENGTH 0.002
///// \brief Simulation time in seconds required to execute the desired trajectory once
#define TRAJECTORY_TIME 1 

/// \brief Approximate maximum acceptable time which can be consunsed by EDLUT in each slot
/// Used in real time only
#define MAX_SIM_SLOT_CONSUMED_TIME (SIM_SLOT_LENGTH*0.5)

/// \brief Maximum delay time that a delay line can have in seconds
/// (The particular delay of each line is specified when calling init_delay())
#define MAX_DELAY_TIME 0.090

/// \brief Number of used robot's joints
/// (depends on the number of joints moved during the performed trajectory)
#define NUM_JOINTS 1

/// \brief Number of RBFs to encode the trajectory in the cerebellum's mossy fibers
/// (the correponsing distribution mathematical expression can be printed with printRBFs())
#define NUM_RBFS 20
/// \brief Number of cerebellum's input variables for the desired trajectory (encoded by mossy fibers)
/// (usually 2 (position and velocity) per joint)
#define NUM_TRAJECTORY_INPUT_VARS (NUM_JOINTS*2) // ECEA
/// \brief Number of cerebellum's input variables for the robot's state (encoded by mossy fibers)
/// (usually 2 (position and velocity) per joint)
#define NUM_ROBOT_STATE_INPUT_VARS (NUM_JOINTS*2) // ICEA
/// \brief Number of cerebellum's mossy fibers used to encode the cerebellar trajectory input variables
/// (depends on NUM_INPUT_VARS and NUM_RBFS)
#define NUM_TRAJECTORY_INPUT_NEURONS (NUM_TRAJECTORY_INPUT_VARS*NUM_RBFS)
/// \brief Number of cerebellum's mossy fibers used to encode the cerebellar robot state variables
/// (depends on NUM_INPUT_VARS and NUM_RBFS)
#define NUM_ROBOT_STATE_INPUT_NEURONS (NUM_ROBOT_STATE_INPUT_VARS*NUM_RBFS)
/// \brief Number of the first network neuron which correspong to a mossy fiber
/// (depends on the neural network definition: EDLUT net input file)
#define FIRST_INPUT_NEURON 0

/// \brief Number of cerebellum's output variables (encoded by DNC cells)
/// (usually 2 (positive and negative torque) per joint)
#define NUM_OUTPUT_VARS (NUM_JOINTS*2)
/// \brief number of cerebellum's DCN cells used to encode one cerebellar output variable
/// (depends on the neural network definition)
#define NUM_NEURONS_PER_OUTPUT_VAR 100
/// \brief Total number of cerebellum's DCN cells used to encode the cerebellar output variables
/// (depends on NUM_OUTPUT_VARS and NUM_NEURONS_PER_OUTPUT_VAR)
#define NUM_OUTPUT_NEURONS (NUM_OUTPUT_VARS*NUM_NEURONS_PER_OUTPUT_VAR)
/// \brief Number of the first network neuron which correspong to a DCN cell
/// (depends on the neural network definition: EDLUT net input file)
#define FIRST_OUTPUT_NEURON (2100 + 1 * NUM_JOINTS * 200) //
/// \brief Number of the first network neuron which correspong to a climbing fiber
/// (depends on the neural network definition: EDLUT net input file)
#define FIRST_LEARNING_NEURON (2100 + 3 * NUM_JOINTS * 200) //
/// \brief Number of cerebellum's climbing fibers which correspond to one cerebellar output variable
/// (depends on the neural network definition)
#define NUM_LEARNING_NEURONS_PER_OUTPUT_VAR 100  //
/// \brief Total number of cerebellum's climbing fibers
/// (depends on NUM_JOINTS and NUM_LEARNING_NEURONS_PER_OUTPUT_VAR)
#define NUM_LEARNING_NEURONS (NUM_JOINTS*2*NUM_LEARNING_NEURONS_PER_OUTPUT_VAR)

/// \brief Maximum applicable torque for each robot's joint (in N/m)
/// (used to calculate the performed trajectory error)
//static const double MAX_ROBOT_JOINT_TORQUE[NUM_JOINTS]={1.0};//{1.0,1.0};//{1,1,1}
static const double MAX_ROBOT_JOINT_TORQUE[2] = { 1.0, 1.0 };//{1.0,1.0};//{1,1,1}

/// \brief proportional and derivative error factor for each robot's joint
/// (used to calculate the performed trajectory error and therefore the corresponding activity of climbing fibers)
//static const double ROBOT_JOINT_ERROR_KP[NUM_JOINTS] ={1.0};//{1.0,1.0};// {10.0*36,250.0*6,250.0*12};//{1,1,1};// proportional error
static const double ROBOT_JOINT_ERROR_KP[2] = { 0.5,0.5 };//{1.0,1.0};// {10.0*36,250.0*6,250.0*12};//{1,1,1};// proportional error
//static const double ROBOT_JOINT_ERROR_KD[NUM_JOINTS] ={0.0,};//{0.0,0.0};//{23.0*36,23.0*6*5,23.0*12*5};//{0,0,0}//{125*2,125,125/2};//23.0*12}; // derivative error;
static const double ROBOT_JOINT_ERROR_KD[2] = { 0.5/(2*3.141592),0.5/(2*3.141592) };//{0.0,0.0};//{23.0*36,23.0*6*5,23.0*12*5};//{0,0,0}//{125*2,125,125/2};//23.0*12}; // derivative error;

/// \brief A structure definiton used to specify radial basis function (RBF) shape
/// (used to encode input variables in mossy fibers)
struct rbf_set
  {
   int num_rbfs;
   double bell_amp;
   double bell_overlap;
   double first_bell_pos, last_bell_pos;
  };


///////////////////////////// SIMULATION MANAGEMENT //////////////////////////

/// \brief Creates and initializes a neural network simulation.
/// In the end, the simulation must be finished (by calling finish_neural_simulation())
/// \param net_file A pointer to the neural-network definition file name to load
/// \param input_weight_file A pointer to the network weight file name 
/// \param input_spike_file A pointer to the neural activity input file name (use NULL in order no to load an activity file)
/// \param output_weight_file A pointer to the weight output file name (use NULL in order no to save network weights)
/// \param output_spike_file A pointer to the output neural activity file name (use NULL in order no to save neural activity)
/// \param weight_save_period Time interval used to save network weights in \a output_weight_file
/// \param real_time_simulation Indicates if the simulation must be performed in real time (1=real time, 0=normal simulation)
/// \return A pointer to the created neural network
EXTERN_C Simulation *create_neural_simulation(const char *net_file, const char *input_weight_file, const char *input_spike_file, const char *output_weight_file, const char *output_spike_file, double weight_save_period, int number_of_openmp_queues);

/// \brief Finishes and deletes a neural network simulation.
/// Neural simulation output data is stored in the previously-specified files
/// \param neural_sim Pointer to a Simulation created by create_neural_simulation()
EXTERN_C void finish_neural_simulation(Simulation *neural_sim);

/// \brief Performs a single network simulation time slot.
/// The neural network is simulated since the last simulated time till \a slot_end_time
/// \param neural_sim Pointer to a Simulation created by create_neural_simulation()
/// \param slot_end_time last time of the time slot to be simulated
/// \return Error occurred during the function execution (0 if it is successfully executed)
/// \note If the real_time_simulation parameter is set to 1 (true) when the neural simulation
/// \note was crated (create_neural_simulation function), the processing of some events can
/// \note be skipped in order not to exceed the computation time limit in a time slot (SIM_SLOT_LENGTH)
EXTERN_C int run_neural_simulation_slot(Simulation *neural_sim, double next_step_sim_time);

/// \brief Saves the neural network weights.
/// All the neural network weights are save in a text file which name is specified when creating the network.
/// \param neural_sim Pointer to a Simulation created by create_neural_simulation()
/// \pre Weights must be saved before finishing the simulation
EXTERN_C void save_neural_weights(Simulation *neural_sim); 

/// \brief Resets the neural network simulation
/// The neural activity remaining in the network from the last slot simulation is discarded.
/// This function is called at the beginning of a new trajectory execution.
/// \param neural_sim Pointer to a Simulation created by create_neural_simulation()
EXTERN_C void reset_neural_simulation(Simulation *neural_sim);

/// \brief Returns the number of spikes processed by the neural network
/// This value is related to the consumed computation time
/// \param neural_sim Pointer to a Simulation created by create_neural_simulation()
/// \return Number of processed spikes
EXTERN_C long get_neural_simulation_spike_counter_for_each_slot_time();

/// \brief Returns the number of events processed by the neural network
/// This value is used for statistics and to obtain information about the simulation
/// \param neural_sim Pointer to a Simulation created by create_neural_simulation()
/// \return Number of processed events
EXTERN_C long long get_neural_simulation_event_counter(Simulation *neural_sim);

/// \brief Returns the total number of events processed by the simulator
/// This value is used for statistics and to obtain information about the simulation
/// \param neural_sim Pointer to a Simulation created by create_neural_simulation()
/// \return Total number of processed events
EXTERN_C long long get_accumulated_heap_occupancy_counter(Simulation *neural_sim);

///////////////////////////// DELAY LINES FOR THE CONTROL LOOP //////////////////////////

/// \brief The structure of a delay line
///  A pointer to a structure of this type is passed as parameter to all the delay functions
struct delay
  {
   double buffer[(int)(MAX_DELAY_TIME/SIM_SLOT_LENGTH+1.5)][NUM_OUTPUT_VARS];//[NUM_JOINTS]; // Circular buffer
   int length; // the useful length of the line is length-1
   // This index points to the place where the new element will be stored.
   // index+1 is the oldest element in the buffer
   int index;
  };

/// \brief The structure of a delay line per joint
///  A pointer to a structure of this type is passed as parameter to all the delay functions
struct delay_joint
  {
   double buffer_joint[(int)(MAX_DELAY_TIME/SIM_SLOT_LENGTH+1.5)][NUM_JOINTS];//[NUM_JOINTS]; // Circular buffer
   int length_joint; // the useful length of the line is length-1
   // This index points to the place where the new element will be stored.
   // index+1 is the oldest element in the buffer
   int index_joint;
  };

/// \brief Initializes and clear a delay line.
/// A specificed delay line is initialized to a particular length and cleared
/// (values set to 0).
/// \param d Pointer to the delay structure to be init.
/// \param del_time Delay introduced by the line in seconds.
/// \pre The \a d strucure must be previously allocated.
EXTERN_C void init_delay(struct delay *d, double del_time);

/// \brief Inserts a new element in the delay line and get the oldest element.
/// The delay line is shifted one position
/// \param d Pointer to the delay structure to be shifed.
/// \param elem Pointer to the new elem to be inserted in the line.
///  Each delay line element is a group of NUM_OUTPUT_VARS] doubles, therefore \a elem must
///  point to at least NUM_JOINT doubles.
/// \return Pointer to the oldest element group if the line.
/// \post The returned value points to NUM_OUTPUT_VARS] doubles which are allocated in the
///  delay-line buffer. These values can be used until this function is called again.
EXTERN_C double *delay_line(struct delay *d, double *elem);


/// \brief Initializes and clear a delay line.
/// A specificed delay line is initialized to a particular length and cleared
/// (values set to 0).
/// \param d Pointer to the delay structure to be init.
/// \param del_time Delay introduced by the line in seconds.
/// \pre The \a d strucure must be previously allocated.
EXTERN_C void init_delay_joint(struct delay_joint *d, double del_time);

/// \brief Inserts a new element in the delay line and get the oldest element.
/// The delay line is shifted one position
/// \param d Pointer to the delay structure to be shifed.
/// \param elem Pointer to the new elem to be inserted in the line.
///  Each delay line element is a group of NUM_JOINT doubles, therefore \a elem must
///  point to at least NUM_JOINT doubles.
/// \return Pointer to the oldest element group if the line.
/// \post The returned value points to NUM_JOINT doubles which are allocated in the
///  delay-line buffer. These values can be used until this function is called again.
EXTERN_C double *delay_line_joint(struct delay_joint *d, double *elem);

///////////////////////////// INPUT TRAJECTORY //////////////////////////

/// \brief Calculates the value of a Gaussian function
/// The value is calculated accoring to the expression: f(x) = a e^{- { \frac{(x-b)^2 }{ 2 c^2} } }
/// This value is used to calculate the RBFs shape and therefore the current injected in
///  mossy fibers depending on the input variables.
/// \param a Amplitide
/// \param b Center
/// \param c Width
/// \param x Position
/// \return Gaussian function value
double gaussian_function(double a, double b, double c, double x);

/// \brief Calculates the desired robot's trajectory to be performed
/// This function returns the position, velocity and acceleration of a point of the trajectory
/// \param inp Pointer to a vector where the trajectory point will be stored
///            In the first NUM_JOINTS positions of \a inp the positions for each robot's joint are stored.
///            In the next NUM_JOINTS positions of \a inp the velocities are stored.
///            In the next NUM_JOINTS positions of \a inp the accelerations are stored.
///			 
/// \param amplitude trajectory amplitude
/// \param tsimul time corresponing to the trajectory point which must be calculated (0.0=trajectoy beginning)
/// \param num_rep time corresponing to the frequency of the sinusoidal entry

EXTERN_C void calculate_input_trajectory_sine_inverse(double *inp, double amplitude, double tsimul,double num_rep);
/// \brief Calculates the  inverse desired robot's trajectory to be performed
/// This function returns the position, velocity and acceleration of a point of the trajectory
/// \param inp Pointer to a vector where the trajectory point will be stored
///            In the first NUM_JOINTS positions of \a inp the positions for each robot's joint are stored.
///            In the next NUM_JOINTS positions of \a inp the velocities are stored.
///            In the next NUM_JOINTS positions of \a inp the accelerations are stored.
///			 
/// \param amplitude trajectory amplitude
/// \param tsimul time corresponing to the trajectory point which must be calculated (0.0=trajectoy beginning)
/// \param num_rep time corresponing to the frequency of the sinusoidal entry

EXTERN_C void calculate_input_trajectory_sine(double *inp, double amplitude, double tsimul,double num_rep);
/// \brief Calculates the desired robot's trajectory to be performed
/// This function returns the position, velocity and acceleration of a point of the trajectory
/// \param inp Pointer to a vector where the trajectory point will be stored
///            In the first NUM_JOINTS positions of \a inp the positions for each robot's joint are stored.
///            In the next NUM_JOINTS positions of \a inp the velocities are stored.
///            In the next NUM_JOINTS positions of \a inp the accelerations are stored.
///			 
/// \param amplitude trajectory amplitude
/// \param tsimul time corresponing to the trajectory point which must be calculated (0.0=trajectoy beginning)

EXTERN_C void calculate_input_trajectory(double *inp, double amplitude, double tsimul);

/// \brief Calculates the desired error value to be injected through IO. This value is base on a Gaussian function
/// The value is calculated accoring to the expression: f(x) = a e^{- { \frac{(x-b)^2 }{ 2 c^2} } }
/// This value is used to calculate the error shape and therefore the current injected in
///  inferior olive cells.
///			   In the first NUM_JOINTS positions of \a inp the injected error shapes for each joint are stored.
///            In the next  NUM_JOINTS positions of \a inp the diff values of the injected error shapes are stored.
///            In the next  NUM_JOINTS positions of \a inp the second diff values of the injected error shapes are stored.
/// \param amplitude trajectory amplitude (a)
/// \param trajectory time 
/// \param tsimul time corresponing to the trajectory point which must be calculated (0.0=trajectoy beginning)(x)
/// \param *centerpos Centers of the gaussian shapes conforming the positive error part(b).
/// \param *centerneg Centers of the gaussian shapes conforming the negative error part (b).
/// \param *sigma Width (c)
EXTERN_C void calculate_input_error(double *inp, double amplitude,double trajectory_time,double tsimul,double centerpos[],double centerneg[], double sigma[]);

/// \brief Calculates the inverse desired error value to be injected through IO. This value is base on a Gaussian function
/// The value is calculated accoring to the expression: f(x) = a e^{- { \frac{(x-b)^2 }{ 2 c^2} } }
/// This value is used to calculate the error shape and therefore the current injected in
///  inferior olive cells.
///			   In the first NUM_JOINTS positions of \a inp the injected error shapes for each joint are stored.
///            In the next  NUM_JOINTS positions of \a inp the diff values of the injected error shapes are stored.
///            In the next  NUM_JOINTS positions of \a inp the second diff values of the injected error shapes are stored.
/// \param amplitude trajectory amplitude (a)
/// \param trajectory time 
/// \param tsimul time corresponing to the trajectory point which must be calculated (0.0=trajectoy beginning)(x)
/// \param *centerpos Centers of the gaussian shapes conforming the positive error part(b).
/// \param *centerneg Centers of the gaussian shapes conforming the negative error part (b).
/// \param *sigma Width (c)


EXTERN_C void calculate_input_error_inverse(double *inp, double amplitude,double trajectory_time,double tsimul,double centerpos[],double centerneg[], double sigma[]);

/// \brief Calculates a pure positive desired error value to be injected through IO. This value is base on a Gaussian function
/// The value is calculated accoring to the expression: f(x) = a e^{- { \frac{(x-b)^2 }{ 2 c^2} } }
/// This value is used to calculate the error shape and therefore the current injected in
///  inferior olive cells.
///			   In the first NUM_JOINTS positions of \a inp the injected error shapes for each joint are stored.
///            In the next  NUM_JOINTS positions of \a inp the diff values of the injected error shapes are stored.
///            In the next  NUM_JOINTS positions of \a inp the second diff values of the injected error shapes are stored.
/// \param amplitude trajectory amplitude (a)
/// \param trajectory time 
/// \param tsimul time corresponing to the trajectory point which must be calculated (0.0=trajectoy beginning)(x)
/// \param *centerpos Centers of the gaussian shape (b).
/// \param *sigma Width (c)
/// \param number_of_rep Number of times that the double gaussian function is going to be repeated along the trajectory time
EXTERN_C void calculate_input_error_pos(double *inp, double amplitude,double trajectory_time,double tsimul,double centerpos[],double sigma[]);

/// \brief Calculates a pure negative desired error value to be injected through IO. This value is base on a Gaussian function
/// The value is calculated accoring to the expression: f(x) = a e^{- { \frac{(x-b)^2 }{ 2 c^2} } }
/// This value is used to calculate the error shape and therefore the current injected in
///  inferior olive cells.
///			   In the first NUM_JOINTS positions of \a inp the injected error shapes for each joint are stored.
///            In the next  NUM_JOINTS positions of \a inp the diff values of the injected error shapes are stored.
///            In the next  NUM_JOINTS positions of \a inp the second diff values of the injected error shapes are stored.
/// \param amplitude trajectory amplitude (a)
/// \param trajectory time 
/// \param tsimul time corresponing to the trajectory point which must be calculated (0.0=trajectoy beginning)(x)
/// \param *centerpos Centers of the gaussian shape (b).
/// \param *sigma Width (c)
/// \param number_of_rep Number of times that the double gaussian function is going to be repeated along the trajectory time

EXTERN_C void calculate_input_error_pos_inverse(double *inp, double amplitude,double trajectory_time,double tsimul,double centerpos[],double sigma[]);

/// \brief Calculates the desired error value variations to be injected through IO and it is repeated each iteration. This value is base on a Gaussian function
/// The value is calculated accoring to the expression: f(x) = a e^{- { \frac{(x-b)^2 }{ 2 c^2} } }
/// This value is used to calculate the error shape and therefore the current injected in
///  inferior olive cells.
///			   In the first NUM_JOINTS positions of \a inp the injected error shapes for each joint are stored.
///            In the next  NUM_JOINTS positions of \a inp the diff values of the injected error shapes are stored.
///            In the next  NUM_JOINTS positions of \a inp the second diff values of the injected error shapes are stored.
/// \param amplitude trajectory amplitude (a)
/// \param trajectory time 
/// \param tsimul time corresponing to the trajectory point which must be calculated (0.0=trajectoy beginning)(x)
/// \param *centerpos Centers of the gaussian shapes conforming the positive error part(b).
/// \param *centerneg Centers of the gaussian shapes conforming the negative error part (b).
/// \param *sigma Width (c)
/// \param *number_of_rep number of repetitions

EXTERN_C void calculate_input_error_repetitions(double *inp, double amplitude,double trajectory_time,double tsimul,double centerpos[],double centerneg[],double sigma[],int number_of_rep);
EXTERN_C void calculate_input_error_repetitions_inverse(double *inp, double amplitude,double trajectory_time,double tsimul,double centerpos[],double centerneg[],double sigma[],int number_of_rep);
EXTERN_C void calculate_input_error_repetitions_pos(double *inp, double amplitude,double trajectory_time,double tsimul,double centerpos[],double sigma[],int number_of_rep);
EXTERN_C void calculate_input_error_repetitions_pos_inverse(double *inp, double amplitude,double trajectory_time,double tsimul,double centerpos[],double sigma[],int number_of_rep);

/// \brief Calculates maximum and minimum values of the trajectory positions, velocities and accelerations.
/// These values are used to set the rages of the input variables which will be encoded in mossy fiber activity
/// \param amplitude Amplitide of the trajectory
/// \param min_traj_amplitude Pointer to an array in which the minimum values for the position, velocity and acceleration will be stored.
/// \param max_traj_amplitude Pointer to an array in which the maximum values for the position, velocity and acceleration will be stored.
EXTERN_C void calculate_input_trajectory_max_amplitude(double trajectory_time, double amplitude, double *min_traj_amplitude, double *max_traj_amplitude);

/// \brief Calculates the RBF (Gaussian bell) width for a given overlap
/// This value is used to automatically configure RBFs shape
/// \param rbf_distance Distance between two consecutive RBFs
/// \param overlap Overlap between RBFs (0,1)
/// \return The RBF c parameter (bell width) depending on the desired intersection between RBFs
double calculate_RBF_width(double rbf_distance, double overlap);

/// \brief Prints the mathematical expression corresponding to an RBF set.
/// Creates a Derive mathematical expression which can be plotted to check the correct overlapping of RBF functions.
/// example:
/// \code{.cpp}
/// struct rbf_set rbfs;
/// rbfs.bell_amp=1.0; rbfs.bell_overlap=0.2; rbfs.first_bell_pos=-0.1; rbfs.last_bell_pos=0.1; rbfs.num_rbfs=20;
/// printRBFs(&rbfs);
/// \endcode
/// \param rbfs RBF definition structure.
EXTERN_C void printRBFs(struct rbf_set *rbfs);

/// \brief Generates the population neural activity during a simulation slot for a specified input variable.
/// The generated neural activity is inserted in the simulation to be precessed in the
/// next call to run_neural_simulation_slot().
/// \param neural_sim Pointer to a Simulation created by create_neural_simulation().
/// \param cur_slot_time The simulation time of the beginning of the slot that will be simulated next.
/// \param rbfs Pointer to the RBF definitions that will be used for the encoding into neural activity
/// \param input_var Input variable to be encoded
/// \param first_input_neuron Number (index) of the first mossy fiber which encode this input variable
/// \param last_spk_times Pointer to an array which contains the times in which each mossy fiber trasmitted the last spike (this array is updated by this function).
/// \param max_spk_freq Maximum firing frequency for a mossy fiber
void generate_activityRBF(Simulation *sim, double cur_slot_time, struct rbf_set *rbfs, double input_var, long first_input_neuron, double *last_spk_times, double max_spk_freq);

/// \brief Generates the population neural activity during a simulation slot for a specified input variable set.
/// The generated neural activity is inserted in the simulation to be precessed in the
/// next call to run_neural_simulation_slot().
/// \param neural_sim Pointer to a Simulation created by create_neural_simulation().
/// \param cur_slot_time The simulation time of the beginning of the slot that will be simulated next.
/// \param input_var Pointer to an array which contains the input variables to be encoded
/// \param min_traj_amplitude Pointer to an array in which the minimum values for the position and velocity will be stored.
/// \param max_traj_amplitude Pointer to an array in which the maximum values for the position and velocity will be stored.
EXTERN_C void generate_input_traj_activity(Simulation *neural_sim, double cur_slot_time, double *input_vars, double *min_traj_amplitude, double *max_traj_amplitude);

/// \brief Generates the population neural activity during a simulation slot for a specified input variable set.
/// The generated neural activity is inserted in the simulation to be precessed in the
/// next call to run_neural_simulation_slot().
/// \param neural_sim Pointer to a Simulation created by create_neural_simulation().
/// \param cur_slot_time The simulation time of the beginning of the slot that will be simulated next.
/// \param robot_state_vars Pointer to an array which contains the robot input variables to be encoded
/// \param min_traj_amplitude Pointer to an array in which the minimum values for the position and velocity will be stored.
/// \param max_traj_amplitude Pointer to an array in which the maximum values for the position and velocity will be stored.
EXTERN_C void generate_robot_state_activity(Simulation *neural_sim, double cur_slot_time, double *robot_state_vars, double *min_traj_amplitude, double *max_traj_amplitude);

/////////////////////// GENERATE LEARNING ACTIVITY ///////////////////

/// \brief Calculates and weighs the robot's obtained errors in positions and velocities.
/// The input values should be obtained from the control loop variables. 
/// This function is called at each simulation step.
/// \param desired_position Desired Position of the corresponding robot link.
/// \param desired_velocity Desired Velocity of the corresponding robot link.
/// \param actual_position Actual Position of the corresponding robot link.
/// \param actual_velocity Actual Velocity of the corresponding robot link.
/// \param kp Proportial Gain Value				
/// \param kv Derivative Gain Value
/// \return Error per link: The final combined and weighted error value per robot link
double compute_PD_error(double desired_position, double desired_velocity, double actual_position, double actual_velocity, double kp, double kd);

/// \brief Calculates and weighs the robot's obtained errors in positions 
/// The input values should be obtained from the control loop variables. 
/// This function is called at each simulation step.
/// \param desired_position Desired Position of the corresponding robot link.
/// \param actual_position Actual Position of the corresponding robot link.
/// \param kp Proportial Gain Value				
/// \return Error per link: The final weighted error value per robot link
double compute_P_error(double desired_position, double actual_position, double kp);

/// \brief Calculates the learning signal from the torque error using a sigmoid function.
/// This function is called by calculate_learning_signals().
/// \param error_torque Actual generated torque error value.
/// \param max_error_torque Maximum torque value for the correponding joint.
/// \return Value related to the error which can be used as learning signal.
double error_sigmoid(double error_torque, double max_error_torque);

/// \brief Generates the population neural activity from a specified inferior olive input current.
/// This function must be called at each simulation step.
/// \param cur_slot_init The simulation time of the beginning of the slot that will be simulated next.
/// \param input_current Input current to be injected in the inferior olive neurons related to the performed error.
/// \param first_input_neuron Number (index) of network neuron corresponding to the first climbing fiber to receive the activity.
/// \param num_learning_neurons Number of neurons (climbing fibers), belonging to the neural network, that will be activated.
/// \param last_spk_times Pointer to an array containing the last time that each climbing fiber was active (this array is updated by this function).
/// \param max_spk_freq Maximum firing frecuency of climbing fibers (inferior olive neurons).
void generate_stochastic_activity(double cur_slot_init, double input_current, long first_learning_neuron, long num_learning_neurons, double *last_spk_times, double max_spk_freq);

/// \brief Calculates the error signal from the robot's actual and desired joint position and volocity.
/// This function is called in the control loop.
/// \param input_vars Desired positions and velocities (what the robot's links should have).
/// \param state_vars Actual positions and velocities  (what the robot's links currently have).
/// \param error_vars Pointer to an array in which the ponderated error for each robot link will be stored.
EXTERN_C void calculate_error_signals(double *input_vars, double *state_vars, double *error_vars);

/// \brief Calculates the error signal from the head's actual position and eye's actual positions.
/// This function is called in the control loop.
/// \param input_vars Desired HEAD's positions (where the HEAD  should be).
/// \param state_vars Actual EYES' positions  (where the eyes' currently are).
/// \param error_vars Pointer to an array in which the ponderated error for each pair of muscle in VOR plant will be stored.
EXTERN_C void calculate_error_signals_vor(double *input_vars, double *state_vars, double *error_vars);

/// \brief Calculates the actuator learning signals from the robot's joint performed error.
/// Depending on the error sign the agonist or the antagonist learning is activated.
/// This function is called in the control loop.
/// \param error_vars Pointer to an array which must contain the ponderated error for each robot's link 
/// \param output_vars Pointer to an array which must contain the splitted cerebellar corrective torque
///   values (agonist/antagonist muscle). NUM_JOINTS*2 values in total.
/// \param learning_vars Pointer to an array in which the calculated learning variables will be stored.
///   They are coded as splitted input current values (agonist/antagonist muscle), which correspond to
///   the estimated current to be injected in the corresponding inferior olive neurons.
EXTERN_C void calculate_learning_signals(double *error_vars, double *output_vars, double *learning_vars);

/// \brief This function injects activity in the neural network (learning activity) depending
///        on the climbing-fiber-realted input current which is estimated from the obtained
///        error in the control loop. 
/// This function must be called at each simulation step.
/// \param neural_sim Pointer to a Simulation created by create_neural_simulation().
/// \param cur_slot_init The simulation time of the beginning of the slot that will be simulated next.
/// \param learning_vars Pointer to an array which contains the calculated learning variables.
///     They are splitted input currents (agonist/antagonist muscle) retated to the estimated error signal
///     (injected current in their corresponding inferior olive neurons)
EXTERN_C void generate_learning_activity(Simulation *neural_sim, double cur_slot_init, double *learning_vars);


//////////////////////////// GENERATE OUTPUT /////////////////////////

/// \brief Computes the output variables from the neural activity that the network generates.
/// This function is called at each simulation step.
/// \param neural_sim Pointer to a Simulation created by create_neural_simulation().
/// \output_vars Calculated Output variables correponding to the network output activity.
/// \return Number of generated spikes
EXTERN_C int compute_output_activity(Simulation *neural_sim, double *output_vars);

///////////////////////////// VARIABLES LOG //////////////////////////

/// \brief Comment string used in the text files
///  It is used to add comments in the log file
#define COMMENT_CHARS "%"

/// \brief The structure of one log register
///  This is an internal strucutre used by struct log
struct log_reg
  {
   float cereb_input_vars[NUM_JOINTS*3];
   float robot_state_vars[NUM_JOINTS*3];
   float robot_torque_vars[NUM_JOINTS];
   float cereb_output_vars[NUM_OUTPUT_VARS];
   float cereb_learning_vars[NUM_OUTPUT_VARS];
   float robot_error_vars[NUM_JOINTS];
   float time;
   float consumed_time;
   unsigned long spk_counter;
  };

/// \brief The structure of one log register for VOR 3DOF
///  This is an internal strucutre used by struct log
struct log_reg_vor
{
	float vor_head_vars[NUM_JOINTS*2];
	float vor_output_vars[NUM_JOINTS*2];
	float avatar_head_vars[NUM_JOINTS * 2];
	float avatar_output_vars[NUM_JOINTS * 2];
	float cereb_output_vars[NUM_OUTPUT_VARS];
	float cereb_learning_vars[NUM_OUTPUT_VARS];
	float error_vars[NUM_JOINTS];
	float time;
	float consumed_time;
	unsigned long spk_counter;
	

};

/// \brief Structure where the variables are logged during the simulation
///  It's size will depend on SIM_SLOT_LENGTH, the trajectory time and the maximum number of trajectory repetitions
///  The user has to declare one instance of this strcuture for each log process
struct log
  {
   int nregs;
   struct log_reg *regs;
  };

/// \brief Structure where the variables are logged during the simulation for VOR 2DOF
///  It's size will depend on SIM_SLOT_LENGTH, the trajectory time and the maximum number of trajectory repetitions
///  The user has to declare one instance of this strcuture for each log process
struct log_vor
{
	int nregs; // It shows how many inner-loop iterations have been done
	int outer_loop_index; // It shows how many outer-loop iterations have been done
	struct log_reg_vor *regs;
};

/// \brief Creates and initializes a log register.
/// This function must be called once at the beginnig of the logging process.
/// \param log Pointer to an allocated log structure. This structure will be initialized
///        and will contain the log registers.
/// \param total_traj_executions Number of repetitions of the trajectory that the robot
///        will perform.
/// \param trajectory_time Time that the robot takes to perform a single trajectory
///        execution in seconds of simulated time.
/// \return Error occurred during the function execution (log memory allocation)
///         (0 if it is successfully executed)

EXTERN_C int create_log(struct log *log, int total_traj_executions, int trajectory_time);

/// \brief Creates and initializes a log register for VOR 3DOF.
/// This function must be called once at the beginnig of the logging process.
/// \param log Pointer to an allocated log structure. This structure will be initialized
///        and will contain the log registers.
/// \param total_traj_executions Number of repetitions of the trajectory that the robot
///        will perform.
/// \param trajectory_time Time that the robot takes to perform a single trajectory
///        execution in seconds of simulated time.
/// \return Error occurred during the function execution (log memory allocation)
///         (0 if it is successfully executed)

EXTERN_C int create_log_vor(struct log_vor *log_vor, int total_traj_executions, int trajectory_time);

/// \brief Stores all the interesting variables during the simulation process of VOR. 
///   These variables are logged during the simulation process.
/// This function must be called at each simulation step.
/// \param log_vor Pointer to an initialized log_vor structure where the variables will be stored.
/// \param time Simulation time of the simulation process.
/// \param vor_head_vars Desired Position of the head.
/// \param vor_output_vars Actual Position VOR plant.
/// \param cereb_output_vars Cerebellum output variables (corerctive torque values).
/// \param cereb_learning_vars Learning variables calculated from the performed error.
/// \param error_vars Actual Position Velocity and Acceleration Values-Desired Position Velocity and Acceleration Values.
/// \param vor_head_vars_xy entry signal transformed to x and y coordinates
/// \param vor_output_vars_xy output signal transformed to x and y coordinates
/// \param elapsed_time Computation time consumed by a control loop iteration.
/// \param spk_counter Processed events during a control loop iteration.
/// \pre The log must be previously initialized (calling init_log())
EXTERN_C void log_vars_vor(struct log_vor *log_vor, double time, double *vor_head_vars, double *vor_output_vars, double *cereb_output_vars, double *cereb_learning_vars, double *error_vars, float elapsed_time, unsigned long spk_counter);


/// \brief Stores all the interesting variables during the simulation process of VOR with an avatar. This process is done in three steps. 
///   These variables are logged during the simulation process.
/// This function must be called at each inner-outer simulation step.
/// \param log_vor Pointer to an initialized log_vor structure where the variables will be stored.
/// \param time Simulation time of the simulation process.
/// \param vor_head_vars Desired Position of the head.
/// \param vor_output_vars Actual Position VOR plant.
/// \param cereb_output_vars Cerebellum output variables (corerctive torque values).
/// \param elapsed_time Computation time consumed by a control loop iteration.
/// \param spk_counter Processed events during a control loop iteration.
/// \pre The log must be previously initialized (calling init_log())
EXTERN_C void log_vars_vor_first_etage_inner_outer_loop(struct log_vor *log_vor, double time, double *vor_head_vars, double *vor_output_vars, double *cereb_output_vars, float elapsed_time, unsigned long spk_counter);


/// \brief Stores all the interesting variables during the simulation process of VOR with an avatar. This process is done in three steps. 
///   These variables are logged during the simulation process.
/// This function must be called at each inner-outer simulation step.
/// \param index index inside the log_vor structure that must be used
/// \param log_vor Pointer to an initialized log_vor structure where the variables will be stored.
/// \param cereb_learning_vars Learning variables calculated from the performed error.
/// \param error_vars Actual Position Velocity and Acceleration Values-Desired Position Velocity and Acceleration Values.
/// \pre The log must be previously initialized (calling init_log())
EXTERN_C void log_vars_vor_second_etage_inner_outer_loop(struct log_vor *log_vor, int index, double *cereb_learning_vars, double *error_vars);


/// \brief Updates the outer-loop values with a new value and interpolate the previous ones using the inner-loop values
/// This function must be called at each communication event with the avatar, once it answer with the real positions and velocities for the head and the eyes.
/// \param log_vor Pointer to an initialized log_vor structure where the variables will be stored.
/// \param index index of the new value inside the log_vor strucuture.
/// \param avatar_head_vars Real avatar position of the head.
/// \param avatar_output_vars Real avatar position and velocity of the eyes.
EXTERN_C void update_outer_loop_log_vars_vor(struct log_vor *log_vor, int index, float *avatar_head_vars, float *avatar_output_vars);


 //input_aux_vars, cerebellar_aux_vars, cerebellar_output_vars, delayed_cerebellar_learning_vars, error_vars, VOR_AXIS_ROTATION, slot_elapsed_time, get_neural_simulation_spike_counter_for_each_slot_time()); // Store vars into RAM

/// \brief Stores all the interesting variables during the simulation process. 
///   These variables are logged during the simulation process.
/// This function must be called at each simulation step.
/// \param log Pointer to an initialized log structure where the variables will be stored.
/// \param time Simulation time of the simulation process.
/// \param input_vars Desired Position Velocity and Acceleration Values.
/// \param state_vars Actual Position Velocity and Acceleration Values.
/// \param torque_vars Torque Values calculated by the crude inverse dynamics.
/// \param output_vars Cerebellum output variables (corerctive torque values).
/// \param learning_vars Learning variables calculated from the performed error.
/// \param error_vars Actual Position Velocity and Acceleration Values-Desired Position Velocity and Acceleration Values.
/// \param elapsed_time Computation time consumed by a control loop iteration.
/// \param spk_counter Processed events during a control loop iteration.
/// \pre The log must be previously initialized (calling init_log())
EXTERN_C void log_vars(struct log *log, double time, double *input_vars, double *state_vars, double *torque_vars, double *output_vars, double *learning_vars, double *error_vars, float elapsed_time, unsigned long spk_counter);

/// \brief Stores all the interesting variables during the simulation process. 
///   These variables are logged during the simulation process.
/// This function must be called at each simulation step.
/// \param log Pointer to an initialized log structure where the variables will be stored.
/// \param time Simulation time of the simulation process.
/// \param input_vars Desired ERROR, First diff and Second diff Values.
/// \param output_vars_normalized Cerebellum output variables  normalized(corerctive torque values).
/// \param output_vars Cerebellum output variables (corerctive torque values).
/// \param learning_vars Learning variables calculated from the performed error.
/// \param error_vars Actual ERROR (first diff and second diff)Values-Desired ERROR (first diff and second diff)Values.
/// \param elapsed_time Computation time consumed by a control loop iteration.
/// \param spk_counter Processed events during a control loop iteration.
/// \pre The log must be previously initialized (calling init_log())
EXTERN_C void log_vars_reduced(struct log *log, double time, double *input_vars,double *output_vars_normalized, double *output_vars, double *learning_vars, double *error_vars, float elapsed_time, unsigned long spk_counter);


/// \brief Saves the previously stored log registers in a text file.
/// This function must be called at the end of the logging process.
/// \param log Pointer to an initialized log structure where the variables have been stored.
/// \param file_name Pointer to an array containing the name of the output log file.
/// \pre The log must be previously initialized (calling init_log())
/// \return Error occurred during the function execution (0 if it is successfully executed)
EXTERN_C int save_and_finish_log(struct log *log, char *file_name);

/// \brief Saves the previously stored log registers in a text file VOR 3DOF experiment. 
/// This function must be called at the end of the logging process.
/// \param log Pointer to an initialized log structure where the variables have been stored.
/// \param file_name Pointer to an array containing the name of the output log file.
/// \pre The log must be previously initialized (calling init_log())
/// \return Error occurred during the function execution (0 if it is successfully executed)
EXTERN_C int save_and_finish_log_vor(struct log_vor *log_vor, char *file_name);

/// \brief Calculates and insertes in the event queue the input activity for one trajectory. 
/// This function must be called at in each trajectory if the input activity is not loaded when the simulation started. 
/// \param neural_sim simulation object.
/// \param time time when starts the trajectory.
/// \pre The input activity can not be previously load with other function.
EXTERN_C void calculate_input_activity_for_one_trajectory(Simulation *neural_sim, double time);

/// \brief Initializes the watchdog parameters for the real time simulation.
/// This function must be called befor the simulation of the first trajectory. 
/// \param neural_sim simulation object.
/// \param new_slot_time simulation step time (usually 2 ms).
/// \param max_simulation_delay amount of time that the simulation can be executed in advance. Time since and output spike in the neural network generate an 
///        input spike as a response (delay in parallel fiber learning rule). Usually between 100 and 150 ms.
/// \param new_first_section parameter between 0 and 1 that fix limit between 0 and 1 in the RestrictionLevel of RealTimeRestriction objects. 
/// \param new_second_section parameter between 0 and 1 that fix limit between 1 and 2 in the RestrictionLevel of RealTimeRestriction objects.
/// \param new_third_section parameter between 0 and 1 that fix limit between 2 and 3 in the RestrictionLevel of RealTimeRestriction objects.
EXTERN_C void enable_real_time_restriction(Simulation * neural_sim, double new_slot_time, double max_simulation_delay, float new_first_section, float new_second_section, float new_third_section);

/// \brief A secondary thread will perform the watchdog. This thread is blocked inside the function until start_real_time_restriction function is executed by the main thread.
/// This function must be called by a secondary thread before the simulation the the first trajectory.
/// \param neural_sim simulation object.
EXTERN_C void enable_real_time_restriction_in_secondary_thread(Simulation * neural_sim);

/// \brief When the main thread execute this function, the secondary thread that perform the watchdog is unlocked and start to measure the real time.
/// This function must be called just before the simulation of the first trajectory.
/// \param neural_sim simulation object.
EXTERN_C void start_real_time_restriction(Simulation * neural_sim);

/// \brief When the main thread execute this function, the boundary time in the watchdog is increased in a simulation step.
/// This function must be called each simulation step by the main thread.
/// \param neural_sim simulation object.
EXTERN_C void next_step_real_time_restriction(Simulation * neural_sim);

/// \brief When the main thread execute this function, the secondary thread that perform the watchdog returns from the infinite loop of the watchdog.
/// This function must be called just after the simulation of the last trajectory.
/// \param neural_sim simulation object.
EXTERN_C void stop_real_time_restriction(Simulation * neural_sim);

/// \brief A secondary thread will perform the communication with the avatar. This secondary thread is lock inside this function until the main thread execute stop_connection_with_avatar function.
/// This function must be called by a secondary thread before the simulation the the first trajectory.
/// \param neural_sim simulation object.
/// \param communicationAvatarVOR object used to make the communication with the avatar.
/// \param var_log strucutre where is stored all the information sent to and received from the avatar.
/// \param Avatar_IP avatar IP direcction. The avatar enables a server socket in this IP direcction and wait for input connections.
/// \param Avatar_port avatar TPC port. The avatar enables a server socket in this TCP port and wait for input connections.
/// \param robot_processing_delay time spends by the avatar in process an input command.
EXTERN_C void enable_and_start_connection_with_avatar_in_seconday_thread(Simulation * neural_sim, CommunicationAvatarVOR * communicationAvatarVOR, struct log_vor * var_log, string Avatar_IP, unsigned short Avatar_port, int robot_processing_delay1, int robot_processing_delay2);

/// \brief When the main thread execute this function, the secondary thread that perform the communication with the avatar returns from the infinite loop of the communication.
/// This function must be called just after the simulation of the last trajectory.
/// \param communicationAvatarVOR object used to make the communication with the avatar.
EXTERN_C void stop_connection_with_avatar(CommunicationAvatarVOR * communicationAvatarVOR);

EXTERN_C float calculate_ideal_MAE(log_vor var_log_vor, int first_index, int last_index);

EXTERN_C float calculate_avatar_MAE(log_vor var_log_vor, int first_index, int last_index);

#endif /*_EDLUT_INTERFACE_H_*/
