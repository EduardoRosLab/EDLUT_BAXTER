/***************************************************************************
 *                    C_interface_for_robot_control.cpp                    *
 *                           -------------------                           *
 * copyright        : (C) 2013 by Richard R. Carrillo and Niceto R. Luque  *
 * email            : rcarrillo at ugr.es                                  *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/
 
 /*!
 * \file C_interface_for_robot_control.cpp
 *
 * \author Richard R. Carrillo
 * \author Niceto R. Luque
 * \date 27 of May 2016
 *
 * This file defines the interface functions to access EDLUT's functionality.
 */

#include <time.h>
#define _USE_MATH_DEFINES
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

#include "../../include/simulation/Simulation.h"

#include "../../include/spike/Network.h"

#include "../../include/spike/InputSpike.h"

#include "../../include/communication/ArrayOutputSpikeDriver.h"
#include "../../include/communication/FileInputSpikeDriver.h"
#include "../../include/communication/FileOutputSpikeDriver.h"
#include "../../include/communication/FileOutputWeightDriver.h"

#include "../../include/simulation/EventQueue.h"

#include "../../include/spike/EDLUTFileException.h"
#include "../../include/spike/EDLUTException.h"
#include "../../include/communication/ConnectionException.h"


#include "../../include/interface/C_interface_for_robot_control.h"

#include "../../include/simulation/RealTimeRestriction.h"
#include "../../include/simulation/RandomGenerator.h"
#include "../../include/simulation/MediananFilter.h"

#include "../../include/openmp/openmp.h"

#include "../../include/vor_model/CommunicationAvatarVOR.h"

///////////////////////////// SIMULATION MANAGEMENT //////////////////////////
long N_one_slot_internal_spikes2;




extern "C" Simulation *create_neural_simulation(const char *net_file, const char *input_weight_file, const char *input_spike_file, const char *output_weight_file, const char *output_spike_file, double weight_save_period, int number_of_openmp_queues)
  {
   Simulation *neural_sim; // Neural-simulator object (EDLUT)
   try
     {
      neural_sim=new Simulation(net_file, input_weight_file, -1.0, 0.0, number_of_openmp_queues); // End-simulation time not specified

      if(input_spike_file)
         neural_sim->AddInputSpikeDriver(new FileInputSpikeDriver(input_spike_file));

      if(output_weight_file) // Driver to save neural-network weights
        { 
         neural_sim->AddOutputWeightDriver(new FileOutputWeightDriver(output_weight_file));
         neural_sim->SetSaveStep((float)weight_save_period);
        }

      neural_sim->AddOutputSpikeDriver(new ArrayOutputSpikeDriver()); // OutputSpikeDriver used to send activity to the robot interface. This output driver must be inserted first

      if(output_spike_file) // Driver to write output activity
         neural_sim->AddMonitorActivityDriver(new FileOutputSpikeDriver(output_spike_file,false)); // Neuron potentials are not written

      neural_sim->InitSimulation();
     }
   catch(EDLUTException exc)
     {
      cerr << exc;
      neural_sim=NULL;
     }     
   return(neural_sim);
  }

extern "C" void finish_neural_simulation(Simulation *neural_sim)
  {
   // EDLUT interface drivers
   FileInputSpikeDriver *neural_activity_input_file;
   FileOutputWeightDriver *neural_weight_output_file;
   ArrayOutputSpikeDriver *neural_activity_output_array;
   FileOutputSpikeDriver *neural_activity_output_file;

   neural_activity_input_file=(FileInputSpikeDriver *)neural_sim->GetInputSpikeDriver(0);
   if(neural_activity_input_file)
     {
      neural_sim->RemoveInputSpikeDriver(neural_activity_input_file);
      delete neural_activity_input_file;
     }

   neural_weight_output_file=(FileOutputWeightDriver *)neural_sim->GetOutputWeightDriver(0);
   if(neural_weight_output_file)
      {
       neural_sim->RemoveOutputWeightDriver(neural_weight_output_file);
       delete neural_weight_output_file;
      }

   neural_activity_output_array=(ArrayOutputSpikeDriver *)neural_sim->GetOutputSpikeDriver(0); // The first output spike driver in the list is the ArrayOutputSpikeDriver
   if(neural_activity_output_array)
     {
      neural_sim->RemoveOutputSpikeDriver(neural_activity_output_array); // remove the driver from the simulation output-driver list
      delete neural_activity_output_array;
     }

   neural_activity_output_file=(FileOutputSpikeDriver *)neural_sim->GetMonitorActivityDriver(0); // The first monitor-activity driver in the list is the FileOutputSpikeDriver
   if(neural_activity_output_file)
     {
      neural_sim->RemoveMonitorActivityDriver(neural_activity_output_file);
      delete neural_activity_output_file;
     }
   delete neural_sim;
  }

extern "C" int run_neural_simulation_slot(Simulation *neural_sim, double next_step_sim_time)
  {
	int ret=0;
	N_one_slot_internal_spikes2=neural_sim->GetTotalSpikeCounter();

	try{
		neural_sim->RunSimulationSlot(next_step_sim_time);
		N_one_slot_internal_spikes2=neural_sim->GetTotalSpikeCounter()-N_one_slot_internal_spikes2;
		ret=0;
	}
	catch(ConnectionException exc){
		cerr << exc << endl;
		ret=1;
	}
	catch(EDLUTFileException exc){
		cerr << exc << endl;
		ret=exc.GetErrorNum();
	}
	catch(EDLUTException exc){
		cerr << exc << endl;
		ret=exc.GetErrorNum();
	}
	return(ret);
}

extern "C" void reset_neural_simulation(Simulation *neural_sim)
  {
	  for(int i=0; i<neural_sim->GetNumberOfQueues(); i++){
		neural_sim->GetQueue()->RemoveSpikes(i);
	  }
  }

extern "C" void save_neural_weights(Simulation *neural_sim)
  {
   neural_sim->SaveWeights();
  }

extern "C" long get_neural_simulation_spike_counter_for_each_slot_time()
  {
	  return N_one_slot_internal_spikes2;
  }

extern "C" long long get_neural_simulation_event_counter(Simulation *neural_sim)
  {
   return(neural_sim->GetSimulationUpdates());
  }

extern "C" long long get_accumulated_heap_occupancy_counter(Simulation *neural_sim)
  {
   return(neural_sim->GetHeapAcumSize());
  }

///////////////////////////// DELAY LINES FOR THE CONTROL LOOP //////////////////////////

void init_delay(struct delay *d, double del_time)
  {
   int nelem, npos;
   if(del_time>MAX_DELAY_TIME)
      del_time=MAX_DELAY_TIME;
   d->length=(int)(del_time/SIM_SLOT_LENGTH+1.5); // +1 because one position of the line is wasted to return the oldest element. +0.5 to round the size
   d->index=0;
   for(npos=0;npos<d->length;npos++)
      for(nelem=0;nelem<NUM_OUTPUT_VARS;nelem++) //NUM_JOINTS
         d->buffer[npos][nelem]=0;
  }

double *delay_line(struct delay *d, double *elem)
  {
   int nelem,next_index;
   double *elems;
   next_index=(d->index+1)%d->length; // oldest used element position
   elems=d->buffer[next_index];
   for(nelem=0;nelem<NUM_OUTPUT_VARS;nelem++)// NUM_JOINTS
      d->buffer[d->index][nelem]=elem[nelem];
   d->index=next_index; // make index point to the returned element
   return(elems);
  }
/////////////////////DELAY LINES FOR THE CONTROL LOOP PER JOINT NOT PER MICROZONE//////////////////////////
void init_delay_joint(struct delay_joint *d, double del_time)
  {
   int nelem, npos;
   if(del_time>MAX_DELAY_TIME)
      del_time=MAX_DELAY_TIME;
   d->length_joint=(int)(del_time/SIM_SLOT_LENGTH+1.5); // +1 because one position of the line is wasted to return the oldest element. +0.5 to round the size
   d->index_joint=0;
   for(npos=0;npos<d->length_joint;npos++)
      for(nelem=0;nelem<NUM_JOINTS;nelem++) //NUM_JOINTS
         d->buffer_joint[npos][nelem]=0.0;  }

double *delay_line_joint(struct delay_joint *d, double *elem)
  {
   int nelem,next_index;
   double *elems;
   next_index=(d->index_joint+1)%d->length_joint; // oldest used element position
   elems=d->buffer_joint[next_index];
   for(nelem=0;nelem<NUM_JOINTS;nelem++)// NUM_JOINTS
      d->buffer_joint[d->index_joint][nelem]=elem[nelem];
   d->index_joint=next_index; // make index point to the returned element
   return(elems);
  }

///////////////////////////// INPUT TRAJECTORY //////////////////////////

double gaussian_function(double a, double b, double c, double x)
  {
   return(a*exp(-(x-b)*(x-b)/(2*c*c)));
  }

extern "C" void calculate_input_trajectory_sine_inverse(double *inp, double amplitude, double tsimul,double num_rep)
  {
   int njoint;
   for(njoint=0;njoint<NUM_JOINTS;njoint++)
     {
      // sine trajectory (qd[0] = 0 and qd[1] = 0 and q[0] = 0 and q[1] = 2*pi)
      inp[njoint]=-amplitude*sin(2.0*M_PI*tsimul*num_rep + njoint*M_PI/4.0);
      inp[njoint+NUM_JOINTS]=-2.0*amplitude* M_PI*num_rep*cos(2*M_PI*tsimul*num_rep);
      inp[njoint+2*NUM_JOINTS]=4.0*amplitude*M_PI*M_PI*num_rep*num_rep*sin(2.0*M_PI*num_rep*tsimul);
     }
  }
extern "C" void calculate_input_trajectory_sine(double *inp, double amplitude, double tsimul,double num_rep)
  {
   int njoint;
   for(njoint=0;njoint<NUM_JOINTS;njoint++)
     {
      // sine trajectory (qd[0] = 0 and qd[1] = 0 and q[0] = 0 and q[1] = 2*pi)
      inp[njoint]=amplitude*sin(2.0*M_PI*tsimul*num_rep + njoint*M_PI/4.0);
      inp[njoint+NUM_JOINTS]=2.0*amplitude* M_PI*num_rep*cos(2*M_PI*tsimul*num_rep);
      inp[njoint+2*NUM_JOINTS]=-4.0*amplitude*M_PI*M_PI*num_rep*num_rep*sin(2.0*M_PI*num_rep*tsimul);
     }
  }

extern "C" void calculate_input_trajectory(double *inp, double amplitude, double tsimul)
  {
   int njoint;
   for(njoint=0;njoint<NUM_JOINTS;njoint++)
     {
      // modified trajectory by means of a cubic spline: -4·pi·t^3 + 6·pi·t^2 (qd[0] = 0 and qd[1] = 0 and q[0] = 0 and q[1] = 2*pi)
      inp[njoint]=amplitude*sin((-4*M_PI*tsimul*tsimul*tsimul + 6*M_PI*tsimul*tsimul) + njoint*M_PI/4);
      inp[njoint+NUM_JOINTS]=amplitude*12*M_PI*tsimul*(1-tsimul)*cos(4*M_PI*tsimul*tsimul*tsimul - 6*M_PI*tsimul*tsimul - M_PI*njoint/4);
      inp[njoint+2*NUM_JOINTS]=amplitude*(12*M_PI*(1-2*tsimul)*cos(4*M_PI*tsimul*tsimul*tsimul - 6*M_PI*tsimul*tsimul - M_PI*njoint/4) + 144*M_PI*M_PI*tsimul*tsimul*(tsimul-1)*(tsimul-1)*sin(4*M_PI*tsimul*tsimul*tsimul - 6*M_PI*tsimul*tsimul - M_PI*njoint/4));
     }
  }

extern "C" void calculate_input_error(double *inp, double amplitude,double trajectory_time,double tsimul,double centerpos[],double centerneg[],double sigma[])
  {
   int njoint;

   for(njoint=0;njoint<NUM_JOINTS;njoint++)
     {  centerpos[njoint]=trajectory_time*2.0/3;
		centerneg[njoint]=trajectory_time*1.0/3;
		sigma[njoint]=trajectory_time*0.1;//trajectory_time*0.1;
		 
		 inp[njoint]=amplitude*exp(-(tsimul - centerpos[njoint])*(tsimul - centerpos[njoint])/(2*sigma[njoint]*sigma[njoint]))-amplitude*exp(-(tsimul - centerneg[njoint])*(tsimul - centerneg[njoint])/(2*sigma[njoint]*sigma[njoint]));
		 inp[njoint+NUM_JOINTS]=amplitude*((2*centerpos[njoint]-2*tsimul)/(2*sigma[njoint]*sigma[njoint]))*exp(-(centerpos[njoint]-tsimul)*(centerpos[njoint]-tsimul)/(2*sigma[njoint]*sigma[njoint]))-amplitude*((2*centerneg[njoint]-2*tsimul)/(2*sigma[njoint]*sigma[njoint]))*exp(-(centerneg[njoint]-tsimul)*(centerneg[njoint]-tsimul)/(2*sigma[njoint]*sigma[njoint]));	
		 inp[njoint+2*NUM_JOINTS]=amplitude*(((2*centerpos[njoint]-2*tsimul)*(2*centerpos[njoint]-2*tsimul)/(4*sigma[njoint]*sigma[njoint]*sigma[njoint]*sigma[njoint]))*exp(-(centerpos[njoint]-tsimul)*(centerpos[njoint]-tsimul)/(2*sigma[njoint]*sigma[njoint]))-(1/sigma[njoint]*sigma[njoint])*exp(-(centerpos[njoint]-tsimul)*(centerpos[njoint]-tsimul)/(2*sigma[njoint]*sigma[njoint])))-amplitude*(((2*centerneg[njoint]-2*tsimul)*(2*centerneg[njoint]-2*tsimul)/(4*sigma[njoint]*sigma[njoint]*sigma[njoint]*sigma[njoint]))*exp(-(centerneg[njoint]-tsimul)*(centerneg[njoint]-tsimul)/(2*sigma[njoint]*sigma[njoint]))-(1/sigma[njoint]*sigma[njoint])*exp(-(centerneg[njoint]-tsimul)*(centerneg[njoint]-tsimul)/(2*sigma[njoint]*sigma[njoint])));
          }
  }
extern "C" void calculate_input_error_inverse(double *inp, double amplitude,double trajectory_time,double tsimul,double centerpos[],double centerneg[],double sigma[])
  {
   int njoint;

   for(njoint=0;njoint<NUM_JOINTS;njoint++)
     {  centerpos[njoint]=trajectory_time*2.0/3;
		centerneg[njoint]=trajectory_time*1.0/3;
		sigma[njoint]=trajectory_time*0.1;//trajectory_time*0.1;
		 
		 inp[njoint]=-amplitude*exp(-(tsimul - centerpos[njoint])*(tsimul - centerpos[njoint])/(2*sigma[njoint]*sigma[njoint]))+amplitude*exp(-(tsimul - centerneg[njoint])*(tsimul - centerneg[njoint])/(2*sigma[njoint]*sigma[njoint]));
		 inp[njoint+NUM_JOINTS]=-amplitude*((2*centerpos[njoint]-2*tsimul)/(2*sigma[njoint]*sigma[njoint]))*exp(-(centerpos[njoint]-tsimul)*(centerpos[njoint]-tsimul)/(2*sigma[njoint]*sigma[njoint]))+amplitude*((2*centerneg[njoint]-2*tsimul)/(2*sigma[njoint]*sigma[njoint]))*exp(-(centerneg[njoint]-tsimul)*(centerneg[njoint]-tsimul)/(2*sigma[njoint]*sigma[njoint]));	
		 inp[njoint+2*NUM_JOINTS]=-amplitude*(((2*centerpos[njoint]-2*tsimul)*(2*centerpos[njoint]-2*tsimul)/(4*sigma[njoint]*sigma[njoint]*sigma[njoint]*sigma[njoint]))*exp(-(centerpos[njoint]-tsimul)*(centerpos[njoint]-tsimul)/(2*sigma[njoint]*sigma[njoint]))-(1/sigma[njoint]*sigma[njoint])*exp(-(centerpos[njoint]-tsimul)*(centerpos[njoint]-tsimul)/(2*sigma[njoint]*sigma[njoint])))+amplitude*(((2*centerneg[njoint]-2*tsimul)*(2*centerneg[njoint]-2*tsimul)/(4*sigma[njoint]*sigma[njoint]*sigma[njoint]*sigma[njoint]))*exp(-(centerneg[njoint]-tsimul)*(centerneg[njoint]-tsimul)/(2*sigma[njoint]*sigma[njoint]))-(1/sigma[njoint]*sigma[njoint])*exp(-(centerneg[njoint]-tsimul)*(centerneg[njoint]-tsimul)/(2*sigma[njoint]*sigma[njoint])));
          }
  }
extern "C" void calculate_input_error_pos(double *inp, double amplitude,double trajectory_time,double tsimul,double centerpos[],double sigma[])
  {
   int njoint;

   for(njoint=0;njoint<NUM_JOINTS;njoint++)
     {  centerpos[njoint]=trajectory_time*1.0/2.0;
		
		sigma[njoint]=trajectory_time*0.1;//trajectory_time*0.1;
		 
		 inp[njoint]=amplitude*exp(-(tsimul - centerpos[njoint])*(tsimul - centerpos[njoint])/(2*sigma[njoint]*sigma[njoint]));
		 inp[njoint+NUM_JOINTS]=amplitude*((2*centerpos[njoint]-2*tsimul)/(2*sigma[njoint]*sigma[njoint]))*exp(-(centerpos[njoint]-tsimul)*(centerpos[njoint]-tsimul)/(2*sigma[njoint]*sigma[njoint]));	
		 inp[njoint+2*NUM_JOINTS]=amplitude*(((2*centerpos[njoint]-2*tsimul)*(2*centerpos[njoint]-2*tsimul)/(4*sigma[njoint]*sigma[njoint]*sigma[njoint]*sigma[njoint]))*exp(-(centerpos[njoint]-tsimul)*(centerpos[njoint]-tsimul)/(2*sigma[njoint]*sigma[njoint]))-(1/sigma[njoint]*sigma[njoint])*exp(-(centerpos[njoint]-tsimul)*(centerpos[njoint]-tsimul)/(2*sigma[njoint]*sigma[njoint])));
          }
  }
extern "C" void calculate_input_error_pos_inverse(double *inp, double amplitude,double trajectory_time,double tsimul,double centerpos[],double sigma[])
  {
   int njoint;

   for(njoint=0;njoint<NUM_JOINTS;njoint++)
     {  centerpos[njoint]=trajectory_time*1.0/2.0;
		
		sigma[njoint]=trajectory_time*0.1;//trajectory_time*0.1;
		 
		 inp[njoint]=-amplitude*exp(-(tsimul - centerpos[njoint])*(tsimul - centerpos[njoint])/(2*sigma[njoint]*sigma[njoint]));
		 inp[njoint+NUM_JOINTS]=-amplitude*((2*centerpos[njoint]-2*tsimul)/(2*sigma[njoint]*sigma[njoint]))*exp(-(centerpos[njoint]-tsimul)*(centerpos[njoint]-tsimul)/(2*sigma[njoint]*sigma[njoint]));	
		 inp[njoint+2*NUM_JOINTS]=-amplitude*(((2*centerpos[njoint]-2*tsimul)*(2*centerpos[njoint]-2*tsimul)/(4*sigma[njoint]*sigma[njoint]*sigma[njoint]*sigma[njoint]))*exp(-(centerpos[njoint]-tsimul)*(centerpos[njoint]-tsimul)/(2*sigma[njoint]*sigma[njoint]))-(1/sigma[njoint]*sigma[njoint])*exp(-(centerpos[njoint]-tsimul)*(centerpos[njoint]-tsimul)/(2*sigma[njoint]*sigma[njoint])));
          }
  }
extern "C" void calculate_input_error_repetitions(double *inp, double amplitude,double trajectory_time,double tsimul,double centerpos[],double centerneg[],double sigma[],int number_of_rep)
  {
   int njoint;
   int nrepetition;
   double center_pos_init;
   double center_neg_init;
   double factor;
   double inp_aux=0.0;
   double inp_aux_vel=0.0; 
   double inp_aux_acel=0.0; 
   

   factor=(trajectory_time/number_of_rep);
   center_pos_init=factor*(2.0/3.0);
   center_neg_init=factor*(1.0/3.0);
   
   for(njoint=0;njoint<NUM_JOINTS;njoint++)
     {	
	sigma[njoint]=factor*0.1;   
		for (nrepetition=0;nrepetition<number_of_rep;nrepetition++)
		{
		 centerpos[(njoint*number_of_rep)+nrepetition]=center_pos_init+(trajectory_time/number_of_rep)*(nrepetition);
		 centerneg[(njoint*number_of_rep)+nrepetition]=center_neg_init+(trajectory_time/number_of_rep)*(nrepetition);
		 inp_aux+=(amplitude*exp(-(tsimul - centerpos[(njoint*number_of_rep)+nrepetition])*(tsimul - centerpos[(njoint*number_of_rep)+nrepetition])/(2*sigma[njoint]*sigma[njoint])))-(amplitude*exp(-(tsimul - centerneg[(njoint*number_of_rep)+nrepetition])*(tsimul - centerneg[(njoint*number_of_rep)+nrepetition])/(2*sigma[njoint]*sigma[njoint])));
		 inp_aux_vel+=(amplitude*((2*centerpos[(njoint*number_of_rep)+nrepetition]-2*tsimul)/(2*sigma[njoint]*sigma[njoint]))*exp(-(centerpos[(njoint*number_of_rep)+nrepetition]-tsimul)*(centerpos[(njoint*number_of_rep)+nrepetition]-tsimul)/(2*sigma[njoint]*sigma[njoint]))-amplitude*((2*centerneg[(njoint*number_of_rep)+nrepetition]-2*tsimul)/(2*sigma[njoint]*sigma[njoint]))*exp(-(centerneg[(njoint*number_of_rep)+nrepetition]-tsimul)*(centerneg[(njoint*number_of_rep)+nrepetition]-tsimul)/(2*sigma[njoint]*sigma[njoint])));	
		 inp_aux_acel+=(amplitude*(((2*centerpos[(njoint*number_of_rep)+nrepetition]-2*tsimul)*(2*centerpos[(njoint*number_of_rep)+nrepetition]-2*tsimul)/(4*sigma[njoint]*sigma[njoint]*sigma[njoint]*sigma[njoint]))*exp(-(centerpos[(njoint*number_of_rep)+nrepetition]-tsimul)*(centerpos[(njoint*number_of_rep)+nrepetition]-tsimul)/(2*sigma[njoint]*sigma[njoint]))-(1/sigma[njoint]*sigma[njoint])*exp(-(centerpos[(njoint*number_of_rep)+nrepetition]-tsimul)*(centerpos[(njoint*number_of_rep)+nrepetition]-tsimul)/(2*sigma[njoint]*sigma[njoint])))-amplitude*(((2*centerneg[(njoint*number_of_rep)+nrepetition]-2*tsimul)*(2*centerneg[(njoint*number_of_rep)+nrepetition]-2*tsimul)/(4*sigma[njoint]*sigma[njoint]*sigma[njoint]*sigma[njoint]))*exp(-(centerneg[(njoint*number_of_rep)+nrepetition]-tsimul)*(centerneg[(njoint*number_of_rep)+nrepetition]-tsimul)/(2*sigma[njoint]*sigma[njoint]))-(1/sigma[njoint]*sigma[njoint])*exp(-(centerneg[(njoint*number_of_rep)+nrepetition]-tsimul)*(centerneg[(njoint*number_of_rep)+nrepetition]-tsimul)/(2*sigma[njoint]*sigma[njoint]))));
          
		}
		 
		 inp[njoint]=inp_aux;
		 inp[njoint+NUM_JOINTS]=inp_aux_vel;
		 inp[njoint+2*NUM_JOINTS]=inp_aux_acel;
		 inp_aux=0.0;
		 inp_aux_vel=0.0; 
		 inp_aux_acel=0.0;
		}
  }
extern "C" void calculate_input_error_repetitions_inverse(double *inp, double amplitude,double trajectory_time,double tsimul,double centerpos[],double centerneg[],double sigma[],int number_of_rep)
  {
   int njoint;
   int nrepetition;
   double center_pos_init;
   double center_neg_init;
   double factor;
   double inp_aux=0.0;
   double inp_aux_vel=0.0; 
   double inp_aux_acel=0.0; 
   

   factor=(trajectory_time/number_of_rep);
   center_pos_init=factor*(2.0/3.0);
   center_neg_init=factor*(1.0/3.0);
   
   for(njoint=0;njoint<NUM_JOINTS;njoint++)
     {	
	sigma[njoint]=factor*0.1;   
		for (nrepetition=0;nrepetition<number_of_rep;nrepetition++)
		{
		 centerpos[(njoint*number_of_rep)+nrepetition]=center_pos_init+(trajectory_time/number_of_rep)*(nrepetition);
		 centerneg[(njoint*number_of_rep)+nrepetition]=center_neg_init+(trajectory_time/number_of_rep)*(nrepetition);
		 inp_aux+=-amplitude*exp(-(tsimul - centerpos[(njoint*number_of_rep)+nrepetition])*(tsimul - centerpos[(njoint*number_of_rep)+nrepetition])/(2*sigma[njoint]*sigma[njoint]))+amplitude*exp(-(tsimul - centerneg[(njoint*number_of_rep)+nrepetition])*(tsimul - centerneg[(njoint*number_of_rep)+nrepetition])/(2*sigma[njoint]*sigma[njoint]));
		 inp_aux_vel+=-amplitude*((2*centerpos[(njoint*number_of_rep)+nrepetition]-2*tsimul)/(2*sigma[njoint]*sigma[njoint]))*exp(-(centerpos[(njoint*number_of_rep)+nrepetition]-tsimul)*(centerpos[(njoint*number_of_rep)+nrepetition]-tsimul)/(2*sigma[njoint]*sigma[njoint]))+amplitude*((2*centerneg[(njoint*number_of_rep)+nrepetition]-2*tsimul)/(2*sigma[njoint]*sigma[njoint]))*exp(-(centerneg[(njoint*number_of_rep)+nrepetition]-tsimul)*(centerneg[(njoint*number_of_rep)+nrepetition]-tsimul)/(2*sigma[njoint]*sigma[njoint]));	
		 inp_aux_acel+=-amplitude*(((2*centerpos[(njoint*number_of_rep)+nrepetition]-2*tsimul)*(2*centerpos[(njoint*number_of_rep)+nrepetition]-2*tsimul)/(4*sigma[njoint]*sigma[njoint]*sigma[njoint]*sigma[njoint]))*exp(-(centerpos[(njoint*number_of_rep)+nrepetition]-tsimul)*(centerpos[(njoint*number_of_rep)+nrepetition]-tsimul)/(2*sigma[njoint]*sigma[njoint]))-(1/sigma[njoint]*sigma[njoint])*exp(-(centerpos[(njoint*number_of_rep)+nrepetition]-tsimul)*(centerpos[(njoint*number_of_rep)+nrepetition]-tsimul)/(2*sigma[njoint]*sigma[njoint])))+amplitude*(((2*centerneg[(njoint*number_of_rep)+nrepetition]-2*tsimul)*(2*centerneg[(njoint*number_of_rep)+nrepetition]-2*tsimul)/(4*sigma[njoint]*sigma[njoint]*sigma[njoint]*sigma[njoint]))*exp(-(centerneg[(njoint*number_of_rep)+nrepetition]-tsimul)*(centerneg[(njoint*number_of_rep)+nrepetition]-tsimul)/(2*sigma[njoint]*sigma[njoint]))-(1/sigma[njoint]*sigma[njoint])*exp(-(centerneg[(njoint*number_of_rep)+nrepetition]-tsimul)*(centerneg[(njoint*number_of_rep)+nrepetition]-tsimul)/(2*sigma[njoint]*sigma[njoint])));
          
		}
		 
		 inp[njoint]=inp_aux;
		 inp[njoint+NUM_JOINTS]=inp_aux_vel;
		 inp[njoint+2*NUM_JOINTS]=inp_aux_acel;
		 inp_aux=0.0;
		 inp_aux_vel=0.0; 
		 inp_aux_acel=0.0;
		}
  }
extern "C" void calculate_input_error_repetitions_pos(double *inp, double amplitude,double trajectory_time,double tsimul,double centerpos[],double sigma[],int number_of_rep)
  {
   int njoint;
   int nrepetition;
   double center_pos_init;
   double factor;
   double inp_aux=0.0;
   double inp_aux_vel=0.0; 
   double inp_aux_acel=0.0; 
   

   factor=(trajectory_time/number_of_rep);
   center_pos_init=factor*(1.0/2.0);

   
   for(njoint=0;njoint<NUM_JOINTS;njoint++)
     {	
	sigma[njoint]=factor*0.1;   
		for (nrepetition=0;nrepetition<number_of_rep;nrepetition++)
		{
		 centerpos[(njoint*number_of_rep)+nrepetition]=center_pos_init+(trajectory_time/number_of_rep)*(nrepetition);
	     inp_aux+=0.5+0.5*amplitude*exp(-(tsimul - centerpos[(njoint*number_of_rep)+nrepetition])*(tsimul - centerpos[(njoint*number_of_rep)+nrepetition])/(2*sigma[njoint]*sigma[njoint]));
		 inp_aux_vel+=0.5*amplitude*((2*centerpos[(njoint*number_of_rep)+nrepetition]-2*tsimul)/(2*sigma[njoint]*sigma[njoint]))*exp(-(centerpos[(njoint*number_of_rep)+nrepetition]-tsimul)*(centerpos[(njoint*number_of_rep)+nrepetition]-tsimul)/(2*sigma[njoint]*sigma[njoint]));	
		 inp_aux_acel+=0.5*amplitude*(((2*centerpos[(njoint*number_of_rep)+nrepetition]-2*tsimul)*(2*centerpos[(njoint*number_of_rep)+nrepetition]-2*tsimul)/(4*sigma[njoint]*sigma[njoint]*sigma[njoint]*sigma[njoint]))*exp(-(centerpos[(njoint*number_of_rep)+nrepetition]-tsimul)*(centerpos[(njoint*number_of_rep)+nrepetition]-tsimul)/(2*sigma[njoint]*sigma[njoint]))-(1/sigma[njoint]*sigma[njoint])*exp(-(centerpos[(njoint*number_of_rep)+nrepetition]-tsimul)*(centerpos[(njoint*number_of_rep)+nrepetition]-tsimul)/(2*sigma[njoint]*sigma[njoint])));
          
		}
		 
		 inp[njoint]=inp_aux;
		 inp[njoint+NUM_JOINTS]=inp_aux_vel;
		 inp[njoint+2*NUM_JOINTS]=inp_aux_acel;
		 inp_aux=0.0;
		 inp_aux_vel=0.0; 
		 inp_aux_acel=0.0;
		}
  }
extern "C" void calculate_input_error_repetitions_pos_inverse(double *inp, double amplitude,double trajectory_time,double tsimul,double centerpos[],double sigma[],int number_of_rep)
  {
   int njoint;
   int nrepetition;
   double center_pos_init;
   double factor;
   double inp_aux=0.0;
   double inp_aux_vel=0.0; 
   double inp_aux_acel=0.0; 
   

   factor=(trajectory_time/number_of_rep);
   center_pos_init=factor*(1.0/2.0);

   
   for(njoint=0;njoint<NUM_JOINTS;njoint++)
     {	
	sigma[njoint]=factor*0.1;   
		for (nrepetition=0;nrepetition<number_of_rep;nrepetition++)
		{
		 centerpos[(njoint*number_of_rep)+nrepetition]=center_pos_init+(trajectory_time/number_of_rep)*(nrepetition);
	     inp_aux+=-0.5-0.5*amplitude*exp(-(tsimul - centerpos[(njoint*number_of_rep)+nrepetition])*(tsimul - centerpos[(njoint*number_of_rep)+nrepetition])/(2*sigma[njoint]*sigma[njoint]));
		 inp_aux_vel+=-0.5*amplitude*((2*centerpos[(njoint*number_of_rep)+nrepetition]-2*tsimul)/(2*sigma[njoint]*sigma[njoint]))*exp(-(centerpos[(njoint*number_of_rep)+nrepetition]-tsimul)*(centerpos[(njoint*number_of_rep)+nrepetition]-tsimul)/(2*sigma[njoint]*sigma[njoint]));	
		 inp_aux_acel+=-0.5*amplitude*(((2*centerpos[(njoint*number_of_rep)+nrepetition]-2*tsimul)*(2*centerpos[(njoint*number_of_rep)+nrepetition]-2*tsimul)/(4*sigma[njoint]*sigma[njoint]*sigma[njoint]*sigma[njoint]))*exp(-(centerpos[(njoint*number_of_rep)+nrepetition]-tsimul)*(centerpos[(njoint*number_of_rep)+nrepetition]-tsimul)/(2*sigma[njoint]*sigma[njoint]))-(1/sigma[njoint]*sigma[njoint])*exp(-(centerpos[(njoint*number_of_rep)+nrepetition]-tsimul)*(centerpos[(njoint*number_of_rep)+nrepetition]-tsimul)/(2*sigma[njoint]*sigma[njoint])));
          
		}
		 
		 inp[njoint]=inp_aux;
		 inp[njoint+NUM_JOINTS]=inp_aux_vel;
		 inp[njoint+2*NUM_JOINTS]=inp_aux_acel;
		 inp_aux=0.0;
		 inp_aux_vel=0.0; 
		 inp_aux_acel=0.0;
		}
  }
//extern "C" void calculate_input_trajectory(double *inp, double amplitude, double tsimul)
/*
     PURPOSE:
          Inverse kinematic for Schunk robot(6DOF). x, y z position are given by geometric methods, yaw pich and roll are fixed as a constant. 
		  Inverse  kinematics   (l1, l2 are the corresponding lenghts of the links)
			q1=atan(y/x);
			q3=acos((x^2+y^2+z^2-l1^2-l2^2)/(2*l1*l2));
			q2=atan(z/sqrt(x^2+y^2))-atan((l2*sin(acos((x^2+y^2+z^2-l1^2-l2^2)/(2*l1*l2))))/(l1+l2*cos(acos((x^2+y^2+z^2-l1^2-l2^2)/(2*l1*l2)))));

		The benchmark trajectory is an eight-like trajectory whose equations are given by:
		cartesian coordinates
			y1 = 0.1 * sin(2*pi *(-2*t^3+3*t^2) )+0.21502;
			z1 = 0.1 * sin(4*pi *(-2*t^3+3*t^2) )+0.18502;
			x1 =0.1; 
		
		the simulation time t(lineal) has been modified by means of a cubic spline so as to obtain a zero initial velocity and a zero final trajectory values
		this is convenient for controlling the real robot.
		cubic spline  t=(-4*pi*t^3+6*pi*t^2) where (qd[0] = 0 and qd[1] = 0 and q[0] = 0 and q[1] = 2*pi)
		this system can be rotated according to a certain angle.this is a rotation of the reference system (Y rotation)
				

     CALLING SEQUENCE:
           calculate_input_trajectory_shucnk ( *inputs, eight_plane_angle, tsimul);
         
     INPUTS:
          inputs				buffer containing position, velocity and acceleration trajectories
          eight_plane_angle     angle in radian,the eight trajectory is located in a certain plane, this plane can be rotated along Y axis
		  tsimul				simulation step 
          
     OUTPUT:
          input					buffer containing position, velocity and acceleration trajectories
*/  
/*
{
//auxiliary Variables
	double aux1A,aux1B,aux1C,auxK,auxL,auxM;
    double rotsimul=0.0;

	//POSITIONS

	aux1A=(sin(2.0*M_PI*(3.0*tsimul*tsimul - 2.0*tsimul*tsimul*tsimul))/10.0 + 10751.0/50000.0);
	aux1B=(cos(rotsimul)/10.0 + sin(rotsimul)*(sin(4.0*M_PI*(3.0*tsimul*tsimul - 2.0*tsimul*tsimul*tsimul))/10.0 + 9251.0/50000.0));
	aux1C=(sin(rotsimul)/10.0 - cos(rotsimul)*(sin(4.0*M_PI*(3.0*tsimul*tsimul - 2.0*tsimul*tsimul*tsimul))/10.0 + 9251.0/50000.0));
	auxK=cos(2.0*M_PI*(3.0*tsimul*tsimul - 2.0*tsimul*tsimul*tsimul));
	auxL=(6.0*tsimul - 6.0*tsimul*tsimul);
	auxM=cos(4.0*M_PI*(3.0*tsimul*tsimul - 2.0*tsimul*tsimul*tsimul));


	//POSITIONS

	inp[0]=atan(aux1A/aux1B);
	inp[1]=-atan((61.0*sqrt(1.0 - (4000000.0*(aux1A*aux1A+ aux1B*aux1B+ aux1C*aux1C- 8621.0/40000.0)*(aux1A*aux1A+ aux1B*aux1B+ aux1C*aux1C- 8621.0/40000.0))/182329.0))/(200.0*((10*aux1A*aux1A)/7.0 + (10.0*aux1B*aux1B)/7.0 + (10.0*aux1C*aux1C)/7.0 + 1179.0/28000.0))) - atan(aux1C/sqrt(aux1A*aux1A+ aux1B*aux1B));
	inp[2]=acos((2000.0*aux1A*aux1A)/427.0 + (2000.0*aux1B*aux1B)/427.0 + (2000.0*aux1C*aux1C)/427.0 - 8621.0/8540.0);  

	//VELOCITIES

	inp[3]=((M_PI*cos(2.0*M_PI*(- 2.0*tsimul*tsimul*tsimul + 3.0*tsimul*tsimul))*(- 6.0*tsimul*tsimul + 6.0*tsimul))/(5.0*(cos(rotsimul)/10.0 + sin(rotsimul)*(sin(4.0*M_PI*(- 2.0*tsimul*tsimul*tsimul + 3.0*tsimul*tsimul))/10.0 + 9251.0/50000.0))) - (2.0*M_PI*cos(4.0*M_PI*(- 2.0*tsimul*tsimul*tsimul + 3.0*tsimul*tsimul))*sin(rotsimul)*(- 6.0*tsimul*tsimul + 6.0*tsimul)*(sin(2.0*M_PI*(- 2.0*tsimul*tsimul*tsimul + 3.0*tsimul*tsimul))/10.0 + 10751.0/50000.0))/(5.0*aux1B*aux1B))/((sin(2.0*M_PI*(- 2.0*tsimul*tsimul*tsimul + 3.0*tsimul*tsimul))/10.0 + 10751.0/50000.0)*(sin(2.0*M_PI*(- 2.0*tsimul*tsimul*tsimul + 3.0*tsimul*tsimul))/10.0 + 10751.0/50000.0)/aux1B*aux1B + 1.0);
	inp[4]= ((((2.0*M_PI*auxK*auxL*aux1A)/5.0 + (4.0*M_PI*auxM*sin(rotsimul)*auxL*aux1B)/5.0)*aux1C)/(2.0*(aux1A*aux1A + aux1B*aux1B)*sqrt(aux1A*aux1A + aux1B*aux1B)) + (2.0*M_PI*cos(rotsimul)*auxM*auxL)/(5.0*sqrt(aux1A*aux1A + aux1B*aux1B)))/(aux1C*aux1C/(aux1A*aux1A + aux1B*aux1B) + 1.0) - ((61.0*sqrt(1.0 - (4000000.0*(aux1A*aux1A + aux1B*aux1B + aux1C*aux1C - 8621.0/40000.0)*(aux1A*aux1A + aux1B*aux1B + aux1C*aux1C - 8621.0/40000.0))/182329.0)*((4.0*M_PI*auxK*auxL*aux1A)/7.0 - (8.0*M_PI*cos(rotsimul)*auxM*auxL*aux1C)/7.0 + (8.0*M_PI*auxM*sin(rotsimul)*auxL*aux1B)/7.0))/(200.0*((10.0*aux1A*aux1A)/7.0 + (10.0*aux1B*aux1B)/7.0 + (10.0*aux1C*aux1C)/7.0 + 1179.0/28000.0)*((10.0*aux1A*aux1A)/7.0 + (10.0*aux1B*aux1B)/7.0 + (10.0*aux1C*aux1C)/7.0 + 1179.0/28000.0)) + (20000.0*((2.0*M_PI*auxK*auxL*aux1A)/5.0 - (4.0*M_PI*cos(rotsimul)*auxM*auxL*aux1C)/5.0 + (4.0*M_PI*auxM*sin(rotsimul)*auxL*aux1B)/5.0)*(aux1A*aux1A + aux1B*aux1B + aux1C*aux1C - 8621.0/40000.0))/(2989.0*sqrt(1.0 - (4000000.0*(aux1A*aux1A + aux1B*aux1B + aux1C*aux1C - 8621.0/40000.0)*(aux1A*aux1A + aux1B*aux1B + aux1C*aux1C - 8621.0/40000.0))/182329.0)*((10.0*aux1A*aux1A)/7.0 + (10.0*aux1B*aux1B)/7.0 + (10.0*aux1C*aux1C)/7.0 + 1179.0/28000.0)))/((3721.0*((4000000.0*(aux1A*aux1A + aux1B*aux1B + aux1C*aux1C - 8621.0/40000.0)*(aux1A*aux1A + aux1B*aux1B + aux1C*aux1C - 8621.0/40000.0))/182329.0 - 1.0))/(40000.0*((10.0*aux1A*aux1A)/7.0 + (10.0*aux1B*aux1B)/7.0 + (10.0*aux1C*aux1C)/7.0 + 1179.0/28000.0)*((10.0*aux1A*aux1A)/7.0 + (10.0*aux1B*aux1B)/7.0 + (10.0*aux1C*aux1C)/7.0 + 1179.0/28000.0)) - 1.0);
	inp[5]=-((800.0*M_PI*cos(2.0*M_PI*(- 2.0*tsimul*tsimul*tsimul + 3.0*tsimul*tsimul))*(- 6.0*tsimul*tsimul + 6.0*tsimul)*(sin(2.0*M_PI*(- 2.0*tsimul*tsimul*tsimul + 3.0*tsimul*tsimul))/10.0 + 10751.0/50000.0))/427.0 - (1600.0*M_PI*cos(rotsimul)*cos(4.0*M_PI*(- 2.0*tsimul*tsimul*tsimul + 3.0*tsimul*tsimul))*(- 6.0*tsimul*tsimul + 6.0*tsimul)*(sin(rotsimul)/10.0 - cos(rotsimul)*(sin(4.0*M_PI*(- 2.0*tsimul*tsimul*tsimul + 3.0*tsimul*tsimul))/10.0 + 9251.0/50000.0)))/427.0 + (1600.0*M_PI*cos(4*M_PI*(- 2.0*tsimul*tsimul*tsimul + 3.0*tsimul*tsimul))*sin(rotsimul)*(- 6.0*tsimul*tsimul + 6.0*tsimul)*(cos(rotsimul)/10.0 + sin(rotsimul)*(sin(4.0*M_PI*(- 2.0*tsimul*tsimul*tsimul + 3.0*tsimul*tsimul))/10.0 + 9251.0/50000.0)))/427.0)/sqrt(- ((2000.0*aux1A*aux1A)/427.0 + (2000.0*aux1B*aux1B)/427.0 + (2000.0*aux1C*aux1C)/427.0 - 8621.0/8540.0)*((2000.0*aux1A*aux1A)/427.0 + (2000*aux1B*aux1B)/427.0 + (2000.0*aux1C*aux1C)/427.0 - 8621.0/8540.0) + 1.0);
 
	//ACCELERATIONS

	inp[6]=-((M_PI*auxK*(12.0*tsimul - 6.0))/(5.0*aux1B) + (2.0*M_PI*M_PI*sin(2.0*M_PI*(3.0*tsimul*tsimul - 2.0*tsimul*tsimul*tsimul))*auxL*auxL)/(5.0*aux1B) - (2.0*M_PI*auxM*sin(rotsimul)*(12.0*tsimul - 6.0)*aux1A)/(5.0*aux1B*aux1B) - (8.0*M_PI*M_PI*sin(rotsimul)*sin(4.0*M_PI*(3.0*tsimul*tsimul - 2.0*tsimul*tsimul*tsimul))*auxL*auxL*aux1A)/(5.0*aux1B*aux1B) + (4.0*M_PI*M_PI*auxK*auxM*sin(rotsimul)*auxL*auxL)/(25.0*aux1B*aux1B) - (8.0*M_PI*M_PI*auxM*auxM*sin(rotsimul)*sin(rotsimul)*auxL*auxL*aux1A)/(25.0*aux1B*aux1B*aux1B))/(aux1A*aux1A/aux1B*aux1B + 1.0) - (((M_PI*auxK*auxL)/(5.0*aux1B) - (2.0*M_PI*auxM*sin(rotsimul)*auxL*aux1A)/(5.0*aux1B*aux1B))*((2.0*M_PI*auxK*auxL*aux1A)/(5.0*aux1B*aux1B) - (4.0*M_PI*auxM*sin(rotsimul)*auxL*aux1A*aux1A)/(5.0*aux1B*aux1B*aux1B)))/(aux1A*aux1A/aux1B*aux1B + 1.0)*(aux1A*aux1A/aux1B*aux1B + 1.0);	
	inp[7]= (((((2.0*M_PI*auxK*auxL*aux1A)/5.0 + (4.0*M_PI*auxM*sin(rotsimul)*auxL*aux1B)/5.0)*aux1C*aux1C)/(aux1B*aux1B + aux1A*aux1A)*(aux1B*aux1B + aux1A*aux1A) + (4.0*M_PI*cos(rotsimul)*auxM*auxL*aux1C)/(5.0*(aux1A*aux1A + aux1B*aux1B)))*((((2.0*M_PI*auxK*auxL*aux1A)/5.0 + (4.0*M_PI*auxM*sin(rotsimul)*auxL*aux1B)/5.0)*aux1C)/(2.0*(aux1B*aux1B + aux1A*aux1A)*sqrt(aux1B*aux1B + aux1A*aux1A)) + (2.0*M_PI*cos(rotsimul)*auxM*auxL)/(5.0*sqrt(aux1B*aux1B + aux1A*aux1A))))/(aux1C*aux1C/(aux1B*aux1B + aux1A*aux1A) + 1.0)*(aux1C*aux1C/(aux1B*aux1B + aux1A*aux1A) + 1.0) - ((20000.0*((2.0*M_PI*auxK*auxL*aux1A)/5.0 - (4.0*M_PI*cos(rotsimul)*auxM*auxL*aux1C)/5.0 + (4.0*M_PI*auxM*sin(rotsimul)*auxL*aux1B)/5.0)*((2.0*M_PI*auxK*auxL*aux1A)/5.0 - (4.0*M_PI*cos(rotsimul)*auxM*auxL*aux1C)/5.0 + (4*M_PI*auxM*sin(rotsimul)*auxL*aux1B)/5))/(2989.0*sqrt(1 - (4000000.0*(aux1B*aux1B + aux1C*aux1C + aux1A*aux1A - 8621.0/40000.0)*(aux1B*aux1B + aux1C*aux1C + aux1A*aux1A - 8621.0/40000.0))/182329.0)*((10.0*aux1A*aux1A)/7.0 + (10.0*aux1B*aux1B)/7.0 + (10.0*aux1C*aux1C)/7.0 + 1179.0/28000.0)) - (61.0*sqrt(1 - (4000000.0*(aux1A*aux1A + aux1B*aux1B + aux1C*aux1C - 8621.0/40000.0)*(aux1A*aux1A + aux1B*aux1B + aux1C*aux1C - 8621.0/40000.0))/182329.0)*((4.0*M_PI*auxK*auxL*aux1A)/7.0 - (8.0*M_PI*cos(rotsimul)*auxM*auxL*aux1C)/7.0 + (8.0*M_PI*auxM*sin(rotsimul)*auxL*aux1B)/7.0)*((4.0*M_PI*auxK*auxL*aux1A)/7.0 - (8.0*M_PI*cos(rotsimul)*auxM*auxL*aux1C)/7.0 + (8.0*M_PI*auxM*sin(rotsimul)*auxL*aux1B)/7.0))/(100.0*((10.0*aux1B*aux1B)/7.0 + (10*aux1C*aux1C)/7.0 + (10.0*aux1A*aux1A)/7.0 + 1179.0/28000.0)*((10.0*aux1B*aux1B)/7.0 + (10.0*aux1C*aux1C)/7.0 + (10.0*aux1A*aux1A)/7.0 + 1179.0/28000.0)*((10.0*aux1B*aux1B)/7.0 + (10.0*aux1C*aux1C)/7.0 + (10.0*aux1A*aux1A)/7.0 + 1179.0/28000.0)) + (61.0*sqrt(1.0 - (4000000.0*(aux1A*aux1A + aux1B*aux1B + aux1C*aux1C - 8621.0/40000.0)*(aux1A*aux1A + aux1B*aux1B + aux1C*aux1C - 8621.0/40000.0))/182329.0)*((4.0*M_PI*M_PI*auxK*auxK*auxL*auxL)/35.0 + (16.0*M_PI*M_PI*cos(rotsimul)*cos(rotsimul)*auxM*auxM*auxL*auxL)/35.0 + (16.0*M_PI*M_PI*auxM*auxM*sin(rotsimul)*sin(rotsimul)*auxL*auxL)/35.0 - (4.0*M_PI*auxK*(12.0*tsimul - 6.0)*aux1A)/7.0 - (8.0*M_PI*M_PI*sin(2.0*M_PI*(3.0*tsimul*tsimul - 2.0*tsimul*tsimul*tsimul))*auxL*auxL*aux1A)/7.0 + (8.0*M_PI*cos(rotsimul)*auxM*(12.0*tsimul - 6.0)*aux1C)/7.0 + (32.0*M_PI*M_PI*cos(rotsimul)*sin(4.0*M_PI*(3.0*tsimul*tsimul - 2.0*tsimul*tsimul*tsimul))*auxL*auxL*aux1C)/7.0 - (8.0*M_PI*auxM*sin(rotsimul)*(12.0*tsimul - 6.0)*aux1B)/7.0 - (32.0*M_PI*M_PI*sin(rotsimul)*sin(4.0*M_PI*(3.0*tsimul*tsimul - 2.0*tsimul*tsimul*tsimul))*auxL*auxL*aux1B)/7.0))/(200.0*((10.0*aux1B*aux1B)/7.0 + (10.0*aux1C*aux1C)/7.0 + (10.0*aux1A*aux1A)/7.0 + 1179.0/28000.0)*((10.0*aux1B*aux1B)/7.0 + (10.0*aux1C*aux1C)/7.0 + (10.0*aux1A*aux1A)/7.0 + 1179.0/28000.0)) + (20000.0*(aux1A*aux1A + aux1B*aux1B + aux1C*aux1C - 8621.0/40000.0)*((2.0*M_PI*M_PI*auxK*auxK*auxL*auxL)/25.0 + (8.0*M_PI*M_PI*cos(rotsimul)*cos(rotsimul)*auxM*auxM*auxL*auxL)/25.0 + (8.0*M_PI*M_PI*auxM*auxM*sin(rotsimul)*sin(rotsimul)*auxL*auxL)/25.0 - (2.0*M_PI*auxK*(12.0*tsimul - 6.0)*aux1A)/5.0 - (4.0*M_PI*M_PI*sin(2.0*M_PI*(3.0*tsimul*tsimul - 2.0*tsimul*tsimul*tsimul))*auxL*auxL*aux1A)/5.0 + (4.0*M_PI*cos(rotsimul)*auxM*(12.0*tsimul - 6.0)*aux1C)/5.0 + (16.0*M_PI*M_PI*cos(rotsimul)*sin(4.0*M_PI*(3.0*tsimul*tsimul - 2.0*tsimul*tsimul*tsimul))*auxL*auxL*aux1C)/5.0 - (4.0*M_PI*auxM*sin(rotsimul)*(12.0*tsimul - 6.0)*aux1B)/5.0 - (16.0*M_PI*M_PI*sin(rotsimul)*sin(4.0*M_PI*(3.0*tsimul*tsimul - 2.0*tsimul*tsimul*tsimul))*auxL*auxL*aux1B)/5.0))/(2989.0*sqrt(1.0 - (4000000.0*(aux1B*aux1B + aux1C*aux1C + aux1A*aux1A - 8621.0/40000.0)*(aux1B*aux1B + aux1C*aux1C + aux1A*aux1A - 8621.0/40000.0))/182329.0)*((10.0*aux1A*aux1A)/7.0 + (10.0*aux1B*aux1B)/7.0 + (10.0*aux1C*aux1C)/7.0 + 1179.0/28000.0)) + (80000000000.0*((2.0*M_PI*auxK*auxL*aux1A)/5.0 - (4.0*M_PI*cos(rotsimul)*auxM*auxL*aux1C)/5.0 + (4.0*M_PI*auxM*sin(rotsimul)*auxL*aux1B)/5.0)*((2.0*M_PI*auxK*auxL*aux1A)/5.0 - (4.0*M_PI*cos(rotsimul)*auxM*auxL*aux1C)/5.0 + (4.0*M_PI*auxM*sin(rotsimul)*auxL*aux1B)/5.0)*(aux1A*aux1A + aux1B*aux1B + aux1C*aux1C - 8621.0/40000.0)*(aux1A*aux1A + aux1B*aux1B + aux1C*aux1C - 8621.0/40000.0))/(544981381.0*(1.0 - (4000000.0*(aux1B*aux1B + aux1C*aux1C + aux1A*aux1A - 8621.0/40000.0)*(aux1B*aux1B + aux1C*aux1C + aux1A*aux1A - 8621.0/40000.0))/182329.0)*sqrt(1.0 - (4000000.0*(aux1B*aux1B + aux1C*aux1C + aux1A*aux1A - 8621.0/40000.0)*(aux1B*aux1B + aux1C*aux1C + aux1A*aux1A - 8621.0/40000.0))/182329.0)*((10.0*aux1A*aux1A)/7.0 + (10.0*aux1B*aux1B)/7.0 + (10.0*aux1C*aux1C)/7 + 1179.0/28000.0)) - (40000.0*((2.0*M_PI*auxK*auxL*aux1A)/5.0 - (4.0*M_PI*cos(rotsimul)*auxM*auxL*aux1C)/5.0 + (4.0*M_PI*auxM*sin(rotsimul)*auxL*aux1B)/5.0)*((4.0*M_PI*auxK*auxL*aux1A)/7.0 - (8.0*M_PI*cos(rotsimul)*auxM*auxL*aux1C)/7.0 + (8.0*M_PI*auxM*sin(rotsimul)*auxL*aux1B)/7.0)*(aux1A*aux1A + aux1B*aux1B + aux1C*aux1C - 8621.0/40000.0))/(2989.0*sqrt(1.0 - (4000000.0*(aux1B*aux1B + aux1C*aux1C + aux1A*aux1A - 8621.0/40000.0)*(aux1B*aux1B + aux1C*aux1C + aux1A*aux1A - 8621.0/40000.0))/182329.0)*((10.0*aux1B*aux1B)/7.0 + (10.0*aux1C*aux1C)/7.0 + (10.0*aux1A*aux1A)/7.0 + 1179.0/28000.0)*((10.0*aux1B*aux1B)/7.0 + (10.0*aux1C*aux1C)/7.0 + (10.0*aux1A*aux1A)/7.0 + 1179.0/28000.0)))/((3721.0*((4000000.0*(aux1A*aux1A + aux1B*aux1B + aux1C*aux1C - 8621.0/40000.0)*(aux1A*aux1A + aux1B*aux1B + aux1C*aux1C - 8621.0/40000.0))/182329.0 - 1.0))/(40000.0*((10.0*aux1B*aux1B)/7.0 + (10.0*aux1C*aux1C)/7.0 + (10.0*aux1A*aux1A)/7.0 + 1179.0/28000.0)*((10.0*aux1B*aux1B)/7.0 + (10.0*aux1C*aux1C)/7.0 + (10.0*aux1A*aux1A)/7.0 + 1179.0/28000.0)) - 1.0) - ((3.0*((2.0*M_PI*auxK*auxL*aux1A)/5.0 + (4.0*M_PI*auxM*sin(rotsimul)*auxL*aux1B)/5.0)*((2.0*M_PI*auxK*auxL*aux1A)/5.0 + (4.0*M_PI*auxM*sin(rotsimul)*auxL*aux1B)/5.0)*aux1C)/(4.0*(aux1B*aux1B + aux1A*aux1A)*(aux1B*aux1B + aux1A*aux1A)*sqrt(aux1B*aux1B + aux1A*aux1A)) + (aux1C*((2.0*M_PI*auxK*(12.0*tsimul - 6.0)*aux1A)/5.0 - (8.0*M_PI*M_PI*auxM*auxM*sin(rotsimul)*sin(rotsimul)*auxL*auxL)/25.0 - (2.0*M_PI*M_PI*auxK*auxK*auxL*auxL)/25.0 + (4.0*M_PI*M_PI*sin(2.0*M_PI*(3.0*tsimul*tsimul - 2.0*tsimul*tsimul*tsimul))*auxL*auxL*aux1A)/5.0 + (4.0*M_PI*auxM*sin(rotsimul)*(12.0*tsimul - 6.0)*aux1B)/5.0 + (16.0*M_PI*M_PI*sin(rotsimul)*sin(4.0*M_PI*(3.0*tsimul*tsimul - 2.0*tsimul*tsimul*tsimul))*auxL*auxL*aux1B)/5.0))/(2.0*(aux1B*aux1B + aux1A*aux1A)*sqrt(aux1B*aux1B + aux1A*aux1A)) + (2.0*M_PI*cos(rotsimul)*auxM*(12.0*tsimul - 6.0))/(5.0*sqrt(aux1B*aux1B + aux1A*aux1A)) + (8.0*M_PI*M_PI*cos(rotsimul)*sin(4.0*M_PI*(3.0*tsimul*tsimul - 2.0*tsimul*tsimul*tsimul))*auxL*auxL)/(5.0*sqrt(aux1B*aux1B + aux1A*aux1A)) + (2.0*M_PI*cos(rotsimul)*auxM*((2.0*M_PI*auxK*auxL*aux1A)/5.0 + (4.0*M_PI*auxM*sin(rotsimul)*auxL*aux1B)/5.0)*auxL)/(5.0*(aux1B*aux1B + aux1A*aux1A)*sqrt(aux1B*aux1B + aux1A*aux1A)))/(aux1C*aux1C/(aux1A*aux1A + aux1B*aux1B) + 1.0) + (((200.0*((2.0*M_PI*auxK*auxL*aux1A)/5.0 - (4.0*M_PI*cos(rotsimul)*auxM*auxL*aux1C)/5.0 + (4.0*M_PI*auxM*sin(rotsimul)*auxL*aux1B)/5.0)*(aux1A*aux1A + aux1B*aux1B + aux1C*aux1C - 8621.0/40000.0))/(49.0*((10.0*aux1B*aux1B)/7.0 + (10.0*aux1C*aux1C)/7.0 + (10.0*aux1A*aux1A)/7.0 + 1179.0/28000.0)*((10.0*aux1B*aux1B)/7.0 + (10.0*aux1C*aux1C)/7.0 + (10.0*aux1A*aux1A)/7.0 + 1179.0/28000.0)) - (3721.0*((4000000.0*(aux1A*aux1A + aux1B*aux1B + aux1C*aux1C - 8621.0/40000.0)*(aux1A*aux1A + aux1B*aux1B + aux1C*aux1C - 8621.0/40000.0))/182329.0 - 1.0)*((4.0*M_PI*auxK*auxL*aux1A)/7.0 - (8.0*M_PI*cos(rotsimul)*auxM*auxL*aux1C)/7.0 + (8.0*M_PI*auxM*sin(rotsimul)*auxL*aux1B)/7.0))/(20000.0*((10.0*aux1B*aux1B)/7.0 + (10.0*aux1C*aux1C)/7.0 + (10.0*aux1A*aux1A)/7.0 + 1179.0/28000.0)*((10.0*aux1B*aux1B)/7.0 + (10.0*aux1C*aux1C)/7.0 + (10.0*aux1A*aux1A)/7.0 + 1179.0/28000.0)*((10.0*aux1B*aux1B)/7.0 + (10.0*aux1C*aux1C)/7.0 + (10.0*aux1A*aux1A)/7.0 + 1179.0/28000.0)))*((61.0*sqrt(1.0 - (4000000.0*(aux1A*aux1A + aux1B*aux1B + aux1C*aux1C - 8621.0/40000.0)*(aux1A*aux1A + aux1B*aux1B + aux1C*aux1C - 8621.0/40000.0))/182329.0)*((4.0*M_PI*auxK*auxL*aux1A)/7.0 - (8.0*M_PI*cos(rotsimul)*auxM*auxL*aux1C)/7.0 + (8.0*M_PI*auxM*sin(rotsimul)*auxL*aux1B)/7.0))/(200.0*((10.0*aux1B*aux1B)/7.0 + (10.0*aux1C*aux1C)/7.0 + (10.0*aux1A*aux1A)/7.0 + 1179.0/28000.0)*((10.0*aux1B*aux1B)/7.0 + (10.0*aux1C*aux1C)/7.0 + (10.0*aux1A*aux1A)/7.0 + 1179.0/28000.0)) + (20000.0*((2.0*M_PI*auxK*auxL*aux1A)/5.0 - (4.0*M_PI*cos(rotsimul)*auxM*auxL*aux1C)/5.0 + (4.0*M_PI*auxM*sin(rotsimul)*auxL*aux1B)/5.0)*(aux1A*aux1A + aux1B*aux1B + aux1C*aux1C - 8621.0/40000.0))/(2989.0*sqrt(1.0 - (4000000.0*(aux1B*aux1B + aux1C*aux1C + aux1A*aux1A - 8621.0/40000.0)*(aux1B*aux1B + aux1C*aux1C + aux1A*aux1A - 8621.0/40000.0))/182329.0)*((10.0*aux1A*aux1A)/7.0 + (10.0*aux1B*aux1B)/7.0 + (10.0*aux1C*aux1C)/7.0 + 1179.0/28000.0))))/((3721.0*((4000000.0*(aux1B*aux1B + aux1C*aux1C + aux1A*aux1A - 8621.0/40000.0)*(aux1B*aux1B + aux1C*aux1C + aux1A*aux1A - 8621.0/40000.0))/182329.0 - 1.0))/(40000.0*((10.0*aux1B*aux1B)/7.0 + (10.0*aux1C*aux1C)/7.0 + (10.0*aux1A*aux1A)/7.0 + 1179.0/28000.0)*((10.0*aux1B*aux1B)/7.0 + (10.0*aux1C*aux1C)/7.0 + (10.0*aux1A*aux1A)/7.0 + 1179.0/28000.0)) - 1.0)*((3721.0*((4000000.0*(aux1B*aux1B + aux1C*aux1C + aux1A*aux1A - 8621.0/40000.0)*(aux1B*aux1B + aux1C*aux1C + aux1A*aux1A - 8621.0/40000.0))/182329.0 - 1.0))/(40000.0*((10.0*aux1B*aux1B)/7.0 + (10.0*aux1C*aux1C)/7.0 + (10.0*aux1A*aux1A)/7.0 + 1179.0/28000.0)*((10.0*aux1B*aux1B)/7.0 + (10.0*aux1C*aux1C)/7.0 + (10.0*aux1A*aux1A)/7.0 + 1179.0/28000.0)) - 1.0);
	inp[8]=-((160.0*M_PI*M_PI*auxK*auxK*auxL*auxL)/427.0 + (640.0*M_PI*M_PI*cos(rotsimul)*cos(rotsimul)*auxM*auxM*auxL*auxL)/427.0 + (640.0*M_PI*M_PI*auxM*auxM*sin(rotsimul)*sin(rotsimul)*auxL*auxL)/427.0 - (800.0*M_PI*auxK*(12.0*tsimul - 6.0)*aux1A)/427.0 - (1600.0*M_PI*M_PI*sin(2.0*M_PI*(3.0*tsimul*tsimul - 2.0*tsimul*tsimul*tsimul))*auxL*auxL*aux1A)/427.0 + (1600.0*M_PI*cos(rotsimul)*auxM*(12.0*tsimul - 6.0)*aux1C)/427.0 + (6400.0*M_PI*M_PI*cos(rotsimul)*sin(4*M_PI*(3.0*tsimul*tsimul - 2.0*tsimul*tsimul*tsimul))*auxL*auxL*aux1C)/427.0 - (1600.0*M_PI*auxM*sin(rotsimul)*(12.0*tsimul - 6.0)*aux1B)/427.0 - (6400.0*M_PI*M_PI*sin(rotsimul)*sin(4.0*M_PI*(3.0*tsimul*tsimul - 2.0*tsimul*tsimul*tsimul))*auxL*auxL*aux1B)/427.0)/sqrt(1.0 - ((2000.0*aux1A*aux1A)/427.0 + (2000.0*aux1B*aux1B)/427.0 + (2000.0*aux1C*aux1C)/427.0 - 8621.0/8540.0)*((2000.0*aux1A*aux1A)/427.0 + (2000.0*aux1B*aux1B)/427.0 + (2000.0*aux1C*aux1C)/427.0 - 8621.0/8540.0)) - (((800.0*M_PI*auxK*auxL*aux1A)/427.0 - (1600.0*M_PI*cos(rotsimul)*auxM*auxL*aux1C)/427.0 + (1600.0*M_PI*auxM*sin(rotsimul)*auxL*aux1B)/427.0)*((800.0*M_PI*auxK*auxL*aux1A)/427.0 - (1600.0*M_PI*cos(rotsimul)*auxM*auxL*aux1C)/427.0 + (1600.0*M_PI*auxM*sin(rotsimul)*auxL*aux1B)/427.0)*((2000.0*aux1A*aux1A)/427.0 + (2000.0*aux1B*aux1B)/427.0 + (2000.0*aux1C*aux1C)/427.0 - 8621.0/8540.0))/(1.0 - ((2000.0*aux1A*aux1A)/427.0 + (2000.0*aux1B*aux1B)/427.0 + (2000.0*aux1C*aux1C)/427.0 - 8621.0/8540.0)*((2000.0*aux1A*aux1A)/427.0 + (2000.0*aux1B*aux1B)/427.0 + (2000.0*aux1C*aux1C)/427.0 - 8621.0/8540.0))*sqrt(1.0 - ((2000.0*aux1A*aux1A)/427.0 + (2000.0*aux1B*aux1B)/427.0 + (2000.0*aux1C*aux1C)/427.0 - 8621.0/8540.0)*((2000.0*aux1A*aux1A)/427.0 + (2000.0*aux1B*aux1B)/427.0 + (2000.0*aux1C*aux1C)/427.0 - 8621.0/8540.0));
  }
*/

extern "C" void calculate_input_trajectory_max_amplitude(double trajectory_time, double amplitude, double *min_traj_amplitude, double *max_traj_amplitude)
  {
   double inp[NUM_JOINTS*3],traj_val;
   double tsimul;
   int njoint,nmagnit;
   for(nmagnit=0;nmagnit<3;nmagnit++)
     {
      min_traj_amplitude[nmagnit]=-log(0.0L); // +Infinite
      max_traj_amplitude[nmagnit]=log(0.0L); // -Infinite
     }
   for(tsimul=0.0;tsimul<trajectory_time;tsimul+=SIM_SLOT_LENGTH)
     {
      calculate_input_trajectory(inp,amplitude,tsimul);
      for(njoint=0;njoint<NUM_JOINTS;njoint++)
         for(nmagnit=0;nmagnit<3;nmagnit++)
           {
            traj_val=inp[njoint+nmagnit*NUM_JOINTS];
            if(traj_val > max_traj_amplitude[nmagnit])
               max_traj_amplitude[nmagnit]=traj_val;
            if(traj_val < min_traj_amplitude[nmagnit])
               min_traj_amplitude[nmagnit]=traj_val;
           }
     }
  }



double calculate_RBF_width(double rbf_distance, double overlap)
  {
   double half_rbf_pos_inc=rbf_distance/2;
   return(sqrt(half_rbf_pos_inc*half_rbf_pos_inc/(-2*log(overlap))));
  }
 
extern "C" void printRBFs(struct rbf_set *rbfs)
  {
   int nneu;
   double rbf_cur_pos,rbf_pos_inc,rbf_width;

   rbf_cur_pos=rbfs->first_bell_pos;
   rbf_pos_inc=(rbfs->last_bell_pos - rbfs->first_bell_pos)/(rbfs->num_rbfs-1);
   rbf_width=calculate_RBF_width(rbf_pos_inc,rbfs->bell_overlap);
   printf("[");
   for(nneu=0;nneu<rbfs->num_rbfs;nneu++,rbf_cur_pos+=rbf_pos_inc)
      printf("%g*exp(-(x-%g)^2/(2*%g^2));",rbfs->bell_amp, rbf_cur_pos, rbf_width);
   printf("0]\n");  
  }

void generate_activityRBF(Simulation *sim, double cur_slot_time, struct rbf_set *rbfs, double input_var, long first_input_neuron, double *last_spk_times, double max_spk_freq)
  {
   int nneu;
   double rbf_cur_pos,rbf_pos_inc,neu_i_current,rbf_width;
   double spk_per,cur_spk_time;
   Neuron *cur_neuron;
   rbf_cur_pos=rbfs->first_bell_pos;
   rbf_pos_inc=(rbfs->last_bell_pos - rbfs->first_bell_pos)/(rbfs->num_rbfs-1);
   rbf_width=calculate_RBF_width(rbf_pos_inc,0.2);
   for(nneu=0;nneu<rbfs->num_rbfs;nneu++,rbf_cur_pos+=rbf_pos_inc)
     {
      neu_i_current=gaussian_function(rbfs->bell_amp, rbf_cur_pos, rbf_width, input_var);
      spk_per=1/(max_spk_freq*neu_i_current);
      cur_neuron=sim->GetNetwork()->GetNeuronAt(first_input_neuron+nneu);
      cur_spk_time=last_spk_times[nneu]+spk_per;
      if(cur_spk_time<cur_slot_time)
         cur_spk_time=cur_slot_time;
      for(;cur_spk_time<cur_slot_time+SIM_SLOT_LENGTH;cur_spk_time+=spk_per)
        {
		 sim->GetQueue()->InsertInputSpikeEvent(cur_spk_time, cur_neuron);
         last_spk_times[nneu]=cur_spk_time;
        }
     }
  }

extern "C" void generate_input_traj_activity(Simulation *neural_sim, double cur_slot_time, double *input_vars, double *min_traj_amplitude, double *max_traj_amplitude)
  {
   static double last_spk_times[NUM_TRAJECTORY_INPUT_NEURONS]={0.0};
   double max_spk_freq=100.0;
   long first_neuron_of_var,num_first_neuron_in_input;
   struct rbf_set rbfs_pos,rbfs_vel; //,rbfs_acc;
   int nvar;
   rbfs_pos.bell_amp=1.0;
   rbfs_pos.bell_overlap=0.2;
   rbfs_pos.first_bell_pos=min_traj_amplitude[0];
   rbfs_pos.last_bell_pos=max_traj_amplitude[0];
   rbfs_pos.num_rbfs=NUM_RBFS;

   rbfs_vel.bell_amp=1.0;
   rbfs_vel.bell_overlap=0.2;
   rbfs_vel.first_bell_pos=min_traj_amplitude[1];
   rbfs_vel.last_bell_pos=max_traj_amplitude[1];
   rbfs_vel.num_rbfs=NUM_RBFS;
/*
   rbfs_acc.bell_amp=1.0;
   rbfs_acc.bell_overlap=0.2;
   rbfs_acc.first_bell_pos=-traj_amplitude*4*M_PI*M_PI;
   rbfs_acc.last_bell_pos=traj_amplitude*4*M_PI*M_PI;
   rbfs_acc.num_rbfs=NUM_RBFS;
*/
   // Position of joints
   for(nvar=0;nvar<NUM_JOINTS;nvar++)
     {
      num_first_neuron_in_input=NUM_RBFS*nvar;
      first_neuron_of_var=FIRST_INPUT_NEURON+num_first_neuron_in_input;
      generate_activityRBF(neural_sim,cur_slot_time,&rbfs_pos,input_vars[nvar],first_neuron_of_var,last_spk_times+num_first_neuron_in_input,max_spk_freq);
     } 
   // Velocity of joints
   for(nvar=0;nvar<NUM_JOINTS;nvar++)
     {
      num_first_neuron_in_input=(NUM_RBFS*NUM_JOINTS)*1 + NUM_RBFS*nvar;
      first_neuron_of_var=FIRST_INPUT_NEURON+num_first_neuron_in_input;
      generate_activityRBF(neural_sim,cur_slot_time,&rbfs_vel,input_vars[nvar+NUM_JOINTS],first_neuron_of_var,last_spk_times+num_first_neuron_in_input,max_spk_freq);
     } 
   // Acceleration of joints
/*
   for(nvar=0;nvar<NUM_JOINTS;nvar++)
     {
      num_first_neuron_in_input=(NUM_RBFS*NUM_JOINTS)*2 + NUM_RBFS*nvar;
      first_neuron_of_var=FIRST_INPUT_NEURON+num_first_neuron_in_input;
      generate_activityRBF(sim,cur_slot_time,&rbfs_acc,input_vars[nvar+NUM_JOINTS*2],first_neuron_of_var,last_spk_times+num_first_neuron_in_input,max_spk_freq);
     } 
*/
  }

extern "C" void generate_robot_state_activity(Simulation *neural_sim, double cur_slot_time, double *robot_state_vars, double *min_traj_amplitude, double *max_traj_amplitude)
  {
   static double last_spk_times[NUM_ROBOT_STATE_INPUT_NEURONS]={0.0};
   double max_spk_freq=100.0;
   long first_neuron_of_var,num_first_neuron_in_input,first_robot_state_neuron;
   struct rbf_set rbfs_pos,rbfs_vel; //,rbfs_acc;
   int nvar;
   rbfs_pos.bell_amp=1.0;
   rbfs_pos.bell_overlap=0.2;
   rbfs_pos.first_bell_pos=min_traj_amplitude[0];
   rbfs_pos.last_bell_pos=max_traj_amplitude[0];
   rbfs_pos.num_rbfs=NUM_RBFS;

   rbfs_vel.bell_amp=1.0;
   rbfs_vel.bell_overlap=0.2;
   rbfs_vel.first_bell_pos=min_traj_amplitude[1];
   rbfs_vel.last_bell_pos=max_traj_amplitude[1];
   rbfs_vel.num_rbfs=NUM_RBFS;
/*
   rbfs_acc.bell_amp=1.0;
   rbfs_acc.bell_overlap=0.2;
   rbfs_acc.first_bell_pos=-traj_amplitude*4*M_PI*M_PI;
   rbfs_acc.last_bell_pos=traj_amplitude*4*M_PI*M_PI;
   rbfs_acc.num_rbfs=NUM_RBFS;
*/
   first_robot_state_neuron=FIRST_INPUT_NEURON+(NUM_RBFS*NUM_JOINTS)*2; // The first input neuron which encodes the robot's state is the following to the last input trajectory neuron
   // Position of joints
   for(nvar=0;nvar<NUM_JOINTS;nvar++)
     {
      num_first_neuron_in_input=NUM_RBFS*nvar;
      first_neuron_of_var=first_robot_state_neuron+num_first_neuron_in_input;
      generate_activityRBF(neural_sim,cur_slot_time,&rbfs_pos,robot_state_vars[nvar],first_neuron_of_var,last_spk_times+num_first_neuron_in_input,max_spk_freq);
     } 
   // Velocity of joints
   for(nvar=0;nvar<NUM_JOINTS;nvar++)
     {
      num_first_neuron_in_input=(NUM_RBFS*NUM_JOINTS)*1 + NUM_RBFS*nvar;
      first_neuron_of_var=first_robot_state_neuron+num_first_neuron_in_input;
      generate_activityRBF(neural_sim,cur_slot_time,&rbfs_vel,robot_state_vars[nvar+NUM_JOINTS],first_neuron_of_var,last_spk_times+num_first_neuron_in_input,max_spk_freq);
     } 
   // Acceleration of joints
/*
   for(nvar=0;nvar<NUM_JOINTS;nvar++)
     {
      num_first_neuron_in_input=(NUM_RBFS*NUM_JOINTS)*2 + NUM_RBFS*nvar;
      first_neuron_of_var=FIRST_INPUT_NEURON+num_first_neuron_in_input;
      generate_activityRBF(sim,cur_slot_time,&rbfs_acc,robot_state_vars[nvar+NUM_JOINTS*2],first_neuron_of_var,last_spk_times+num_first_neuron_in_input,max_spk_freq);
     } 
*/
  }


/////////////////////// GENERATE LEARNING ACTIVITY ///////////////////

double compute_PD_error(double desired_position, double desired_velocity, double actual_position, double actual_velocity, double kp, double kd)
 {
  double t_error;
  t_error=kp*(actual_position-desired_position) + kd*(actual_velocity-desired_velocity);
  return(t_error);
 }

double compute_P_error(double desired_position, double actual_position, double kp)
{
	double t_error;
	t_error = kp*(actual_position - desired_position);
	return(t_error);
}

double error_sigmoid(double error_torque, double max_error_torque)
  {
  return 0.2+0.8*(1-exp(-error_torque*90/max_error_torque));
  }



void generate_stochastic_activity(Simulation *sim, double cur_slot_init, double input_current, long first_learning_neuron, long num_learning_neurons, double *last_spk_times, double max_spk_freq)
  {
   int nneu,spk,max_spk_amplitude,medium_spk_amplitude_up,medium_spk_amplitude_down,min_spk_amplitude,num_spk;

   // Generating Bursting, number of spike generated according to the received error.
   max_spk_amplitude=3;
   medium_spk_amplitude_up=3;
   medium_spk_amplitude_down=2;
   min_spk_amplitude=1;


	if(input_current>=0.75)
		num_spk= max_spk_amplitude;
	else if((input_current <0.75) &&(input_current >0.50)  )
		num_spk= medium_spk_amplitude_up;
	else if((input_current <=0.50) &&(input_current >0.25)  )
		num_spk= medium_spk_amplitude_down;
	else 
		num_spk= min_spk_amplitude;
   
   Neuron *cur_neuron;

   for(nneu=0;nneu<num_learning_neurons;nneu++){
	  double factor_time_since_spk=(cur_slot_init-last_spk_times[nneu])*max_spk_freq;
	  if(factor_time_since_spk>1.0){
		factor_time_since_spk=1.0;
	  }

	  if (factor_time_since_spk*input_current*SIM_SLOT_LENGTH*max_spk_freq > RandomGenerator::drand()){
			cur_neuron=sim->GetNetwork()->GetNeuronAt(first_learning_neuron+nneu);
	        double cur_spk_time=cur_slot_init+SIM_SLOT_LENGTH;
			for(spk=0;spk<num_spk;spk++){
				sim->GetQueue()->InsertInputSpikeEvent(cur_spk_time, cur_neuron);
				cur_spk_time+=SIM_SLOT_LENGTH;
				
			}
			last_spk_times[nneu]=cur_spk_time;
	  }

    }
  }

extern "C" void calculate_error_signals(double *input_vars, double *state_vars, double *error_vars)
  {
   int num_joint;
   for(num_joint=0;num_joint<NUM_JOINTS;num_joint++)
     {
      double desired_position, desired_velocity, actual_position, actual_velocity;
      desired_position=input_vars[num_joint];
      desired_velocity=input_vars[NUM_JOINTS+num_joint];
      actual_position=state_vars[num_joint];
      actual_velocity=state_vars[NUM_JOINTS+num_joint];
      error_vars[num_joint]=compute_PD_error(desired_position, desired_velocity, actual_position, actual_velocity, ROBOT_JOINT_ERROR_KP[num_joint], ROBOT_JOINT_ERROR_KD[num_joint]);
     }
  }

extern "C" void calculate_error_signals_vor(double *input_vars, double *state_vars, double *error_vars)
{
	int num_joint;
	for (num_joint = 0; num_joint<NUM_JOINTS; num_joint++)
	{
		double desired_position, actual_position;
		desired_position = input_vars[num_joint];
		actual_position = state_vars[num_joint];

		error_vars[num_joint] = compute_P_error(desired_position, actual_position,  ROBOT_JOINT_ERROR_KP[num_joint]);
	}
}

extern "C" void calculate_learning_signals(double *error_vars, double *output_vars, double *learning_vars)
  {
   int num_joint;
   double torque_error;
   double i_neu_posit=0.0;
   double i_neu_negat=0.0;
   const double i_base=0.0; // 0.15;
   const double max_low_torque_factor=0.2;
   for(num_joint=0;num_joint<NUM_JOINTS;num_joint++)
     {
      double max_torque, corrective_torque_posit, corrective_torque_negat;
      torque_error=error_vars[num_joint];
      max_torque=MAX_ROBOT_JOINT_TORQUE[num_joint];
      corrective_torque_posit=output_vars[num_joint*2];
      corrective_torque_negat=output_vars[num_joint*2+1];
      // use a sigmoid to adapt torque error signals
      if(torque_error<=0.0)
        {
         i_neu_posit = error_sigmoid(-torque_error,max_torque);
         if(corrective_torque_negat > max_torque*max_low_torque_factor)
            i_neu_negat=0.0;
         else
            i_neu_negat=i_base;
        }
      else
         if(torque_error>0.0)
           {
            i_neu_negat = error_sigmoid(torque_error,max_torque);
            if(corrective_torque_posit > max_torque*max_low_torque_factor)
               i_neu_posit=0.0;
            else 
               i_neu_posit=i_base;
           }
      learning_vars[num_joint*2]=i_neu_posit;
      learning_vars[num_joint*2+1]=i_neu_negat;
     }
  }

extern "C" void generate_learning_activity(Simulation *neural_sim, double cur_slot_init, double *learning_vars)
  {
   static double last_spk_times[NUM_LEARNING_NEURONS]={0.0};
   int num_joint;
   const double max_spk_freq=5;//25; // maximum learning-neuron firing frequency
   for(num_joint=0;num_joint<NUM_JOINTS;num_joint++)
     {
      long first_learning_neuron, first_neuron_in_learning_neurons;
      double i_neu_posit, i_neu_negat;
      i_neu_posit=learning_vars[num_joint*2];
      i_neu_negat=learning_vars[num_joint*2+1];
      first_neuron_in_learning_neurons=(NUM_LEARNING_NEURONS_PER_OUTPUT_VAR*2)*num_joint;
      first_learning_neuron=FIRST_LEARNING_NEURON+first_neuron_in_learning_neurons;
      generate_stochastic_activity(neural_sim, cur_slot_init, i_neu_posit, first_learning_neuron, NUM_LEARNING_NEURONS_PER_OUTPUT_VAR, last_spk_times+first_neuron_in_learning_neurons, max_spk_freq);
      generate_stochastic_activity(neural_sim, cur_slot_init, i_neu_negat, first_learning_neuron+NUM_LEARNING_NEURONS_PER_OUTPUT_VAR, NUM_LEARNING_NEURONS_PER_OUTPUT_VAR, last_spk_times+first_neuron_in_learning_neurons+NUM_LEARNING_NEURONS_PER_OUTPUT_VAR, max_spk_freq);
     } 
  }

//////////////////////////// GENERATE OUTPUT /////////////////////////



extern "C" int compute_output_activity(Simulation *neural_sim, double *output_vars)
  {
	  static MediananFilter filter[NUM_OUTPUT_VARS];


   long nspks, spkneu;
   double spktime;
   static double current_time_check=0;
   double max_time_check=current_time_check;
   int noutputvar;
   ArrayOutputSpikeDriver *neural_activity_output_array;
   const double tau_time_constant=0.001; // 20ms
   //const double kernel_amplitude=sqrt(0.00000002/tau_time_constant); // normalization coefficient
   const double kernel_amplitude = 0.012*150.0; //2*3.141592*25; // normalization coefficient
// Update value of output vars to the current time
   for(noutputvar=0;noutputvar<NUM_OUTPUT_VARS;noutputvar++){
//      output_vars[noutputvar]*=exp(-SIM_SLOT_LENGTH/tau_time_constant);
      output_vars[noutputvar]=0.0;
   }
 
   nspks=0;
   neural_activity_output_array=(ArrayOutputSpikeDriver *)neural_sim->GetOutputSpikeDriver(0); // The first output spike driver in the list is the ArrayOutputSpikeDriver
   while(neural_activity_output_array->RemoveBufferedSpike(spktime,spkneu))
     {
	  if(spktime < current_time_check){
         printf("Unexpected spike time:%lf of neuron: %li (current simulation time: %f)\n",spktime,spkneu,current_time_check);
	  }
	  if(spktime>max_time_check){
		 max_time_check=spktime;
	  }

	  if(spkneu >= FIRST_OUTPUT_NEURON && spkneu < FIRST_OUTPUT_NEURON+NUM_OUTPUT_NEURONS){
         noutputvar=(spkneu-FIRST_OUTPUT_NEURON)/NUM_NEURONS_PER_OUTPUT_VAR;
         output_vars[noutputvar]+=kernel_amplitude;
      }
      nspks++;
   }

//   for (noutputvar = 0; noutputvar<NUM_OUTPUT_VARS; noutputvar++)
//	   output_vars[noutputvar] = filter[noutputvar].medianfilter(output_vars[noutputvar]);
   
   
   current_time_check=max_time_check;
   neural_activity_output_array->OutputBuffer.clear();
   return(nspks);
  }




///////////////////////////// VARIABLES LOG //////////////////////////

extern "C" int create_log(struct log *log, int total_traj_executions, int trajectory_time)
  {
  //Total number of registers that will be stored in the computer memory during the simulation
   int n_log_regs=total_traj_executions*(int)((trajectory_time-((SIM_SLOT_LENGTH)/2.0))/(float)(SIM_SLOT_LENGTH) + 1);
   int errorn;
   log->nregs=0;
   log->regs=(struct log_reg *)malloc(sizeof(struct log_reg)*n_log_regs);
   if(log)
      errorn=0;
   else
      errorn=-1;
   return(errorn);
  }
extern "C" int create_log_vor(struct log_vor *log_vor, int total_traj_executions, int trajectory_time)
{
	//Total number of registers that will be stored in the computer memory during the simulation
	int n_log_regs = total_traj_executions*(int)((trajectory_time - ((SIM_SLOT_LENGTH) / 2.0)) / (float)(SIM_SLOT_LENGTH)+1);
	int errorn;
	log_vor->nregs = 0;
	log_vor->outer_loop_index = -1;
	log_vor->regs = (struct log_reg_vor *)malloc(sizeof(struct log_reg_vor)*n_log_regs);
	if (log_vor)
		errorn = 0;
	else
		errorn = -1;
	return(errorn);
}

extern "C" void log_vars_vor(struct log_vor *log_vor, double time, double *vor_head_vars, double *vor_output_vars, double *cereb_output_vars, double *cereb_learning_vars, double *error_vars, float elapsed_time, unsigned long spk_counter)
{
	int nvar;
	log_vor->regs[log_vor->nregs].time = (float)time;
	log_vor->regs[log_vor->nregs].spk_counter = spk_counter;
	log_vor->regs[log_vor->nregs].consumed_time = elapsed_time;
	for (nvar = 0; nvar < NUM_JOINTS*2; nvar++)
		log_vor->regs[log_vor->nregs].vor_head_vars[nvar] = vor_head_vars ? (float)vor_head_vars[nvar]:0; 
	for (nvar = 0; nvar<NUM_JOINTS*2; nvar++)
		log_vor->regs[log_vor->nregs].vor_output_vars[nvar] = vor_output_vars ? (float)vor_output_vars[nvar]:0;
	for (nvar = 0; nvar < NUM_JOINTS * 2; nvar++)
		log_vor->regs[log_vor->nregs].avatar_head_vars[nvar] = vor_head_vars ? (float)vor_head_vars[nvar] : 0; //we copy the inner-loop results in the outer-loop results. When the avatar answers, these datas are updates.
	for (nvar = 0; nvar<NUM_JOINTS * 2; nvar++)
		log_vor->regs[log_vor->nregs].avatar_output_vars[nvar] = vor_output_vars ? (float)vor_output_vars[nvar] : 0;//we copy the inner-loop results in the outer-loop results. When the avatar answers, these datas are updates.
	for (nvar = 0; nvar<NUM_OUTPUT_VARS; nvar++)
		log_vor->regs[log_vor->nregs].cereb_output_vars[nvar] = cereb_output_vars ? (float)cereb_output_vars[nvar]:0;
	for (nvar = 0; nvar<NUM_OUTPUT_VARS; nvar++)
		log_vor->regs[log_vor->nregs].cereb_learning_vars[nvar] = cereb_learning_vars ? (float)cereb_learning_vars[nvar]:0;
	for (nvar = 0; nvar<NUM_JOINTS; nvar++)
		log_vor->regs[log_vor->nregs].error_vars[nvar] = error_vars ? (float)error_vars[nvar]:0;
	
	log_vor->nregs++;

}

extern "C" void log_vars_vor_first_etage_inner_outer_loop(struct log_vor *log_vor, double time, double *vor_head_vars, double *vor_output_vars, double *cereb_output_vars, float elapsed_time, unsigned long spk_counter)
{
	int nvar;
	log_vor->regs[log_vor->nregs].time = (float)time;
	log_vor->regs[log_vor->nregs].spk_counter = spk_counter;
	log_vor->regs[log_vor->nregs].consumed_time = elapsed_time;
	for (nvar = 0; nvar < NUM_JOINTS * 2; nvar++)
		log_vor->regs[log_vor->nregs].vor_head_vars[nvar] = vor_head_vars ? (float)vor_head_vars[nvar] : 0;
	for (nvar = 0; nvar<NUM_JOINTS * 2; nvar++)
		log_vor->regs[log_vor->nregs].vor_output_vars[nvar] = vor_output_vars ? (float)vor_output_vars[nvar] : 0;
	for (nvar = 0; nvar < NUM_JOINTS * 2; nvar++)
		log_vor->regs[log_vor->nregs].avatar_head_vars[nvar] = 0;// vor_head_vars ? (float)vor_head_vars[nvar] : 0; //we copy the inner-loop results in the outer-loop results. When the avatar answers, these datas are updates.
	for (nvar = 0; nvar < NUM_JOINTS * 2; nvar++)
		log_vor->regs[log_vor->nregs].avatar_output_vars[nvar] = 0;// vor_output_vars ? (float)vor_output_vars[nvar] : 0;//we copy the inner-loop results in the outer-loop results. When the avatar answers, these datas are updates.
	for (nvar = 0; nvar<NUM_OUTPUT_VARS; nvar++)
		log_vor->regs[log_vor->nregs].cereb_output_vars[nvar] = cereb_output_vars ? (float)cereb_output_vars[nvar] : 0;
	for (nvar = 0; nvar<NUM_OUTPUT_VARS; nvar++)
		log_vor->regs[log_vor->nregs].cereb_learning_vars[nvar] = 0;
	for (nvar = 0; nvar<NUM_JOINTS; nvar++)
		log_vor->regs[log_vor->nregs].error_vars[nvar] = 0;

	log_vor->nregs++;
}

extern "C" void log_vars_vor_second_etage_inner_outer_loop(struct log_vor *log_vor, int index, double *cereb_learning_vars, double *error_vars)
{
	int nvar;
	for (nvar = 0; nvar<NUM_OUTPUT_VARS; nvar++)
		log_vor->regs[index].cereb_learning_vars[nvar] = cereb_learning_vars ? (float)cereb_learning_vars[nvar] : 0;
	for (nvar = 0; nvar<NUM_JOINTS; nvar++)
		log_vor->regs[index].error_vars[nvar] = error_vars ? (float)error_vars[nvar] : 0;
}

//INTERPOLATION OF REAL VALUES USING THE IDEAL ONES.
//extern "C" void update_outer_loop_log_vars_vor(struct log_vor *log_vor, int index, float *avatar_head_vars, float *avatar_output_vars)
//{
//	if (log_vor->outer_loop_index < 0){
//		log_vor->outer_loop_index = 0;
//	}
//
//	if (log_vor->outer_loop_index < index){
//		//update the real avatar values (outer values) with the real values that the avatar has answered
//		int nvar;
//		for (nvar = 0; nvar < NUM_JOINTS * 2; nvar++)
//			log_vor->regs[index].avatar_head_vars[nvar] = avatar_head_vars[nvar];
//		for (nvar = 0; nvar < NUM_JOINTS * 2; nvar++)
//			log_vor->regs[index].avatar_output_vars[nvar] = avatar_output_vars[nvar];
//
//		if (log_vor->outer_loop_index + 1 < index){
//			//Interpolate the previous values since the last outer-loop value using the inner-loop values
//			for (int i = log_vor->outer_loop_index + 1; i < index; i++){
//				for (nvar = 0; nvar < NUM_JOINTS * 2; nvar++)
//					log_vor->regs[i].avatar_head_vars[nvar] = log_vor->regs[i - 1].avatar_head_vars[nvar] + (log_vor->regs[i].vor_head_vars[nvar] - log_vor->regs[i - 1].vor_head_vars[nvar]);
//				for (nvar = 0; nvar < NUM_JOINTS * 2; nvar++)
//					log_vor->regs[i].avatar_output_vars[nvar] = log_vor->regs[i - 1].avatar_output_vars[nvar] + (log_vor->regs[i].vor_output_vars[nvar] - log_vor->regs[i - 1].vor_output_vars[nvar]);
//			}
//		}
//
//		log_vor->outer_loop_index = index; //update the outer loop index
//	}
//}

//LINEAL INTERPOLATION OF REAL VALUES USING THE REAL ONES.
extern "C" void update_outer_loop_log_vars_vor(struct log_vor *log_vor, int index, float *avatar_head_vars, float *avatar_output_vars)
{
	if (log_vor->outer_loop_index < 0){
		log_vor->outer_loop_index = 0;
	}

	if (log_vor->outer_loop_index < index){
		//update the real avatar values (outer values) with the real values that the avatar has answered
		int nvar;
		for (nvar = 0; nvar < NUM_JOINTS * 2; nvar++)
			log_vor->regs[index].avatar_head_vars[nvar] = avatar_head_vars[nvar];
		for (nvar = 0; nvar < NUM_JOINTS * 2; nvar++)
			log_vor->regs[index].avatar_output_vars[nvar] = avatar_output_vars[nvar];

		if (log_vor->outer_loop_index + 1 < index){
			float head[NUM_JOINTS * 2];
			float eyes[NUM_JOINTS * 2];
			for (nvar = 0; nvar < NUM_JOINTS * 2; nvar++){
				head[nvar] = (log_vor->regs[index].avatar_head_vars[nvar] - log_vor->regs[log_vor->outer_loop_index].avatar_head_vars[nvar]) / (index - log_vor->outer_loop_index);
				eyes[nvar] = (log_vor->regs[index].avatar_output_vars[nvar] - log_vor->regs[log_vor->outer_loop_index].avatar_output_vars[nvar]) / (index - log_vor->outer_loop_index);
			}
			//Interpolate the previous values since the last outer-loop value using the inner-loop values
			for (int i = log_vor->outer_loop_index + 1; i < index; i++){
				for (nvar = 0; nvar < NUM_JOINTS * 2; nvar++)
					log_vor->regs[i].avatar_head_vars[nvar] = log_vor->regs[i - 1].avatar_head_vars[nvar] + head[nvar];
				for (nvar = 0; nvar < NUM_JOINTS * 2; nvar++)
					log_vor->regs[i].avatar_output_vars[nvar] = log_vor->regs[i - 1].avatar_output_vars[nvar] + eyes[nvar];
			}
		}

		log_vor->outer_loop_index = index; //update the outer loop index
	}
}


extern "C" void log_vars(struct log *log, double time, double *input_vars, double *state_vars, double *torque_vars, double *output_vars, double *learning_vars, double *error_vars, float elapsed_time, unsigned long spk_counter)
  {
   int nvar;
   log->regs[log->nregs].time=(float)time;
   log->regs[log->nregs].spk_counter=spk_counter;
   log->regs[log->nregs].consumed_time=elapsed_time;
   for(nvar=0;nvar<NUM_JOINTS*3;nvar++)
      log->regs[log->nregs].cereb_input_vars[nvar]=input_vars?(float)input_vars[nvar]:0;
   for(nvar=0;nvar<NUM_JOINTS*3;nvar++)
      log->regs[log->nregs].robot_state_vars[nvar]=state_vars?(float)state_vars[nvar]:0;
   for(nvar=0;nvar<NUM_JOINTS;nvar++)
      log->regs[log->nregs].robot_torque_vars[nvar]=torque_vars?(float)torque_vars[nvar]:0;
   for(nvar=0;nvar<NUM_OUTPUT_VARS;nvar++)
      log->regs[log->nregs].cereb_output_vars[nvar]=output_vars?(float)output_vars[nvar]:0;
   for(nvar=0;nvar<NUM_OUTPUT_VARS;nvar++)
      log->regs[log->nregs].cereb_learning_vars[nvar]=learning_vars?(float)learning_vars[nvar]:0;
   for(nvar=0;nvar<NUM_JOINTS;nvar++)
      log->regs[log->nregs].robot_error_vars[nvar]=error_vars?(float)error_vars[nvar]:0;
   log->nregs++;
  }
extern "C" void log_vars_reduced(struct log *log, double time, double *input_vars,double *output_vars_normalized,double *output_vars, double *learning_vars, double *error_vars, float elapsed_time, unsigned long spk_counter)
  {
   
   int nvar;
   log->regs[log->nregs].time=(float)time;
   log->regs[log->nregs].spk_counter=spk_counter;
   log->regs[log->nregs].consumed_time=elapsed_time;
   for(nvar=0;nvar<NUM_JOINTS*2;nvar++)
      log->regs[log->nregs].cereb_input_vars[nvar]=input_vars?(float)input_vars[nvar]:0; 

   for (nvar = 0; nvar<NUM_JOINTS * 2; nvar++)
      log->regs[log->nregs].robot_state_vars[nvar]=output_vars?(float)output_vars_normalized[nvar]:0;
 
   for(nvar=0;nvar<NUM_JOINTS;nvar++)
      log->regs[log->nregs].robot_torque_vars[nvar]=output_vars?(float)output_vars[2*nvar]-(float)output_vars[2*(nvar)+1]:0;
   
   for(nvar=0;nvar<NUM_OUTPUT_VARS;nvar++)
      log->regs[log->nregs].cereb_output_vars[nvar]=output_vars?(float)output_vars[nvar]:0;
   
   for(nvar=0;nvar<NUM_OUTPUT_VARS;nvar++)
      log->regs[log->nregs].cereb_learning_vars[nvar]=learning_vars?(float)learning_vars[nvar]:0;

   for(nvar=0;nvar<NUM_JOINTS;nvar++)
      log->regs[log->nregs].robot_error_vars[nvar]=error_vars?(float)error_vars[nvar]:0;

   log->nregs++;
}
extern "C" int save_and_finish_log(struct log *log, char *file_name)
  {
   int ret;
   int cur_reg,cur_var;
   FILE *fd;
   fd=fopen(file_name,"wt");
   if(fd)
     {
      time_t current_time;
      char *cur_time_string;
      ret=0;
      fprintf(fd,"%s Log file generated by save_log()",COMMENT_CHARS);
      // Get current time
      current_time = time(NULL);
      if(current_time != ((time_t)-1))
        {
         cur_time_string = ctime(&current_time);
         if(cur_time_string)
            fprintf(fd," on %s",cur_time_string);
        }
      fprintf(fd,"\n%s Numer of registers: %i. Columns per register: time consumed_time spk_counter %i_input_vars %i_state_vars %i_torque_vars %i_output_vars %i_learning_vars %i_error_vars\n\n",COMMENT_CHARS,log->nregs,NUM_JOINTS*3,NUM_JOINTS*3,NUM_JOINTS,NUM_OUTPUT_VARS,NUM_OUTPUT_VARS,NUM_JOINTS);
      for(cur_reg=0;cur_reg<log->nregs;cur_reg++)
        {
         ret|=!fprintf(fd,"%10.10g ",log->regs[cur_reg].time);
         ret|=!fprintf(fd,"%10.10g ",log->regs[cur_reg].consumed_time);
         ret|=!fprintf(fd,"%lu ",log->regs[cur_reg].spk_counter);
         for(cur_var=0;cur_var<NUM_JOINTS*3;cur_var++)
            ret|=!fprintf(fd,"%10.10g ",log->regs[cur_reg].cereb_input_vars[cur_var]);
         for(cur_var=0;cur_var<NUM_JOINTS*3;cur_var++)
            ret|=!fprintf(fd,"%10.10g ",log->regs[cur_reg].robot_state_vars[cur_var]);
         for(cur_var=0;cur_var<NUM_JOINTS;cur_var++)
            ret|=!fprintf(fd,"%10.10g ",log->regs[cur_reg].robot_torque_vars[cur_var]);
         for(cur_var=0;cur_var<NUM_OUTPUT_VARS;cur_var++)
            ret|=!fprintf(fd,"%10.10g ",log->regs[cur_reg].cereb_output_vars[cur_var]);
         for(cur_var=0;cur_var<NUM_OUTPUT_VARS;cur_var++)
            ret|=!fprintf(fd,"%10.10g ",log->regs[cur_reg].cereb_learning_vars[cur_var]);
         for(cur_var=0;cur_var<NUM_JOINTS;cur_var++)
            ret|=!fprintf(fd,"%10.10g ",log->regs[cur_reg].robot_error_vars[cur_var]);
         fprintf(fd,"\n");
        }
      fclose(fd);
      free(log->regs);
     }
   else
      ret=1;
   return(ret);
  }

extern "C" int save_and_finish_log_vor(struct log_vor *log_vor, char *file_name)
{
	int ret;
	int cur_reg, cur_var;
	FILE *fd;
	fd = fopen(file_name, "wt");
	if (fd)
	{
		time_t current_time;
		char *cur_time_string;
		ret = 0;
		fprintf(fd, "%s Log file generated by save_log_vor()", COMMENT_CHARS);
		// Get current time
		current_time = time(NULL);
		if (current_time != ((time_t)-1))
		{
			cur_time_string = ctime(&current_time);
			if (cur_time_string)
				fprintf(fd, " on %s", cur_time_string);
		}
		fprintf(fd, "\n%s Numer of registers: %i. Columns per register: time consumed_time spk_counter %i_vor_head_vars %i_vor_output_vars %i_avatar_head_vars %i_avatar_output_vars %i_cereb_output_vars %i_cereb_learning_vars %i_error_vars \n\n", COMMENT_CHARS, log_vor->nregs, NUM_JOINTS * 2, NUM_JOINTS * 2, NUM_JOINTS * 2, NUM_JOINTS * 2, NUM_OUTPUT_VARS, NUM_OUTPUT_VARS, NUM_JOINTS, 2);
		for (cur_reg = 0; cur_reg<log_vor->nregs; cur_reg++)
		{
			ret |= !fprintf(fd, "%10.10g ", log_vor->regs[cur_reg].time);
			ret |= !fprintf(fd, "%10.10g ", log_vor->regs[cur_reg].consumed_time);
			ret |= !fprintf(fd, "%lu ", log_vor->regs[cur_reg].spk_counter);
			for (cur_var = 0; cur_var<NUM_JOINTS*2 ; cur_var++)
				ret |= !fprintf(fd, "%10.10g ", log_vor->regs[cur_reg].vor_head_vars[cur_var]);
			for (cur_var = 0; cur_var<NUM_JOINTS*2 ; cur_var++)
				ret |= !fprintf(fd, "%10.10g ", log_vor->regs[cur_reg].vor_output_vars[cur_var]);
			for (cur_var = 0; cur_var<NUM_JOINTS * 2; cur_var++)
				ret |= !fprintf(fd, "%10.10g ", log_vor->regs[cur_reg].avatar_head_vars[cur_var]);
			for (cur_var = 0; cur_var<NUM_JOINTS * 2; cur_var++)
				ret |= !fprintf(fd, "%10.10g ", log_vor->regs[cur_reg].avatar_output_vars[cur_var]);
			for (cur_var = 0; cur_var<NUM_OUTPUT_VARS ; cur_var++)
				ret |= !fprintf(fd, "%10.10g ", log_vor->regs[cur_reg].cereb_output_vars[cur_var]);
			for (cur_var = 0; cur_var<NUM_OUTPUT_VARS; cur_var++)
				ret |= !fprintf(fd, "%10.10g ", log_vor->regs[cur_reg].cereb_learning_vars[cur_var]);
			for (cur_var = 0; cur_var<NUM_JOINTS; cur_var++)
				ret |= !fprintf(fd, "%10.10g ", log_vor->regs[cur_reg].error_vars[cur_var]);

			
			fprintf(fd, "\n");
		}
		fclose(fd);
		free(log_vor->regs);
	}
	else
		ret = 1;
	return(ret);
}

extern "C" void calculate_input_activity_for_one_trajectory(Simulation *neural_sim, double time){
	int i=0;
	int j=0;
	int n=100;
	for(double t=0; t<0.999; t+=0.002){
		neural_sim->GetQueue()->InsertInputSpikeEvent(time+t, neural_sim->GetNetwork()->GetNeuronAt(n));
		neural_sim->GetQueue()->InsertInputSpikeEvent(time+t, neural_sim->GetNetwork()->GetNeuronAt(n+1));
		neural_sim->GetQueue()->InsertInputSpikeEvent(time+t, neural_sim->GetNetwork()->GetNeuronAt(n+2));
		neural_sim->GetQueue()->InsertInputSpikeEvent(time+t, neural_sim->GetNetwork()->GetNeuronAt(n+3));
		n+=4;

		if(i<19){
			neural_sim->GetQueue()->InsertInputSpikeEvent(time+t, neural_sim->GetNetwork()->GetNeuronAt(j));
			neural_sim->GetQueue()->InsertInputSpikeEvent(time+t, neural_sim->GetNetwork()->GetNeuronAt(j+1));
			neural_sim->GetQueue()->InsertInputSpikeEvent(time+t, neural_sim->GetNetwork()->GetNeuronAt(j+2));
			neural_sim->GetQueue()->InsertInputSpikeEvent(time+t, neural_sim->GetNetwork()->GetNeuronAt(j+3));
		}
		i++;
		if(i==20){
			i=0;
			j+=4;
		}
	}
}

extern "C" void enable_real_time_restriction(Simulation * neural_sim, double slot_time, double max_simulation_delay, float first_section, float second_section, float third_section){
	neural_sim->RealTimeRestrictionObject->SetParameterWatchDog(slot_time, max_simulation_delay, first_section, second_section, third_section);
}


extern "C" void enable_real_time_restriction_in_secondary_thread(Simulation * neural_sim){
	neural_sim->RealTimeRestrictionObject->Watchdog();
}

extern "C" void start_real_time_restriction(Simulation * neural_sim){
	neural_sim->RealTimeRestrictionObject->StartWatchDog();
}

extern "C" void next_step_real_time_restriction(Simulation * neural_sim){
	neural_sim->RealTimeRestrictionObject->NextStepWatchDog();
}

extern "C" void stop_real_time_restriction(Simulation * neural_sim){
	neural_sim->RealTimeRestrictionObject->StopWatchDog();
}


extern "C" void enable_and_start_connection_with_avatar_in_seconday_thread(Simulation * neural_sim, CommunicationAvatarVOR * communicationAvatarVOR, struct log_vor * var_log, string Avatar_IP, unsigned short Avatar_port, int robot_processing_delay1, int robot_processing_delay2){
	communicationAvatarVOR->Communication(Avatar_IP, Avatar_port, neural_sim->RealTimeRestrictionObject, var_log, robot_processing_delay1, robot_processing_delay2);

}

extern "C" void stop_connection_with_avatar(CommunicationAvatarVOR * communicationAvatarVOR){
	communicationAvatarVOR->StopConnection();
}

extern "C" float calculate_ideal_MAE(log_vor var_log_vor, int first_index, int last_index){
	float MAE = 0;
	int N_elements = last_index - first_index;

	if (last_index < var_log_vor.nregs){
		for (int i = first_index; i <= last_index; i++){
			MAE += abs(var_log_vor.regs[i].vor_head_vars[0] - var_log_vor.regs[i].vor_output_vars[0]);
		}
		MAE /= N_elements;
	}
	else{
		MAE = -1;
	}

	return MAE;
}

extern "C" float calculate_avatar_MAE(log_vor var_log_vor, int first_index, int last_index){
	float MAE = 0;
	int N_elements = last_index - first_index;

	if (last_index < var_log_vor.nregs){
		for (int i = first_index; i <= last_index; i++){
//			MAE += abs(var_log_vor.regs[i].avatar_head_vars[0] - var_log_vor.regs[i].avatar_output_vars[0]);
			MAE += abs(var_log_vor.regs[i].avatar_head_vars[1] - var_log_vor.regs[i].avatar_output_vars[1]);
		}
		MAE /= N_elements;
	}
	else{
		MAE = -1;
	}

	return MAE;
}