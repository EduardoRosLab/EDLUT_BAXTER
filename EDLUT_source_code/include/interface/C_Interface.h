/***************************************************************************
 *                            C_Interface.h                             *
 *                           -------------------                           *
 * copyright            : (C) 2009 by Martin Nilsson, Jesus Garrido 
                                                    and Richard Carrillo   *
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

#ifndef LSAM_INTERFACE_H_
#define LSAM_INTERFACE_H_

/*!
 * \file LSAM_Interface.h
 *
 * \author Martin Nilsson
 * \author Jesus Garrido
 * \author Richard Carrido
 * \date October 2009
 *
 * This file declares a set of functions to communicate LSAM with EDLUT.
 */

class Simulation;

/** Initializes the simulation and catches errors.
Initializes the simulation, reads in tables, and catches errors.
For a detailed description of the
EDLUT simulator and its switches, please refer to the EDLUT
documentation.

\param count		is the number of items in the argument vector.
\param parameters	is the vector of switches and their parameters.

Obligatory switches and switch values:
\param "-time Simulation_Time"		 sets the total simulation time (in seconds).
\param "-nf Network_File"		 sets the network description file.
\param "-wf Weights_File"		 sets the weights file.

Optional switches and switch values:
\param "-info"				 shows network information.
\param "-sf File_Name"			 saves the final weights in file File_Name.
\param "-wt Save_Weight_Step"		 sets the step time between weights saving.
\param "-st Step_Time"			 sets the step time in simulation in seconds.
\param "-log File_Name"			 saves the activity register in file File_Name.
\param "-if Input_File"		 	 adds the Input_File file to the input sources of the simulation.
\param "-of Output_File"		 adds the Output_File file to the output targets of the simulation.
\param "-ic IPDirection:Port Server|Client"	 adds the connection as a server or a client at 
                                                 the specified address to the input sources of the simulation.
\param "-oc IPDirection:Port Server|Client"	 adds the connection as a server or a client at
                                                 the specified address to the output targets of the simulation.	 
\param "-ioc IPDirection:Port Server|Client"	 adds the connection as a server or a client at the specified
                                                 address to the input sources and in the output targets of the simulation.

\returns a simulation object, holding the state of simulation. */

Simulation * InitializeAndCatch(int count, char *parameters[]);

/*!
 * \brief Runs simulation preemptably and catches errors.
 * 
 * Similar to RunSimulationSlot(), but catches errors.
 * 
 * \pre The activity has been previously loaded.
 * \param sim is the simulation to run.
 * \param preempt_time is the deadline for the simulation.
 * \returns zero if successful, non-zero (error code) if there was an error.
 */
int SimulateSlotAndCatch(Simulation * sim, double preempt_time);

/* ---------- LSAM interface routines */

/** Initialize simulator.
Creates and initializes a simulation object that holds the state of
simulation for one subnetwork.  For a detailed description of the
EDLUT simulator and its switches, please refer to the EDLUT
documentation.
\param count is the number of EDLUT switches (initialization options).
\param parameters is the vector of EDLUT switches.
\returns a simulation object, holding the state of simulation. */

extern "C" Simulation *LSAM_des_initialize(int count, char *parameters[]);

/** Check if a neuron exists.
Checks if an index corresponds to an existing neuron in the
simulated network.
\param Simul is the simulation object that holds the state of
simulation for one subnetwork.
\param neuron_index is the index of the neuron to be checked.
\returns one if the neuron exists, 0 otherwise. */

extern "C" int LSAM_des_checkIfNeuronExists(Simulation *Simul, int neuron_index);

/** Inject spike in simulation network.
This procedure injects a spike in the network. Normally, the spike is injected in
a dummy neuron in the network, but it can be injected in any neuron.
\param Simul is the simulation object that holds the state of
simulation for one subnetwork.
\param neuron_index is the index of the neuron where the spike should be injected. */

extern "C" void LSAM_des_injectSpike(Simulation *Simul, double time, int neuron_index);

/** Run simulator until a specified time.
Runs a simulation until simulation time reaches a specified limit.
\param Simul is the simulation object that holding the state of simulation
for one subnetwork.
\param preempt_time is the ending time of the simulation.
Simulation can be continued later.
\returns an integer indicating success (zero) or failure (non-zero). */

extern "C" int LSAM_des_simulate(Simulation *Simul, double preempt_time);

/** Obtain the number of spikes generated by a neuron.
Counts the number of spikes generated by a neuron. Normally, the spikes are counted in
output neurons, but any neuron's spike counter may be read.
\param Simul is the simulation object that holds the state of
simulation for one subnetwork.
\param neuron_index is the index of the neuron whose spike counter should be read.
\returns the value of the spike counter. */

extern "C" int LSAM_des__getSpikeCounter(Simulation *Simul, int neuron_index);

/** Set neuron's spike counter.
Sets the spike counter of a neuron. Normally, the counter in
output neurons are set to zero, but any neuron's spike counter may be set to any value.
\param Simul is the simulation object that holds the state of
simulation for one subnetwork.
\param neuron_index is the index of the neuron whose spike counter should be read.
\param value is the value the neuron's counter should be set to. */

extern "C" void LSAM_des_setSpikeCounter(Simulation *Simul, int neuron_index, int value);

/** Obtain the total number of spikes generated.
Returns the total number of spikes handled by the simulator
since the initialization of the simulation object.
\param Simul is the simulation object that holds the state of
simulation for one subnetwork.
\returns the number of events. */

extern "C" int LSAM_des_getTotalSpikeCounter(Simulation *Simul);

#endif /*LSAM_INTERFACE_H_*/

