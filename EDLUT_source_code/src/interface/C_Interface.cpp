/***************************************************************************
 *                           C_Interface.cpp                            *
 *                           -------------------                           *
 * copyright            : (C) 2009 by Martin Nilsson, Jesus Garrido        * 
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

#include "../../include/interface/C_Interface.h"

#include "../../include/simulation/Simulation.h"
#include "../../include/simulation/ParameterException.h"
#include "../../include/simulation/EventQueue.h"
#include "../../include/simulation/ParamReader.h"

#include "../../include/spike/Network.h"
#include "../../include/spike/Neuron.h"
#include "../../include/spike/InputSpike.h"
#include "../../include/spike/EDLUTException.h"
#include "../../include/spike/EDLUTFileException.h"

#include "../../include/communication/ConnectionException.h"


Simulation * InitializeAndCatch(int ac, char *av[]) {
        Simulation * Simul;

	try {
		ParamReader Reader(ac, av);
	        Simul = Reader.CreateAndInitializeSimulation();
	} catch (ParameterException Exc){
		cerr << Exc << endl;
		cerr << av[0] << " -time Simulation_Time -nf Network_File -wf Weights_File";
		cerr << " [-info] [-sf Final_Weights_File] [-wt Save_Weight_Step] [-st Simulation_Step_Time] [-log Activity_Register_File] [-if Input_File] ";
		cerr << "[-ic IPAddress:Port Server|Client] [-of Output_File] [-oc IPAddress:Port Server|Client] [-ioc IPAddress:Port Server|Client]" << endl;	
		Simul = NULL;
	} catch (ConnectionException Exc){
		cerr << Exc << endl;
		Simul = NULL;
	} catch (EDLUTFileException Exc){
		cerr << Exc << endl;
		Simul = NULL;
	} catch (EDLUTException Exc){
		cerr << Exc << endl;
		Simul = NULL;
	}
	return Simul;
}



int SimulateSlotAndCatch(Simulation * sim, double preempt_time) {
	try {
		sim->RunSimulationSlot(preempt_time);
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



/* ---------- Begin LSAM interface routines */

extern "C" Simulation * LSAM_des_initialize(int ac, char *av[]) {
  return InitializeAndCatch(ac, av);
}

extern "C" int LSAM_des_checkIfNeuronExists(Simulation * Simul, int neuron_index) {
  if (neuron_index >= Simul->GetNetwork()->GetNeuronNumber()) return 0;
  if (Simul->GetNetwork()->GetNeuronAt(neuron_index) == NULL) return 0;
  return 1;
}

extern "C" void LSAM_des_injectSpike(Simulation * Simul, double time, int neuron_index) {
	Simul->GetQueue()->InsertEvent(new InputSpike(time, Simul->GetNetwork()->GetNeuronAt(neuron_index)->get_OpenMP_queue_index(), Simul->GetNetwork()->GetNeuronAt(neuron_index)));
}

extern "C" int LSAM_des_simulate(Simulation * Simul, double preempt_time) {
  return SimulateSlotAndCatch(Simul, preempt_time);
}

extern "C" int LSAM_des_getSpikeCounter(Simulation * Simul, int neuron_index) {
  return Simul->GetNetwork()->GetNeuronAt(neuron_index)->GetSpikeCounter();	
}
		
extern "C" void LSAM_des_setSpikeCounter(Simulation * Simul, int neuron_index, int value) {
  Simul->GetNetwork()->GetNeuronAt(neuron_index)->SetSpikeCounter(value);
}
		
extern "C" int LSAM_des_getTotalSpikeCounter(Simulation * Simul) {
  return Simul->GetTotalSpikeCounter();	
}

/* ---------- End of LSAM interface routines */



