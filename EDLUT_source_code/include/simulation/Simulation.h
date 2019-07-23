/***************************************************************************
 *                           Simulation.h                                  *
 *                           -------------------                           *
 * copyright            : (C) 2009 by Jesus Garrido and Richard Carrillo   *
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

#ifndef SIMULATION_H_
#define SIMULATION_H_

/*!
 * \file Simulation.h
 *
 * \author Jesus Garrido
 * \author Richard Carrido
 * \date August 2008
 *
 * This file declares a class which abstracts a simulation of an spiking neural network.
 */
 
#include <list>

#include "../spike/EDLUTException.h"
#include "./RealTimeRestriction.h"

#include "./PrintableObject.h"

/*!
 * This constant defines how many events are processed before the RunSimulationSlot method
 * checks that the specified MaxSlotConsumedTime is not violated. A higher number increases
 * the computation overhead but reduces the excess time that this method execution could
 * consume.
 */
#define NUM_EVENTS_PER_TIME_SLOT_CHECK 5UL

class Network;
class EventQueue;
class InputSpikeDriver;
class InputCurrentDriver;
class OutputSpikeDriver;
class OutputWeightDriver;
class Spike;
class Neuron;


/*!
 * \class Simulation
 *
 * \brief Spiking neural network simulation.
 *
 * This class abstract the behaviour of a spiking neural network simulation. It loads the network,
 * gets the input spikes and finally produces the output spikes. Moreover, this class generates some
 * simulation statistics.
 *
 * \author Jesus Garrido
 * \author Richard Carrillo
 * \date August 2008
 */
class Simulation : public PrintableObject{
	
	private:

		/*!
		 * Neural network to simulate.
		 */
		Network * Net;
		
		/*!
		 * Vector of event queue used in the events for each OpenMP thread.
		 */
		EventQueue * Queue;
		
		/*!
		 * Input of activity (SPIKES).
		 */
		list<InputSpikeDriver *> InputSpike;

		/*!
		* Input of activity (CURRENTS).
		*/
		list<InputCurrentDriver *> InputCurrent;
		
		/*!
		 * Output of activity (SPIKE).
		 */
		list<OutputSpikeDriver *> OutputSpike;
		
		/*!
		 * Output of weights.
		 */
		list<OutputWeightDriver *> OutputWeight;
		
		/*!
		 * Monitor of activity.
		 */
		list<OutputSpikeDriver *> MonitorSpike;
		
		/*!
		 * Total time of simulation.
		 */
		double Totsimtime;
		
		/*!
		 * Simulation step
		 */
		double SimulationStep;

		
		/*!
		 * Save weight step
		 */
		double SaveWeightStep;
		
		/*!
		 * Vector or current simulation Time for each OpenMP thread
		 */
		double * CurrentSimulationTime;
		
		
		/*!
		 * Vector of end of simulation for each OpenMP thread
		 */
		bool * EndOfSimulation;
		
		/*!
		 * Vector of stop of simulation for each OpenMP thread
		 */
		bool * StopOfSimulation;

		/*!
		 * Vector of syncronize events of simulation for each OpenMP thread
		 */
		bool * SynchronizeThread;

		/*!
		 * Vector of number of realized updates for each OpenMP thread
		 */
		long long * Updates;
		
		/*!
		 * Vector of sumatory of heap size for each OpenMP thread
		 */
		long long * Heapoc;

		/*!
		 * Vector of total spike count for each OpenMP thread
		 */
		long long * TotalSpikeCounter;

		/*!
		 * Vector of total propated spike count for each OpenMP thread
		 */
		long long * TotalPropagateCounter;

		/*!
		 * Vector of total propated spike event count for each OpenMP thread
		 */
		long long * TotalPropagateEventCounter;

		/*!
 		 * Number of OpenMP queues. 
 		 */
		int NumberOfQueues;

		/*!
 		 * Min interpropagation time between neuron of differents OpenMP queues. This value is used
		 * to create the SynchronizeActivityEvent that insert the buffer of events inside each queue.
 		 */
		double MinInterpropagationTime;

	public:

		/*!
 		 * Real time restriction object used to control a real robot. 
 		 */
		RealTimeRestriction * RealTimeRestrictionObject;

		/*!
		* This variable sets if at least one neuron is monitored
		*/
		bool monitore_neurons;

	protected:
		
		/*!
		 * \brief It realizes the main process of simulation.
		 * 
		 * It propagates the spikes after the input is loaded. This propagation ends when the
		 * simulation time is higher than the total simulation time or when no more activity is
		 * produced.
		 * 
		 * \pre The activity has been previously loaded.
		 * 
		 * \throw EDLUTException If something wrong happends in the spike propagation process.
		 */
		void RunSimulationStep() throw (EDLUTException);
				
	public:
	
		/*Simulation(Network * Net, InputDriver * NewInput, OutputDriver * NewOutput, long SimulationTime);*/
		
		/*!
		 * \brief Class constructor with parameters.
		 * 
		 * It creates a new object an initilizes its elements.
		 * 
		 * \param NetworkFile Network description file name. The network will be loaded from this file.
		 * \param WeightsFile Weights description file name. The network synaptic weights will be loaded from this file.
		 * \param SimulationTime Simulation total time.
		 * \param NewSimulationStep Simulation step time.
		 * 
		 * throw EDLUTException If something wrong happens.
		 */
		Simulation(const char * NetworkFile, const char * WeightsFile, double SimulationTime, double NewSimulationStep=0.00, int NewNumberOfQueues=1) throw (EDLUTException);
		
		/*!
		 * \brief Copy constructor of the class.
		 * 
		 * It creates a new object copied of the parameter.
		 * 
		 * \param ant Current simulation object (copy source).
		 */
		Simulation(const Simulation & ant);
		
		/*!
		 * \brief Object destructor.
		 * 
		 * It destroys an object of Simulation class.
		 */
		~Simulation();
		
		/*!
		 * \brief It sets the saving weights step time.
		 * 
		 * It sets the saving weights step time.
		 * 
		 * \param NewSaveStep The saving step time (in seconds). 0 values don't save the weights.
		 */
		void SetSaveStep(float NewSaveStep);
		
		/*!
		 * \brief It gets the saving weights step time.
		 * 
		 * It gets the saving weights step time.
		 * 
		 * \return The saving step time (in seconds). 0 values don't save the weights.
		 */
		double GetSaveStep();
		
		/*!
		 * \brief It sets the simulation step time.
		 * 
		 * It sets the saving step time.
		 * 
		 * \param NewSimulationStep The simulation step time (in seconds). 0 values don't simulate by steps.
		 */
		void SetSimulationStep(double NewSimulationStep);
		
		/*!
		 * \brief It gets the simulation step time.
		 * 
		 * It gets the simulation step time.
		 * 
		 * \return The simulation step time (in seconds). 0 values don't simulate by step.
		 */
		double GetSimulationStep();

		/*!
		 * \brief It ends the simulation before the next event for one OpenMP thread.
		 * 
		 * It ends the simulation before the next event for one OpenMP thread.
		 * 
		 * \param indexThread index of the OpenMP thread.
		 */
		void EndSimulation(int indexThread);

		/*!
		 * \brief It stops the simulation before the next event for one OpenMP thread.
		 *
		 * It stops the simulation before the next event for one OpenMP thread.
		 * 
		 * \param indexThread index of the OpenMP thread.
		 */
		void StopSimulation(int indexThread);

		/*!
		 * \brief It sets the varible SynchronizeThread to true for a specified OpenMP queue.
		 *
		 * It sets the varible SynchronizeThread to true for a specified OpenMP queue.
		 * 
		 * \param indexThread index of the OpenMP queue.
		 */
		void SetSynchronizeSimulationEvent(int indexThread);

		/*!
		 * \brief It sets the varible SynchronizeThread to false for a specified OpenMP queue.
		 *
		 * It sets the varible SynchronizeThread to false for a specified OpenMP queue.
		 * 
		 * \param indexThread index of the OpenMP queue.
		 */
		void ResetSynchronizeSimulationEvent(int indexThread);

		/*!
		 * \brief It gets the varible SynchronizeThread for a specified OpenMP queue.
		 *
		 * It gets the varible SynchronizeThread for a specified OpenMP queue.
		 * 
		 * \param indexThread index of the OpenMP queue.
		 */
		bool GetSynchronizeSimulationEvent(int indexThread);

		/*!
		 * \brief It synchronize all OpenMP queues.
		 *
		 * It synchronize all OpenMP queues.
		 * 
		 */				 
		void SynchronizeThreads();

		/*!
		 * \brief It adds a new input spike driver to the input driver list.
		 * 
		 * It adds a new input spike driver to the input driver list.
		 * 
		 * \param NewInput The input spike driver to add.
		 */
		 void AddInputSpikeDriver(InputSpikeDriver * NewInput);
		 
		/*!
		 * \brief It return the specified element of the InputSpikeDriver list.
		 * 
		 * It returns the InputSpikeDriver pointer in the specified position of the Simulation's
		 * InputSpikeDriver list.
		 * 
		 * \param ElementPosition The position of the list. First element is in position 0.
		 * \return The InputSpikeDriver in the specified position of the list.
		 */
		 InputSpikeDriver *GetInputSpikeDriver(unsigned int ElementPosition);
		 
		/*!
		 * \brief It removes an input spike driver of the input driver list.
		 * 
		 * It removes an input spike driver of the input driver list.
		 * 
		 * \param NewInput The input driver to remove.
		 */
		 void RemoveInputSpikeDriver(InputSpikeDriver * NewInput);

		 /*!
		 * \brief It adds a new input current driver to the input driver list.
		 *
		 * It adds a new input current driver to the input driver list.
		 *
		 * \param NewInput The input driver to add.
		 */
		 void AddInputCurrentDriver(InputCurrentDriver * NewInput);

		 /*!
		 * \brief It return the specified element of the InputCurrentDriver list.
		 *
		 * It returns the InputCurrentDriver pointer in the specified position of the Simulation's
		 * InputSpikeDriver list.
		 *
		 * \param ElementPosition The position of the list. First element is in position 0.
		 * \return The InputCurrentDriver in the specified position of the list.
		 */
		 InputCurrentDriver *GetInputCurrentDriver(unsigned int ElementPosition);

		 /*!
		 * \brief It removes an input current driver of the input driver list.
		 *
		 * It removes an input current driver of the input driver list.
		 *
		 * \param NewInput The input driver to remove.
		 */
		 void RemoveInputCurrentDriver(InputCurrentDriver * NewInput);
		 
		/*!
		 * \brief It adds a new output driver to the output driver list.
		 * 
		 * It adds a new output driver to the output driver list.
		 * 
		 * \param NewOutput The Output driver to add.
		 */
		 void AddOutputSpikeDriver(OutputSpikeDriver * NewOutput);
		
		/*!
		 * \brief It return the specified element of the OutputSpikeDriver list.
		 * 
		 * It returns the OutputSpikeDriver pointer in the specified position of
		 * the Simulation's InputSpikeDriver list.
		 * 
		 * \param ElementPosition The position of the list. First element is in position 0.
		 * \return The OutputSpikeDriver in the specified position of the list.
		 */
		 OutputSpikeDriver *GetOutputSpikeDriver(unsigned int ElementPosition);

		/*!
		 * \brief It removes an output driver of the output driver list.
		 * 
		 * It removes an output driver of the output driver list.
		 * 
		 * \param NewOutput The output driver to remove.
		 */
		 void RemoveOutputSpikeDriver(OutputSpikeDriver * NewOutput);
		 
		 /*!
		 * \brief It adds a new output driver to the monitor activity list.
		 * 
		 * It adds a new output driver to the monitor activity list.
		 * 
		 * \param NewMonitor The Output driver to add.
		 */
		 void AddMonitorActivityDriver(OutputSpikeDriver * NewMonitor);
		
		/*!
		 * \brief It return the specified element of the MonitorActivityDriver list.
		 * 
		 * It returns the OutputSpikeDriver pointer in the specified position of
		 * the Simulation's MonitorActivityDriver list.
		 * 
		 * \param ElementPosition The position of the list. First element is in position 0.
		 * \return The OutputSpikeDriver in the specified position of the list.
		 */
		 OutputSpikeDriver *GetMonitorActivityDriver(unsigned int ElementPosition);

		/*!
		 * \brief It removes an output driver of the output monitor list.
		 * 
		 * It removes an output driver of the output monitor list.
		 * 
		 * \param NewMonitor The output driver to remove.
		 */
		 void RemoveMonitorActivityDriver(OutputSpikeDriver * NewMonitor);
		 
		 /*!
		 * \brief It adds a new output weight driver to the output driver list.
		 * 
		 * It adds a new output driver to the output driver list.
		 * 
		 * \param NewOutput The Output driver to add.
		 */
		 void AddOutputWeightDriver(OutputWeightDriver * NewOutput);
		
		/*!
		 * \brief It return the specified element of the OutputWeightDriver list.
		 * 
		 * It returns the OutputWeightDriver pointer in the specified position of
		 * the Simulation's OutputWeightDriver list.
		 * 
		 * \param ElementPosition The position of the list. First element is in position 0.
		 * \return The OutputWeightDriver in the specified position of the list.
		 */
		 OutputWeightDriver *GetOutputWeightDriver(unsigned int ElementPosition);

		/*!
		 * \brief It removes an output weight driver of the output driver list.
		 * 
		 * It removes an output driver of the output driver list.
		 * 
		 * \param NewOutput The output driver to remove.
		 */
		 void RemoveOutputWeightDriver(OutputWeightDriver * NewOutput);
		
		/*!
		 * \brief It loads the simulation.
		 *
		 * It loads the simulation (inputs from file, saving weight events, communication events...).
		 *
		 * \throw EDLUTException If something wrong happens.
		 */
		void InitSimulation() throw (EDLUTException);

		/*!
		 * \brief It runs the simulation.
		 * 
		 * It runs the simulation.
		 * 
		 * \throw EDLUTException If something wrong happens.
		 */
		void RunSimulation() throw (EDLUTException);
		
		/*!
		 * \brief It runs the simulation until simulation time > preempt_time.
		 *
		 * It runs the simulation until simulation time > preempt_time.
		 *
		 * \throw EDLUTException If something wrong happens.
		 *
		 * \param preempt_time is the ending time of the time slot.
		 */
		void RunSimulationSlot(double preempt_time) throw (EDLUTException);


		/*!
		 * \brief It writes a spike in the activity log and in the outputs.
		 * 
		 * It writes a spike in the activity log and in the outputs.
		 * 
		 * \param spike The spike to be wrotten.
		 */
		void WriteSpike(const Spike * spike);
		
		/*!
		 * \brief It writes a neuron potential.
		 * 
		 * It writes a neuron potential.
		 * 
		 * \param time The event time.
		 * \param neuron The neuron of the event.
		 */
		void WriteState(float time, Neuron * neuron);
		
		/*!
		 * \brief It saves the current synaptic weights in the output weights driver.
		 * 
		 * It saves the current synaptic weights in the output weights driver.
		 */
		void SaveWeights();
		
		/*!
		 * \brief It sends the happened events.
		 * 
		 * It sends the happened events.
		 */
		void SendOutput();
		
		/*!
		 * \brief Set total spike count.
		 *
		 * Sets the total spike counter.
		 *
		 * \param indexThread index of the OpenMP queue.
		 * \param value is value to set the counter to.
		 */
		void SetTotalSpikeCounter(int indexThread, long long value);

		/*!
		 * \brief Set total spike count.
		 *
		 * Sets the total spike counter.
		 *
		 * \param indexThread index of the OpenMP queue.
		 */
		void IncrementTotalSpikeCounter(int indexThread);

		/*!
		 * \brief It gets the spike count for a OpenMP queue.
		 *
		 * It gets the spike count for a OpenMP queue.
		 *
		 * \param indexThread index of the OpenMP queue.
		 *
		 * \returns the spike count.
		 */
		long long GetTotalSpikeCounter(int indexThread);

		/*!
		 * \brief It gets the total spike count for all OpenMP queues.
		 *
		 * It gets the total spike count for all OpenMP queues.
		 *
		 * \returns the total spike count.
		 */
		long long GetTotalSpikeCounter();

		/*!
		 * \brief It gets the propagate spike count for a OpenMP queue.
		 *
		 * It gets the propagate spike count for a OpenMP queue.
		 *
		 * \param indexThread index of the OpenMP queue.
		 *
		 * \returns the propagate spike count.
		 */
		long long GetTotalPropagateCounter(int indexThread);

		/*!
		 * \brief It gets the total propagate spike count for all OpenMP queues.
		 *
		 * It gets the total propagate spike count for all OpenMP queues.
		 *
		 * \param indexThread index of the OpenMP queue.
		 *
		 * \returns the total propagate spike count.
		 */
		long long GetTotalPropagateCounter();

		/*!
		 * \brief It gets the propagate spike event count for a OpenMP queues.
		 *
		 * It gets the propagate spike event count for a OpenMP queues.
		 *
		 * \param indexThread index of the OpenMP queue.
		 *
		 * \returns the propagate spike event count.
		 */
		long long GetTotalPropagateEventCounter(int indexThread);

		/*!
		 * \brief It gets the total propagate spike event count for all OpenMP queues.
		 *
		 * It gets the total propagate spike event count for all OpenMP queues.
		 *
		 * \param indexThread index of the OpenMP queue.
		 *
		 * \returns the total propagate spike evet count.
		 */
		long long GetTotalPropagateEventCounter();

		/*!
		 * \brief It sets the propagate spike count for a OpenMP queue.
		 *
		 * It sets the propagate spike count for a OpenMP queue.
		 *
		 * \param indexThread index of the OpenMP queue.
		 * \param value is value to set the counter to.
		 */
		void SetTotalPropagateCounter(int indexThread, long long value); 

		/*!
		 * \brief It sets the propagate spike event count for a OpenMP queue.
		 *
		 * It sets the propagate spike event count for a OpenMP queue.
		 *
		 * \param indexThread index of the OpenMP queue.
		 * \param value is value to set the counter to.
		 */
		void SetTotalPropagateEventCounter(int indexThread, long long value); 

		/*!
		 * \brief It increments the propagate spike count for a OpenMP queue.
		 *
		 * It increments the propagate spike count for a OpenMP queue.
		 *
		 * \param indexThread index of the OpenMP queue.
		 */
		void IncrementTotalPropagateCounter(int indexThread); 

		/*!
		 * \brief It increments the propagate spike count for a OpenMP queue.
		 *
		 * It increments the propagate spike count for a OpenMP queue.
		 *
		 * \param indexThread index of the OpenMP queue.
		 * \param increment increment.
		 */
		void IncrementTotalPropagateCounter(int indexThread, int increment); 

		/*!
		 * \brief It increments the propagate spike event count for a OpenMP queue.
		 *
		 * It increments the propagate spike event count for a OpenMP queue.
		 *
		 * \param indexThread index of the OpenMP queue.
		 */
		void IncrementTotalPropagateEventCounter(int indexThread); 

		/*!
		 * \brief It gets the input activity.
		 * 
		 * It gets the input activity.
		 */		
		void GetInput();
		
		/*!
		 * \brief It gets the simulated network.
		 * 
		 * It gets the simulated network.
		 * 
		 * \return The simulated network.
		 */
		Network * GetNetwork() const;
		
		/*!
		 * \brief It gets the event queue.
		 * 
		 * It gets the event queue.
		 * 
		 * \return The simulated network.
		 */
		EventQueue * GetQueue() const;
		
		/*!
		 * \brief It gets the stablished simulation time.
		 * 
		 * It gets the total simulation time.
		 * 
		 * \return The total simulation time.
		 */
		double GetTotalSimulationTime() const;
		
		/*!
		 * \brief It gets the number of simulation updates realized for a OpenMP queue.
		 * 
		 * It gets the number of simulation updates realized for a OpenMP queue.
		 *
		 * \param indexThread index of the OpenMP queue.
		 * 
		 * \return The number of simulation updates.
		 */
		long long GetSimulationUpdates(int indexThread) const;

		/*!
		 * \brief It gets the total number of simulation updates realized.
		 * 
		 * It gets the total number of simulation updates realized.
		 * 
		 * \return The number of simulation updates.
		 */
		long long GetSimulationUpdates() const;
		
		/*!
		 * \brief It gets the sumatory of event queue sizes for a OpenMP queue.
		 * 
		 * It gets the sumatory of event queue sizes for a OpenMP queue.
		 * 
		 * \param indexThread index of the OpenMP queue.
		 * 		
		 * \return The number of acumulated event queue sizes.
		 */
		long long GetHeapAcumSize(int indexThread) const;

		/*!
		 * \brief It gets the sumatory of event queue sizes.
		 * 
		 * It gets the sumatory of event queue sizes.
		 * 
		 * \return The number of acumulated event queue sizes.
		 */
		long long GetHeapAcumSize() const;

		/*!
		 * \brief It prints the information of the object.
		 *
		 * It prints the information of the object.
		 *
		 * \param out The output stream where it prints the object to.
		 * \return The output stream.
		 */
		virtual ostream & PrintInfo(ostream & out);

		/*!
		 * \brief It gets the number of OpenMP queues.
		 * 
		 * It gets the number of OpenMP queues.
		 * 
		 * \return The number of OpenMP queues.
		 */
		int GetNumberOfQueues();

		/*!
		 * \brief It gets the minimum interpropagation delay between neuron of different OpenMP queues.
		 * 
		 * It gets the minimum interpropagation delay between neuron of different OpenMP queues.
		 * 
		 * \return The minimum interpropagation delay
		 */
		double GetMinInterpropagationTime();
	

		/*!
		 * \brief It associate a OpenMP queue thread with a GPU.
		 * 
		 * It associate a OpenMP queue thread with a GPU.
		 * 
		 * \param OpenMP_index index of the OpenMP queue.

		 */
		void SelectGPU(int OpenMP_index);
};

#endif /*SIMULATION_H_*/
