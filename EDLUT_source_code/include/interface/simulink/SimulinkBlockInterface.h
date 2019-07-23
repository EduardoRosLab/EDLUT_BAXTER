/***************************************************************************
 *                            SimulinkBlockInterface.h                     *
 *                           -------------------                           *
 * copyright            : (C) 2010 by Jesus Garrido                        *
 *                                                                         *
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

#ifndef SIMULINKBLOCKINTERFACE_H_
#define SIMULINKBLOCKINTERFACE_H_

/*!
 * \file SimulinkBlockInterface.h
 *
 * \author Jesus Garrido
 * \date December 2010
 *
 * This file declares a class for interfacing a simulink block with an EDLUT simulation object.
 */

class Simulation;
class InputBooleanArrayDriver;
class OutputBooleanArrayDriver;
class FileOutputSpikeDriver;
class FileOutputWeightDriver;

/*!
 * \class SimulinkBlockInterface
 *
 * \brief Class for interfacing a simulink block with an EDLUT simulation object.
 *
 * This class abstract methods for running a simulation with EDLUT inside a simulink block.
 *
 * \note The block parameters must be the following:
 * - 1st parameter: Network description file.
 * - 2nd parameter: Weight description file.
 * - 3rd parameter: Log activity file.
 * - 4th parameter: Input map -> Vector mapping each input line with an input cell.
 * - 5th parameter: Output map -> Vector mapping each output line with an output cell:
 *
 * \author Jesus Garrido
 * \date December 2010
 */
class SimulinkBlockInterface {

	private:

		/*!
		 * Simulation object.
		 */
		Simulation * Simul;

		/*!
		 * Input array driver.
		 */
		InputBooleanArrayDriver * InputDriver;

		/*!
		 * Output array driver.
		 */
		OutputBooleanArrayDriver * OutputDriver;

		/*!
		 * Log file driver.
		 */
		FileOutputSpikeDriver * FileDriver;

		/*!
		 * Output weight driver.
		 */
		FileOutputWeightDriver * WeightDriver;



	public:
		/*!
		 * \brief Class constructor.
		 *
		 * It creates a new object..
		 */
		SimulinkBlockInterface();

		/*!
		 * \brief Class destructor.
		 *
		 * It destroys a new object..
		 */
		~SimulinkBlockInterface();

		/*!
		 * \brief Initialize a new simulation object
		 *
		 * It initializes a new simulation object with the values obtained from the simulink
		 * block parameters.
		 *
		 * \param S Pointer to the simulink struct of this block.
		 *
		 * \note This function is though to be called from mdlStart function.
		 */
		void InitializeSimulation(SimStruct *S);

		/*!
		 * \brief Simulate the next simulation step.
		 *
		 * It simulates the next simulation step.
		 *
		 * \param S Pointer to the simulink struct of this block.
		 * \param tid Current simulation time.
		 *
		 * \note This function is though to be called from mdlUpdate function. First, it gets
		 * the inputs from the simulation block and then it simulates the step.
		 */
		void SimulateStep(SimStruct *S, int_T tid);

		/*!
		 * \brief Assign the simulation outputs to the block outputs.
		 *
		 * It assigns the simulation outputs to the block outputs.
		 *
		 * \param S Pointer to the simulink struct of this block.
		 *
		 * \note This function is though to be called from mdlOutputs function. It doesn't simulate
		 * the step, it only assigns the outputs.
		 */
		void AssignOutputs(SimStruct *S);

};

#include "../../../src/interface/simulink/SimulinkBlockInterface.cpp"

#endif /* SIMULINKBLOCKINTERFACE_H_ */
