/***************************************************************************
 *                           NeuronState.h                                 *
 *                           -------------------                           *
 * copyright            : (C) 2010 by Jesus Garrido                        *
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

#ifndef NEURONSTATE_H_
#define NEURONSTATE_H_

/*!
 * \file NeuronState.h
 *
 * \author Jesus Garrido
 * \date February 2010
 *
 * This file declares a class which abstracts the current state of a cell.
 */

#define NO_SPIKE_PREDICTED -1

/*!
 * \class NeuronModel
 *
 * \brief Spiking neuron current state.
 *
 * This class abstracts the state of a cell and defines the state variables of
 * that cell.
 *
 * \author Jesus Garrido
 * \date February 2010
 */

class NeuronState {

	private:
		/*!
		 * \brief Number of state variables.
		 */
		unsigned int NumberOfVariables;

		/*!
	   	 * \brief Neuron state variables.
	   	 */
	   	float * StateVars;

	   	/*!
	   	 * \brief Last update time
	   	 */
	   	double LastUpdate;

	   	/*!
	   	 * Next spike predicted time.
	   	 */
	   	double PredictedSpike;

	   	/*!
	   	 * End of the event prediction.
	   	 */
	   	double PredictionEnd;


	protected:
	   	/*!
		 * \brief Time since last spike fired.
		 */
		double LastSpikeTime;

	public:

		/*!
		 * \brief Default constructor with parameters.
		 *
		 * It generates a new state of a cell.
		 *
		 * \param NumVariables Number of the state variables this model needs.
		 */
		NeuronState(unsigned int NumVariables);

		/*!
		 * \brief Copies constructor.
		 *
		 * It generates a new objects which copies the parameter.
		 *
		 * \param OldState State being copied.
		 */
		NeuronState(const NeuronState & OldState);

		/*!
		 * \brief It sets the state variable in a specified position.
		 *
		 * It sets the state variable in a specified position.
		 *
		 * \param position The position of the state variable.
		 * \param NewValue The new value of that state variable.
		 */
		void SetStateVariableAt(unsigned int position,float NewValue);

		/*!
		 * \brief It sets the time when the last update happened.
		 *
		 * It sets the time when the last update happened.
		 *
		 * \param NewTime The time when the last update happened.
		 */
		void SetLastUpdateTime(double NewTime);

		/*!
		 * \brief It sets the time when the next predicted spike will happen.
		 *
		 * It sets the time when the next predicted spike will happen.
		 *
		 * \param NextPredictedTime The time when the next spike is predicted. If no spike is predicted, it returns -1.
		 */
		void SetNextPredictedSpikeTime(double NextPredictedTime);

		/*!
		 * \brief It sets the time when the refractory period finishes.
		 *
		 * It sets the time when the refractory period finishes.
		 *
		 * \param NextRefractoryPeriod The new refractory period.
		 */
		void SetEndRefractoryPeriod(double NextRefractoryPeriod);

		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		virtual ~NeuronState();

		/*!
		 * \brief It gets the number of state variables.
		 *
		 * It gets the number of state variables.
		 *
		 * \return The number of state variables of this model.
		 */
		unsigned int GetNumberOfVariables();

		/*!
		 * \brief It gets the state variable in a specified position.
		 *
		 * It gets the state variable in a specified position.
		 *
		 * \param position The position of the state variable.
		 * \return The value of the position-th state variable.
		 */
		float GetStateVariableAt(unsigned int position);

		/*!
		 * \brief It gets the time when the last update happened.
		 *
		 * It gets the time when the last update happened.
		 *
		 * \return The time when the last update happened.
		 */
		double GetLastUpdateTime();

		/*!
		 * \brief It gets the time when the next predicted spike will happen.
		 *
		 * It gets the time when the next predicted spike will happen.
		 *
		 * \return The time when the next spike is predicted. If no spike is predicted, it returns NO_SPIKE_PREDICTED.
		 */
		double GetNextPredictedSpikeTime();

		/*!
		 * \brief It gets the time when the refractory period finishes.
		 *
		 * It gets the time when the refractory period finishes.
		 *
		 * \return The refractory period.
		 */
		double GetEndRefractoryPeriod();

		/*!
		 * \brief It gets the time since the last spike was fired.
		 *
		 * It gets the time since the last spike was fired.
		 *
		 * \return The time since the last spike fired.
		 */
		double GetLastSpikeTime();

		/*!
		 * \brief It gets the number of variables that you can print in this state.
		 *
		 * It gets the number of variables that you can print in this state.
		 *
		 * \return The number of variables that you can print in this state.
		 */
		virtual unsigned int GetNumberOfPrintableValues();

		/*!
		 * \brief It gets a value to be printed from this state.
		 *
		 * It gets a value to be printed from this state.
		 *
		 * \return The value at position-th position in this state.
		 */
		virtual double GetPrintableValuesAt(unsigned int position);

		/*!
		 * \brief Add elapsed time to spikes.
		 *
		 * It adds the elapsed time to spikes.
		 *
		 * \param ElapsedTime The time since the last update.
		 */
		virtual void AddElapsedTime(float ElapsedTime);


		/*!
		 * \brief It adds a new fired spike to the state.
		 *
		 * It adds a new fired spike to the state. Only changes the last spike time.
		 */
		virtual void NewFiredSpike();

};

#endif /* NEURONSTATE_H_ */

