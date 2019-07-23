/***************************************************************************
 *                           VectorNeuronState.h                           *
 *                           -------------------                           *
 * copyright            : (C) 2012 by Jesus Garrido and Francisco Naveros  *
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

#ifndef VECTORNEURONSTATE_H_
#define VECTORNEURONSTATE_H_

/*!
 * \file VectorNeuronState.h
 *
 * \author Jesus Garrido
 * \author Francisco Naveros
 * \date February 2012
 *
 * This file declares a class which abstracts the current state of a cell vector.
 *
 * \note: This class is a modification of previous NeuronState class. In this new class,
 * it is generated a only object for a neuron model cell vector instead of a object for
 * each cell.
 */

#define NO_SPIKE_PREDICTED -1

/*!
 * \class VectorNeuronState
 *
 * \brief Spiking neuron current state.
 *
 * This class abstracts the state of a cell vector and defines the state variables of
 * that cell vector.
 *
 * \author Francisco Naveros
 * \date February 2012
 */



class VectorNeuronState {

	public:
		/*!
		 * \brief Number of state variables for each cells.
		 */
		unsigned int NumberOfVariables;

		/*!
	   	 * \brief Neuron state variables for all neuron model cell vector.
	   	 */
	   	float * VectorNeuronStates;

	   	/*!
	   	 * \brief Last update time for all neuron model cell vector.
	   	 */
	   	double * LastUpdate;

	   	/*!
	   	 * brief Next spike predicted time for all neuron model cell vector (only for event-driven methods).
	   	 */
	   	double * PredictedSpike;

	   	/*!
	   	 * brief End of the event prediction for all neuron model cell vector (only for event-driven methods).
	   	 */
	   	double * PredictionEnd;


		/*!
	   	 * brief It is for a time-driven method or for a event-driven method.
	   	 */
		bool TimeDriven;

	   	/*!
		 * \brief Time since last spike fired for all neuron model cell vector.
		 */
		double * LastSpikeTime;

	   	/*!
		 * \brief The cell number inside the vector.
		 */
		int SizeStates;


		/*!
		 * \brief Time-driven methods in CPU use this vector to indicate which neurons have to
		 * generate a internal spike after a update event.
		 */
		bool * InternalSpike;

		/*!
		 * \brief Time-driven methods in CPU use this vector to indicate which neurons have to
		 * generate a internal spike after a update event.
		 */
		int * InternalSpikeIndexs;
		int NInternalSpikeIndexs;



		/*!
		 * brief It is for a time-driven method or for a event-driven method.
		 */
		bool Is_Monitored;


		/*!
		 * brief It is used to store the state of a GPU neuron model. 
		 */
		bool Is_GPU;


		/*!
		 * brief It is used to store initial state of the neuron model. 
		 */
		float * InitialState;
	

		/*!
		 * \brief Default constructor with parameters.
		 *
		 * It generates a new state of a cell vector.
		 *
		 * \param NumVariables Number of the state variables this model needs.
		 * \param isTimeDriven It is for a time-driven or a event-driven method.
		 */
		VectorNeuronState(unsigned int NumVariables, bool isTimeDriven);

		/*!
		 * \brief Default constructor with parameters.
		 *
		 * It generates a new state of a cell vector.
		 *
		 * \param NumVariables Number of the state variables this model needs.
		 * \param isTimeDriven It is for a time-driven or a event-driven method.
		 * \param Is_GPU
		 */
		VectorNeuronState(unsigned int NumVariables, bool isTimeDriven, bool Is_GPU);

		/*!
		 * \brief Copies constructor.
		 *
		 * It generates a new objects which copies the parameter.
		 *
		 * \param OldState State being copied.
		 */
		VectorNeuronState(const VectorNeuronState & OldState);

		/*!
		 * \brief Copies constructor.
		 *
		 * It generates a new objects which copies the parameter for only a neuron.
		 *
		 * \param OldState State being copied.
		 * \param index cell index. 
		 */
		VectorNeuronState(const VectorNeuronState & OldState, int index);

		/*!
		 * \brief It sets the state variable for a cell in a specified position.
		 *
		 * It sets the state variable for a cell in a specified position.
		 *
		 * \param index The cell index inside the vector.
		 * \param position The position of the state variable.
		 * \param NewValue The new value of that state variable.
		 */
		void SetStateVariableAt(int index, int position, float NewValue);

		/*!
		 * \brief It increments the state variable for a cell in a specified position.
		 *
		 * It increments the state variable for a cell in a specified position.
		 *
		 * \param index The cell index inside the vector.
		 * \param position The position of the state variable.
		 * \param Increments The increments of that state variable.
		 */
		void IncrementStateVariableAt(int index, int position, float Increment);

		/*!
		 * \brief It increments the state variable for a cell in a specified position.
		 *
		 * It increments the state variable for a cell in a specified position.
		 *
		 * \param index The cell index inside the vector.
		 * \param position The position of the state variable.
		 * \param Increments The increments of that state variable.
		 */
		//void IncrementStateVariableAtCPU(int index, int position, float Increment);
		inline void IncrementStateVariableAtCPU(int index, int position, float Increment){
			VectorNeuronStates[index*NumberOfVariables + position]+= Increment;
		}

		/*!
		 * \brief It increments the state variable for a cell in a specified position.
		 *
		 * It increments the state variable for a cell in a specified position.
		 *
		 * \param index The cell index inside the vector.
		 * \param position The position of the state variable.
		 * \param Increments The increments of that state variable.
		 */
		//void IncrementStateVariableAtGPU(int index, int position, float Increment);
		inline void IncrementStateVariableAtGPU(int index, int position, float Increment){
			this->VectorNeuronStates[this->SizeStates*position + index]+= Increment;
		}

		/*!
		 * \brief It sets the time when the last update happened for a cell.
		 *
		 * It sets the time when the last update happened for a cell.
		 *
		 * \param index The cell index inside the vector.
		 * \param NewTime The time when the last update happened.
		 */
		void SetLastUpdateTime(int index, double NewTime);

		/*!
		* \brief It sets the time when the last update happened for all cells.
		*
		* It sets the time when the last update happened for all cells.
		*
		* \param NewTime The time when the last update happened.
		*/
		void SetLastUpdateTime(double NewTime);

		/*!
		 * \brief It sets the time when the next predicted spike will happen for a cell.
		 *
		 * It sets the time when the next predicted spike will happen for a cell.
		 *
		 * \param index The cell index inside the vector.
		 * \param NextPredictedTime The time when the next spike is predicted. If no spike is predicted, it returns -1.
		 */
		void SetNextPredictedSpikeTime(int index, double NextPredictedTime);

		/*!
		 * \brief It sets the time when the refractory period finishes for a cell.
		 *
		 * It sets the time when the refractory period finishes for a cell.
		 *
		 * \param index The cell index inside the vector.
		 * \param NextRefractoryPeriod The new refractory period.
		 */
		void SetEndRefractoryPeriod(int index, double NextRefractoryPeriod);

		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		virtual ~VectorNeuronState();

		/*!
		 * \brief It gets the number of state variables.
		 *
		 * It gets the number of state variables.
		 *
		 * \return The number of state variables of this model.
		 */
		unsigned int GetNumberOfVariables();

		/*!
		 * \brief It gets the state variable for a cell in a specified position.
		 *
		 * It gets the state variable for a cell in a specified position.
		 *
		 * \param index The cell index inside the vector.
		 * \param position The position of the state variable.
		 * \return The value of the position-th state variable.
		 */
		//virtual float GetStateVariableAt(int index, int position);
		float GetStateVariableAt(int index, int position){
			if (Is_GPU == false){
				return VectorNeuronStates[index*NumberOfVariables + position];
			}
			else{
				return VectorNeuronStates[this->SizeStates*position + index];
			}
		}


		/*!
		 * \brief It gets the pointer to the state variables for a cell.
		 *
		 * It gets the pointer to the state variables for a cell.
		 *
		 * \param index The cell index inside the vector.
		 * \return The pointer of the position-th state variable.
		 */
		//virtual float * GetStateVariableAt(int index);
		float * GetStateVariableAt(int index){
			return VectorNeuronStates + (index*NumberOfVariables);
		}


		/*!
		 * \brief It gets the time when the last update happened for a cell.
		 *
		 * It gets the time when the last update happened for a cell.
		 *
		 * \param index The cell index inside the vector.
		 *
		 * \return The time when the last update happened for a cell.
		 */
		//double GetLastUpdateTime(int index);
		inline double GetLastUpdateTime(int index){
			return this->LastUpdate[index];
		}

		/*!
		 * \brief It gets the time when the next predicted spike will happen for a cell.
		 *
		 * It gets the time when the next predicted spike will happen for a cell.
		 *
		 * \param index The cell index inside the vector.
		 *
		 * \return The time when the next spike is predicted for a cell. If no spike is predicted, it returns NO_SPIKE_PREDICTED.
		 */
		double GetNextPredictedSpikeTime(int index);

		/*!
		 * \brief It gets the time when the refractory period finishes  for a cell.
		 *
		 * It gets the time when the refractory period finishes for a cell.
		 *
		 * \param index The cell index inside the vector.
		 *
		 * \return The refractory period for a cell.
		 */
		//double GetEndRefractoryPeriod(int index);
		inline double GetEndRefractoryPeriod(int index){
			return this->PredictionEnd[index];
		}

		/*!
		 * \brief It gets the time since the last spike was fired for a cell.
		 *
		 * It gets the time since the last spike was fired for a cell.
		 *
		 * \param index The cell index inside the vector.
		 *
		 * \return The time since the last spike fired for a cell.
		 */
		//double GetLastSpikeTime(int index);
		inline double GetLastSpikeTime(int index){
			return this->LastSpikeTime[index];
		}

		/*!
		 * \brief It gets the number of variables that you can print in this state.
		 *
		 * It gets the number of variables that you can print in this state.
		 *
		 * \return The number of variables that you can print in this state.
		 */
		virtual unsigned int GetNumberOfPrintableValues();

		/*!
		 * \brief It gets a value to be printed from this state for a cell.
		 *
		 * It gets a value to be printed from this state for a cell.
		 *
		 * \param index The cell index inside the vector.
		 * \param position inside a neuron state.
		 *
		 * \return The value at position-th position in this state for a cell.
		 */
		virtual double GetPrintableValuesAt(int index, int position);

		/*!
		 * \brief Add elapsed time to spikes for a cell.
		 *
		 * It adds the elapsed time to spikes for a cell.
		 *
		 * \param index The cell index inside the vector.
		 * \param ElapsedTime The time since the last update for a cell.
		 */
		virtual void AddElapsedTime(int index, double ElapsedTime);


		/*!
		 * \brief It adds a new fired spike to the state for a cell.
		 *
		 * It adds a new fired spike to the state for a cell. Only changes the last spike time.
		 *
		 * \param index The cell index inside the vector.
		 */
		virtual void NewFiredSpike(int index);


		/*!
		 * \brief It sets the cell number inside the VectorNeuronState.
		 *
		 * It sets the cell number inside the VectorNeuronState.
		 *
		 * \param size Cell number inside the VectorNeuronState.
		 */
		void SetSizeState(int size);

		/*!
		 * \brief It gets the cell number inside the VectorNeuronState.
		 *
		 * It gets the cell number inside the VectorNeuronState.
		 *
		 * \return The cell number inside the VectorNeuronState.
		 */
		int GetSizeState();

		/*!
		 * \brief It sets if the VectorNeuronState is for a time-driven or a event-driven method.
		 *
		 * It sets if the VectorNeuronState is for a time-driven or a event-driven method.
		 *
		 * \param isTimeDriven it is for a time-driven or a event-driven method.
		 */
		void SetTimeDriven(bool isTimeDriven);

		/*!
		 * \brief It gets if the VectorNeuronState is for a time-driven or a event-driven method.
		 *
		 * It gets if the VectorNeuronState is for a time-driven or a event-driven method.
		 *
		 * \return if the VectorNeuronState is for a time-driven or a event-driven method.
		 */
		bool GetTimeDriven();

		/*!
		 * \brief It initialice all vectors with size size and copy initialization inside VectorNeuronStates
		 * for each cell.
		 *
		 * It initialice all vectors with size size and copy initialization inside VectorNeuronStates
		 * for each cell.
		 *
		 * \param size cell number inside the VectorNeuronState.
		 * \param initialization initial state for each cell.
		 */
		void InitializeStates(int size, float * initialization);

		/*!
		 * \brief It gets the InternalSpike vector.
		 *
		 * It gets the InternalSpike vector.
		 *
		 * \return The InternalSpike vector
		 */
		virtual bool * getInternalSpike();

		/*!
		 * \brief It gets the InternalSpikeIndex vector.
		 *
		 * It gets the InternalSpikeIndex vector.
		 *
		 * \return The InternalSpikeIndex vector
		 */
		int * getInternalSpikeIndexs();
		
		/*!
		 * \brief It gets the NInternalSpikeIndex vector.
		 *
		 * It gets the NInternalSpikeIndex vector.
		 *
		 * \return The NInternalSpikeIndex vector
		 */
		int getNInternalSpikeIndexs();

		/*!
		 * \brief It sets if some neuron is monitored.
		 *
		 * It sets if some neuron is monitored.
		 *
		 * \param monitored 
		 */
		void Set_Is_Monitored(bool monitored);


		/*!
		 * \brief It gets if some neuron is monitored.
		 *
		 * It gets if some neuron is monitored.
		 *
		 * \return if some neuron is monitored. 
		 */
		bool Get_Is_Monitored();


		/*!
		 * \brief It resets the neuron state of a neuron with the initial state.
		 *
		 * It resets the neuron state of a neuron with the initial state.
		 *
		 * \param index The cell index inside the vector. 
		 */
		void ResetState(int index);

};

#endif /* VECTORNEURONSTATE_H_ */

