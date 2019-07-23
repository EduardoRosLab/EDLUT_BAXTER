/***************************************************************************
 *                           VectorBufferedState.h                         *
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

#ifndef VECTORBUFFEREDSTATE_H_
#define VECTORBUFFEREDSTATE_H_

/*!
 * \file VectorBufferedState.h
 *
 * \author Jesus Garrido
 * \author Francisco Naveros
 * \date February 2012
 *
 * This file declares a class which implements the state of a cell vector which
 * stores the last activity happened.
 *
 * \note: This class is a modification of previous BufferedState class. In this new class,
 * it is generated a only object for a neuron model cell vector instead of a object for
 * each cell.
 */

#include "VectorNeuronState.h"

#include "../spike/Interconnection.h"

#include <utility>
#include <list>

using namespace std;

/*!
 * \class VectorBufferedState
 *
 * \brief Spiking neuron current state with activity buffer.
 *
 * This class abstracts the state of a cell vector and stores the last activity happened.
 *
 * \author Jesus Garrido
 * \author Francisco Naveros
 * \date February 2012
 */

typedef pair<double,Interconnection *> InputActivity;

/*!
 * Define a node of the activity list
 */
struct Activity{
	/*!
	 * The input activity
	 */
	InputActivity Spike;

	/*!
	 * The next node in the list
	 */
	struct Activity * NextNode;
};

typedef struct Activity ActivityNode;

class VectorBufferedState: public VectorNeuronState {

	private:

		/*!
		 * \brief First node of the activity list (the oldest one) for all neuron model cell vector.
		 */
		ActivityNode *** FirstElement;

		/*!
		 * \brief Last node of the activity list (the youngest one) for all neuron model cell vector.
		 */
		ActivityNode *** LastElement;

		/*!
		 * \brief Time in which the activity will be removed for all neuron model cell vector.
		 */
		float ** BufferAmplitude;

		/*!
		 * \brief Number of elements inside for all neuron model cell vector.
		 */
		unsigned int ** NumberOfElements;

		/*!
		 * \brief Number of buffers included.
		 */
		unsigned int NumberOfBuffers;


	public:
		/*!
		 * \brief Default constructor with parameters.
		 *
		 * It generates a new state of a cell. The temporal amplitude of each channel must be set by using
		 * SetBufferAmplitude.
		 *
		 * \param NumVariables Number of the state variables this model needs.
		 * \param NumBuffers Number of input channels
		 * \param isTimeDriven It is for a time-driven or a event-driven method.
		 */
		VectorBufferedState(unsigned int NumVariables, unsigned int NumBuffers, bool isTimeDriven);

		/*!
		 * \brief Copies constructor.
		 *
		 * It generates a new objects which copies the parameter.
		 *
		 * \param OldState State being copied.
		 */
		VectorBufferedState(const VectorBufferedState & OldState);

		/*!
		 * \brief It sets the amplitude of the selected buffer for a cell.
		 *
		 * It sets the amplitude of the selected buffer for a cell.
		 *
		 * \param index The cell index inside the vector.
		 * \param NumBuffer Number of the buffer to be set.
		 * \param BufferAmpl Temporal amplitude of the buffer.
		 */
		void SetBufferAmplitude(int index, unsigned int NumBuffer, float BufferAmpl);

		/*!
		 * \brief It adds a new input spike into the buffer of activity for a cell.
		 *
		 * It adds a new input spike into the buffer of activity for a cell. The spike insertion
		 * must be done in ascending order by time.
		 *
		 * \param index The cell index inside the vector.
		 * \param InputConnection Interconnection in which the spike was received.
		 */
		void AddActivity(int index, Interconnection * InputConnection);

		/*!
		 * \brief It removes all the spikes happened before BufferAmplitude time for a cell.
		 *
		 * It removes all the spikes happened before BufferAmplitude time for a cell.
		 *
		 * \param index The cell index inside the vector.
		 */
		void CheckActivity(int index);

		/*!
		 * \brief Add elapsed time to spikes for a cell.
		 *
		 * It adds the elapsed time to spikesfor a cell.
		 *
		 * \param index The cell index inside the vector.
		 * \param ElapsedTime The time since the last update.
		 */
		virtual void AddElapsedTime(int index, double ElapsedTime);

		/*!
		 * \brief Class destructor.
		 *
		 * It destroys an object of this class.
		 */
		virtual ~VectorBufferedState();

		/*!
		 * \brief It gets the number of stored spikes for a cell.
		 *
		 * It gets the number of stored spikes for a cell.
		 *
		 * \param index The cell index inside the vector.
		 * \param NumBuffer Number of the buffer to get the number of stored spikes.
		 *
		 * \return The number of spikes stored in the current state for a cell.
		 */
		unsigned int GetNumberOfSpikes(int index, unsigned int NumBuffer);

		/*!
		 * \brief It gets the time when a spike happened for a cell.
		 *
		 * It gets the time when a spike happened for a cell. The first spike is the first being introduced.
		 *
		 * \param index The cell index inside the vector.
		 * \param Position Position of the spike.
		 * \param NumBuffer Number of the buffer from which the spike will be retrieved.
		 *
		 * \return The time when the Position-th stored spike happened.
		 *
		 * \note This function should be avoided in favour of iterators due to efficiency issues.
		 */
		double GetSpikeTimeAt(int index, unsigned int Position, unsigned int NumBuffer);

		/*!
		 * \brief It gets the connection where a spike happened for a cell.
		 *
		 * It gets the connection where a spike happened for a cell. The first spike is the first being introduced.
		 *
		 * \param index The cell index inside the vector.
		 * \param Position Position of the spike.
		 * \param NumBuffer Number of the buffer from which the spike will be retrieved.
		 *
		 * \return The connection when the Position-th stored spike happened.
		 *
		 * \note This function should be avoided in favour of iterators due to efficiency issues.
		 */
		Interconnection * GetInterconnectionAt(int index, unsigned int Position, unsigned int NumBuffer);

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
void InitializeBufferedStates(int size, float * initialization);


		class Iterator {
			private:
				// Declaring buffer iterator
				ActivityNode * element;

			public:
				/*!
				 * \brief Default class constructor
				 *
				 * Default class constructor. It creates a new null-pointer iterator.
				 */
				Iterator();

				/*!
				 * \brief Copy class constructor
				 *
				 * Copy class constructor. It creates a new pointer pointing to the parameter.
				 *
				 * \param ElemAux Element to be copied.
				 */
				Iterator(const Iterator & ItAux);

				/*!
				 * \brief Copy class constructor
				 *
				 * Copy class constructor. It creates a new pointer pointing to the parameter.
				 *
				 * \param ElemAux Element to be copied.
				 */
				Iterator(ActivityNode * ElemAux);

				/*!
				 * \brief It gets the next element stored in the activity buffer.
				 *
				 * It gets the next element stored in the activity buffer.
				 *
				 * \return An iterator pointing to the next element in the buffer.
				 */
				Iterator & operator++();

				/*!
				 * \brief It compares if two iterators point the same element.
				 *
				 * It compares if two iterators point the same element.
				 *
				 * \return True if the two iterators point the same element. False otherwise.
				 */
				bool operator==(VectorBufferedState::Iterator Aux);

				/*!
				 * \brief It compares if two iterators point different elements.
				 *
				 * It compares if two iterators point different elements.
				 *
				 * \return True if the two iterators point different elements. False otherwise.
				 */
				bool operator!=(VectorBufferedState::Iterator Aux);

				/*! It gets the spike time of the current element pointed by the iterator.
				 *
				 * It gets the spike time of the current element pointed by the iterator.
				 *
				 * \return The spike time of the current element pointed by the iterator.
				 */
				double GetSpikeTime();

				/*! It gets the connection of the current element pointed by the iterator.
				 *
				 * It gets the connection of the current element pointed by the iterator.
				 *
				 * \return The connection of the current element pointed by the iterator.
				 */
				Interconnection * GetConnection();

		};

		/*!
		 * \brief It gets the first element stored in the activity buffer for a cell.
		 *
		 * It gets the first element stored in the activity buffer for a cell.
		 *
		 * \param index The cell index inside the vector.
		 * \param NumBuffer Number of the buffer to iterate.
		 *
		 * \return An iterator pointing to the first element in the buffer for a cell.
		 */
		Iterator Begin(int index, unsigned int NumBuffer);

		/*!
		 * \brief It gets the after-last element stored in the activity buffer for a cell.
		 *
		 * It gets the after-last element stored in the activity buffer for a cell.
		 *
		 * \return An iterator pointing to the after-last element in the buffer for a cell.
		 */
		Iterator End();


};


#endif /* VECTORBUFFEREDSTATE_H_ */
