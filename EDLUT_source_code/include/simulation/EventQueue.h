/***************************************************************************
 *                           EventQueue.h                                  *
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

#ifndef EVENTQUEUE_H_
#define EVENTQUEUE_H_

/*!
 * \file EventQueue.h
 *
 * \author Jesus Garrido
 * \author Richard Carrido
 * \date August 2008
 *
 * This file declares a class which abstracts an event queue by using standard arrays.
 */
 
#include <cstdlib>

//Minimun size of each queue
#define MIN_SIZE 1024
//When a queue reaches his capacitance limit, it is increased in this "RESIZE_FACTOR"
#define RESIZE_FACTOR 2

using namespace std;

class Event;
class Neuron;

/*!
 * \brief Auxiliary struct to take advantage of cache saving event time and pointer in the same array.
 *
 * Auxiliary struct to take advantage of cache saving event time and pointer in the same array.
 */
struct EventForQueue {
	/*!
	* Event that must be stored.
	*/
	Event * EventPtr;
	 
	/*!
	 * Time stamp of the event.
	*/
	double Time;
 };

/*!
 * \class EventQueue
 *
 * \brief Event queue
 *
 * This class abstract the behaviour of an sorted by event time queue by using standard arrays.
 *
 * \author Jesus Garrido
 * \author Richard Carrillo
 * \date August 2008
 */
class EventQueue {
	public:

		/*!
		 * Number of queues, one for each OpenMP queue.
		 */
		int NumberOfQueues;
	
		/*!
		 * Event vector (one for each OpenMP queue).
		 */
		EventForQueue ** Events;

		/*!
		 * Number of elements introduced in the Events array (one position for each OpenMP queue).
		 */
		unsigned int * NumberOfElements;

		/*!
		 * Number of elements allocated in the Event array (one position for each OpenMP queue).
		 */
		unsigned int * AllocatedSize;


		/*!
		 * Spikes vector for event that require synchronization between all the OpenMP queues 
		 * (i.e TimeEventSpikingNeurons_GPU, SaveWeightsEvent, CommunicationEvent and 
		 * SynchronizeActivityEvent.
		 */
		EventForQueue * EventsWithSynchronization;

		/*!
		 * Number of elements introduced in the queue.
		 */
		unsigned int NumberOfElementsWithSynchronization;

		/*!
		 * Number of elements allocated in the queue.
		 */
		unsigned int AllocatedSizeWithSynchronization;


		/*!
		 * Buffer used to store the activity that must be propagated between the different OpenMP queues
		 * in each sinchronization period.
		 */
		Event **** Buffers;

		/*!
		 * Number of elements introduced in each buffer.
		 */
		int ** SizeBuffers;

		/*!
		 * Number of elements allocated in each buffer.
		 */
		int ** AllocatedBuffers;

   

   		/*!
   		 * It swaps the position of two events inside a specific OpenMP queue.
		 *
		 * \param index OpenMP queue index.
		 * \param c1 first object.
		 * \param c2 second object.
   		 */
   		void SwapEvents(int index, unsigned int c1, unsigned int c2);


		/*!
   		 * \brief Resize the event queue to a new size keeping the same elements inside.
		 *
		 * Resize the event queue to a new size keeping the same elements inside.
		 *
		 * \param index OpenMP queue index.
		 * \param NewSize The new size of the event queue.
   		 */
   		void Resize(int index, unsigned int NewSize);

   		/*!
   		 * It swaps the position of two events inside the synchronization queue..
		 *
		 * \param c1 first object.
		 * \param c2 second object.
   		 */
		void SwapEventsWithSynchronization(unsigned int c1, unsigned int c2);

		/*!
   		 * \brief Resize the synchronization event queue to a new size keeping the same elements inside.
		 *
		 * Resize the synchronization event queue to a new size keeping the same elements inside.
		 *
		 * \param NewSize The new size of the synchronization event queue.
   		 */
		void ResizeWithSynchronization(unsigned int NewSize);
   		
   	public:
   	
   		/*!
   		 * \brief Contructor with the number of OpenMP queues that must be created.
   		 * 
   		 * Contructor with the number of OpenMP queues that must be created.
		 *
		 * \param numberOfQueues Number of OpenMP queues.
   		 */
   		EventQueue(int numberOfQueues);
   		
   		/*!
   		 * \brief Object destructor.
   		 * 
   		 * Default object destructor.
   		 */
   		~EventQueue();
   		
   		/*!
   		 * \brief It gets the number of events in a specific queue.
   		 * 
   		 * It gets the number of events in a specific queue.
		 *
		 * \param index OpenMP queue index.
   		 * 
   		 * \return The number of events in a specific queue.
   		 */
   		unsigned int Size(int index) const;

   		
   		/*!
   		 * \brief It inserts a spike in the event queue.
   		 * 
   		 * It inserts a spike in the event queue.
   		 * 
   		 * \param event The new event to insert in the queue.
   		 */
   		void InsertEvent(Event * event);

		/*!
   		 * \brief It inserts a spike in the event queue.
   		 * 
   		 * It inserts a spike in the event queue.
   		 * 
		 * \param index OpenMP queue index.
   		 * \param event The new event to insert in the queue.
   		 */
   		void InsertEvent(int index, Event * event);
   		
   		/*!
   		 * \brief It removes the first event in the queue.
   		 * 
   		 * It removes the first event in the queue. It returns the first event sorted by time.
		 *
		 * \param index OpenMP queue index.
   		 * 
   		 * \return The first event sorted by time.
   		 */
   		Event * RemoveEvent(int index);
   		
   		/*!
   		 * \brief It returns the time of the first event.
   		 * 
   		 * It returns the time of the first event.
		 *
		 * \param index OpenMP queue index.
   		 * 
   		 * \return The time of the first event.
   		 */
   		double FirstEventTime(int index) const;	


		/*!
   		 * \brief It remove all spike events.
   		 * 
   		 * It remove all spike events.
		 *
		 * \param index OpenMP queue index.
   		 */
		void RemoveSpikes(int index);


		
   		/*!
   		 * \brief It inserts a InputSpike in the event queue. 
   		 * 
   		 * It inserts a input spike in the event queue.
   		 * 
		 * \param time of the internal spike.
   		 * \param neuron associated to this input spike.
   		 */
   		void InsertInputSpikeEvent(double time, Neuron * neuron);



   		/*!
   		 * \brief It gets the number of events in the queue with synchronization.
   		 * 
   		 * It gets the number of events in the queue with synchronization.
   		 * 
   		 * \return The number of events in the queue with synchronization.
   		 */
   		unsigned int SizeWithSynchronization() const;

   		
   		/*!
   		 * \brief It inserts a spike in the event queue with synchronization.
   		 * 
   		 * It inserts a spike in the event queue with synchronization.
   		 * 
   		 * \param event The new event to insert in the queue with synchronization.
   		 */
   		void InsertEventWithSynchronization(Event * event);

   		
   		/*!
   		 * \brief It removes the first event in the queue with synchronization.
   		 * 
   		 * It removes the first event in the queue with synchronization. It returns the first event sorted by time.
   		 * 
   		 * \return The first event sorted by time.
   		 */
   		Event * RemoveEventWithSynchronization();
   		
   		/*!
   		 * \brief It returns the time of the first event in the queue with synchronization.
   		 * 
   		 * It returns the time of the first event in the queue with synchronization.
   		 * 
   		 * \return The time of the first event in the queue with synchronization.
   		 */
   		double FirstEventTimeWithSynchronization() const;	

		/*!
   		 * \brief It inserts an event in the buffer.
   		 * 
   		 * It inserts an event in the buffer.
   		 * 
   		 * \param index1 source OpenMP queue index.
		 * \param index2 target OpenMP queue index.
		 * \param NewEvent event inserted in the buffer.
   		 */
		void InsertEventInBuffer(int index1, int index2, Event * NewEvent);

		/*!
   		 * \brief It inserts the events inside the Buffer in the specified OpenMP queue.
   		 * 
   		 * It inserts the events insede the Buffer in the specified OpenMP queue.
   		 * 
		 * \param index target OpenMP queue index.
   		 */
		void InsertBufferInQueue(int index);


		/*!
   		 * \brief It deletes all the event inside the buffer for a specified OpenMP queue. 
   		 * 
   		 * It delete all the event inside the buffer for a specified OpenMP queue.
   		 * 
		 * \param index target OpenMP queue index.
   		 */
		void ResetBuffer(int index);
		
		/*!
   		 * \brief It resizes the buffer corresponding to the source OpenMP queue (index1) and 
		 *  target OpenMP queue (index2)
   		 * 
   		 * It resizes the buffer corresponding to the source OpenMP queue (index1) and 
		 *  target OpenMP queue (index2)
   		 * 
   		 * \param index1 source OpenMP queue index.
		 * \param index2 target OpenMP queue index.
   		 */
		void ResizeBuffer(int index1, int index2);
	
		/*!
   		 * \brief It gets the allocated buffer size corresponding to the source OpenMP queue (index1) and 
		 *  target OpenMP queue (index2)
   		 * 
   		 * It gets the alloctad buffer size.
   		 * 
   		 * \param index1 source OpenMP queue index.
		 * \param index2 target OpenMP queue index.
   		 */
		int GetAllocatedBuffer(int index1, int index2);
		
		/*!
   		 * \brief It gets the number of events inside the buffer corresponding to the source OpenMP
		 * queue (index1) and target OpenMP queue (index2)
   		 * 
   		 * It gets the number of events.
   		 * 
   		 * \param index1 source OpenMP queue index.
		 * \param index2 target OpenMP queue index.
   		 */
		int GetSizeBuffer(int index1, int index2);

		/*!
   		 * \brief It increments the number of events inside the buffer corresponding to the source OpenMP
		 * queue (index1) and target OpenMP queue (index2)
   		 * 
   		 * It increments the number of events.
   		 * 
   		 * \param index1 source OpenMP queue index.
		 * \param index2 target OpenMP queue index.
   		 */
		void IncrementSizeBuffer(int index1, int index2);

		/*!
   		 * \brief It fixed to zero the number of events inside the buffer corresponding to the source OpenMP
		 * queue (index1) and target OpenMP queue (index2)
   		 * 
   		 * It increments the number of events.
   		 * 
   		 * \param index1 source OpenMP queue index.
		 * \param index2 target OpenMP queue index.
   		 */
		void ResetSizeBuffer(int index1, int index2);
		
};

#endif /*EVENTQUEUE_H_*/
