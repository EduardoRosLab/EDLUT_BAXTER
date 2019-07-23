/***************************************************************************
 *                           RealTimeRestriction.h                         *
 *                           -------------------                           *
 * copyright            : (C) 2014 by Francisco Naveros                    *
 * email                : fnaveros@ugr.es                                  *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef REALTIMERESTRICTION_H_
#define REALTIMERESTRICTION_H_

/*!
 * \file RealTimeRestriction.h
 *
 * \author Francisco Naveros
 * \date June 2014
 *
 * This file declares a class which control how much time is consumed in each slot time for the 
 * robot control. This class is used to implement the real time restriction.
 */
 
#if defined(__APPLE_CC__)
  // MAC OS X includes
#	define REAL_TIME_OSX
#elif defined (__linux__)
  // Linux includes
#	define REAL_TIME_LINUX
#elif (defined(_WIN32) || defined(_WIN64))
#	define REAL_TIME_WINNT
#else
#	error "Unsupported platform"
#endif


#if defined(REAL_TIME_OSX)
#	include <mach/mach.h>
#	include <mach/mach_time.h>
#	include <CoreServices/CoreServices.h>
#	include <unistd.h>
#elif defined(REAL_TIME_LINUX)
#	include <time.h>
#elif defined(REAL_TIME_WINNT)
#	include <windows.h>
#endif


#include <iostream>

using namespace std;


//This real time restriction level set what must be done with each event in real time simulations.
enum RealTimeRestrictionLevel { SIMULATION_TOO_FAST, ALL_EVENTS_ENABLED, LEARNING_RULES_DISABLED, SPIKES_DISABLED, ALL_UNESSENTIAL_EVENTS_DISABLED, ALL_EVENTS_DISABLED};



/*!
 * \class RealTimeRestriction
 *
 * \brief This class is used to implement the real time restriction.
 *
 * This class control how much time is consumed in each slot time for the robot control. This class implement
 * a watchdog which increment the value of the variable "RestrictionLevel" to control the priority of the
 * events that must be processes. It uses the variables "first_section", "second_section" and "third_section"
 * to divide the "simulation_step_time" in four sections. When the "slot_elapsed_time" pass from one secction to another,
 * the RestrictionLevel increases. All the events in EDLUT use this resctriction level to know what they must
 * do. For instant, when the restriction level is higher than zero, the learning rules are ignores.
 *
 * \author Francisco Naveros
 * \date June 2014
 */
class RealTimeRestriction{
	public:

		/*!
		* Boolean variable to indicate if this object generates it own internal clock or uses an external clock.
		*/
		bool use_external_clock;

		/*!
		* Value of the external clock.
		*/
		double external_clock;

		/*!
		* Watch dog sleep period.
		*/
		double sleep_period;


		/*!
		 * Auxiliar variables to measure the time with a precission of nanosecon in differents Operative systemes. 
		 */
		#if defined(REAL_TIME_WINNT)
			// Variables for consumed-CPU-time measurement
		LARGE_INTEGER startt, endt, freq, endt_supervision, startt_outer_loop;

		#elif defined(REAL_TIME_OSX)
		uint64_t startt, endt, elapsed,endt_supervision, elapsed_supervision, startt_outer_loop;
			mach_timebase_info_data_t freq;
	
		#elif defined(REAL_TIME_LINUX)
			// Calculate time taken by a request - Link with real-time library -lrtwatchdog
		struct timespec startt, endt, freq,endt_supervision, startt_outer_loop;
		#endif

		/*!
		 * This is the simulation step time size. It fix the minimum time that the watchdog can control.
		 */
		double simulation_step_size;
		double inv_simulation_step_size;

		/*!
		* Number of simulation step sizes
		*/
		long int N_simulation_step_sizes;

		/*!
		 * This is the maximum time that the simulation can be performed in advance respect to the real time without lost the feedback information 
		 * (the delay in the learning rule of parallel fibres: usually between 100 and 150 ms).
		 */
		double max_simulation_time_in_advance;

		/*!
		 * It stores the simulation time of the actual simulation step. This time must be always between the real time measured in 
		 * this watchdog and the real time plus the max_simulation_time_in_advance. The RestricitionLevel variable will adjust its value to maintain
		 * this variable between these boundaries.
		 */
		double simulation_time;

		/*!
		 * Value between 0 and 1 that contain the fraction of max_simulation_time_in_advance that changes the RestricitionLevel from 0 to 1.
		 */
		double first_section;

		/*!
		 * Value between "first_section" and 1 that contain the fraction of max_simulation_time_in_advance that changes the RestricitionLevel from 1 to 2.
		 */
		double second_section;

		/*!
		 * Value between "second_section" and 1 that contain the fraction of max_simulation_time_in_advance that changes the RestricitionLevel from 2 to 3.
		 */
		double third_section;

		/*!
		 * Value that converts first_section in a value between 0 and max_simulation_time_in_advance*(1.0f-first_section)
		 */
		double first_gap_time;

		/*!
		 * Value that converts second_section in a value between first_gap_time and max_simulation_time_in_advance*(1.0f-second_section)
		 */
		double second_gap_time;

		/*!
		 * Value that converts third_section in a value between second_gap_time and max_simulation_time_in_advance*(1.0f-third_section)
		 */
		double third_gap_time;

		/*!
		 * Restriction level fixed by the watchdog. This variable is used by the rest of OpenMP threads to know what must be done with each event.
		 * This variable can take 5 different values:
		 * TOO_FAST: The simulation is been performed to fast and must be stopped until RestrictionLevel >= 0.
		 * ALL_EVENTS_ENABLED: The simulation is been performed perfectly.
		 * LEARNING_RULES_DISABLED: The simulation is been performed a bit slow. The learning rules are disables to outperform the simulation.
		 * SPIKES_DISABLED: The simulation is been performed slow. The learning rules and spikes are disabled to outperform the simulation.
		 * ALL_UNESSENTIAL_EVENTS_DISABLED: The simulation is been performed too slow (near of the real time limit). All the unessential events are disabled to outperform the simulation.
		 */
		RealTimeRestrictionLevel RestrictionLevel;

		/*!
		* Index that calculate in which "simulation step time" is the real time. This index is used to calculate which elements must be sent to the avatar and where to 
		* store the elements received from the avatar.
		*/
		int real_time_simulation_step_index;


		/*!
		* It controls when the watchdog must be stopped.
		*/
		volatile bool Stop;

		/*!
		* It controls when the watchdog must be started.
		*/
		volatile bool Start;


		/*!
		* It computes local and global statistics of the RealTimeRestrictionLevel
		*/
		long int RestrictionLevelLocalStatistics[5];
		long int RestrictionLevelGlobalStatistics[5];
		long int RestrictionLevelLocalStatisticsCounter;
		long int RestrictionLevelGlobalStatisticsCounter;

	
   	public:
   		

		 /*!
   		 * \brief Default constructor.
   		 * 
   		 * It creates and initializes a new event object.
		 *
   		 */
   		RealTimeRestriction();

   		
   		/*!
   		 * \brief Class destructor.
   		 * 
   		 * It destroies an object of this class.
   		 */
   		~RealTimeRestriction();


   		/*!
   		 * \brief It sets the watchdog parameter.
   		 * 
   		 * It sets the watchdog parameter.
		 * \param new_simulation_step_size
		 * \param new_max_simulation_time_in_advance
		 * \param new_first_section
		 * \param new_second_section
		 * \param new_third_section
   		 */
		void SetParameterWatchDog(double new_simulation_step_size, double new_max_simulation_time_in_advance, float new_first_section, float new_second_section, float new_third_section);


		/*!
		 * \brief It increases the simulation time in a simulation step.
		 *
		 * It increases the simulation time in a simulation step. This function is executed by the main thread.
		 */
		void NextStepWatchDog();

		/*!
		* \brief It increases the simulation time in a simulation step and set the external clock.
		*
		* It increases the simulation time in a simulation step and set the external clock. This function is executed by the main thread.
		*
		* \param new_external_clock
		*/
		void NextStepWatchDog(double new_external_clock);

   		/*!
   		 * \brief It stops the watchdog and force the thread inside the Watchdog function to return.
   		 * 
   		 * It stops the watchdog and force the thread inside the Watchdog function to return. This function is executed by the main thread.
   		 */	
		void StopWatchDog();

		/*!
		* \brief It unlocks the thread inside the Watchdog function.
		*
		* It unlocks the thread inside the Watchdog function. This function is executed by the main thread.
		*/
		void StartWatchDog();

   		/*!
   		 * \brief It performs the watchdog in a infinite loop. 
   		 * 
   		 * It performs the watchdog in a infinite loop. This function is executed by a secondary thread.
   		 */	
		void Watchdog();


		/*!
		* \brief It gets the time_index_after_reset.
		*
		* It gets the time_index_after_reset.
		*
		* \return The time_index_after_reset.
		*/
		int GetRealTimeSimulationStepIndex();

		/*!
		* \brief It sets the external clock option to true.
		*
		* It sets the external clock option to true.
		*/
		void SetExternalClockOption();

		/*!
		* \brief It sets the external clock option to false.
		*
		* It sets the external clock option to true.
		*/
		void SetInternalClockOption();

		/*!
		* \brief It sets the sleep period of the watch dog.
		*
		* It sets the sleep period of the watch dog.
		*/
		void SetSleepPeriod(double new_sleep_period);

		/*!
		* \brief It computes and shows the local statistics of the RealTimeRestrictionLevel since the last local call.
		*
		* It computes and shows the local statistics of the RealTimeRestrictionLevel since the last local call.
		*/
		void ShowLocalStatistics();

		/*!
		* \brief It computes and shows the global statistics of the RealTimeRestrictionLevel.
		*
		* It computes and shows the global statistics of the RealTimeRestrictionLevel.
		*/
		void ShowGlobalStatistics();


};

#endif /*REALTIMERESTRICTION_H_*/
