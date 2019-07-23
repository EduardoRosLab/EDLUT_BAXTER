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

#include "../../include/simulation/RealTimeRestriction.h"

#include <stdio.h>

#if (defined (_WIN32) || defined(_WIN64))
   #include "windows.h"
#else 
   #include <unistd.h>
#endif




RealTimeRestriction::RealTimeRestriction() : use_external_clock(false), external_clock(0.0), sleep_period(0.0), simulation_step_size(0.0), inv_simulation_step_size(0.0), N_simulation_step_sizes(0), max_simulation_time_in_advance(0.0f), first_section(1.0f), second_section(1.0f), third_section(1.0f),
Stop(false), Start(false), RestrictionLevel(ALL_EVENTS_ENABLED), real_time_simulation_step_index(0){
	for (int i=0; i < 5; i++){
		RestrictionLevelLocalStatistics[i]=0;
		RestrictionLevelGlobalStatistics[i]=0;
	}
	RestrictionLevelLocalStatisticsCounter=0;
	RestrictionLevelGlobalStatisticsCounter=0;

}

RealTimeRestriction::~RealTimeRestriction(){
	Stop = true;
}

void RealTimeRestriction::SetParameterWatchDog(double new_simulation_step_size, double new_max_simulation_time_in_advance, float new_first_section, float new_second_section, float new_third_section){
	simulation_step_size = new_simulation_step_size;
	inv_simulation_step_size = 1.0 / new_simulation_step_size;

	if(new_max_simulation_time_in_advance<new_simulation_step_size){
		max_simulation_time_in_advance=new_simulation_step_size;
	}else{
		max_simulation_time_in_advance=new_max_simulation_time_in_advance;
	}

	//Initialize the boundary time.
	N_simulation_step_sizes = 0;
	simulation_time = 0;
#if (defined (_WIN32) || defined(_WIN64))
   #include "windows.h"
#else 
   #include <unistd.h>
#endif
	if(new_third_section<=1 && new_third_section>=0){
		third_section=new_third_section;
	}else{
		third_section=1;
	}

	if(new_second_section<=third_section && new_second_section>=0){
		second_section=new_second_section;
	}else{
		second_section=third_section;
	}

	if(new_first_section<=second_section && new_first_section>=0){
		first_section=new_first_section;
	}else{
		first_section=second_section;
	}


	first_gap_time=max_simulation_time_in_advance*(1.0f-first_section);
	second_gap_time=max_simulation_time_in_advance*(1.0f-second_section);
	third_gap_time=max_simulation_time_in_advance*(1.0f-third_section);


	#if defined(REAL_TIME_WINNT)
		if(!QueryPerformanceFrequency(&this->freq))
			puts("QueryPerformanceFrequency failed");
	#elif defined (REAL_TIME_LINUX)
		if(clock_getres(CLOCK_REALTIME, &this->freq))
			puts("clock_getres failed");
	#elif defined (REAL_TIME_OSX)
		// If this is the first time we've run, get the timebase.
		// We can use denom == 0 to indicate that sTimebaseInfo is
		// uninitialised because it makes no sense to have a zero
		// denominator is a fraction.
		if (this->freq.denom == 0 ) {
			(void) mach_timebase_info(&this->freq);
		}
	#endif

	#if defined(REAL_TIME_WINNT)
   		QueryPerformanceCounter(&startt);
	#elif defined(REAL_TIME_LINUX)
    	clock_gettime(CLOCK_REALTIME, &startt);
	#elif defined(REAL_TIME_OSX)
		startt = mach_absolute_time();
	#endif
}
 



//This function is executed by the main thread, not by the watchdog thread.
void RealTimeRestriction::NextStepWatchDog(){
	//Increment the boundary time in one step.
	N_simulation_step_sizes++;
	simulation_time = N_simulation_step_sizes*this->simulation_step_size;
	
	//Time mesured by the watchdog since the init of the supervision
	double current_time_supervision;

	//If an external clock is used.
	if (this->use_external_clock == true){
		current_time_supervision = this->external_clock;

		//Get the time stamp to internally measure the increment of the time in the external clock.
		#if defined(REAL_TIME_WINNT)
			QueryPerformanceCounter(&startt);
		#elif defined(REAL_TIME_LINUX)
			clock_gettime(CLOCK_REALTIME, &startt);
		#elif defined(REAL_TIME_OSX)
			startt = mach_absolute_time();
		#endif
	}
	else{
		#if defined(REAL_TIME_WINNT)
			QueryPerformanceCounter(&endt_supervision); // measures time
			current_time_supervision = (endt_supervision.QuadPart - startt.QuadPart) / (float)this->freq.QuadPart; // to be logged
		#elif defined(REAL_TIME_LINUX)
			clock_gettime(CLOCK_REALTIME, &endt_supervision);
			// Calculate time it took
			current_time_supervision = (endt_supervision.tv_sec-startt.tv_sec ) + (endt_supervision.tv_nsec-startt.tv_nsec )/float(1e9);
		#elif defined(REAL_TIME_OSX)
			// Stop the clock.
			endt_supervision = mach_absolute_time();
			// Calculate the duration.
			elapsed_supervision = endt_supervision - startt;
			current_time_supervision = 1e-9 * elapsed_supervision * this->freq.numer / this->freq.denom;
		#endif
	}
	
	//Calculate the difference between the boundary_time and the real time.		
	double gap = simulation_time - current_time_supervision;
	//The simulation is too fast, has arrived to the boundary time and must wait.
	if (gap >= max_simulation_time_in_advance){
		//While the RestricitionLevel is "SIMULATION_TOO_FAST", the simulation is stopped.
		RestrictionLevel = SIMULATION_TOO_FAST;
	}else if(gap>first_gap_time){
		//The simulation time is fine.
		RestrictionLevel = ALL_EVENTS_ENABLED;
	}else if(gap>second_gap_time){
		//The simulation time evolves a bit slow. Learning rules are disabled.
		RestrictionLevel = LEARNING_RULES_DISABLED;
	}else if(gap>third_gap_time){
		//The simulation time evolves slow. Learning rules and spike generation and processing are disabled.
		RestrictionLevel = SPIKES_DISABLED;
	}else{
		//The simulation time evolves very slow (near the real time boundary). All unessential events are disabled.
		RestrictionLevel = ALL_UNESSENTIAL_EVENTS_DISABLED;
	}

	RestrictionLevelLocalStatistics[RestrictionLevel] += 1;
	RestrictionLevelLocalStatisticsCounter += 1;

	//Calculate the real time index (used to manage the communication with the avatar).
	real_time_simulation_step_index = current_time_supervision * inv_simulation_step_size;
}

//This function is executed by the main thread, not by the watchdog thread.
void RealTimeRestriction::NextStepWatchDog(double new_external_clock){
	if (this->use_external_clock == true){
		this->external_clock = new_external_clock;
	}else{
		cerr<<"EXTERNAL CLOCK NOT ENABLED"<<endl;
	}
	NextStepWatchDog();
}


void RealTimeRestriction::StopWatchDog(){
	Stop=true;
}

void RealTimeRestriction::StartWatchDog(){
	Start=true;
}

void RealTimeRestriction::Watchdog(){
	if (this->simulation_step_size == 0.0f){
		puts("Real time restriction object parameter not initialized");
	}

	//We wait here until the watchdog is unlocked by the main thread.
	while (!Start);

	//Get the first time stamp for the watchdog.
	#if defined(REAL_TIME_WINNT)
		QueryPerformanceCounter(&startt);
	#elif defined(REAL_TIME_LINUX)
		clock_gettime(CLOCK_REALTIME, &startt);
	#elif defined(REAL_TIME_OSX)
		startt = mach_absolute_time();
	#endif



	double current_time;
	while(!Stop){
		#if defined(REAL_TIME_WINNT)
			QueryPerformanceCounter(&endt); // measures time
			current_time=(endt.QuadPart-startt.QuadPart)/(float)this->freq.QuadPart; // to be logged
		#elif defined(REAL_TIME_LINUX)
			clock_gettime(CLOCK_REALTIME, &endt);
			// Calculate time it took
			 current_time = (endt.tv_sec-startt.tv_sec ) + (endt.tv_nsec-startt.tv_nsec )/float(1e9);
		#elif defined(REAL_TIME_OSX)
			// Stop the clock.
			endt = mach_absolute_time();
			// Calculate the duration.
			elapsed = endt - startt;
			current_time = 1e-9 * elapsed * this->freq.numer / this->freq.denom;
		#endif

		//If an external clock is used, current time just measure the time since the last call to the NextStepWatchDog function and the external clock time
		//must be added to the current_time.
		if (this->use_external_clock == true){
			current_time += this->external_clock;			
		}


		//Calculate the difference between the boundary_time and the real time.		
		double gap = simulation_time - current_time;
		//The simulation is too fast, has arrived to the boundary time and must wait.
		if (gap >= max_simulation_time_in_advance){
			//While the RestricitionLevel is "SIMULATION_TOO_FAST", the simulation is stopped.
			RestrictionLevel = SIMULATION_TOO_FAST;
		}
		else if (gap>first_gap_time){
			//The simulation time is fine.
			RestrictionLevel = ALL_EVENTS_ENABLED;
		}
		else if (gap>second_gap_time){
			//The simulation time evolves a bit slow. Learning rules are disabled.
			RestrictionLevel = LEARNING_RULES_DISABLED;
		}
		else if (gap>third_gap_time){
			//The simulation time evolves slow. Learning rules and spike generation and processing are disabled.
			RestrictionLevel = SPIKES_DISABLED;
		}
		else{
			//The simulation time evolves very slow (near the real time boundary). All unessential events are disabled.
			RestrictionLevel = ALL_UNESSENTIAL_EVENTS_DISABLED;
		}

		RestrictionLevelLocalStatistics[RestrictionLevel] += 1;
		RestrictionLevelLocalStatisticsCounter += 1;

		//Calculate the real time index (used to manage the communication with the avatar).
		real_time_simulation_step_index = current_time * inv_simulation_step_size;

		//sleep the watchdog
		if (this->sleep_period > 0){
			#if (defined (_WIN32) || defined(_WIN64))
				Sleep(this->sleep_period*1000);
			#else
				usleep(this->sleep_period * 1000000);
			#endif
		}
	}
	this->ShowGlobalStatistics();
}



int  RealTimeRestriction::GetRealTimeSimulationStepIndex(){
	return real_time_simulation_step_index;
}


void  RealTimeRestriction::SetExternalClockOption(){
	this->use_external_clock = true;
}


void  RealTimeRestriction::SetInternalClockOption(){
	this->use_external_clock = false;
}


void RealTimeRestriction::SetSleepPeriod(double new_sleep_period){
	this->sleep_period = new_sleep_period;
	if (this->sleep_period < 0.0){
		this->sleep_period = 0.0;
	}
}


void RealTimeRestriction::ShowLocalStatistics(){
	if (RestrictionLevelLocalStatisticsCounter > 0){
		cout << "LOCAL REAL-TIME RESTRICTION LEVEL: 0->" << (RestrictionLevelLocalStatistics[0] * 100.0) / RestrictionLevelLocalStatisticsCounter << "%, 1->" << (RestrictionLevelLocalStatistics[1] * 100.0) / RestrictionLevelLocalStatisticsCounter << "%, 2->" << (RestrictionLevelLocalStatistics[2] * 100.0) / RestrictionLevelLocalStatisticsCounter << "%, 3->" << (RestrictionLevelLocalStatistics[3] * 100.0) / RestrictionLevelLocalStatisticsCounter << "%, 4->" << (RestrictionLevelLocalStatistics[4] * 100.0) / RestrictionLevelLocalStatisticsCounter << "%" << endl;
	}

	for (int i = 0; i < 5; i++){
		RestrictionLevelGlobalStatistics[i] += RestrictionLevelLocalStatistics[i];
		RestrictionLevelLocalStatistics[i]=0;
	}
	RestrictionLevelGlobalStatisticsCounter += RestrictionLevelLocalStatisticsCounter;
	RestrictionLevelLocalStatisticsCounter =0;
}

void RealTimeRestriction::ShowGlobalStatistics(){
	for (int i = 0; i < 5; i++){
		RestrictionLevelGlobalStatistics[i] += RestrictionLevelLocalStatistics[i];
		RestrictionLevelLocalStatistics[i]=0;
	}
	RestrictionLevelGlobalStatisticsCounter += RestrictionLevelLocalStatisticsCounter;
	RestrictionLevelLocalStatisticsCounter =0;

	if (RestrictionLevelGlobalStatisticsCounter > 0){
		cout << "GLOBAL REAL-TIME RESTRICTION LEVEL: 0->" << (RestrictionLevelGlobalStatistics[0] * 100.0) / RestrictionLevelGlobalStatisticsCounter << "%, 1->" << (RestrictionLevelGlobalStatistics[1] * 100.0) / RestrictionLevelGlobalStatisticsCounter << "%, 2->" << (RestrictionLevelGlobalStatistics[2] * 100.0) / RestrictionLevelGlobalStatisticsCounter << "%, 3->" << (RestrictionLevelGlobalStatistics[3] * 100.0) / RestrictionLevelGlobalStatisticsCounter << "%, 4->" << (RestrictionLevelGlobalStatistics[4] * 100.0) / RestrictionLevelGlobalStatisticsCounter << "%" << endl;
	}
}