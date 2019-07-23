/***************************************************************************
 *                           simulator_node.cpp                            *
 *                           -------------------                           *
 * copyright            : (C) 2016 by Jesus Garrido                        *
 * email                : jesusgarrido@ugr.es                              *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

// This is the EDLUT simulator node. 


#include "ros/ros.h"
#include "std_msgs/Time.h"
#include "log4cxx/logger.h"

#include "edlut_ros/ROSTopicInputDriver.h"
#include "edlut_ros/ROSTopicOutputDriver.h"

#include "spike/EDLUTFileException.h"
#include "spike/EDLUTException.h"

#include "communication/FileOutputSpikeDriver.h"
#include "communication/FileInputSpikeDriver.h"
#include "communication/FileOutputWeightDriver.h"
#include "simulation/EventQueue.h"
#include "simulation/SaveWeightsEvent.h"


#include "simulation/Simulation.h"
#include "edlut_ros/ExternalClock.h"

#include <cstring>
#include <ctime>
#include <limits>
#include <signal.h>

#include "omp.h"

static bool stop_simulation;

void rosShutdownHandler(int sig)
{
	stop_simulation = true;
}

int main(int argc, char **argv)
{
	// Set up ROS.
	ros::init(argc, argv, "edlut_simulator", ros::init_options::NoSigintHandler);
	log4cxx::LoggerPtr my_logger = log4cxx::Logger::getLogger(ROSCONSOLE_DEFAULT_NAME);
	my_logger->setLevel(ros::console::g_level_lookup[ros::console::levels::Info]);
	ros::NodeHandle nh;

	signal(SIGINT, rosShutdownHandler);

	// Declare variables that can be modified by launch file or command line.
	std::string network_file, weight_file, input_topic, output_topic, clock_topic;
	bool use_sim_time = false;
	double total_simulation_time, simulation_step_time, sensorial_delay, rt1_gap, rt2_gap, rt3_gap, checking_frequency, rt_statistics_period;
	int max_spike_buffered, num_threads;
	ros::Publisher time_publisher;
	ros::Subscriber clock_subscriber;
	ExternalClock ext_clock;
	double output_delay = 0.0;
	double save_weight_period = 0.0;

	ros::CallbackQueue CallbackQueue;
	nh.setCallbackQueue(&CallbackQueue);

	struct timespec startt, controlt, endt;

	stop_simulation = false;

	// Initialize node parameters from launch file or command line.
	// Use a private node handle so that multiple instances of the node can be run simultaneously
	// while using different parameters.
	ros::NodeHandle private_node_handle_("~");
	private_node_handle_.getParam("network_file", network_file);
	private_node_handle_.getParam("weight_file", weight_file);
	private_node_handle_.getParam("input_topic", input_topic);
	private_node_handle_.getParam("output_topic", output_topic);
	private_node_handle_.getParam("clock_topic", clock_topic);
	private_node_handle_.getParam("Rt1_gap", rt1_gap);
	private_node_handle_.getParam("Rt2_gap", rt2_gap);
	private_node_handle_.getParam("Rt3_gap", rt3_gap);
	private_node_handle_.getParam("number_threads", num_threads);
	private_node_handle_.getParam("step_time", simulation_step_time);
	private_node_handle_.getParam("checking_frequency", checking_frequency);
	private_node_handle_.getParam("max_spike_buffered", max_spike_buffered);
	private_node_handle_.getParam("sensorial_delay", sensorial_delay);
	private_node_handle_.getParam("output_delay", output_delay);
	private_node_handle_.getParam("save_weight_period", save_weight_period);
	private_node_handle_.getParam("rt_statistics_period", rt_statistics_period);

	nh.getParam("use_sim_time", use_sim_time);

	//Subscribe to synchronizer node clock and Publish edlut's node clock
	if (use_sim_time){
		ROS_DEBUG("Simulator: Subscribing to topic /clock_sync");
		clock_subscriber = nh.subscribe("/clock_sync", 1000, &ExternalClock::ClockCallback, &ext_clock);
		// Initialize the publisher in clock topic
		time_publisher  = nh.advertise<rosgraph_msgs::Clock>(clock_topic, 1000);
		ROS_DEBUG("Simulator: Publishing simulation time to topic %s",clock_topic.c_str());
	}


	ROS_INFO("Simulator: Creating simulation with network file %s, weight file %s, and step time %f",
			network_file.c_str(), weight_file.c_str(), simulation_step_time);
	ROS_INFO("Simulator: Using input topic %s and output topic %s",
			input_topic.c_str(), output_topic.c_str());

	try{
		Simulation Simul(network_file.c_str(), weight_file.c_str(), std::numeric_limits<double>::max(), simulation_step_time, num_threads);

		ros::Time InitSimulationRosTime;
		if (!use_sim_time){
			InitSimulationRosTime = ros::Time::now();
		} else {
			InitSimulationRosTime = ros::Time(0.0);
		}

		//Set global param "reference_time" to edlut's initialization time
		nh.setParam("reference_time", InitSimulationRosTime.toSec());

		ROS_DEBUG("Simulation: Reference time: %f",InitSimulationRosTime.toSec());

		Simul.AddInputSpikeDriver(new ROSTopicInputDriver(input_topic, max_spike_buffered, InitSimulationRosTime.toSec()));
		ROSTopicOutputDriver *output_driver = new ROSTopicOutputDriver(output_topic, max_spike_buffered, InitSimulationRosTime.toSec(), output_delay);
		Simul.AddOutputSpikeDriver(output_driver);

		// OUTPUT WEIGHT FILE
		string File_weight="data/output_weight.dat";
		const char *weight_file = File_weight.c_str();
		Simul.AddOutputWeightDriver(new FileOutputWeightDriver(weight_file));
		if (save_weight_period>0.0){
	    Simul.SetSaveStep(save_weight_period);
	    // Add the first save weight event
      Simul.GetQueue()->InsertEventWithSynchronization(new SaveWeightsEvent(Simul.GetSaveStep(), &Simul));
		}
		//Log file.
		// Simul.AddMonitorActivityDriver(new FileOutputSpikeDriver ("data/log_file.dat",false));

		//FileOutputSpikeDriver * driver = new FileOutputSpikeDriver("data/INPUT_ACTIVITY.dat", true);
		//Simul.AddOutputSpikeDriver((OutputSpikeDriver*)driver);
/**
		Simul.AddInputSpikeDriver(new ROSTopicInputDriver(input_topic, &nh, &CallbackQueue, InitSimulationRosTime));
		Simul.AddOutputSpikeDriver(new ROSTopicOutputDriver(output_topic, InitSimulationRosTime));
**/
		clock_t starttotalt=clock();

		if (!use_sim_time){
#ifdef _OPENMP
		omp_set_nested(true);
		// Strict real-time simulation
		Simul.RealTimeRestrictionObject->SetParameterWatchDog(simulation_step_time, (sensorial_delay+output_delay)*0.9, rt1_gap, rt2_gap, rt3_gap);
		Simul.RealTimeRestrictionObject->SetExternalClockOption();
		Simul.RealTimeRestrictionObject->SetSleepPeriod(0.001);
#else
		ROS_FATAL("Simulator: REAL TIME SIMULATION is not available due to the openMP support is disabled");
		exit(0);
#endif
		}

#pragma omp parallel if(!use_sim_time) num_threads(2)
{
		ROS_DEBUG("Simulator: Thread %d in parallel section with simulated time option %d", omp_get_thread_num(), use_sim_time);
		if(omp_get_thread_num()==1){
			ROS_INFO("Simulator: Calling Watchdog");
			Simul.RealTimeRestrictionObject->Watchdog();
		}else{
			ROS_INFO("Simulator: Simulating network...");
			Simul.InitSimulation();
			clock_gettime(CLOCK_REALTIME, &startt);
			//double time;

			if(!use_sim_time){
				ROS_INFO("Simulator: Starting watchdog...");
				Simul.RealTimeRestrictionObject->StartWatchDog();
				ROS_INFO("Simulator: Watchdog started...");
			}

			ros::Time last_print_time (0.0), time(0.0), last_sent_time(0.0);
			ros::Duration simulation_step (Simul.GetSimulationStep());
			ros::WallRate rate(checking_frequency);
			ros::WallRate init_rate(1.0);

			ros::Time rt_statistics_start (0.0);

			if (use_sim_time){
				// Wait until the first signal has been received from the synchronizer node
				while(!ext_clock.FirstReceived()){
					CallbackQueue.callAvailable(ros::WallDuration(0.001));
					rosgraph_msgs::Clock current_time;
					current_time.clock = ros::Time(0.0);
					ROS_DEBUG("Simulator: Publishing simulation time %f", time.toSec());
					time_publisher.publish(current_time);
					init_rate.sleep();
				}

				time = ext_clock.GetLastConfirmedTime() +ros::Duration(sensorial_delay);

				ROS_DEBUG("Simulator: Node synchronized");
			}

			ros::Time last_statistics = rt_statistics_start;

			while (!stop_simulation){
				rt_statistics_start = time;
				if ((rt_statistics_start - last_statistics).toSec() >= rt_statistics_period){
					Simul.RealTimeRestrictionObject->ShowLocalStatistics();
					Simul.RealTimeRestrictionObject->ShowGlobalStatistics();
					last_statistics = rt_statistics_start;
				}
				if(!use_sim_time){
					ROS_DEBUG("TIME %f - LAST PRINT TIME %f ", time.toSec(), last_print_time.toSec());
					if (time>=last_print_time+ros::Duration(1.0)){
						ROS_INFO("Simulator: Simulating until time %f with thread %d", time.toSec(), omp_get_thread_num());
				  		last_print_time = time;
					}
					//ROS_DEBUG("Restriction LEVEL %i", Simul.RealTimeRestrictionObject->RestrictionLevel);
					Simul.RealTimeRestrictionObject->NextStepWatchDog(ros::Time::now().toSec() - InitSimulationRosTime.toSec() - output_delay*0.9);
				}

				ROS_DEBUG("Simulator: Running simulation until time %f", time.toSec());
				Simul.RunSimulationSlot(time.toSec());
				output_driver->SendSpikeGroup();

				if (time>=last_print_time+ros::Duration(1.0)){
					ROS_INFO("Simulator: Simulating until time %f with thread %d", time.toSec(), omp_get_thread_num());
					last_print_time = time;
				}

				if (use_sim_time){
					if (time!=last_sent_time){
						// Publish the simulation time
						rosgraph_msgs::Clock current_time;
						current_time.clock = time;
						ROS_DEBUG("Simulator: Publishing simulation time %f", time.toSec());
						time_publisher.publish(current_time);
						last_sent_time = time;
					}


					CallbackQueue.callAvailable(ros::WallDuration(0.0001));
					ros::Time new_time = ext_clock.GetLastConfirmedTime() +ros::Duration(sensorial_delay);
					if (new_time>time){
						time = new_time;
						ROS_DEBUG("Simulator: Updating simulation time until %f", time.toSec());
					}
					rate.sleep();
				} else {
					time += simulation_step;
				}
			}

			if(!use_sim_time){
				Simul.RealTimeRestrictionObject->StopWatchDog();
			}
			clock_gettime(CLOCK_REALTIME, &endt);
		}
}

		ROS_INFO("Oky doky");

		float TotalWallTime = (endt.tv_sec - startt.tv_sec)+(endt.tv_nsec - startt.tv_nsec) / 1.0e9;

		ROS_INFO("Elapsed time: %f sec",TotalWallTime);
		for(int i=0; i<Simul.GetNumberOfQueues(); i++){
			ROS_INFO("Thread %d --> Number of updates: %lli",i, Simul.GetSimulationUpdates(i));
			ROS_INFO("Thread %d --> Number of InternalSpike: %lli",i, Simul.GetTotalSpikeCounter(i));
			ROS_INFO("Thread %d --> Number of Propagated Spikes and Events: %lli, %lli",i, Simul.GetTotalPropagateCounter(i),Simul.GetTotalPropagateEventCounter(i));
			ROS_INFO("Thread %d --> Mean number of spikes in heap: %f",i, Simul.GetHeapAcumSize(i)/(float)Simul.GetSimulationUpdates(i));
			ROS_INFO("Thread %d --> Updates per second: %f",i, Simul.GetSimulationUpdates(i)/TotalWallTime);
		}
		clock_t endtotalt=clock();
		ROS_INFO("Total CPU elapsed time: %f sec",(endtotalt-starttotalt)/(float)CLOCKS_PER_SEC);
		ROS_INFO("Total Internal Spike: %lli",Simul.GetTotalSpikeCounter());
		ROS_INFO("Total Propagated Spikes and Events: %lli, %lli",Simul.GetTotalPropagateCounter(), Simul.GetTotalPropagateEventCounter());

		//Save final weights in files
		ROS_INFO("SAVING FINAL WEIGHTS");
		Simul.SaveWeights();

	} catch (EDLUTFileException Exc){
		cerr << Exc << endl;
		ros::shutdown();
		return Exc.GetErrorNum();
	} catch (EDLUTException Exc){
		cerr << Exc << endl;
		ros::shutdown();
		return Exc.GetErrorNum();
	}
	ros::shutdown();
	return 0;
} // end main()
