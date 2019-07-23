/***************************************************************************
 *                          SinTrajectoryGenerator.h                       *
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

#ifndef SINTRAJECTORYGENERATOR_H_
#define SINTRAJECTORYGENERATOR_H_

/*!
 * \file SinTrajectoryGenerator.h
 *
 * \author Jesus Garrido
 * \date September 2016
 *
 * This file declares a class generating sinusoidal trajectories according to the current time.
 */

#include <ros/ros.h>
#include "edlut_ros/ExternalClock.h"
#include <ros/callback_queue.h>


#include <vector>

/*!
 * \class SinTrajectoryGenerator
 *
 * \brief Class generating sinusoidal trajectories according to the current time.
 *
 * This class provides the positions and velocities to generate sinusoidal trajectories.
 *
 * \author Jesus Garrido
 * \date September 2016
 */
class SinTrajectoryGenerator {

	private:
	// ROS Node Handler
	ros::NodeHandle NodeHandler;

	//Subscriber and external clock for synchronizer clock
	ros::Subscriber clock_subscriber;
	ExternalClock ext_clock;

	//Callback
	ros::CallbackQueue CallbackQueue;

	//Simulation time active
	bool use_sim_time;

	/*!
	 * Initial time of the trajectory. It is set when calling Reset function.
	 */
	ros::Time InitTime;


	/*!
	 * Vector of the amplitudes
	 */
	std::vector<double> Amplitude;

	/*!
	 * Vector of the frequency for each sin function
	 */
	std::vector<double> Frequency;

	/*!
	 * Vector of the phase for each sin function
	 */
	std::vector<double> Phase;

	/*!
	 * Baxter max position limit per joint
	 */
	std::vector<double> max_pos_amplitude;

	/*!
	 * Baxter min position limit per joint
	 */
	std::vector<double> min_pos_amplitude;

	/*!
	 * Baxter max velocity limit per joint
	 */
	std::vector<double> max_vel_amplitude;

	/*!
	 * Baxter min velocity limit per joint
	 */
	std::vector<double> min_vel_amplitude;
	/*!
	 * \brief This function provides the current position and velocity according to the trajectory.
	 *
	 * This function provides the current position and velocity vectors according to the trajectory functions.
	 */
	void GetState(std::vector<double> & CurrentPosition, std::vector<double> & CurrentVelocity, double ElapsedTime);


	public:
	/*!
	 * \brief Class constructor.
	 *
	 * It creates a new object.
	 *
	 */
	SinTrajectoryGenerator(std::vector<double> Amplitude,
			std::vector<double> Frequency,
			std::vector<double> Phase,
			bool sim_time,
			std::vector<double> max_pos_amplitude,
			std::vector<double> min_pos_amplitude,
			std::vector<double> max_vel_amplitude,
			std::vector<double> min_vel_amplitude);

	/*!
	 * \brief Class desctructor.
	 *
	 * Class desctructor.
	 */
	~SinTrajectoryGenerator();

	/*!
	 * \brief Set the initial time of the trajectory.
	 *
	 * This function has to be called at the time when the trajectory starts.
	 */
	ros::Time ResetGenerator();

	/*!
	 * \brief This function provides the current position and velocity according to the trajectory.
	 *
	 * This function provides the current position and velocity vectors according to the trajectory functions.
	 */
	void GetState(std::vector<double> & CurrentPosition, std::vector<double> & CurrentVelocity);


	/*!
	 * \brief This function provides the starting position of the trajectory.
	 *
	 * This function provides the starting position of the trajectory.
	 */
	std::vector<double> GetStartingPoint();

};

#endif /*SINTRAJECTORYGENERATOR_H_*/
