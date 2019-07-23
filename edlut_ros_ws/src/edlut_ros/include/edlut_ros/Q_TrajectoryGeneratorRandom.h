/***************************************************************************
 *                          Q_TrajectoryGeneratorRandom.h                  *
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

#ifndef Q_TRAJECTORYGENERATORRANDOM_H_
#define Q_TRAJECTORYGENERATORRANDOM_H_

/*!
 * \file Q_TrajectoryGenerator.h
 *
 * \author Jesus Garrido
 * \date September 2016
 *
 * This file declares a class generating sinusoidal trajectories according to the current time.
 */

#include <ros/ros.h>
#include "edlut_ros/ExternalClock.h"
#include <ros/callback_queue.h>
#include <iostream>
#include <fstream>
#include <string>

#include <vector>

/*!
 * \class Q_TrajectoryGenerator
 *
 * \brief Class generating sinusoidal trajectories according to the current time.
 *
 * This class provides the positions and velocities to generate sinusoidal trajectories.
 *
 * \author Jesus Garrido
 * \date September 2016
 */
class Q_TrajectoryGeneratorRandom {

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

	//First iteration
	bool first_iteration;

	/*!
	 * Initial time of the trajectory. It is set when calling Reset function.
	 */
	ros::Time InitTime;

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
	 * Number of samples of Q matrix
	 */
	int samples;


	/*!
	 * Matrix containing joints positions for the desired trajectory
	 */
	std::vector<std::vector<double> > Q_positions;

	/*!
	 * Matrix containing joints velocities for the desired trajectory
	 */
	std::vector<std::vector<double> > Qd_velocities;

	/*!
	 * File containing the joints positions for the desired trajectory
	 */
	std::ifstream file_positions;
	/*!
	 * File containing the joints positions for the desired trajectory
	 */
	std::ifstream file_velocities;


	/*!
	 * Index counter to move through Q_positions rows
	 */
	int index;

	/*!
	 * Last Index counter to move through Q_positions rows
	 */
	int last_index;

	/*!
	 * Auxiliar variable to compute the index
	 */
	double aux_index_computation;

	/*!
	 * Trajectory frequency
	 */
	double trajectory_frequency;

	/*!
	 * Inverse trajectory frequency
	 */
	double inverse_trajectory_frequency;


	/*!
	 * In the target reaching task: number of tasks along the
	 whole circle
	 */
	int number_of_tasks;

	/*!
	 * Samples per task: total_samples / number_of_tasks
	 */
	int task_samples;

	/*!
	 * First sample of task
	 */
	int init_sample;


	/*!
	 * Time counter to know whether to move to next Q_positions row or not
	 */
	double last_time;



	public:
	/*!
	 * \brief Class constructor.
	 *
	 * It creates a new object.
	 *
	 */
	Q_TrajectoryGeneratorRandom(bool sim_time,
			std::vector<double> max_pos_amplitude,
			std::vector<double> min_pos_amplitude,
			std::vector<double> max_vel_amplitude,
			std::vector<double> min_vel_amplitude,
			int samples,
			std::string positions_file_name,
			std::string velocities_file_name,
			double trajectory_frequency,
			int number_of_tasks);

	/*!
	 * \brief Class desctructor.
	 *
	 * Class desctructor.
	 */
	~Q_TrajectoryGeneratorRandom();

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
	// void GetState(std::vector<double> & CurrentPosition, std::vector<double> & CurrentVelocity);

	/*!
	 * \brief This function provides the current position and velocity according to the trajectory.
	 *
	 * This function provides the current position and velocity vectors according to the trajectory functions.
	 */
	void GetState(std::vector<double> & CurrentPosition, std::vector<double> & CurrentVelocity, double current_time);

	/*!
	 * \brief This function provides the starting position of the trajectory.
	 *
	 * This function provides the starting position of the trajectory.
	 */
	std::vector<double> GetStartingPoint();

};

#endif /*Q_TrajectoryGENERATOR_H_*/
