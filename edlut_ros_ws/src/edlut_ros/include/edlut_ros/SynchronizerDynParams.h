/***************************************************************************
 *                          SynchronizerDynParams.h                        *
 *                           -------------------                           *
 * copyright            : (C) 2018 by Ignacio Abadia                       *
 * email                : iabadia@ugr.es                              		 *
 ***************************************************************************/

/***************************************************************************
 *                                                                         *
 *   This program is free software; you can redistribute it and/or modify  *
 *   it under the terms of the GNU General Public License as published by  *
 *   the Free Software Foundation; either version 3 of the License, or     *
 *   (at your option) any later version.                                   *
 *                                                                         *
 ***************************************************************************/

#ifndef SYNCHRONIZERDYNPARAMS_H_
#define SYNCHRONIZERDYNPARAMS_H_

/*!
 * \file SynchronizerDynParams.h
 *
 * \author Ignacio Abadia
 * \date May 2018
 *
 * This file declares a class for dynamically change some parameters of the synchronizer node
 */

#include <ros/ros.h>
#include <dynamic_reconfigure/server.h>
#include <edlut_ros/SynchronizerDynParametersConfig.h>
#include <ros/callback_queue.h>


/*!
 * \class SynchronizerDynParams
 *
 * \brief Class to modify dynamic parameters of the synchronizer node
 *
 * This class allows the user to stop the synchronizer node or make it run until a
 * given time stamp
 * \author Ignacio Abadia
 * \date May 2018
 */

 class SynchronizerDynParams{
 private:
 	bool paused, stop_ts;
 	double time_stamp;

  //! Dynamic reconfigure server.
  dynamic_reconfigure::Server<edlut_ros::SynchronizerDynParametersConfig> dr_srv_;
  //! ROS node handle.
  ros::NodeHandle *nh_;
  //! ROS node handle.
  ros::NodeHandle *pnh;
 public:
	 /*!
		* \brief Class constructor.
		*
		* It creates a new object.
		*
		*/
	 SynchronizerDynParams(ros::NodeHandle *nh, ros::NodeHandle *pnh);

	/*!
	 * \brief Class desctructor.
	 *
	 * Class desctructor.
	 */
	 ~SynchronizerDynParams();



   /*!
    * \brief This function stores the values of the dynamic parameters given by the user
    *
    */
   void callback(edlut_ros::SynchronizerDynParametersConfig &config, uint32_t level);

	 /*!
	  * \brief This function returns whether the synchronizer is stopped or not
	  *
	  */
	 bool GetPaused();

	 /*!
	  * \brief This function returns whether the synchronizer should stop at a given time or not
	  *
	  */
	 bool GetStopTS();

	 /*!
	  * \brief This function returns the value at which the synchronizer should stop
	  *
	  */
	 double GetTimeStamp();
};

#endif /*SYNCHRONIZERDYNPARAMS_H_*/
