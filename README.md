# EDLUT_BAXTER
Cerebellar-SNN control of a Baxter robot. The repository includes EDLUT simulator source code, the ROS package to perform closed-loop cerebellar-SNN control and all configuration files needed. 

There is also a folder containing the results obtained for each of the performed motor tasks. For each of the performed trajectories the following files are included: 
* recorded_desired_positions.txt : This file contains the desired position of each joint at each point of time along the experimental setup performed. 
* recorded_current_positions.txt : This file contains the current robot's position for each joint at each point of time along the experimental setup performed. 
* recorded_joint_MAE.txt : This file contains the Poisition Mean Absolute Error (MAE) (i.e. difference between desired and current position) of each joint for each of the trials performed along the experimental setup. 
* recorded_global_MAE.txt : This file contains the global (i.e. mean of all joints) Poisition MAE of each trial performed along the experimental setup. 
	

##  Requirements
* A computer with at least 8GB of RAM, a multicore CPU, an ENVIDIA GPU with CUDA support, and Linux (Ubuntu 15.10 or Ubuntu 16.04, required for ROS Kinetic).
* A Baxter robot (it can be simulated using Gazebo, although its behaviour is completely different from the real one).


## Installation
* Install and configure ROS (Kinetic distribution) and setup Baxter robot using the SDK Baxter guide: http://sdk.rethinkrobotics.com/wiki/Getting_Started

* Install Gazebo (OPTIONAL): http://sdk.rethinkrobotics.com/wiki/Baxter_Simulator . Notice that the dynamics model of the simulated version is completely different from the real robot, obtaining different results. 

* Download source code from repository https://github.com/EduardoRosLab/EDLUT_BAXTER and copy in home folder. 

* Install EDLUT simulator (this step requires an NVIDIA GPU with CUDA support and CUDA installation):
	* Open a terminal and go to the folder EDLUT_source_code inside EDLUT_BAXTER repository.
	* chmod u+x configure
	* ./configure
	* make
	* sudo bash
	* make install

* Compile the ROS EDLUT-BAXTER package (this step requires the installation of ROS and proper Baxter configuration):
	* Open a terminal and go to the folder ros_ws (where the Baxter setup has been installed as described in http://sdk.rethinkrobotics.com/wiki/Getting_Started): cd ~/ros_ws
	* ./baxter.sh
	* cd ..
  * cd EDLUT_BAXTER/edlut_ros_ws
	* catkin_make

* Copy the config_files folder from EDLUT_BAXTER repository to ~/.ros path. 


## Execution 
### Using real Baxter robot 
* Connect to Baxter robot following http://sdk.rethinkrobotics.com/wiki/Hello_Baxter 
  * Open a terminal: 
  * cd ros_ws/
  * ./baxter.sh 
  * rostopic pub /robot/joint_state_publish_rate std_msgs/UInt16 500 (Set Baxter publishing rate to 500 Hz) 
  * cd ..
  * cd EDLUT_BAXTER/edlut_ros_ws
  * source devel/setup.bash
  * Launch the desired motor task: 
    * roslaunch edlut_ros circle_trajectory.launch (Circle trajectory – Cerebellar-SNN Torque Control)
    * roslaunch edlut_ros eight_trajectory.launch (Eight-like trajectory – Cerebellar-SNN Torque Control)
    * roslaunch edlut_ros target_reaching.launch (Target Reaching – Cerebellar-SNN Torque Control)
    * roslaunch edlut_ros position_control_mode_circle_trajectory.launch (Circle trajectory – Position Control Mode)
    * roslaunch edlut_ros position_control_mode_eight_trajectory.launch (Eight-like trajectory – Position Control Mode)
    * roslaunch edlut_ros position_control_mode_target_reaching.launch (Target Reaching – Position Control Mode)
* To use the monitoring tools run the rqt_reconfigure ROS plugin:
  * Open a terminal: 
  * cd ros_ws/
  * ./baxter.sh 
  * rosrun rqt_reconfigure rqt_reconfigure (A new window will be displayed allowing the user to choose among several monitoring tools) 



### Using simulated Baxter robot on Gazebo 
* Launch Baxter on Gazebo 
  * Open a terminal: 
  * cd ros_ws/
  * ./baxter.sh sim
  * roslaunch baxter_gazebo baxter_world.launch

* Connect to simulated Baxter robot 
  * Open a new terminal: 
  * cd ros_ws/
  * ./baxter.sh sim
  * rostopic pub /robot/joint_state_publish_rate std_msgs/UInt16 500 (Set Baxter publishing rate to 500 Hz)
  * cd ..
  * cd EDLUT_BAXTER/edlut_ros_ws
  * source devel/setup.bash
  * Launch the desired motor task: 
    * roslaunch edlut_ros circle_trajectory.launch (Circle trajectory – Cerebellar-SNN Torque Control)
    * roslaunch edlut_ros eight_trajectory.launch (Eight-like trajectory – Cerebellar-SNN Torque Control)
    * roslaunch edlut_ros target_reaching.launch (Target Reaching – Cerebellar-SNN Torque Control)
    * roslaunch edlut_ros position_control_mode_circle_trajectory.launch (Circle trajectory – Position Control Mode)
    * roslaunch edlut_ros position_control_mode_eight_trajectory.launch (Eight-like trajectory – Position Control Mode)
    * roslaunch edlut_ros position_control_mode_target_reaching.launch (Target Reaching – Position Control Mode)
* To use the monitoring tools run the rqt_reconfigure ROS plugin:
  * Open a terminal: 
  * cd ros_ws/
  * ./baxter.sh sim 
  * rosrun rqt_reconfigure rqt_reconfigure (A new window will be displayed allowing the user to choose among several monitoring tools) 

