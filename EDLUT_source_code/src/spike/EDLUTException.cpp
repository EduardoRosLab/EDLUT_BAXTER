/***************************************************************************
 *                           EDLUTException.cpp                            *
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

#include "../../include/spike/EDLUTException.h"



const TASK_STRUCT EDLUTException::TaskMsgs[] = {
	{ TASK_OK, "Task OK" },
	{ TASK_FILE_INPUT_SPIKE_DRIVER , "Loading the input spikes from file" },
	{ TASK_FILE_INPUT_CURRENT_DRIVER, "Loading the input currents from file" },
	{ TASK_FILE_OUTPUT_SPIKE_DRIVER, "Writing the output spikes in file" },
	{ TASK_INTEGRATION_METHOD_TYPE, "Loading the integration method type" },
	{ TASK_BI_FIXED_STEP_LOAD, "Loading the bi-fixed-step integration method parameters" },
	{ TASK_FIXED_STEP_LOAD, "Loading the fixed-step integration method parameters" },
	{ TASK_LEARNING_RULE_LOAD, "Loading the learning rules from net configuration file" },

	//NETWORK
	{ TASK_NETWORK_LOAD, "Loading the network file" },
	{ TASK_NETWORK_LOAD_NEURON_MODELS, "Loading neuron model from network file" },
	{ TASK_NETWORK_LOAD_LEARNING_RULES, "Loading learning rules from network file" },
	{ TASK_NETWORK_LOAD_SYNAPSES, "Loading synapses from network file" },

	//WEIGHTS
	{ TASK_WEIGHTS_LOAD, "Loading the weight configuration file" },
	{ TASK_WEIGHTS_SAVE, "Saving all the synaptic weights in an file" },

	//EVENT DRIVEN MODELS
	{ TASK_NEURON_MODEL_TABLE, "Scaling neuron tables" },
	{ TASK_NEURON_MODEL_TABLE_LOAD, "Loading event-driven neuron model tables" },
	{ TASK_NEURON_MODEL_TABLE_TABLES_STRUCTURE, "Loading the tables structure from configuration file" },
	{ TASK_TABLE_BASED_MODEL_LOAD, "Loading event-driven neuron model table configuration" },


	//TIME DRIVEN MODELS
	{ TASK_SRM_TIME_DRIVEN_MODEL_LOAD, "Loading the configuration parameters of SRMTimeDrivenModel neuron model" },
	{ TASK_LIF_TIME_DRIVEN_MODEL_1_2_LOAD, "Loading the configuration parameters of LIFTimeDrivenModel_1_2 neuron model" },
	{ TASK_LIF_TIME_DRIVEN_MODEL_1_2_NEW_LOAD, "Loading the configuration parameters of LIFTimeDrivenModel_1_2_NEW neuron model" },
	{ TASK_LIF_TIME_DRIVEN_MODEL_1_2_GPU_LOAD, "Loading the configuration parameters of LIFTimeDrivenModel_1_2_GPU neuron model" },
	{ TASK_LIF_TIME_DRIVEN_MODEL_1_2_GPU_NEW_LOAD, "Loading the configuration parameters of LIFTimeDrivenModel_1_2_GPU_NEW neuron model" },
	{ TASK_LIF_TIME_DRIVEN_MODEL_1_3_LOAD, "Loading the configuration parameters of LIFTimeDrivenModel_1_3 neuron model" },
	{ TASK_LIF_TIME_DRIVEN_MODEL_1_3_NEW_LOAD, "Loading the configuration parameters of LIFTimeDrivenModel_1_3_NEW neuron model" },
	{ TASK_LIF_TIME_DRIVEN_MODEL_1_4_LOAD, "Loading the configuration parameters of LIFTimeDrivenModel_1_4 neuron model" },
	{ TASK_LIF_TIME_DRIVEN_MODEL_1_4_NEW_LOAD, "Loading the configuration parameters of LIFTimeDrivenModel_1_4_NEW neuron model" },
	{ TASK_LIF_TIME_DRIVEN_MODEL_1_4_GPU_LOAD, "Loading the configuration parameters of LIFTimeDrivenModel_1_4_GPU neuron model" },
	{ TASK_LIF_TIME_DRIVEN_MODEL_1_4_GPU_NEW_LOAD, "Loading the configuration parameters of LIFTimeDrivenModel_1_4_GPU_NEW neuron model" },
	{ TASK_ADEX_TIME_DRIVEN_MODEL_LOAD, "Loading the configuration parameters of AdExTimeDrivenModel neuron model" },
	{ TASK_ADEX_TIME_DRIVEN_MODEL_GPU_LOAD, "Loading the configuration parameters of AdExTimeDrivenModel_GPU neuron model" },
	{ TASK_HH_TIME_DRIVEN_MODEL_LOAD, "Loading the configuration parameters of HHTimeDrivenModel neuron model" },
	{ TASK_HH_TIME_DRIVEN_MODEL_GPU_LOAD, "Loading the configuration parameters of HHTimeDrivenModel_GPU neuron model" },
	{ TASK_ORIGINAL_HH_TIME_DRIVEN_MODEL_LOAD, "Loading the configuration parameters of OriginalHHTimeDrivenModel neuron model" },
	{ TASK_IZHIKEVICH_TIME_DRIVEN_MODEL_LOAD, "Loading the configuration parameters of IzhikevichTimeDrivenModel neuron model" },
	{ TASK_IZHIKEVICH_TIME_DRIVEN_MODEL_GPU_LOAD, "Loading the configuration parameters of IzhikevichTimeDrivenModel_GPU neuron model" },
	{ TASK_HH_TIME_DRIVEN_MODEL_LOAD, "Loading the configuration parameters of HHTimeDrivenModel neuron model" },
	{ TASK_HH_TIME_DRIVEN_MODEL_GPU_LOAD, "Loading the configuration parameters of HHTimeDrivenModel_GPU neuron model" },
	{ TASK_TIME_DRIVEN_PURKINJE_CELL_LOAD, "Loading the configuration parameters of TimeDrivenPurkinjeCell neuron model" },
	{ TASK_TIME_DRIVEN_PURKINJE_CELL_GPU_LOAD, "Loading the configuration parameters of TimeDrivenPurkinjeCell_GPU neuron model" },
	{ TASK_TIME_DRIVEN_PURKINJE_CELL_IP_LOAD, "Loading the configuration parameters of TimeDrivenPurkinjeCell_IP neuron model" },
	{ TASK_EGIDIO_GRANULE_CELL_TIME_DRIVEN_LOAD, "Loading the configuration parameters of EgidioGranuleCell_TimeDriven neuron model" },
	{ TASK_EGIDIO_GRANULE_CELL_TIME_DRIVEN_GPU_LOAD, "Loading the configuration parameters of EgidioGranuleCell_TimeDriven_GPU neuron model" },
	{ TASK_TIME_DRIVEN_INFERIOR_OLIVE_CELL_LOAD, "Loading the configuration parameters of TimeDrivenInferiorOliveCell neuron model" },
	
	{ TASK_TIME_DRIVEN_SIN_CURRENT_GENERATOR_LOAD, "Loading the configuration parameters of TimeDrivenInputDevice neuron model" },
	{ TASK_TIME_DRIVEN_SIN_CURRENT_GENERATOR_LOAD, "Loading the configuration parameters of TimeDrivenSinCurrentGenerator neuron model" },
};

const ERROR_STRUCT EDLUTException::ErrorMsgs[] = {
	{ ERROR_OK, "Error OK" },
	{ ERROR_FILE_INPUT_SPIKE_DRIVER_OPEN_FILE, "Can't open the input spikes file" },
	{ ERROR_FILE_INPUT_SPIKE_DRIVER_TOO_MUCH_SPIKES, "Found more input spikes in file that expected" },
	{ ERROR_FILE_INPUT_SPIKE_DRIVER_NEURON_INDEX, "Spike neuron number hasn't been defined" },
	{ ERROR_FILE_INPUT_SPIKE_DRIVER_FEW_SPIKES, "Can't read enough input spikes from the file" },
	{ ERROR_FILE_INPUT_SPIKE_DRIVER_N_SPIKES, "Can't read the number of inputs from the file" },

	{ ERROR_FILE_INPUT_CURRENT_DRIVER_OPEN_FILE, "Can't open the input current file" },
	{ ERROR_FILE_INPUT_CURRENT_DRIVER_TOO_MUCH_CURRENTS, "Found more input currents in file that expected" },
	{ ERROR_FILE_INPUT_CURRENT_DRIVER_NEURON_INDEX, "Current neuron number hasn't been defined" },
	{ ERROR_FILE_INPUT_CURRENT_DRIVER_FEW_CURRENTS, "Can't read enough input currents from the file" },
	{ ERROR_FILE_INPUT_CURRENT_DRIVER_N_CURRENTS, "Can't read the number of inputs from the file" },

	{ ERROR_FILE_OUTPUT_SPIKE_DRIVER_OPEN_FILE, "Can't open the output spikes file" },
	{ ERROR_FILE_OUTPUT_SPIKE_DRIVER_WRITE, "Can't write to file of output spikes" },

	{ ERROR_INTEGRATION_METHOD_TYPE, "The integration method does not exist" },
	{ ERROR_INTEGRATION_METHOD_READ, "Can't read the integration method" },
	
	{ ERROR_BI_FIXED_STEP_STEP_SIZE, "Global integration step size must be greater than zero (in s units)" },
	{ ERROR_BI_FIXED_STEP_READ_STEP, "Can't read the global integration step size (in s units)" },
	{ ERROR_BI_FIXED_STEP_GLOBAL_LOCAL_RATIO, "The global/local step ratio must be greater than zero" },
	{ ERROR_BI_FIXED_STEP_READ_GLOBAL_LOCAL_RATIO, "Can't read the global/local step ratio" },

	{ ERROR_FIXED_STEP_STEP_SIZE, "Integration step size must be greater than zero (in s units)" },
	{ ERROR_FIXED_STEP_READ_STEP, "Can't read the integration step size (in s units)" },


	//NETWORK
	{ ERROR_NETWORK_NEURON_MODEL_TYPE, "Invalid type of the neuron model" },
	{ ERROR_NETWORK_NEURON_MODEL_NUMBER, "Invalid number of neuron types" },
	{ ERROR_NETWORK_NEURON_MODEL_LOAD_NUMBER, "Can't read the number of neuron types" },
	{ ERROR_NETWORK_NUMBER_OF_NEURONS, "The actual number of neurons doesn't match with the specified total" },
	{ ERROR_NETWORK_NEURON_PARAMETERS, "Can't read enough neuron-type specifications from file" },
	{ ERROR_NETWORK_ALLOCATE, "Can't allocate enough memory" },
	{ ERROR_NETWORK_READ_NUMBER_OF_NEURONS, "Can't read the number of neurons from file" },
	{ ERROR_NETWORK_LEARNING_RULE_TYPE, "The type of learning rule referenced in the interconnection definition hasn't been defined" },
	{ ERROR_NETWORK_LEARNING_RULE_LOAD, "Can't read enough learning rules from network file" },
	{ ERROR_NETWORK_LEARNING_RULE_NUMBER, "Can't read the number of learning rules from the network file" },
	{ ERROR_NETWORK_SYNAPSES_FIRST_LEARNING_RULE_INDEX, "The index of first learning rule selected in this synapse exceeds the number of learning rule defined in network file" },
	{ ERROR_NETWORK_SYNAPSES_FIRST_LEARNING_RULE_LOAD, "Can't read the index of first learning rule selected in this synapse (tenth parameter)" },
	{ ERROR_NETWORK_SYNAPSES_SECOND_LEARNING_RULE_INDEX, "The index of second learning rule selected in this synapse exceeds the number of learning rule defined in network file" },
	{ ERROR_NETWORK_SYNAPSES_SECOND_LEARNING_RULE_LOAD, "Can't read the index of second learning rule selected in this synapse (eleventh parameter)" },
	{ ERROR_NETWORK_SYNAPSES_NUMBER, "The number of interconnections doesn't match with the total specified" },
	{ ERROR_NETWORK_SYNAPSES_NEURON_INDEX, "The neuron specified in interconnections doesn't exist" },
	{ ERROR_NETWORK_SYNAPSES_LEARNING_RULE_NON_TRIGGER , "Two learning rules will try to change the same synaptic weight" },
	{ ERROR_NETWORK_SYNAPSES_LEARNING_RULE_TRIGGER, "Two trigger learning rules in the same synpase" },
	{ ERROR_NETWORK_SYNAPSES_LOAD, "Can't read all synapse parameters from network file" },
	{ ERROR_NETWORK_SYNAPSES_LOAD_NUMBER, "Can't read the number of synapses from network file" },
	{ ERROR_NETWORK_SYNAPSES_TYPE, "Neuron model of target neuron does not support this kind of synapse type" },
	{ ERROR_NETWORK_OPEN, "Can't open the network file" },

	//WEIGHTS
	{ ERROR_WEIGHTS_OPEN, "Can't open the weights file" },
	{ ERROR_WEIGHTS_READ, "Can't read enough weights from the weights file" },
	{ ERROR_WEIGHTS_NUMBER, "Too much weights have been defined" },
	{ ERROR_WEIGHTS_SAVE, "Can't write in the file to save all the synaptic weights" },
	{ ERROR_WEIGHTS_SAVE_OPEN, "Can't open the file to save all the synaptic weights" },


	

	//LEARNING RULE 
	{ ERROR_LEARNING_RULE_LOAD, "Can't read enough learning rule parameters from file" },
		//ADDITIVE KERNEL CHANGES
		{ ERROR_ADDITIVE_KERNEL_CHANGE_VALUES, "Kernel peak position (first parameter) must be greater than zero" },

		//COS WEIGHT CHANGE, SIMETRIC COS BUFFERED STATE, SIMETRIC COS STDP WEIGHT CHANGE, SIMETRIC COS WEIGHT CHANGE
		{ ERROR_COS_WEIGHT_CHANGE_TAU, "kernel size (first parameter) must be greater than zero" },
		{ ERROR_COS_WEIGHT_CHANGE_EXPONENT, "exponent (second parameter) must be greater than zero" },

		//SIN BUFFERED WEIGHT CHANGE, SIN WEIGHT CHANGE
		{ ERROR_SIN_WEIGHT_CHANGE_EXPONENT, "Exponent (fourth parameter) must be an even number greater than zero and smaller or equal than twenty" },

		//EXP OPTIMISED BUFFERED WEIGHT CHANGE
		{ ERROR_EXP_OPTIMISED_BUFFERED_WEIGHT_CHANGE_INIT_TIME, "Init time (fourth parameter) must be an equal or greater than zero and smaller than kernel peak position (first parameter)" },


		//SIMETRIC COS SIN WEIGHT CHANGE, SIMETRIC COS SIN STDP WEIGHT CHANGE
		{ ERROR_COS_SIN_WEIGHT_CHANGE_AMPLITUDE, "Distance between central and lateral peaks (first parameter) must be greater than zero" },
		{ ERROR_COS_SIN_WEIGHT_CHANGE_SIGNS, "Central (second paramenter) and lateral (third parameter) amplitudes must have oposite signs" },

		//STDP WEIGHT CHANGE
		{ ERROR_STDP_WEIGHT_CHANGE_TAUS, "LTP tau (second parameter) and LTD tau (fourth parameter) must be greater than zero" },





	//NEURON MODELS
	{ ERROR_NEURON_MODEL_OPEN, "Can't open the neuron model configuration file" },

		//NEURON MODEL TABLE, COMPRESS NEURON MODEL TABLE
		{ ERROR_NEURON_MODEL_TABLE_ALLOCATE, "Can't allocate enough memory" },
		{ ERROR_NEURON_MODEL_TABLE_NOT_ENOUGH_DATA, "The table doesn't appear to be correct (not enough data)" },
		{ ERROR_NEURON_MODEL_TABLE_TOO_BIG, "The table is too big for the current processor/compiler architecture or the table file is corrupt" },
		{ ERROR_NEURON_MODEL_TABLE_EMPTY, "The table in the file of neuron tables is empty" },
		{ ERROR_NEURON_MODEL_TABLE_TABLE_NUMBER, "Can't read enough tables from the file of neuron tables" },
		{ ERROR_NEURON_MODEL_TABLE_VARIABLE_INDEX, "Can't read the numbers of state variables or interpolation flags corresponding to each dimension" },
		{ ERROR_NEURON_MODEL_TABLE_DIMENSION_NUMBER, "Can't read the number of dimensions" },

		//TABLE BASED MODEL, SYNCHRONOUS TABLE BASED MODEL, COMPRESS TABLE BASED MODEL, COMPRESS SYNCHRONOUS TABLE BASED MODEL, SRM TABLE BASED MODEL
		{ ERROR_TABLE_BASED_MODEL_OPEN, "Can't open the neuron type configuration file" },
		{ ERROR_TABLE_BASED_MODEL_NUMBER_OF_STATE_VARIABLES, "Can't read the number of state variables" },
		{ ERROR_TABLE_BASED_MODEL_INDEX, "Can't read the number of table used to update the state variables" },
		{ ERROR_TABLE_BASED_MODEL_INITIAL_VALUES, "Can't read the initial values of state variables" },
		{ ERROR_TABLE_BASED_MODEL_FIRING_INDEX, "Can't read the number of table used for firing prediction" },
		{ ERROR_TABLE_BASED_MODEL_END_FIRING_INDEX, "Can't read the number of table used for end of firing prediction" },
		{ ERROR_TABLE_BASED_MODEL_NUMBER_OF_SYNAPSES, "Can't read the number of synaptic variables per neuron" },
		{ ERROR_TABLE_BASED_MODEL_SYNAPSE_INDEXS, "Can't read the numbers of state variables (indexs) that will be used as synaptic variables" },
		{ ERROR_TABLE_BASED_MODEL_NUMBER_OF_TABLES, "Can't read the numbers of tables" },
		{ ERROR_TABLE_BASED_MODEL_LAST_SPIKE, "Can't read the number of the state variable that will be used as last spike time variable" },
		{ ERROR_TABLE_BASED_MODEL_SEED, "Can't read the number of the state variable that will be used as seed variable" },
		{ ERROR_TABLE_BASED_MODEL_TIME_SCALE, "Can't read the time scale used by the look-up tables" },
		{ ERROR_TABLE_BASED_MODEL_SYNCHRONIZATION_PERIOD, "The synchronization period must be greater or equal than zero. This parameter must be set in s units" },

		
		//SRM TIME DRIVEN MODEL
		{ ERROR_SRM_TIME_DRIVEN_MODEL_NUMBER_OF_CHANNELS, "Can't read the number of channels" },
		{ ERROR_SRM_TIME_DRIVEN_MODEL_TAUS, "Can't read the decay time constant parameters" },
		{ ERROR_SRM_TIME_DRIVEN_MODEL_VR, "Can't read the resting potential" },
		{ ERROR_SRM_TIME_DRIVEN_MODEL_W, "Can't read the synaptic efficacy" },
		{ ERROR_SRM_TIME_DRIVEN_MODEL_R0, "Can't read the spontaneous firing rate" },
		{ ERROR_SRM_TIME_DRIVEN_MODEL_V0, "Can't read the probabilistic threshold potential" },
		{ ERROR_SRM_TIME_DRIVEN_MODEL_VF, "Can't read the potential gain factor" },
		{ ERROR_SRM_TIME_DRIVEN_MODEL_TAUABS, "Can't read the absolute refractory period" },
		{ ERROR_SRM_TIME_DRIVEN_MODEL_TAUREL, "Can't read the relative refractory period" },

		//LIF TIME DRIVEN MODEL 1 2, LIF TIME DRIVEN MODEL 1 2 GPU, LIF TIME DRIVEN MODEL 1 3
		{ ERROR_LIF_TIME_DRIVEN_MODEL_1_2_GREST, "Can't read the resting conductance (nineth parameter). This parameter must be set in S units" },
		{ ERROR_LIF_TIME_DRIVEN_MODEL_1_2_TREF, "Can't read the refractory period (eighth parameter). This parameter must be set in s units" },
		{ ERROR_LIF_TIME_DRIVEN_MODEL_1_2_TINH, "Can't read the GABA receptor time constant (seventh parameter). This parameter must be set in s units" },
		{ ERROR_LIF_TIME_DRIVEN_MODEL_1_2_TEXC, "Can't read the AMPA receptor time constant (sixth parameter). This parameter must be set in s units" },
		{ ERROR_LIF_TIME_DRIVEN_MODEL_1_2_CM, "Can't read the membrane capacitance (fifth parameter). This parameter must be set in F units" },
		{ ERROR_LIF_TIME_DRIVEN_MODEL_1_2_VTHR, "Can't read the firing threshold (fourth parameter). This parameter must be set in V units" },
		{ ERROR_LIF_TIME_DRIVEN_MODEL_1_2_EREST, "Can't read the resting potential (third parameter). This parameter must be set in V units" },
		{ ERROR_LIF_TIME_DRIVEN_MODEL_1_2_EINH, "Can't read the inhibitory reversal potential (second parameter). This parameter must be set in V units" },
		{ ERROR_LIF_TIME_DRIVEN_MODEL_1_2_EEXC, "Can't read the excitatory reversal potential (first parameter). This parameter must be set in V units" },

		//LIF TIME DRIVEN MODEL 1 2 NEW, LIF TIME DRIVEN MODEL 1 2 GPU NEW, LIF TIME DRIVEN MODEL 1 3 NEW
		{ ERROR_LIF_TIME_DRIVEN_MODEL_1_2_NEW_GREST, "Can't read the resting conductance (nineth parameter). This parameter must be set in nS units" },
		{ ERROR_LIF_TIME_DRIVEN_MODEL_1_2_NEW_TREF, "Can't read the refractory period (eighth parameter). This parameter must be set in ms units" },
		{ ERROR_LIF_TIME_DRIVEN_MODEL_1_2_NEW_TINH, "Can't read the GABA receptor time constant (seventh parameter). This parameter must be set in ms units" },
		{ ERROR_LIF_TIME_DRIVEN_MODEL_1_2_NEW_TEXC, "Can't read the AMPA receptor time constant (sixth parameter). This parameter must be set in ms units" },
		{ ERROR_LIF_TIME_DRIVEN_MODEL_1_2_NEW_CM, "Can't read the membrane capacitance (fifth parameter). This parameter must be set in pF units" },
		{ ERROR_LIF_TIME_DRIVEN_MODEL_1_2_NEW_VTHR, "Can't read the firing threshold (fourth parameter). This parameter must be set in mV units" },
		{ ERROR_LIF_TIME_DRIVEN_MODEL_1_2_NEW_EREST, "Can't read the resting potential (third parameter). This parameter must be set in mV units" },
		{ ERROR_LIF_TIME_DRIVEN_MODEL_1_2_NEW_EINH, "Can't read the inhibitory reversal potential (second parameter). This parameter must be set in mV units" },
		{ ERROR_LIF_TIME_DRIVEN_MODEL_1_2_NEW_EEXC, "Can't read the excitatory reversal potential (first parameter). This parameter must be set in mV units" },

		//LIF TIME DRIVEN MODEL 1 4, LIF TIME DRIVEN MODEL 1 4 GPU
		{ ERROR_LIF_TIME_DRIVEN_MODEL_1_4_FGJ, "Can't read the Gap junction factor (twelfth parameter). This parameter must be set in V/nS units" },
		{ ERROR_LIF_TIME_DRIVEN_MODEL_1_4_GREST, "Can't read the resting conductance (eleventh parameter). This parameter must be set in nS units" },
		{ ERROR_LIF_TIME_DRIVEN_MODEL_1_4_TREF, "Can't read the refractory period (tenth parameter). This parameter must be set in s units" },
		{ ERROR_LIF_TIME_DRIVEN_MODEL_1_4_GAP, "Can't read the Gap junction time constant (nineth parameter). This parameter must be set in s units" },
		{ ERROR_LIF_TIME_DRIVEN_MODEL_1_4_TINH, "Can't read the GABA receptor time constant (eighth parameter). This parameter must be set in s units" },
		{ ERROR_LIF_TIME_DRIVEN_MODEL_1_4_TNMDA, "Can't read the NMDA receptor time constant (seventh parameter). This parameter must be set in s units" },
		{ ERROR_LIF_TIME_DRIVEN_MODEL_1_4_TAMPA, "Can't read the AMPA receptor time constant (sixth parameter). This parameter must be set in s units" },
		{ ERROR_LIF_TIME_DRIVEN_MODEL_1_4_CM, "Can't read the membrane capacitance (fifth parameter). This parameter must be set in F units" },
		{ ERROR_LIF_TIME_DRIVEN_MODEL_1_4_VTHR, "Can't read the firing threshold (fourth parameter). This parameter must be set in V units" },
		{ ERROR_LIF_TIME_DRIVEN_MODEL_1_4_EREST, "Can't read the resting potential (third parameter). This parameter must be set in V units" },
		{ ERROR_LIF_TIME_DRIVEN_MODEL_1_4_EINH, "Can't read the inhibitory reversal potential (second parameter). This parameter must be set in V units" },
		{ ERROR_LIF_TIME_DRIVEN_MODEL_1_4_EEXC, "Can't read the excitatory reversal potential (first parameter). This parameter must be set in V units" },

		//LIF TIME DRIVEN MODEL 1 4 NEW, LIF TIME DRIVEN MODEL 1 4 GPU NEW
		{ ERROR_LIF_TIME_DRIVEN_MODEL_1_4_NEW_FGJ, "Can't read the Gap junction factor (twelfth parameter). This parameter must be set in mV/nS units" },
		{ ERROR_LIF_TIME_DRIVEN_MODEL_1_4_NEW_GREST, "Can't read the resting conductance (eleventh parameter). This parameter must be set in nS units" },
		{ ERROR_LIF_TIME_DRIVEN_MODEL_1_4_NEW_TREF, "Can't read the refractory period (tenth parameter). This parameter must be set in ms units" },
		{ ERROR_LIF_TIME_DRIVEN_MODEL_1_4_NEW_GAP, "Can't read the Gap junction time constant (nineth parameter). This parameter must be set in ms units" },
		{ ERROR_LIF_TIME_DRIVEN_MODEL_1_4_NEW_TINH, "Can't read the GABA receptor time constant (eighth parameter). This parameter must be set in ms units" },
		{ ERROR_LIF_TIME_DRIVEN_MODEL_1_4_NEW_TNMDA, "Can't read the NMDA receptor time constant (seventh parameter). This parameter must be set in ms units" },
		{ ERROR_LIF_TIME_DRIVEN_MODEL_1_4_NEW_TAMPA, "Can't read the AMPA receptor time constant (sixth parameter). This parameter must be set in ms units" },
		{ ERROR_LIF_TIME_DRIVEN_MODEL_1_4_NEW_CM, "Can't read the membrane capacitance (fifth parameter). This parameter must be set in pF units" },
		{ ERROR_LIF_TIME_DRIVEN_MODEL_1_4_NEW_VTHR, "Can't read the firing threshold (fourth parameter). This parameter must be set in mV units" },
		{ ERROR_LIF_TIME_DRIVEN_MODEL_1_4_NEW_EREST, "Can't read the resting potential (third parameter). This parameter must be set in mV units" },
		{ ERROR_LIF_TIME_DRIVEN_MODEL_1_4_NEW_EINH, "Can't read the inhibitory reversal potential (second parameter). This parameter must be set in mV units" },
		{ ERROR_LIF_TIME_DRIVEN_MODEL_1_4_NEW_EEXC, "Can't read the excitatory reversal potential (first parameter). This parameter must be set in mV units" },

		//ADEX TIME DRIVEN MODEL, ADEX TIME DRIVEN MODEL GPU
		{ ERROR_ADEX_TIME_DRIVEN_MODEL_TINH, "Can't read the GABA receptor time constant (thirteenth parameter). This parameter must be set in ms units" },
		{ ERROR_ADEX_TIME_DRIVEN_MODEL_TEXC, "Can't read the AMPA receptor time constant (twelfth parameter). This parameter must be set in ms units" },
		{ ERROR_ADEX_TIME_DRIVEN_MODEL_CM, "Can't read the membrane capacitance (eleventh parameter). This parameter must be set in pF units" },
		{ ERROR_ADEX_TIME_DRIVEN_MODEL_GLEAK, "Can't read the leak conductance (tenth parameter). This parameter must be set in nS units" },
		{ ERROR_ADEX_TIME_DRIVEN_MODEL_ELEAK, "Can't read the leak potential (nineth parameter). This parameter must be set in mV units" },
		{ ERROR_ADEX_TIME_DRIVEN_MODEL_ERESET, "Can't read the reset potential (eighth parameter). This parameter must be set in mV units" },
		{ ERROR_ADEX_TIME_DRIVEN_MODEL_EINH, "Can't read the inhibitory reversal potential (seventh parameter). This parameter must be set in mV units" },
		{ ERROR_ADEX_TIME_DRIVEN_MODEL_EEXC, "Can't read the excitatory reversal potential (sixth parameter). This parameter must be set in mV units" },
		{ ERROR_ADEX_TIME_DRIVEN_MODEL_TAUW, "Can't read the adaptation time constant (fifth parameter). This parameter must be set in ms units" },
		{ ERROR_ADEX_TIME_DRIVEN_MODEL_VT, "Can't read the efective threshold potential (fourth parameter). This parameter must be set in mV units" },
		{ ERROR_ADEX_TIME_DRIVEN_MODEL_TSF, "Can't read the threshold slope factor (third parameter). This parameter must be set in mV units" },
		{ ERROR_ADEX_TIME_DRIVEN_MODEL_B, "Can't read the spike trigger adaptation (second parameter). This parameter must be set in pA units" },
		{ ERROR_ADEX_TIME_DRIVEN_MODEL_A, "Can't read the adaptation conductance (first parameter). This parameter must be set in nS units" },

		//HH TIME DRIVEN MODEL, HH TIME DRIVEN MODEL GPU, ORIGINAL HH TIME DRIVEN MODEL
		{ ERROR_HH_TIME_DRIVEN_MODEL_VT, "Can't read the efective threshold potential (thirteenth parameter). This parameter must be set in mV units" },
		{ ERROR_HH_TIME_DRIVEN_MODEL_EK, "Can't read the potasium potential (twelfth parameter). This parameter must be set in mV units" },
		{ ERROR_HH_TIME_DRIVEN_MODEL_ENA, "Can't read the sodium potential (eleventh parameter). This parameter must be set in mV units" },
		{ ERROR_HH_TIME_DRIVEN_MODEL_GKD, "Can't read the maximum value of potasium conductance (tenth parameter). This parameter must be set in nS units" },
		{ ERROR_HH_TIME_DRIVEN_MODEL_GNA, "Can't read the maximum value of sodium conductance (nineth parameter). This parameter must be set in nS units" },
		{ ERROR_HH_TIME_DRIVEN_MODEL_TINH, "Can't read the GABA (inhibitory) receptor time constant (eighth parameter). This parameter must be set in ms units" },
		{ ERROR_HH_TIME_DRIVEN_MODEL_TEXC, "Can't read the AMPA (excitatory) receptor time constant (seventh parameter). This parameter must be set in ms units" },
		{ ERROR_HH_TIME_DRIVEN_MODEL_VTHR, "Can't read the virtual firing threshold (sixth parameter). This parameter must be set in mV units" },
		{ ERROR_HH_TIME_DRIVEN_MODEL_CM, "Can't read the membrane capacitance (fifth parameter). This parameter must be set in pF units" },
		{ ERROR_HH_TIME_DRIVEN_MODEL_GREST, "Can't read the leak conductance (fourth parameter). This parameter must be set in nS units" },
		{ ERROR_HH_TIME_DRIVEN_MODEL_EREST, "Can't read the leak potential (third parameter). This parameter must be set in mV units" },
		{ ERROR_HH_TIME_DRIVEN_MODEL_EINH, "Can't read the inhibitory reversal potential (second parameter). This parameter must be set in mV units" },
		{ ERROR_HH_TIME_DRIVEN_MODEL_EEXC, "Can't read the excitatory reversal potential (first parameter). This parameter must be set in mV units" },

		//IZHIKEVICH TIME DRIVEN MODEL NEW, IZHIKEVICH TIME DRIVEN MODEL GPU NEW
		{ ERROR_IZHIKEVICH_TIME_DRIVEN_MODEL_TINH, "Can't read the GABA (inhibitory) receptor time constant (nineth parameter). This parameter must be set in nS units" },
		{ ERROR_IZHIKEVICH_TIME_DRIVEN_MODEL_TEXC, "Can't read the AMPA (excitatory) receptor time constant (eighth parameter). This parameter must be set in nS units" },
		{ ERROR_IZHIKEVICH_TIME_DRIVEN_MODEL_CM, "Can't read the membrane capacitance (seventh parameter). This parameter must be set in pF units" },
		{ ERROR_IZHIKEVICH_TIME_DRIVEN_MODEL_EINH, "Can't read the inhibitory reversal potential (sixth parameter). This parameter must be set in mV units" },
		{ ERROR_IZHIKEVICH_TIME_DRIVEN_MODEL_EEXC, "Can't read the excitatory reversal potential (fifth parameter). This parameter must be set in mV units" },
		{ ERROR_IZHIKEVICH_TIME_DRIVEN_MODEL_D, "Can't read the parameter d (fourth parameter). This parameter is dimensionless" },
		{ ERROR_IZHIKEVICH_TIME_DRIVEN_MODEL_C, "Can't read the parameter c (third parameter). This parameter is dimensionless" },
		{ ERROR_IZHIKEVICH_TIME_DRIVEN_MODEL_B, "Can't read the parameter b (second parameter). This parameter is dimensionless" },
		{ ERROR_IZHIKEVICH_TIME_DRIVEN_MODEL_A, "Can't read the parameter a (first parameter). This parameter is dimensionless" },

		//TIME_DRIVEN_INPUT_DEVICE
		{ ERROR_TIME_DRIVEN_INPUT_DEVICE_STEP_SIZE, "Can't read the time-driven step size (last parameter) or is negative or zero . This parametr must be set in s units" },

		//TIME_DRIVEN_SIN_CURRENT_GENERATOR
		{ ERROR_TIME_DRIVEN_SIN_CURRENT_GENERATOR_FREQUENCY, "Can't read the frequency constant (first parameter). This parametr must be set in Hz units"},
		{ ERROR_TIME_DRIVEN_SIN_CURRENT_GENERATOR_PHASE, "Can't read the phase constant (second parameter). This parametr must be set in rad units" },
		{ ERROR_TIME_DRIVEN_SIN_CURRENT_GENERATOR_AMPLITUDE, "Can't read the amplitude constant (third parameter). This parametr must be set in ////////REVISAR//////// units" },
		{ ERROR_TIME_DRIVEN_SIN_CURRENT_GENERATOR_OFFSET, "Can't read the offset constant (fourth parameter). This parametr must be set in ///////REVISAR/////// units" },
};



const REPAIR_STRUCT EDLUTException::RepairMsgs[] = {
	{ REPAIR_OK, "Repair OK" },
	{ REPAIR_OPEN_FILE_READ, "Ensure that the file has the proper name and is in the application directory" },
	{ REPAIR_OPEN_FILE_WRITE, "Check for disk problems" },

	{ REPAIR_FILE_INPUT_SPIKE_DRIVER_TOO_MUCH_SPIKES, "Specify the correct number of spikes in the file of input spikes" },
	{ REPAIR_FILE_INPUT_SPIKE_DRIVER_NEURON_INDEX, "Specify a correct neuron number in the file of input spikes" },
	{ REPAIR_FILE_INPUT_SPIKE_DRIVER_FEW_SPIKES, "Define more spikes or correct the number of spikes in the file of input spikes" },
	{ REPAIR_FILE_INPUT_SPIKE_DRIVER_N_SPIKES, "Specify the number of spikes in the file of input spikes" },

	{ REPAIR_FILE_INPUT_CURRENT_DRIVER_TOO_MUCH_CURRENTS, "Specify the correct number of currents in the file of input currents" },
	{ REPAIR_FILE_INPUT_CURRENT_DRIVER_NEURON_INDEX, "Specify a correct neuron number in the file of input currents" },
	{ REPAIR_FILE_INPUT_CURRENT_DRIVER_FEW_CURRENTS, "Define more currents or correct the number of currents in the file of input currents" },
	{ REPAIR_FILE_INPUT_CURRENT_DRIVER_N_CURRENTS, "Specify the number of currents in the file of input currents" },

	{ REPAIR_INTEGRATION_METHOD_TYPE, "Check if the integration method exist in EDLUT: Euler, RK2, RK4, BDF1, BDF2, ..., BDF6, Bifixed_Euler, Bifixed_RK2, Bifixed_RK4, Bifixed_BDF2, etc." },
	{ REPAIR_INTEGRATION_METHOD_READ, "Check if the integration method is defined in the configuration file: Euler, RK2, RK4, BDF1, BDF2, ..., BDF6, Bifixed_Euler, Bifixed_RK2, Bifixed_RK4, Bifixed_BDF2, etc." },

	{ REPAIR_BI_FIXED_STEP, "Set a value greater than zero" },

	{ REPAIR_FIXED_STEP, "Set a value greater than zero" },

	//NETOWORK
	{ REPAIR_NETWORK_NEURON_MODEL_TYPE, "Check if this neuron model is defined in EDLUT" },
	{ REPAIR_NETWORK_NEURON_MODEL_NUMBER, "Specify the correct number of neuron types or change the type of some neurons" },
	{ REPAIR_NETWORK_NEURON_MODEL_LOAD_NUMBER, "Specify the number of neuron types" },
	{ REPAIR_NETWORK_NUMBER_OF_NEURONS, "Specify correctly the number of neurons in the configuration file" },
	{ REPAIR_NETWORK_NEURON_PARAMETERS, "Define five parameters: number of neurons, neuron model, neuron configuration model, output neuron, monitored neuron" },
	{ REPAIR_NETWORK_ALLOCATE, "Free more memory or use a smaller network" },
	{ REPAIR_NETWORK_READ_NUMBER_OF_NEURONS, "Specify the number of neurons in the configuration file" },
	{ REPAIR_NETWORK_LEARNING_RULE_TYPE, "Select a type of learning rule defined in EDLUT" },
	{ REPAIR_NETWORK_LEARNING_RULE_LOAD, "Set a correct number of learning rules in network file" },
	{ REPAIR_NETWORK_LEARNING_RULE_NUMBER, "Set a correct number of learning rules in network file" },
	{ REPAIR_NETWORK_SYNAPSES_LEARNING_RULE_INDEX, "Select a correct learning rule index" },
	{ REPAIR_NETWORK_SYNAPSES_NUMBER, "Specify the correct number of interconnections in the network file" },
	{ REPAIR_NETWORK_SYNAPSES_NEURON_INDEX, "Define the neuron or correct the interconnection neuron index" },
	{ REPAIR_NETWORK_SYNAPSES_LEARNING_RULE, "Set one learning rule as trigger and the other one as non trigger" },
	{ REPAIR_NETWORK_SYNAPSES_LOAD, "Specify elevent parameters: first source neuron index, number of source neurons, first target neuron index, number of target neurons, number of replications, synapse delay, synapse delay increment, synapsis type, maximum weight, first learning rule index, second learning rule indes (optional)" },
	{ REPAIR_NETWORK_SYNAPSES_LOAD_NUMBER, "Specify the number of synapses in network file" },
	{ REPAIR_NETWORK_SYNAPSES_TYPE, "Specify a correct synapse type" },
	{ REPAIR_NETWORK_OPEN, "Ensure that the network file has the proper name and is in the application directory" },

	//WEIGHTS
	{ REPAIR_WEIGHTS_OPEN, "Ensure that the weights file has the proper name and is in the application directory" },
	{ REPAIR_WEIGHTS_READ, "Specify more weights in the weights file: number of weights (first parameter) and weights in nS (second parameter)" },
	{ REPARI_WEIGHTS_NUMBER, "Specify the correct number of weights in the weights file" },
	{ REPAIR_WEIGHTS_SAVE, "Check for disk problems" },

	//LEARNING RULES
	{ REPAIR_LEARNING_RULE_VALUES, "Specify a correct value for this parameter" },

		{ REPAIR_ADDITIVE_KERNEL_CHANGE_LOAD, "Specify three learning rule parameters: kernel peak position (s), direct increment/decrement (nS), kernel amplitude (nS)" },

		{ REPAIR_COS_WEIGHT_CHANGE_LOAD, "Specify four learning rule parameters: kernel size (s), exponent (adimensional), direct increment/decrement (nS), kernel amplitude (nS)" },

		{ REPAIR_SIN_WEIGHT_CHANGE_LOAD, "Specify the fourth parameter: exponent (adimensional)" },
		
		{ REPAIR_EXP_OPTIMISED_BUFFERED_WEIGHT_CHANGE_LOAD, "Specify the fourth parameter: init time (s)" },

		{ REPAIR_COS_SIN_WEIGHT_CHANGE_LOAD, "Specify three learning rule parameters: distance between central an lateral peaks (s), central amplitude (nS), lateral amplitude (nS)" },

		{ REPAIR_STDP_WEIGHT_CHANGE_LOAD, "Specify four learning rule parameters: LTP (nS), LTP tau (s), LTD (nS), LTD tau (nS)" },

	//NEURON MODELS
	{ REPAIR_NEURON_MODEL_VALUES, "Specify a correct value for this parameter in the configuration file or check for errors in previous lines" },
	{ REPAIR_NEURON_MODEL_NAME, "Check if the configuration file exist and has the same name that in the network file" },

		//NEURON MODEL TABLE, COMPRESS NEURON MODEL TABLE
		{ REPAIR_NEURON_MODEL_TABLE_ALLOCATE, "Free more memory or use smaller tables" },
		{ REPAIR_NEURON_MODEL_TABLE_NOT_ENOUGH_DATA, "Generate a correct file of neuron tables" },
		{ REPAIR_NEURON_MODEL_TABLE_TABLES_STRUCTURE, "Specify the event-driven neuron model tables strucuture in the configuration file or check for errors in the previous lines" },

		//TABLE BASED MODEL, SYNCHRONOUS TABLE BASED MODEL, COMPRESS TABLE BASED MODEL, COMPRESS SYNCHRONOUS TABLE BASED MODEL
		{ REPAIR_TABLE_BASED_MODEL_TIME_SCALE, "Specify the time scale in \"Second\" or \"Milisecond\"" },


};




unsigned long long EDLUTException::CalculateErrorValue(){
	return ((((TaskCode)&(unsigned long long)0xFFFF) << 32) | (((ErrorCode)&(unsigned long long)0xFFFF) << 16) | (((RepairCode)&(unsigned long long)0xFFFF)));
}

EDLUTException::EDLUTException(TASK_CODE task, ERROR_CODE error, REPAIR_CODE repair):TaskCode(task), ErrorCode(error), RepairCode(repair){
	this->ErrorNum = CalculateErrorValue();
}

unsigned long long EDLUTException::GetErrorNum() const{
	return this->ErrorNum;
}
		
const char * EDLUTException::GetTaskMsg() const{
	int N_Tasks = sizeof(TaskMsgs) / sizeof(TASK_STRUCT);
	int i;
	for (i = 0; i < N_Tasks; i++){
		if (this->TaskCode == TaskMsgs[i].code){
			break;
		}
	}
	if (i < N_Tasks){
		return TaskMsgs[i].msg;
	}
	else{
		printf("TASK CODE DOES NOT EXIST.\n");
		return "";
	}
}
		
const char * EDLUTException::GetErrorMsg() const{
	int N_Errors = sizeof(ErrorMsgs) / sizeof(ERROR_STRUCT);
	int i;
	for (i = 0; i < N_Errors; i++){
		if (this->ErrorCode == ErrorMsgs[i].code){
			break;
		}
	}
	if (i < N_Errors){
		return ErrorMsgs[i].msg;
	}
	else{
		printf("ERROR CODE DOES NOT EXIST.\n");
		return "";
	}
}
		
const char * EDLUTException::GetRepairMsg() const{
	int N_Repairs = sizeof(RepairMsgs) / sizeof(REPAIR_STRUCT);
	int i;
	for (i = 0; i < N_Repairs; i++){
		if (this->RepairCode == RepairMsgs[i].code){
			break;
		}
	}
	if (i < N_Repairs){
		return RepairMsgs[i].msg;
	}
	else{
		printf("REPAIR CODE DOES NOT EXIST.\n");
		return "";
	}
}		

void EDLUTException::display_error() const{

	if(this->ErrorNum){
		cerr << "Error while: " << this->GetTaskMsg() << endl;
		cerr << "Error message (" << this->ErrorNum << "): " << this->GetErrorMsg() << endl;
		cerr << "Try to: " << this->GetRepairMsg() << endl;
	}
}

ostream & operator<< (ostream & out, EDLUTException Exception){
	if(Exception.GetErrorNum()){
		out << "Error while: " << Exception.GetTaskMsg() << endl;
		out << "Error message " << Exception.GetErrorNum() << ": " << Exception.GetErrorMsg() << endl;
		out << "Try to: " << Exception.GetRepairMsg() << endl;
	}
	
	return out;
}

