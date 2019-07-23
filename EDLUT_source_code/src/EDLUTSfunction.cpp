/***************************************************************************
 *                           EDLUTSFunction.cpp                               *
 *                           -------------------                           *
 * copyright            : (C) 2010 by Jesus Garrido                        *
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

/*
 * You must specify the S_FUNCTION_NAME as the name of your S-function.
 */

#define S_FUNCTION_NAME  EDLUTKernel
#define S_FUNCTION_LEVEL 2

/*
 * Need to include simstruc.h for the definition of the SimStruct and
 * its associated macro definitions.
 *
 * The following headers are included by matlabroot/simulink/include/simstruc.h
 * when compiling as a MEX file:
 *
 *   matlabroot/extern/include/tmwtypes.h    - General types, e.g. real_T
 *   matlabroot/extern/include/mex.h         - MATLAB MEX file API routines
 *   matlabroot/extern/include/matrix.h      - MATLAB MEX file API routines
 *
 * The following headers are included by matlabroot/simulink/include/simstruc.h
 * when compiling your S-function with RTW:
 *
 *   matlabroot/extern/include/tmwtypes.h    - General types, e.g. real_T
 *   matlabroot/rtw/c/libsrc/rt_matrx.h      - Macros for MATLAB API routines
 *
 */
// #include "vld.h"

#include "simstruc.h"

#include "../include/interface/simulink/SimulinkBlockInterface.h"

// Define input parameters
// Define input parameters
#define PARAMNET ssGetSFcnParam(S,0) 	// Network description file
#define PARAMWEIGHT ssGetSFcnParam(S,1) 	// Weight description file
#define PARAMLOG ssGetSFcnParam(S,2) 	// Log activity output file
#define PARAMTDSTEP ssGetSFcnParam(S,3)	// Time-driven simulation step
#define PARAMINPUT ssGetSFcnParam(S,4) 	// Input map - Vector mapping each input line with an input cell
#define PARAMOUTPUT ssGetSFcnParam(S,5) 	// Output map - Vector mapping each output line with an output cell
#define PARAMSAVINGWEIGHTFILE ssGetSFcnParam(S,6)	// File where the synaptic weights will be stored.
#define PARAMSAVINGWEIGHTTIME ssGetSFcnParam(S,7)	// Step time between consecutive weight saving.
#define NUMBEROFPARAMETERS 8

/*====================*
 * S-function methods *
 *====================*/

/*=====================================*
 * Configuration and execution methods *
 *=====================================*/

/* Function: mdlInitializeSizes ===============================================
 * Abstract:
 *    The sizes information is used by Simulink to determine the S-function
 *    block's characteristics (number of inputs, outputs, states, etc.).
 *
 *    The direct feedthrough flag can be either 1=yes or 0=no. It should be
 *    set to 1 if the input, "u", is used in the mdlOutput function. Setting
 *    this to 0 is akin to making a promise that "u" will not be used in the
 *    mdlOutput function. If you break the promise, then unpredictable results
 *    will occur.
 *
 *    The NumContStates, NumDiscStates, NumInputs, NumOutputs, NumRWork,
 *    NumIWork, NumPWork NumModes, and NumNonsampledZCs widths can be set to:
 *       DYNAMICALLY_SIZED    - In this case, they will be set to the actual
 *                              input width, unless you are have a
 *                              mdlSetWorkWidths to set the widths.
 *       0 or positive number - This explicitly sets item to the specified
 *                              value.
 */
static void mdlInitializeSizes(SimStruct *S)
{
	ssPrintf("Initializing sizes\n");

    int_T nInputPorts  = 1;  /* number of input ports  */
    int_T nOutputPorts = 1;  /* number of output ports */
    int_T needsInput   = 1;  /* direct feed through    */

    int_T inputPortIdx  = 0;
    int_T outputPortIdx = 0;

    ssPrintf("Setting number of parameters\n");
    ssSetNumSFcnParams(S, NUMBEROFPARAMETERS);  /* Number of expected parameters */
    if (ssGetNumSFcnParams(S) != ssGetSFcnParamsCount(S)) {
        /*
         * If the the number of expected input parameters is not equal
         * to the number of parameters entered in the dialog box return.
         * Simulink will generate an error indicating that there is a
         * parameter mismatch.
         */
        return;
    }

    /*
     * Configure tunability of parameters.  By default, all parameters are
     * tunable (changeable) during simulation.  If there are parameters that
     * cannot change during simulation, such as any parameters that would change
     * the number of ports on the block, the sample time of the block, or the
     * data type of a signal, mark these as non-tunable using a call like this:
     *
     *    ssSetSFcnParamTunable(S, 0, 0);
     *
     * which sets parameter 0 to be non-tunable (0).
     *
     */
    ssPrintf("Setting parameters as non-tuneable\n");
    for (int i=0; i<NUMBEROFPARAMETERS; ++i){
    	ssSetSFcnParamTunable(S, i, 0); // Set every parameter as non tuneable
    }


    // Check parameter types
    // 4th parameter -> Input array map. Unsigned int array.
    ssPrintf("Setting input cells as int parameter\n");
    DTypeId  dtype4 = ssGetDTypeIdFromMxArray(PARAMINPUT);
	if (!(dtype4 == SS_UINT8 || dtype4 == SS_UINT16 || dtype4 == SS_UINT32)){
		real_T * InputCells = (real_T *)mxGetData(PARAMINPUT);
		unsigned int NumberOfElements = (unsigned int) mxGetNumberOfElements(PARAMINPUT);
		
		for (unsigned int i=0; i<NumberOfElements; ++i){
			if ((int)InputCells[i] != InputCells[i] || InputCells[i]<0){
				ssSetErrorStatus(S, "Invalid input cell map - Real type");
				return;
			}
		}		
	}
    
    // 5th parameter -> Output array map. Unsigned int array.
    ssPrintf("Setting output cells as int parameter\n");
    DTypeId  dtype5 = ssGetDTypeIdFromMxArray(PARAMOUTPUT);
	if (!(dtype5 == SS_UINT8 || dtype5 == SS_UINT16 || dtype5 == SS_UINT32)){
		real_T * OutputCells = (real_T *)mxGetData(PARAMOUTPUT);
		unsigned int NumberOfElements = (unsigned int) mxGetNumberOfElements(PARAMOUTPUT);
		
		for (unsigned int i=0; i<NumberOfElements; ++i){
			if ((int)OutputCells[i] != OutputCells[i] || OutputCells[i] < 0 ){
				ssSetErrorStatus(S, "Invalid output cell map - Real type");
				return;
			}
		}		
	}

    /* Register the number and type of states the S-Function uses */
    ssPrintf("Setting continuous and discrete states\n");
    ssSetNumContStates(S, 0);   /* number of continuous states           */
    ssSetNumDiscStates(S, 0);   /* number of discrete states             */


    /*
     * Configure the input ports. First set the number of input ports.
     */
    ssPrintf("Setting input ports\n");
    if (!ssSetNumInputPorts(S, nInputPorts)) return;

    // Input Port Dimension will be the length of 4th parameter
    ssSetInputPortWidth(S, inputPortIdx, mxGetNumberOfElements(PARAMINPUT));

    // Set input port memory area as contiguo
    ssSetInputPortRequiredContiguous(S, inputPortIdx, 0);;

    // Set input port as boolean
    if(!ssSetInputPortDataType(S, 0, SS_BOOLEAN)) return;

    /*
     * Set direct feedthrough flag (1=yes, 0=no).
     * A port has direct feedthrough if the input is used in either
     * the mdlOutputs or mdlGetTimeOfNextVarHit functions.
     * See matlabroot/simulink/src/sfuntmpl_directfeed.txt.
     */
    ssSetInputPortDirectFeedThrough(S, inputPortIdx, needsInput);

    /*
     * Configure the output ports. First set the number of output ports.
     */
    ssPrintf("Setting output ports\n");
    if (!ssSetNumOutputPorts(S, nOutputPorts)) return;

    // Output Port Dimension will be the length of 5th parameter
    ssSetOutputPortWidth(S,outputPortIdx,mxGetNumberOfElements(PARAMOUTPUT));

    // Set output port memory area as contiguo
    // ssSetOutputPortRequiredContiguous(S, outputPortIdx, 1);

    // Set output port as boolean
    if(!ssSetOutputPortDataType(S, 0, SS_BOOLEAN)) return;

    /*
     * Set the number of sample times. This must be a positive, nonzero
     * integer indicating the number of sample times or it can be
     * PORT_BASED_SAMPLE_TIMES. For multi-rate S-functions, the
     * suggested approach to setting sample times is via the port
     * based sample times method. When you create a multirate
     * S-function, care needs to be taking to verify that when
     * slower tasks are preempted that your S-function correctly
     * manages data as to avoid race conditions. When port based
     * sample times are specified, the block cannot inherit a constant
     * sample time at any port.
     */
    ssSetNumSampleTimes(S, 1);   /* number of sample times                */

    /*
     * Set size of the work vectors.
     */
    ssPrintf("Setting static variables\n");
    ssSetNumRWork(S, DYNAMICALLY_SIZED);   /* number of real work vector elements   */
    ssSetNumIWork(S, DYNAMICALLY_SIZED);   /* number of integer work vector elements*/
    // Pointer work vector element: Simulation object pointer
    ssSetNumPWork(S, DYNAMICALLY_SIZED);   /* number of pointer work vector elements*/
    ssSetNumModes(S, DYNAMICALLY_SIZED);   /* number of mode work vector elements   */
    ssSetNumNonsampledZCs( S, DYNAMICALLY_SIZED);   /* number of nonsampled zero crossings   */

    /*
     * All options have the form SS_OPTION_<name> and are documented in
     * matlabroot/simulink/include/simstruc.h. The options should be
     * bitwise or'd together as in
     *   ssSetOptions(S, (SS_OPTION_name1 | SS_OPTION_name2))
     */

	ssSetOptions(S, SS_OPTION_DISALLOW_CONSTANT_SAMPLE_TIME);
    ssSetOptions(S, SS_OPTION_EXCEPTION_FREE_CODE);
    ssPrintf("Initialization finished\n");

} /* end mdlInitializeSizes */

#define MDL_SET_WORK_WIDTHS   /* Change to #undef to remove function */
#if defined(MDL_SET_WORK_WIDTHS) && defined(MATLAB_MEX_FILE)
  /* Function: mdlSetWorkWidths ===============================================
   * Abstract:
   *      The optional method, mdlSetWorkWidths is called after input port
   *      width, output port width, and sample times of the S-function have
   *      been determined to set any state and work vector sizes which are
   *      a function of the input, output, and/or sample times. This method
   *      is used to specify the nonzero work vector widths via the macros
   *      ssNumContStates, ssSetNumDiscStates, ssSetNumRWork, ssSetNumIWork,
   *      ssSetNumPWork, ssSetNumModes, and ssSetNumNonsampledZCs.
   *
   *      Run-time parameters are registered in this method using methods 
   *      ssSetNumRunTimeParams, ssSetRunTimeParamInfo, and related methods.
   *
   *      If you are using mdlSetWorkWidths, then any work vectors you are
   *      using in your S-function should be set to DYNAMICALLY_SIZED in
   *      mdlInitializeSizes, even if the exact value is known at that point.
   *      The actual size to be used by the S-function should then be specified
   *      in mdlSetWorkWidths.
   */
  static void mdlSetWorkWidths(SimStruct *S)
  {
	  if (ssGetSampleTime(S, 0) == CONTINUOUS_SAMPLE_TIME) {
		  ssSetErrorStatus(S,"This block cannot be assigned a continuous sample time\n");
	  }

	  ssPrintf("Setting static variables\n");
	  ssSetNumRWork(S, 0);   /* number of real work vector elements   */
      ssSetNumIWork(S, 0);   /* number of integer work vector elements*/
      // Pointer work vector element: Simulation object pointer
      ssSetNumPWork(S, 1);   /* number of pointer work vector elements*/
      ssSetNumModes(S, 0);   /* number of mode work vector elements   */
      ssSetNumNonsampledZCs( S, 0);   /* number of nonsampled zero crossings   */
  }
#endif /* MDL_SET_WORK_WIDTHS */


/* Function: mdlInitializeSampleTimes =========================================
 * Abstract:
 *
 *    This function is used to specify the sample time(s) for your S-function.
 *    You must register the same number of sample times as specified in
 *    ssSetNumSampleTimes. If you specify that you have no sample times, then
 *    the S-function is assumed to have one inherited sample time.
 *
 *    The sample times are specified as pairs "[sample_time, offset_time]"
 *    via the following macros:
 *      ssSetSampleTime(S, sampleTimePairIndex, sample_time)
 *      ssSetOffsetTime(S, offsetTimePairIndex, offset_time)
 *    Where sampleTimePairIndex starts at 0.
 *
 *    The valid sample time pairs are (upper case values are macros defined
 *    in simstruc.h):
 *
 *      [CONTINUOUS_SAMPLE_TIME,  0.0                       ]
 *      [CONTINUOUS_SAMPLE_TIME,  FIXED_IN_MINOR_STEP_OFFSET]
 *      [discrete_sample_period,  offset                    ]
 *      [VARIABLE_SAMPLE_TIME  ,  0.0                       ]
 *
 *    Alternatively, you can specify that the sample time is inherited from the
 *    driving block in which case the S-function can have only one sample time
 *    pair:
 *
 *      [INHERITED_SAMPLE_TIME,  0.0                       ]
 *    or
 *      [INHERITED_SAMPLE_TIME,  FIXED_IN_MINOR_STEP_OFFSET]
 *
 *    The following guidelines may help aid in specifying sample times:
 *
 *      o A continuous function that changes during minor integration steps
 *        should register the [CONTINUOUS_SAMPLE_TIME, 0.0] sample time.
 *      o A continuous function that does not change during minor integration
 *        steps should register the
 *              [CONTINUOUS_SAMPLE_TIME, FIXED_IN_MINOR_STEP_OFFSET]
 *        sample time.
 *      o A discrete function that changes at a specified rate should register
 *        the discrete sample time pair
 *              [discrete_sample_period, offset]
 *        where
 *              discrete_sample_period > 0.0 and
 *              0.0 <= offset < discrete_sample_period
 *      o A discrete function that changes at a variable rate should
 *        register the variable step discrete [VARIABLE_SAMPLE_TIME, 0.0]
 *        sample time. The mdlGetTimeOfNextVarHit function is called to get
 *        the time of the next sample hit for the variable step discrete task.
 *        Note, the VARIABLE_SAMPLE_TIME can be used with variable step
 *        solvers only.
 *      o Discrete blocks which can operate in triggered subsystems.  For your
 *        block to operate correctly in a triggered subsystem or a periodic
 *        system it must register [INHERITED_SAMPLE_TIME, 0.0]. In a triggered
 *        subsystem after sample times have been propagated throughout the
 *        block diagram, the assigned sample time to the block will be
 *        [INHERITED_SAMPLE_TIME, INHERITED_SAMPLE_TIME]. Typically discrete
 *        blocks which can be periodic or reside within triggered subsystems
 *        need to register the inherited sample time and the option
 *        SS_OPTION_DISALLOW_CONSTANT_SAMPLE_TIME. Then in mdlSetWorkWidths, they
 *        need to verify that they were assigned a discrete or triggered
 *        sample time. To do this:
 *          mdlSetWorkWidths:
 *            if (ssGetSampleTime(S, 0) == CONTINUOUS_SAMPLE_TIME) {
 *              ssSetErrorStatus(S, "This block cannot be assigned a "
 *                               "continuous sample time");
 *            }
 *
 *    If your function has no intrinsic sample time, then you should indicate
 *    that your sample time is inherited according to the following guidelines:
 *
 *      o A function that changes as its input changes, even during minor
 *        integration steps should register the [INHERITED_SAMPLE_TIME, 0.0]
 *        sample time.
 *      o A function that changes as its input changes, but doesn't change
 *        during minor integration steps (i.e., held during minor steps) should
 *        register the [INHERITED_SAMPLE_TIME, FIXED_IN_MINOR_STEP_OFFSET]
 *        sample time.
 *
 *    To check for a sample hit during execution (in mdlOutputs or mdlUpdate),
 *    you should use the ssIsSampleHit or ssIsContinuousTask macros.
 *    For example, if your first sample time is continuous, then you
 *    used the following code-fragment to check for a sample hit. Note,
 *    you would get incorrect results if you used ssIsSampleHit(S,0,tid).
 *        if (ssIsContinuousTask(S,tid)) {
 *        }
 *    If say, you wanted to determine if the third (discrete) task has a hit,
 *    then you would use the following code-fragment:
 *        if (ssIsSampleHit(S,2,tid) {
 *        }
 *
 */
static void mdlInitializeSampleTimes(SimStruct *S)
{
    /* Register one pair for each sample time */
	ssPrintf("Initializing sample times\n");
	ssSetSampleTime(S, 0, INHERITED_SAMPLE_TIME);
    ssSetOffsetTime(S, 0, 0.0);

	time_T Step = ssGetSampleTime(S, 0);

    if (ssGetSampleTime(S, 0) == CONTINUOUS_SAMPLE_TIME) {
    	ssSetErrorStatus(S, "This block cannot be assigned a continuous sample time");
    }
    ssPrintf("Sample times initialized\n");

} /* end mdlInitializeSampleTimes */


#define MDL_SET_INPUT_PORT_DATA_TYPE   /* Change to #undef to remove function */
#if defined(MDL_SET_INPUT_PORT_DATA_TYPE) && defined(MATLAB_MEX_FILE)
  /* Function: mdlSetInputPortDataType =========================================
   * Abstract:
   *    This method is called with the candidate data type id for a dynamically
   *    typed input port.  If the proposed data type is acceptable, the method
   *    should go ahead and set the actual port data type using
   *    ssSetInputPortDataType.  If the data type is unacceptable an error
   *    should generated via ssSetErrorStatus.  Note that any other dynamically
   *    typed input or output ports whose data types are implicitly defined by
   *    virtue of knowing the data type of the given port can also have their
   *    data types set via calls to ssSetInputPortDataType or
   *    ssSetOutputPortDataType.
   *
   *    See matlabroot/simulink/include/simstruc_types.h for built-in
   *    type defines: SS_DOUBLE, SS_BOOLEAN, etc.
   *
   *    See matlabroot/simulink/src/sfun_dtype_io.c for an example.
   */
  static void mdlSetInputPortDataType(SimStruct *S, int portIndex,DTypeId dType)
  {
	  ssPrintf("Checking input port data type\n");
	  if (portIndex==0 && dType!=SS_BOOLEAN){
		  ssSetErrorStatus(S, "This block must be assigned a boolean input type");
	  }
	  ssPrintf("Input port data type verified\n");
  } /* mdlSetInputPortDataType */
#endif /* MDL_SET_INPUT_PORT_DATA_TYPE */


#define MDL_SET_OUTPUT_PORT_DATA_TYPE  /* Change to #undef to remove function */
#if defined(MDL_SET_OUTPUT_PORT_DATA_TYPE) && defined(MATLAB_MEX_FILE)
  /* Function: mdlSetOutputPortDataType ========================================
   * Abstract:
   *    This method is called with the candidate data type id for a dynamically
   *    typed output port.  If the proposed data type is acceptable, the method
   *    should go ahead and set the actual port data type using
   *    ssSetOutputPortDataType.  If the data type is unacceptable an error
   *    should generated via ssSetErrorStatus.  Note that any other dynamically
   *    typed input or output ports whose data types are implicitly defined by
   *    virtue of knowing the data type of the given port can also have their
   *    data types set via calls to ssSetInputPortDataType or
   *    ssSetOutputPortDataType.
   *
   *    See matlabroot/simulink/src/sfun_dtype_io.c for an example.
   */
  static void mdlSetOutputPortDataType(SimStruct *S,int portIndex,DTypeId dType)
  {
	  ssPrintf("Checking output port data type\n");
	  if (portIndex==0 && dType!=SS_BOOLEAN){
		  ssSetErrorStatus(S, "This block must be assigned a boolean output type");
	  }
	  ssPrintf("Output port data type verified\n");
  } /* mdlSetOutputPortDataType */
#endif /* MDL_SET_OUTPUT_PORT_DATA_TYPE */


#define MDL_SET_DEFAULT_PORT_DATA_TYPES /* Change to #undef to remove function*/
#if defined(MDL_SET_DEFAULT_PORT_DATA_TYPES) && defined(MATLAB_MEX_FILE)
  /* Function:  mdlSetDefaultPortDataTypes =====================================
   * Abstract:
   *    This method is called when there is not enough information in your
   *    model to uniquely determine the input and output data types
   *    for your block. When this occurs, Simulink's data type propagation
   *    engine calls this method to ask you to set your S-function default
   *    data type for any dynamically typed input and output ports.
   *
   *    If you do not provide this method and you have dynamically typed
   *    ports where Simulink does not have enough information to propagate
   *    data types to your S-function, then Simulink will assign the
   *    data type to the largest known port data type of your S-function.
   *    If there are no known data types, then Simulink will set the
   *    data type to double.
   *
   *    See matlabroot/simulink/src/sfun_dtype_io.c for an example.
   */
  static void mdlSetDefaultPortDataTypes(SimStruct *S)
  {
	  ssPrintf("Setting default port data type\n");
	  ssSetInputPortDataType(S,0,SS_BOOLEAN);
	  ssSetOutputPortDataType(S,0,SS_BOOLEAN);
	  ssPrintf("Default port data type set\n");
  } /* mdlSetDefaultPortDataTypes */
#endif /* MDL_SET_DEFAULT_PORT_DATA_TYPES */


#define MDL_INITIALIZE_CONDITIONS   /* Change to #undef to remove function */
#if defined(MDL_INITIALIZE_CONDITIONS)
  /* Function: mdlInitializeConditions ========================================
   * Abstract:
   *    In this function, you should initialize the continuous and discrete
   *    states for your S-function block.  The initial states are placed
   *    in the state vector, ssGetContStates(S) or ssGetDiscStates(S).
   *    You can also perform any other initialization activities that your
   *    S-function may require. Note, this method will be called at the
   *    start of simulation and if it is present in an enabled subsystem
   *    configured to reset states, it will be call when the enabled subsystem
   *    restarts execution to reset the states.
   *
   *    You can use the ssIsFirstInitCond(S) macro to determine if this is
   *    is the first time mdlInitializeConditions is being called.
   */
  static void mdlInitializeConditions(SimStruct *S)
  {


	  ssPrintf("Initializing simulation\n");
	  if (ssIsFirstInitCond(S)){
		  SimulinkBlockInterface * Simul = new SimulinkBlockInterface();
		  ssPrintf("Initializing simulation interface object\n");
		  Simul->InitializeSimulation(S);
		  ssPrintf("Saving simulation interface pointer\n");
		  ssSetPWorkValue(S, 0, Simul);
		  ssPrintf("Simulation interface object inicialized\n");
	  } else {
		  // Reset simulation
		  
	  }
	  ssPrintf("Simulation initialized\n");
  }
#endif /* MDL_INITIALIZE_CONDITIONS */


#define MDL_START  /* Change to #undef to remove function */
#if defined(MDL_START)
  /* Function: mdlStart =======================================================
   * Abstract:
   *    This function is called once at start of model execution. If you
   *    have states that should be initialized once, this is the place
   *    to do it.
   */
  static void mdlStart(SimStruct *S)
  {
	  ssPrintf("Starting simulation\n");	  
  }
#endif /*  MDL_START */


/* Function: mdlOutputs =======================================================
 * Abstract:
 *    In this function, you compute the outputs of your S-function
 *    block. Generally outputs are placed in the output vector(s),
 *    ssGetOutputPortSignal.
 */
static void mdlOutputs(SimStruct *S, int_T tid)
{
	//ssPrintf("Setting simulation outputs\n");
	SimulinkBlockInterface * Simul = (SimulinkBlockInterface *) ssGetPWorkValue(S,0);
	Simul->AssignOutputs(S);
	//ssPrintf("Simulation outputs set\n");
} /* end mdlOutputs */


#define MDL_UPDATE  /* Change to #undef to remove function */
#if defined(MDL_UPDATE)
  /* Function: mdlUpdate ======================================================
   * Abstract:
   *    This function is called once for every major integration time step.
   *    Discrete states are typically updated here, but this function is useful
   *    for performing any tasks that should only take place once per
   *    integration step.
   */
  static void mdlUpdate(SimStruct *S, int_T tid)
  {
	  //ssPrintf("Updating simulation step\n");
	  SimulinkBlockInterface * Simul = (SimulinkBlockInterface *) ssGetPWorkValue(S,0);
	  Simul->SimulateStep(S,tid);
	  //ssPrintf("Simulation step updated\n");
  }
#endif /* MDL_UPDATE */


/* Function: mdlTerminate =====================================================
 * Abstract:
 *    In this function, you should perform any actions that are necessary
 *    at the termination of a simulation.  For example, if memory was allocated
 *    in mdlStart, this is the place to free it.
 *
 *    Suppose your S-function allocates a few few chunks of memory in mdlStart
 *    and saves them in PWork. The following code fragment would free this
 *    memory.
 *        {
 *            int i;
 *            for (i = 0; i<ssGetNumPWork(S); i++) {
 *                if (ssGetPWorkValue(S,i) != NULL) {
 *                    free(ssGetPWorkValue(S,i));
 *                }
 *            }
 *        }
 */
static void mdlTerminate(SimStruct *S)
{
	ssPrintf("Finishing simulation\n");
	SimulinkBlockInterface * Simul = (SimulinkBlockInterface *) ssGetPWorkValue(S,0);
	delete (SimulinkBlockInterface *) Simul;
	ssPrintf("Simulation finished\n");
}

/*=============================*
 * Required S-function trailer *
 *=============================*/

#ifdef  MATLAB_MEX_FILE    /* Is this file being compiled as a MEX-file? */
#include "simulink.c"      /* MEX-file interface mechanism */
#else
#include "cg_sfun.h"       /* Code generation registration function */
#endif
