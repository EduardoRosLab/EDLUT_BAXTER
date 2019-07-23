#                                               -*- Autoconf -*-
# Process this file with autoconf to produce a configure script.
# By defult all the options are enabled and search the paths. 
# Parameters: --with-matlab=/path/matlab --without-matlab --without-simulink
# --enable-robot --enable-debugging --enable-profiling --disable-openmp --with-cuda=/cudapath
# --without-cuda


AC_PREREQ([2.69])
AC_INIT([edlutkernel], [2.0], [jgarridoalcazar@gmail.com])

m4_include([m4/matlab.m4])
m4_include([m4/simulink.m4])
m4_include([m4/robot.m4])
m4_include([m4/mexext.m4])
m4_include([m4/cuda.m4])


# Checks for programs.
AC_PROG_CXX
AC_PROG_CC
AC_PROG_AWK

AC_SUBST(compiler,$CXX)
AC_SUBST(ccompiler, $CC)

# Define default language
AC_LANG([C++])

# Checks for libraries.

# Checks for header files.
AC_CHECK_HEADERS([arpa/inet.h fcntl.h limits.h netdb.h netinet/in.h stdint.h string.h sys/socket.h unistd.h wchar.h])

# Checks for typedefs, structures, and compiler characteristics.
AC_CHECK_HEADER_STDBOOL
AC_TYPE_INT16_T
AC_TYPE_INT32_T
AC_TYPE_INT64_T
AC_TYPE_INT8_T
AC_TYPE_UINT16_T
AC_TYPE_UINT32_T
AC_TYPE_UINT64_T
AC_TYPE_UINT8_T

# Checks for library functions.
AC_FUNC_MALLOC
AC_CHECK_FUNCS([memset pow socket sqrt])

AC_CHECK_PROG([dox],doxygen)

# Checks for openMP dependences. It sets OPENMP_CXXFLAGS, that need to be used when compiling programs.
AC_OPENMP

AC_SUBST(openmp_cxxflags,$OPENMP_CXXFLAGS)

AS_IF([test "x$OPENMP_CXXFLAGS" == "x"],
 [AC_SUBST(use_openmp, false)],
 [AC_SUBST(use_openmp, true)])



################################################################################

# Checks for Matlab installation and if so, defines `HAVE_MATLAB'. Set output variable 
# `MATLAB' to the top-level Matlab installation. The user can specify an alternative directory with the
# `--with-matlab' option for `configure' or by setting `MATLAB' in the environment.
AX_MATLAB

AS_IF([test "x$MATLAB" == "x"],
 [AC_SUBST(use_matlab, false)
 AC_SUBST(mextarget, '')],
 [AC_SUBST(use_matlab, true)
 AC_SUBST(mextarget,mex)])
 
AC_SUBST(matlab, $MATLAB) 


# Check for Simulink. The user can enable or disable Simulink support with the
# `--with-simulink' option for `configure'.  Simulink support is always disabled
# if Matlab could not be found. If this macro supports Simulink, it defines `HAVE_SIMULINK'.
AX_SIMULINK

AS_IF([test "x$SIMULINK" == "x"],
 [AC_SUBST(use_simulink, false)
 AC_SUBST(sfunctiontarget)],
 [AC_SUBST(use_simulink, true)
 AC_SUBST(sfunctiontarget,sfunction)])


# Check for MEX-file extension. Set output variable `MEXEXT' to the MEX-file extension, for
# example, `mexsol' for Sun Solaris.  Please note that `MEXEXT' has no leading period.
# The user can override the test by setting `MEXEXT' in the environment.
AX_MEXEXT
AC_SUBST(mexext, $MEXEXT)


################################################################################

# Checks for Cuda installation and if so, defines `HAVE_CUDA'. Set output variable 
# `CUDA' to the top-level Matlab installation. The user can specify an alternative directory with the
# `--with-cuda' option for `configure' or by setting `CUDA' in the environment.
AX_CUDA

AS_IF([test "x$CUDA" == "x"],
 [AC_SUBST(use_cuda, false)],
 [AC_SUBST(use_cuda, true)
 AC_SUBST(cuda, $CUDA)
 AC_SUBST(cuda_lib, $CUDA_LIB)
 AC_SUBST(cudaversion, $CUDA_VERSION)
 AC_SUBST(cudamajor, $CUDA_MAJOR)
 AC_SUBST(cudaminor, $CUDA_MINOR)])
 
 
################################################################################

# Checks for robot simulator installation and if so, defines `enable_robot' to true. 
# For the moment, --with-matlab option is required in order to compile the robot simulator.

AX_ROBOT

AC_SUBST(robot_generation, $USEROBOT)
AS_IF([test "x$USEROBOT" == "xno"],
 [AC_SUBST(robot_generation, false)
 AC_SUBST(robottarget)],
 [AC_SUBST(robot_generation, true)
 AC_SUBST(robottarget,\$\(robottarget\))])
 
 
 
################################################################################

# Enable profiling. Automatically set debugging to true
AC_ARG_ENABLE(profiling,
 [--enable-profiling  Enable profiling information. This is a debugging feature which should not be usually enabled],
 [AC_MSG_NOTICE([Generating profiling information.])
 enable_profiling=true],
 [AC_MSG_NOTICE([Ignoring profiling information.])
 enable_profiling=false])
AC_SUBST(profile_code, $enable_profiling)

# Enable debugging. If Enable debugging then optimization are reduced.
AC_ARG_ENABLE(debugging,
 [--enable-debugging  Enable debugging information. This is a debugging feature which should not be usually enabled],
 [AC_MSG_NOTICE([Generating debugging code configuration.])
 enable_optimize=false],
 [AS_IF([test "$enable_profiling" = true],
  [AC_MSG_NOTICE([Forcing debugging code configuration since profiling is activated.])
  enable_optimize=false],
  [AC_MSG_NOTICE([Generating release code configuration.])
  enable_optimize=true])])
AC_SUBST(optimize_code, $enable_optimize)


##############################################################################
# Checking architecture
AC_SUBST(architecture,`getconf LONG_BIT`)


AC_CONFIG_FILES([makefile])
AC_CONFIG_FILES([rules.makefile])
AC_CONFIG_FILES([flags.makefile])
#cp confdefs.h EdlutConfig.h
AC_OUTPUT
