//Library for CUDA
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <driver_types.h>
#include <stdio.h>
#include <cstdlib>

#include <iostream> 
using namespace std; 

static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))
