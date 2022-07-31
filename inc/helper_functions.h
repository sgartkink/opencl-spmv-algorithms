#ifndef _HELPER_FUNCTIONS_H
#define _HELPER_FUNCTIONS_H

#define CL_TARGET_OPENCL_VERSION 220
#include <CL/cl.h>

#include "mmio.h"
#include "enums.h"

char* read_source_from_cl_file(const char *file, size_t *size) 
{
    cl_int status;
    FILE *fp;
    char *source;
    int i;

    fp = fopen(file, "rb");
    if (fp == NULL) 
    {
        printf("Could not open kernel file\n");
        exit(-1);
    }
    
    status = fseek(fp, 0, SEEK_END);
    if (status != 0) 
    {
        printf("Error seeking to end of file\n");
        exit(-1);
    }
    
    *size = ftell(fp);
    if (*size < 0) 
    {
        printf("Error getting file position\n");
        exit(-1);
    }

    rewind(fp);

    source = (char *)malloc(*size + 1);

    for (i = 0; i < *size+1; i++) 
    {
        source[i] = '\0';
    }

    if (source == NULL) 
    {
        printf("Error allocating space for the kernel source\n");
        exit(-1);
    }

    fread(source, 1, *size, fp);
    source[*size] = '\0';

    return source;
} 

void readProgramBuildInfo(cl_program program, cl_device_id device)
{
    size_t log_size;
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &log_size);

    char *log = (char *)malloc(log_size);

    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

    printf("%s\n", log);
    
    free(log);
}

cl_int get_device_ids(cl_device_id *device_ids, cl_uint *number_of_devices)
{
    cl_int error;
    cl_uint number_of_platforms;
    cl_platform_id *platform_ids;
    cl_uint max_number_of_devices;
    
    error = clGetPlatformIDs(0, NULL, &number_of_platforms);
    
    if (0 == number_of_platforms)
    {
        printf("No OpenCL platform founds\n");
        return error;
    }
    
    if (error != CL_SUCCESS)
    {
        printf("clGetPlatformIDs error %d\n", error);
        return error;
    }
    
    platform_ids = (cl_platform_id*)malloc(sizeof(cl_platform_id *) * number_of_platforms);
    error = clGetPlatformIDs(number_of_platforms, platform_ids, NULL);
    
    if (error != CL_SUCCESS)
    {
        printf("clGetPlatformIDs error %d\n", error);
        return error;
    }

    for (cl_uint platform_number = 0; platform_number < number_of_platforms; ++platform_number)
    {
        error = clGetDeviceIDs(platform_ids[platform_number], CL_DEVICE_TYPE_GPU, 0, NULL, &max_number_of_devices);

        if (0 == max_number_of_devices)
        {
            printf("No OpenCL devices found on the platform\n");
            continue;
        }
        
        if (max_number_of_devices < *number_of_devices)
        {
            *number_of_devices = max_number_of_devices;
        }

        error = clGetDeviceIDs(platform_ids[platform_number], CL_DEVICE_TYPE_GPU, *number_of_devices, device_ids, NULL);
        
        break;
    }
    
    free(platform_ids);
    
    return error;
}

#endif /* _HELPER_FUNCTIONS_H */
