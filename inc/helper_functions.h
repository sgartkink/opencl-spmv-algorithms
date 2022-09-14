#ifndef _HELPER_FUNCTIONS_H
#define _HELPER_FUNCTIONS_H

#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <math.h>

#include "mmio.h"
#include "enums.h"

#define EPSILON 0.0000001

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
        return NULL;
    }
    
    status = fseek(fp, 0, SEEK_END);
    if (status != 0) 
    {
        printf("Error seeking to end of file\n");
        return NULL;
    }
    
    *size = ftell(fp);
    if (*size < 0) 
    {
        printf("Error getting file position\n");
        return NULL;
    }

    rewind(fp);

    source = (char *)malloc(*size + 1);

    if (source == NULL) 
    {
        printf("Error allocating space for the kernel source\n");
        return NULL;
    }
    
    for (i = 0; i < *size+1; i++) 
    {
        source[i] = '\0';
    }

    fread(source, 1, *size, fp);
    source[*size] = '\0';

    return source;
} 

void read_build_program_info(cl_program program, cl_device_id device)
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

/*!
 * \brief File must be opened before calling this function.
 */
bool read_size_of_matrices_from_file(FILE *file, int *number_of_rows, int *number_of_columns, int *number_of_nonzeroes)
{
    MM_typecode matcode;
    
    if (file == NULL)
    {
        return false;
    }
    
    if (mm_read_banner(file, &matcode) != 0)
    {
        printf("Could not process Matrix Market banner.\n");
        return false;
    }

    /*  This is how one can screen matrix types if their application */
    /*  only supports a subset of the Matrix Market data types.      */
    if (mm_is_complex(matcode) && mm_is_matrix(matcode) && mm_is_sparse(matcode))
    {
        printf("Sorry, this application does not support ");
        printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
        return false;
    }

    /* find out size of sparse matrix */
    if (mm_read_mtx_crd_size(file, number_of_rows, number_of_columns, number_of_nonzeroes) != 0) 
    {
        return false;
    }
    
    return true;
}

bool check_result(const char *filename, cl_double *vect, cl_double *result)
{
    FILE *file;
    int number_of_rows;
    int number_of_columns;
    int number_of_nonzeroes;
    int i;
    cl_double *data;
    
    file = fopen(filename, "r");

    if (file == NULL) 
    {
        perror(filename);
        return false;
    }
    
    if (read_size_of_matrices_from_file(file, &number_of_rows, &number_of_columns, &number_of_nonzeroes) == false)
    {
        fclose(file);
        return false;
    }
    
    data = (cl_double *)calloc(number_of_rows, sizeof(cl_double));
    
    for (i = 0; i < number_of_nonzeroes; i++)
    {
        double value;
        int current_row;
        int current_col;
        
        fscanf(file, "%d %d %lg\n", &current_row, &current_col, &value);
        current_row--; // adjust from 1-based to 0-based
        current_col--; 
        data[current_row] += value * vect[current_col];
    }
    
    for (i = 0; i < number_of_rows; ++i)
    {
        if (fabs(data[i] - result[i]) > EPSILON)
        {
            printf("wrong value at index %d: %f - %f\n", i, data[i], result[i]);
            free(data);
            fclose(file);
            return false;
        }
    }
    
    fclose(file);
    free(data);
    
    return true;
}
    
#endif /* _HELPER_FUNCTIONS_H */
