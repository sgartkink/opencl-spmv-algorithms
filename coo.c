#define CL_TARGET_OPENCL_VERSION 220
#include <CL/cl.h>
#include <stdio.h>                                                                                                                                               
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>

#include "mmio.h"
#include "helper_functions.h"

int main(int argc, char *argv[])
{
    (void)argc;
    (void)argv;
    
    cl_int error;
    cl_uint platformNumber = 0;
    
    error = clGetPlatformIDs(0, NULL, &platformNumber);
    
    if (0 == platformNumber)
    {
        printf("No OpenCL platform founds\n");
        return 1;
    }
    
    if (error != CL_SUCCESS)
    {
        printf("clGetPlatformIDs error %d\n", error);
        return 1;
    }
    
    cl_platform_id *platformIds = (cl_platform_id*)malloc(sizeof(cl_platform_id *) * platformNumber);
    
    error = clGetPlatformIDs(platformNumber, platformIds, NULL);
    
    if (error != CL_SUCCESS)
    {
        printf("clGetPlatformIDs error %d\n", error);
        return 1;
    }

    for (cl_uint i = 0; i < platformNumber; ++i)
    {
        cl_uint deviceNumber;
        
        error = clGetDeviceIDs(platformIds[i], CL_DEVICE_TYPE_GPU, 0, NULL, &deviceNumber);

        if (0 == deviceNumber)
        {
            printf("No OpenCL devices found on the platform\n");
            continue;
        }

        cl_device_id* deviceIds = (cl_device_id*)malloc(sizeof(cl_device_id*) * deviceNumber);
        
        error = clGetDeviceIDs(platformIds[i], CL_DEVICE_TYPE_ALL, deviceNumber, deviceIds, &deviceNumber);
        
        for (cl_uint j = 0; j < deviceNumber; ++j)
        {
            int rows_nr, cols_nr, nonzeros_nr;
            MM_typecode matcode;
            FILE *f;
            cl_int *rows, *cols;
            cl_double *data_double;
            cl_int *data_int;
            
            if ((f = fopen("databases/cant.mtx", "r")) == NULL) 
            {
                return 0;
            }
            
            if (mm_read_banner(f, &matcode) != 0)
            {
                printf("Could not process Matrix Market banner.\n");
                return 0;
            }

            /*  This is how one can screen matrix types if their application */
            /*  only supports a subset of the Matrix Market data types.      */
            if (mm_is_complex(matcode) && mm_is_matrix(matcode) && 
                    mm_is_sparse(matcode))
            {
                printf("Sorry, this application does not support ");
                printf("Market Market type: [%s]\n", mm_typecode_to_str(matcode));
                return 0;
            }

            /* find out size of sparse matrix .... */
            if (mm_read_mtx_crd_size(f, &rows_nr, &cols_nr, &nonzeros_nr) != 0) 
            {
                return 0;
            }

            /* reseve memory for matrices */
            rows = malloc(nonzeros_nr * sizeof(cl_int));
            cols = malloc(nonzeros_nr * sizeof(cl_int));
            data_int = malloc(nonzeros_nr * sizeof(cl_int));
            data_double = malloc(nonzeros_nr * sizeof(cl_double));

            /* NOTE: when reading in doubles, ANSI C requires the use of the "l"  */
            /* specifier as in "%lg", "%lf", "%le", otherwise errors will occur   */
            /* (ANSI C X3.159-1989, Sec. 4.9.6.2, p. 136 lines 13-15)             */

            for (int i = 0; i < nonzeros_nr; i++)
            {
                fscanf(f, "%d %d %lg\n", &rows[i], &cols[i], &data_double[i]);
                rows[i]--;  /* adjust from 1-based to 0-based */
                cols[i]--;
                data_int[i] = (int)data_double[i];
            }
        
            if (f != stdin) 
            {
                fclose(f);
            }
             
            size_t vectorSize[1] = { nonzeros_nr };
            size_t localWorkSize[1] = { 1 };
            cl_uint work_dim = 1;
    
            cl_int *vect = (cl_int*)malloc(sizeof(cl_int*) * cols_nr);
            for (int i = 0; i < cols_nr; ++i) {
                vect[i] = 2;
            }
            cl_int *output = (cl_int*)malloc(sizeof(cl_int) * rows_nr);
            
            cl_context context = clCreateContext(0, deviceNumber, deviceIds, NULL, NULL, NULL);
            
            if (NULL == context)
            {
                printf("context is null\n");
                return 2;
            }
            
            cl_command_queue commandQueue = clCreateCommandQueueWithProperties(context, deviceIds[0], 0, &error);
            
            if (error != CL_SUCCESS)
            {
                printf("clCreateCommandQueueWithProperties error %d\n", error);
                return 2;
            }
            
            cl_mem buffer_row = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_int) * nonzeros_nr, NULL, &error);
            cl_mem buffer_col = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_int) * nonzeros_nr, NULL, &error);
            cl_mem buffer_data = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_int) * nonzeros_nr, NULL, &error);
            cl_mem buffer_vect = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_int) * cols_nr, NULL, &error);
            cl_mem buffer_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_int) * rows_nr, NULL, &error);
            
            if (error != CL_SUCCESS)
            {
                printf("clCreateBuffer error %d\n", error);
                return 2;
            }
            
            size_t size;
            const char *source = read_source_from_cl_file("kernels/Coo.cl", &size);
            
            cl_program program = clCreateProgramWithSource(context, 1, &source, &size, &error);
            
            if (error != CL_SUCCESS)
            {
                printf("clCreateProgramWithSource error %d\n", error);
                return 2;
            }
            
            error = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
            
            if (error != CL_SUCCESS)
            {
                readProgramBuildInfo(program, deviceIds[0]);
                return 2;
            }
            
            cl_kernel kernel = clCreateKernel(program, "coo", &error);
            
            if (error != CL_SUCCESS)
            {
                printf("clCreateKernel error %d\n", error);
                return 2;
            }
            
            error  = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&buffer_row);
            error |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&buffer_col);
            error |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&buffer_data);
            error |= clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&buffer_vect);
            error |= clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&buffer_output);
            error |= clSetKernelArg(kernel, 5, sizeof(int), (void*)&nonzeros_nr);
            
            if (error != CL_SUCCESS)
            {
                printf("clSetKernelArg errror\n");
                return 2;
            }
            
            error  = clEnqueueWriteBuffer(commandQueue, buffer_row, CL_FALSE, 0, sizeof(cl_int) * nonzeros_nr, rows, 0, NULL, NULL);
            error |= clEnqueueWriteBuffer(commandQueue, buffer_col, CL_FALSE, 0, sizeof(cl_int) * nonzeros_nr, cols, 0, NULL, NULL);
            error |= clEnqueueWriteBuffer(commandQueue, buffer_data, CL_FALSE, 0, sizeof(cl_int) * nonzeros_nr, data_int, 0, NULL, NULL);
            error |= clEnqueueWriteBuffer(commandQueue, buffer_vect, CL_FALSE, 0, sizeof(cl_int) * cols_nr, vect, 0, NULL, NULL);
            
            if (error != CL_SUCCESS)
            {
                printf("clEnqueueWriteBuffer error %d\n", error);
                return 2;
            }
            clFinish(commandQueue);
            
            cl_event nd_range_kernel_event;

            clock_t start = clock();
            error = clEnqueueNDRangeKernel(commandQueue, kernel, work_dim, NULL, vectorSize, localWorkSize, 0, NULL, &nd_range_kernel_event);
            
            clWaitForEvents(1, &nd_range_kernel_event);
            clFinish(commandQueue);
            
            clock_t end = clock();
            float ms = (float)(end - start) / (CLOCKS_PER_SEC / 1000);
            printf("Your calculations took %.2lf ms to run.\n", ms);
            
            if (error != CL_SUCCESS)
            {
                printf("clEnqueueNDRangeKernel error %d\n", error);
                return 2;
            }
            
            error = clEnqueueReadBuffer(commandQueue, buffer_output, CL_TRUE, 0, sizeof(cl_int) * rows_nr, output, 0, NULL, NULL);
            clFinish(commandQueue);
            
            if (error != CL_SUCCESS)
            {
                printf("clEnqueueReadBuffer error %d\n", error);
                return 2;
            }

//             for (size_t k = 0; k < rows_nr; ++k)
//             {
//                 printf("%ld: %d\n", k, output[k]);
//             }

            clReleaseMemObject(buffer_row);
            clReleaseMemObject(buffer_col);
            clReleaseMemObject(buffer_data);
            clReleaseMemObject(buffer_vect);
            
            free(rows);
            free(cols);
            free(data_int);
            free(data_double);
            
            clFlush(commandQueue);
            clReleaseCommandQueue(commandQueue);
            clReleaseKernel(kernel);
            clReleaseProgram(program);
            clReleaseContext(context);
            
            i = platformNumber;
            break;
        }
        
        free(deviceIds);
    }
    
    free(platformIds);
    
    return 0;
}
