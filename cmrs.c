#define CL_TARGET_OPENCL_VERSION 220
#include <CL/cl.h>
#include <stdio.h>                                                                                                                                               
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include <math.h>

#include "mmio.h"
#include "helper_functions.h"
#include "enums.h"

#define DEVICES_DEFAULT_SIZE 8

int main(int argc, char *argv[])
{
    cl_int error;
    cl_uint number_of_devices = DEVICES_DEFAULT_SIZE;
    cl_device_id device_ids[DEVICES_DEFAULT_SIZE];
    
    if (get_device_ids(&device_ids[0], &number_of_devices) != CL_SUCCESS)
    {
        return OpenCLDeviceError;
    }
        
    for (cl_uint device_number = 0; device_number < number_of_devices; ++device_number)
    {
        int rows_nr; 
        int cols_nr;
        int nonzeros_nr;
        MM_typecode matcode;
        FILE *f;
        cl_int *cols;
        cl_double *data_double;
        cl_int *data_int;
        cl_int *strip_ptr;
        cl_int *row_in_strip;
        int height = 16;
        
        if ((f = fopen("databases/cant/cant.mtx-sorted", "r")) == NULL) 
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
        
        size_t vectorSize[1] = { 256 };
        size_t localWorkSize[1] = { height };
        cl_uint work_dim = 1;

        /* reseve memory for matrices */
        cols = (cl_int *)malloc(nonzeros_nr * sizeof(cl_int));
        data_int = (cl_int *)malloc(nonzeros_nr * sizeof(cl_int));
        data_double = (cl_double *)malloc(nonzeros_nr * sizeof(cl_double));
        strip_ptr = (cl_int *)malloc((ceil(rows_nr / height) + 1) * sizeof(cl_int));
        row_in_strip = (cl_int *)malloc(nonzeros_nr * sizeof(cl_int));
        
        strip_ptr[0] = 0;
        
        int previous_row = 1;
        int current_strip_row = 0;
        int strip_index = 1;
        for (int i = 0; i < nonzeros_nr; i++)
        {
            int current_row;
            int current_col;
            fscanf(f, "%d %d %lg\n", &current_row, &current_col, &data_double[i]);
            
            current_col--;
            data_int[i] = (int)data_double[i];
            cols[i] = current_col;
            
            if (previous_row == current_row)
            {
                row_in_strip[i] = current_strip_row;
            }
            else 
            {
                previous_row = current_row;
                
                if (current_strip_row == height - 1)
                {
                    strip_ptr[strip_index] = i;
                    strip_index++;
                    current_strip_row = 0;
                }
                else 
                {
                    current_strip_row++;
                }
                
                row_in_strip[i] = current_strip_row;
            }
        }
        
        if (f != stdin) 
        {
            fclose(f);
        }
        
        strip_ptr[(int)(ceil(rows_nr / height))] = nonzeros_nr;

        cl_int *vect = (cl_int*)malloc(sizeof(cl_int*) * cols_nr);
        for (int i = 0; i < cols_nr; ++i) {
            vect[i] = 1;
        }
        cl_int *output = (cl_int*)malloc(sizeof(cl_int) * rows_nr);
        
        cl_context context = clCreateContext(0, device_number, device_ids, NULL, NULL, NULL);
        
        if (NULL == context)
        {
            printf("context is null\n");
            return 2;
        }
        
        cl_command_queue commandQueue = clCreateCommandQueueWithProperties(context, device_ids[0], 0, &error);
        
        if (error != CL_SUCCESS)
        {
            printf("clCreateCommandQueueWithProperties error %d\n", error);
            return 2;
        }
        
        cl_mem buffer_data = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_int) * nonzeros_nr, NULL, &error);
        cl_mem buffer_indices = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_int *) * nonzeros_nr, NULL, &error);
        cl_mem buffer_vect = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_int) * cols_nr, NULL, &error);
        cl_mem buffer_strip_ptr = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_int) * (ceil(rows_nr / height) + 1), NULL, &error);
        cl_mem buffer_row_in_strip = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_int) * nonzeros_nr, NULL, &error);
        cl_mem buffer_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_int) * rows_nr, NULL, &error);
        
        if (error != CL_SUCCESS)
        {
            printf("clCreateBuffer error %d\n", error);
            return 2;
        }
        
        size_t size;
        const char *source = read_source_from_cl_file("kernels/Cmrs.cl", &size);
        
        cl_program program = clCreateProgramWithSource(context, 1, &source, &size, &error);
        
        if (error != CL_SUCCESS)
        {
            printf("clCreateProgramWithSource error %d\n", error);
            return 2;
        }
        
        error = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
        
        if (error != CL_SUCCESS)
        {
            readProgramBuildInfo(program, device_ids[0]);
            return 2;
        }
        
        cl_kernel kernel = clCreateKernel(program, "cmrs", &error);
        
        if (error != CL_SUCCESS)
        {
            printf("clCreateKernel error %d\n", error);
            return 2;
        }
        
        int strip_ptr_size = (ceil(rows_nr / height) + 1);
        
        error  = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&buffer_data);
        error |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&buffer_indices);
        error |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&buffer_strip_ptr);
        error |= clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&buffer_row_in_strip);
        error |= clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&buffer_vect);
        error |= clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&buffer_output);
        error |= clSetKernelArg(kernel, 6, sizeof(int), (void*)&strip_ptr_size);
        error |= clSetKernelArg(kernel, 7, sizeof(int), (void*)&height);
        error |= clSetKernelArg(kernel, 8, localWorkSize[0] * height * sizeof(cl_int), NULL);
        
        if (error != CL_SUCCESS)
        {
            printf("clSetKernelArg errror\n");
            return 2;
        }
        
        error = clEnqueueWriteBuffer(commandQueue, buffer_data, CL_FALSE, 0, sizeof(cl_int) * nonzeros_nr, data_int, 0, NULL, NULL);
        error |= clEnqueueWriteBuffer(commandQueue, buffer_indices, CL_FALSE, 0, sizeof(cl_int) * nonzeros_nr, cols, 0, NULL, NULL);
        error |= clEnqueueWriteBuffer(commandQueue, buffer_strip_ptr, CL_FALSE, 0, sizeof(cl_int) * (ceil(rows_nr / height) + 1), strip_ptr, 0, NULL, NULL);
        error |= clEnqueueWriteBuffer(commandQueue, buffer_row_in_strip, CL_FALSE, 0, sizeof(cl_int) * nonzeros_nr, row_in_strip, 0, NULL, NULL);
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

        clReleaseMemObject(buffer_data);
        clReleaseMemObject(buffer_indices);
        clReleaseMemObject(buffer_vect);
        clReleaseMemObject(buffer_output);

        free(cols);
        free(data_int);
        free(data_double);
        free(strip_ptr);
        free(row_in_strip);
        
        clFlush(commandQueue);
        clReleaseCommandQueue(commandQueue);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseContext(context);
        
        break;
    }
    
    return 0;
}
