#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdio.h>                                                                                                                                               
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include <math.h>

#include "helper_functions.h"
#include "enums.h"

#define DEVICES_DEFAULT_SIZE 8

int main(int argc, char *argv[])
{
    (void)argc;
    (void)argv;

    cl_int error;
    cl_uint number_of_devices = DEVICES_DEFAULT_SIZE;
    cl_device_id device_ids[DEVICES_DEFAULT_SIZE];
    
    if (get_device_ids(&device_ids[0], &number_of_devices) != CL_SUCCESS)
    {
        return OpenCLDeviceError;
    }
        
    for (cl_uint device_number = 0; device_number < number_of_devices; ++device_number)
    {
        int number_of_rows; 
        int number_of_columns;
        int number_of_nonzeroes;
        int strip_ptr_size;
        FILE *file;
        int i;
        cl_int *cols;
        cl_int *strip_ptr;
        cl_int *row_in_strip;
        cl_double *data;
        cl_double *vect;
        cl_double *output;
        int height = 8;
        const char *filename = "databases/cant-sorted.mtx";
        struct timespec start_time;
        struct timespec end_time;
        
        size_t global_work_size[1] = { 8192 };
        size_t local_work_size[1] = { height * 4 };
        cl_uint work_dim = 1;
        
        
        /* prepare data for calculations */
        
        file = fopen(filename, "r");
        
        if (file == NULL) 
        {
            perror(filename);
            return FileError;
        }
        
        if (read_size_of_matrices_from_file(file, &number_of_rows, &number_of_columns, &number_of_nonzeroes) == false)
        {
            fclose(file);
            return FileError;
        }
        
        strip_ptr_size = ((int)ceil((double)number_of_rows / (double)height) + 1);

        cols         = (cl_int *)malloc(number_of_nonzeroes * sizeof(cl_int));
        data         = (cl_double *)malloc(number_of_nonzeroes * sizeof(cl_double));
        strip_ptr    = (cl_int *)malloc(strip_ptr_size * sizeof(cl_int));
        row_in_strip = (cl_int *)malloc(number_of_nonzeroes * sizeof(cl_int));
        
        strip_ptr[0] = 0;
        
        int previous_row = 1;
        int current_strip_row = 0;
        int strip_index = 1;
        for (i = 0; i < number_of_nonzeroes; i++)
        {
            int current_row;
            int current_col;
            
            fscanf(file, "%d %d %lg\n", &current_row, &current_col, &data[i]);
            
            current_col--;
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

        fclose(file);
        
        strip_ptr[strip_ptr_size - 1] = number_of_nonzeroes;
        
        vect = (cl_double*)malloc(sizeof(cl_double) * number_of_columns);
        for (i = 0; i < number_of_columns; ++i) 
        {
            vect[i] = 2.0;
        }
        
        output = (cl_double*)malloc(sizeof(cl_double) * number_of_rows);
        
        
        /* prepare OpenCL program */
        
        cl_context context = clCreateContext(0, number_of_devices, device_ids, NULL, NULL, NULL);
        
        if (NULL == context)
        {
            printf("context is null\n");
            return OpenCLProgramError;
        }
        
        cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, device_ids[0], 0, &error);
        
        if (error != CL_SUCCESS)
        {
            printf("clCreateCommandQueueWithProperties error %d\n", error);
            return OpenCLProgramError;
        }
        
        cl_mem buffer_data         = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_double) * number_of_nonzeroes, NULL, &error);
        cl_mem buffer_indices      = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_int) * number_of_nonzeroes, NULL, &error);
        cl_mem buffer_vect         = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_double) * number_of_columns, NULL, &error);
        cl_mem buffer_strip_ptr    = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_int) * strip_ptr_size, NULL, &error);
        cl_mem buffer_row_in_strip = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_int) * number_of_nonzeroes, NULL, &error);
        cl_mem buffer_output       = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_double) * number_of_rows, NULL, &error);
        
        if (error != CL_SUCCESS)
        {
            printf("clCreateBuffer error %d\n", error);
            return OpenCLProgramError;
        }
        
        size_t size_of_cl_file;
        char *source = read_source_from_cl_file("kernels/Cmrs.cl", &size_of_cl_file);
        
        if (source == NULL)
        {
            return FileError;
        }
        
        cl_program program = clCreateProgramWithSource(context, 1, (const char **)&source, &size_of_cl_file, &error);
        
        if (error != CL_SUCCESS)
        {
            printf("clCreateProgramWithSource error %d\n", error);
            return OpenCLProgramError;
        }
        
        error = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
        
        if (error != CL_SUCCESS)
        {
            read_build_program_info(program, device_ids[0]);
            return OpenCLProgramError;
        }
        
        cl_kernel kernel = clCreateKernel(program, "cmrs", &error);
        
        if (error != CL_SUCCESS)
        {
            printf("clCreateKernel error %d\n", error);
            return OpenCLProgramError;
        }
        
        
        /* set data to kernel */
        
        const int N = strip_ptr_size - 1;
        
        error  = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&buffer_data);
        error |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&buffer_indices);
        error |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&buffer_strip_ptr);
        error |= clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&buffer_row_in_strip);
        error |= clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&buffer_vect);
        error |= clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&buffer_output);
        error |= clSetKernelArg(kernel, 6, sizeof(int), (void*)&N);
        error |= clSetKernelArg(kernel, 7, sizeof(int), (void*)&height);
        error |= clSetKernelArg(kernel, 8, local_work_size[0] * height * sizeof(cl_double), NULL);
        
        if (error != CL_SUCCESS)
        {
            printf("clSetKernelArg errror\n");
            return OpenCLProgramError;
        }
        
        error  = clEnqueueWriteBuffer(command_queue, buffer_data, CL_FALSE, 0, sizeof(cl_double) * number_of_nonzeroes, data, 0, NULL, NULL);
        error |= clEnqueueWriteBuffer(command_queue, buffer_indices, CL_FALSE, 0, sizeof(cl_int) * number_of_nonzeroes, cols, 0, NULL, NULL);
        error |= clEnqueueWriteBuffer(command_queue, buffer_strip_ptr, CL_FALSE, 0, sizeof(cl_int) * strip_ptr_size, strip_ptr, 0, NULL, NULL);
        error |= clEnqueueWriteBuffer(command_queue, buffer_row_in_strip, CL_FALSE, 0, sizeof(cl_int) * number_of_nonzeroes, row_in_strip, 0, NULL, NULL);
        error |= clEnqueueWriteBuffer(command_queue, buffer_vect, CL_FALSE, 0, sizeof(cl_double) * number_of_columns, vect, 0, NULL, NULL);
        
        if (error != CL_SUCCESS)
        {
            printf("clEnqueueWriteBuffer error %d\n", error);
            return OpenCLProgramError;
        }
        clFinish(command_queue);
        
        
        /* run program */
        
        cl_event nd_range_kernel_event;

        clock_gettime(CLOCK_MONOTONIC, &start_time);
        error = clEnqueueNDRangeKernel(command_queue, kernel, work_dim, NULL, global_work_size, local_work_size, 0, NULL, &nd_range_kernel_event);
        clWaitForEvents(1, &nd_range_kernel_event);
        clFinish(command_queue);
        
        clock_gettime(CLOCK_MONOTONIC, &end_time);
        double ms = (double)(end_time.tv_nsec - start_time.tv_nsec) / 1000000 + (double)(end_time.tv_sec - start_time.tv_sec) * 1000;

        calculate_and_print_performance(ms, number_of_nonzeroes);
        calculate_and_print_speed(ms, number_of_nonzeroes);
        
        if (error != CL_SUCCESS)
        {
            printf("clEnqueueNDRangeKernel error %d\n", error);
            return OpenCLProgramError;
        }
        
        
        /* read output */
        
        error = clEnqueueReadBuffer(command_queue, buffer_output, CL_TRUE, 0, sizeof(cl_double) * number_of_rows, output, 0, NULL, NULL);
        clFinish(command_queue);
        
        if (error != CL_SUCCESS)
        {
            printf("clEnqueueReadBuffer error %d\n", error);
            return OpenCLProgramError;
        }
        
        if (check_result(filename, vect, output) == true)
        {
            printf("result is ok\n");
        }
        else
        {
            printf("result is wrong\n");
        }
        
//         for (i = 0; i < number_of_rows; ++i)
//         {
//             printf("%d: %d\n", i, output[i]);
//         }


        /* release memory */

        clReleaseMemObject(buffer_data);
        clReleaseMemObject(buffer_indices);
        clReleaseMemObject(buffer_vect);
        clReleaseMemObject(buffer_strip_ptr);
        clReleaseMemObject(buffer_row_in_strip);
        clReleaseMemObject(buffer_output);

        free(cols);
        free(data);
        free(strip_ptr);
        free(row_in_strip);
        free(vect);
        free(output);
        free(source);
        
        clFlush(command_queue);
        clReleaseCommandQueue(command_queue);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseContext(context);
        
        break;
    }
    
    return Success;
}
