#define CL_TARGET_OPENCL_VERSION 220
#include <CL/cl.h>
#include <stdio.h>                                                                                                                                               
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>

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
        int i;
        FILE *file;
        cl_int *rows;
        cl_int *cols;
        cl_int *data;
        cl_int *vect;
        cl_int *output;
        const char *filename = "databases/cant.mtx";
        struct timespec start_time;
        struct timespec end_time;
        
        size_t global_work_size[1];
        size_t local_work_size[1] = { 1 };
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
        
        global_work_size[0] = number_of_nonzeroes;

        rows = (cl_int *)malloc(number_of_nonzeroes * sizeof(cl_int));
        cols = (cl_int *)malloc(number_of_nonzeroes * sizeof(cl_int));
        data = (cl_int *)malloc(number_of_nonzeroes * sizeof(cl_int));

        for (i = 0; i < number_of_nonzeroes; i++)
        {
            double value;
            
            fscanf(file, "%d %d %lg\n", &rows[i], &cols[i], &value);
            rows[i]--;  // adjust from 1-based to 0-based
            cols[i]--;
            data[i] = (int)value;
        }
    
        fclose(file);

        vect = (cl_int*)malloc(sizeof(cl_int*) * number_of_columns);
        for (i = 0; i < number_of_columns; ++i) 
        {
            vect[i] = 2;
        }
        
        output = (cl_int*)malloc(sizeof(cl_int) * number_of_rows);
        
        
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
        
        cl_mem buffer_row    = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_int) * number_of_nonzeroes, NULL, &error);
        cl_mem buffer_col    = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_int) * number_of_nonzeroes, NULL, &error);
        cl_mem buffer_data   = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_int) * number_of_nonzeroes, NULL, &error);
        cl_mem buffer_vect   = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_int) * number_of_columns, NULL, &error);
        cl_mem buffer_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_int) * number_of_rows, NULL, &error);
        
        if (error != CL_SUCCESS)
        {
            printf("clCreateBuffer error %d\n", error);
            return OpenCLProgramError;
        }
        
        size_t size_of_cl_file;
        char *source = read_source_from_cl_file("kernels/Coo.cl", &size_of_cl_file);
        
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
        
        cl_kernel kernel = clCreateKernel(program, "coo", &error);
        
        if (error != CL_SUCCESS)
        {
            printf("clCreateKernel error %d\n", error);
            return OpenCLProgramError;
        }
        
        
        /* set data to kernel */
        
        error  = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&buffer_row);
        error |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&buffer_col);
        error |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&buffer_data);
        error |= clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&buffer_vect);
        error |= clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&buffer_output);
        error |= clSetKernelArg(kernel, 5, sizeof(int), (void*)&number_of_nonzeroes);
        
        if (error != CL_SUCCESS)
        {
            printf("clSetKernelArg errror\n");
            return OpenCLProgramError;
        }
        
        error  = clEnqueueWriteBuffer(command_queue, buffer_row, CL_FALSE, 0, sizeof(cl_int) * number_of_nonzeroes, rows, 0, NULL, NULL);
        error |= clEnqueueWriteBuffer(command_queue, buffer_col, CL_FALSE, 0, sizeof(cl_int) * number_of_nonzeroes, cols, 0, NULL, NULL);
        error |= clEnqueueWriteBuffer(command_queue, buffer_data, CL_FALSE, 0, sizeof(cl_int) * number_of_nonzeroes, data, 0, NULL, NULL);
        error |= clEnqueueWriteBuffer(command_queue, buffer_vect, CL_FALSE, 0, sizeof(cl_int) * number_of_columns, vect, 0, NULL, NULL);
        
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
        printf("Your calculations took %.2lf ms to run.\n", ms);
        printf("Number of operations %d, PERFORMANCE %lf GFlops\n", 
               2 * number_of_nonzeroes, 
               (2 * number_of_nonzeroes) / ms * 1e-6);
        
        if (error != CL_SUCCESS)
        {
            printf("clEnqueueNDRangeKernel error %d\n", error);
            return OpenCLProgramError;
        }
        
        
        /* read output */
        
        error = clEnqueueReadBuffer(command_queue, buffer_output, CL_TRUE, 0, sizeof(cl_int) * number_of_rows, output, 0, NULL, NULL);
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

        clReleaseMemObject(buffer_row);
        clReleaseMemObject(buffer_col);
        clReleaseMemObject(buffer_data);
        clReleaseMemObject(buffer_vect);
        
        free(rows);
        free(cols);
        free(data);
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
