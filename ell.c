#define CL_TARGET_OPENCL_VERSION 300
#include <CL/cl.h>
#include <stdio.h>                                                                                                                                               
#include <stdlib.h>
#include <string.h>
#include <stdbool.h>
#include <time.h>
#include <limits.h>

#include "helper_functions.h"
#include "enums.h"

#define DEVICES_DEFAULT_SIZE 8

void compute_using_cpu(cl_double *data, cl_double *vect, cl_int *cols, int number_of_rows, int longest_col, int number_of_nonzeroes, cl_double **result);

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
        cl_int *cols;
        cl_double *data;
        cl_double *vect;
        cl_double *output;
        cl_double *output_cpu;
        const char *filename = "databases/cant-sorted.mtx";
        struct timespec start_time;
        struct timespec end_time;
        
        size_t global_work_size[1] = { 4096 };
        size_t local_work_size[1] = { 16 };
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

        int longest_col = 0;
        int shortest_col = INT_MAX;
        int previous_row = 1;
        int current_col_len = 0;
        int sum_of_col_len = 0;
        for (i = 0; i < number_of_nonzeroes; ++i)
        {
            int current_row;
            int current_col;
            double value;
            
            fscanf(file, "%d %d %lg\n", &current_row, &current_col, &value);
            
            if (current_row == previous_row)
            {
                current_col_len++;
            }
            else 
            {
                previous_row = current_row;
                
                if (current_col_len > longest_col)
                {
                    longest_col = current_col_len;
                }
                if (current_col_len < shortest_col)
                {
                    shortest_col = current_col_len;
                }
                sum_of_col_len += current_col_len;
                
                current_col_len = 1;
            }
        }

        double average_col_len = (double)sum_of_col_len / (double)number_of_rows;
        printf("average column length %lf, shortest col %d, longest col %d\n", average_col_len, shortest_col, longest_col);

        if (fseek(file, 0, SEEK_SET) != 0) 
        {
            perror(filename);
            return FileError;
        }
        
        if (read_size_of_matrices_from_file(file, &number_of_rows, &number_of_columns, &number_of_nonzeroes) == false)
        {
            fclose(file);
            return FileError;
        }
        
        cols = (cl_int *)malloc(longest_col * number_of_rows * sizeof(cl_int));
        data = (cl_double *)malloc(longest_col * number_of_rows * sizeof(cl_double));
        
        previous_row = 1;
        int current_index = 0;
        int nonzeroes_in_row = 0;
        for (i = 0; i < number_of_nonzeroes; ++i)
        {
            int current_row;
            int current_col;
            double value;
            
            fscanf(file, "%d %d %lg\n", &current_row, &current_col, &value);
            
            current_col--;
            
            if (previous_row == current_row)
            {
                data[current_index] = value;
                cols[current_index] = current_col;
                current_index++;
                nonzeroes_in_row++;
            }
            else 
            {
                long k;
                int diff = current_row - previous_row;
                previous_row = current_row;
                
                for (k = nonzeroes_in_row; k < (long)longest_col * (long)diff; ++k)
                {
                    cols[current_index] = 0;
                    current_index++;
                }
                
                nonzeroes_in_row = 1;
                cols[current_index] = current_col;
                data[current_index] = value;
                current_index++;
            }
        }
        
        for (i = nonzeroes_in_row; i < longest_col; ++i)
        {
            cols[current_index] = 0;
            current_index++;
        }
        
        fclose(file);

        vect = (cl_double*)malloc(sizeof(cl_double) * number_of_columns);
        for (i = 0; i < number_of_columns; ++i) 
        {
            vect[i] = i;
        }
        
        output = (cl_double*)malloc(sizeof(cl_double) * number_of_rows);
        output_cpu = (cl_double*)malloc(sizeof(cl_double) * number_of_rows);
        
        
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
        
        cl_mem buffer_data    = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_double) * longest_col * number_of_rows, NULL, &error);
        cl_mem buffer_indices = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_int) * longest_col * number_of_rows, NULL, &error);
        cl_mem buffer_vect    = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_double) * number_of_columns, NULL, &error);
        cl_mem buffer_output  = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_double) * number_of_rows, NULL, &error);
        
        if (error != CL_SUCCESS)
        {
            printf("clCreateBuffer error %d\n", error);
            return OpenCLProgramError;
        }
        
        size_t size_of_cl_file;
        char *source = read_source_from_cl_file("kernels/Ell.cl", &size_of_cl_file);
        
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
        
        cl_kernel kernel = clCreateKernel(program, "ell", &error);
        
        if (error != CL_SUCCESS)
        {
            printf("clCreateKernel error %d\n", error);
            return OpenCLProgramError;
        }
        
        
        /* set data to kernel */
        
        error  = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&buffer_data);
        error |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&buffer_indices);
        error |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&buffer_vect);
        error |= clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&buffer_output);
        error |= clSetKernelArg(kernel, 4, sizeof(int), (void*)&number_of_rows);
        error |= clSetKernelArg(kernel, 5, sizeof(int), (void*)&longest_col);
        error |= clSetKernelArg(kernel, 6, local_work_size[0] * sizeof(cl_double), NULL);
        
        if (error != CL_SUCCESS)
        {
            printf("clSetKernelArg errror\n");
            return OpenCLProgramError;
        }
        
        error  = clEnqueueWriteBuffer(command_queue, buffer_data, CL_FALSE, 0, sizeof(cl_double) * longest_col * number_of_rows, data, 0, NULL, NULL);
        error |= clEnqueueWriteBuffer(command_queue, buffer_indices, CL_FALSE, 0, sizeof(cl_int) * longest_col * number_of_rows, cols, 0, NULL, NULL);
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


        /* CPU */

        compute_using_cpu(data, vect, cols, number_of_rows, longest_col, number_of_nonzeroes, &output_cpu);


        if (check_result(filename, vect, output_cpu) == true)
        {
            printf("cpu result is ok\n");
        }
        else
        {
            printf("cpu result is wrong\n");
        }


        /* release memory */

        clReleaseMemObject(buffer_data);
        clReleaseMemObject(buffer_indices);
        clReleaseMemObject(buffer_vect);
        clReleaseMemObject(buffer_output);

        free(cols);
        free(data);
        free(vect);
        free(output);
        free(output_cpu);
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

void compute_using_cpu(cl_double *data, cl_double *vect, cl_int *cols, int number_of_rows, int longest_col, int number_of_nonzeroes, cl_double **result)
{
    int i;
    struct timespec start_time;
    struct timespec end_time;

    clock_gettime(CLOCK_MONOTONIC, &start_time);

    #pragma omp parallel for shared(data, vect, cols, number_of_rows, longest_col, result) private(i)
    for (i = 0; i < number_of_rows; ++i)
    {
        int offset = i * longest_col;
        int k;

        for (k = 0; k < longest_col; ++k)
        {
            int element_index = offset + k;
            (*result)[i] += data[element_index] * vect[cols[element_index]];
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end_time);
    double ms = (double)(end_time.tv_nsec - start_time.tv_nsec) / 1000000 + (double)(end_time.tv_sec - start_time.tv_sec) * 1000;

    printf("\nCPU calculations\n");
    calculate_and_print_performance(ms, number_of_nonzeroes);
}
