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
        int number_of_slices;
        int row_indices_size;
        int number_of_groups;
        int i;
        FILE *file;
        cl_int *cols;
        cl_double *data;
        cl_int *row_indices;
        cl_double *vect;
        cl_double *output;
        const char *filename = "databases/cant-sorted.mtx";
        struct timespec start_time;
        struct timespec end_time;

        const int max_rows_to_check = 32;

        size_t global_work_size[1];
        size_t local_work_size[1] = { max_rows_to_check };
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

        number_of_groups = (int)ceil((float)number_of_rows / (float)max_rows_to_check);
        global_work_size[0] = number_of_groups * max_rows_to_check;

        if (number_of_rows % max_rows_to_check == 0)
        {
            number_of_slices = number_of_rows / max_rows_to_check;
        }
        else
        {
            number_of_slices = (number_of_rows / max_rows_to_check) + 1;
        }

        row_indices_size = number_of_slices + 1;
        row_indices    = (cl_int *)malloc(row_indices_size * sizeof(cl_int));
        row_indices[0] = 0;

        int longest_col = 0;
        int previous_row = 0;
        int current_col_len = 0;
        int rows_checked = 0;
        int row_indices_index = 0;
        long elements_sum = 0;
        for (i = 0; i < number_of_nonzeroes; i++)
        {
            int current_row;
            int current_col;
            double value;

            fscanf(file, "%d %d %lg\n", &current_row, &current_col, &value);

            current_row--;

            if (current_row == previous_row)
            {
                current_col_len++;
            }
            else
            {
                rows_checked++;

                if (current_col_len > longest_col)
                {
                    longest_col = current_col_len;
                }

                if (rows_checked == max_rows_to_check)
                {
                    elements_sum += longest_col * max_rows_to_check;
                    row_indices[row_indices_index + 1] = elements_sum;
                    longest_col = 1;
                    row_indices_index++;
                    rows_checked = 0;
                }

                current_col_len = 1;
                previous_row = current_row;
            }
        }

        if (rows_checked != max_rows_to_check)
        {
            if (current_col_len > longest_col)
            {
                longest_col = current_col_len;
            }

            elements_sum += longest_col * max_rows_to_check;
            row_indices[number_of_slices] = elements_sum;
        }

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

        cols = (cl_int *)calloc(elements_sum, sizeof(cl_int));
        data = (cl_double *)calloc(elements_sum, sizeof(cl_double));

        previous_row = 0;
        rows_checked = 0;
        row_indices_index = 0;
        int start_index = row_indices[row_indices_index];
        int current_index = start_index;

        for (i = 0; i < number_of_nonzeroes; i++)
        {
            int current_row;
            int current_col;
            double value;

            fscanf(file, "%d %d %lg\n", &current_row, &current_col, &value);

            current_col--;
            current_row--;

            if (previous_row == current_row)
            {
                data[current_index] = value;
                cols[current_index] = current_col;
                current_index += max_rows_to_check;
            }
            else
            {
                rows_checked++;

                if (rows_checked == max_rows_to_check)
                {
                    rows_checked = 0;
                    row_indices_index++;
                    start_index = row_indices[row_indices_index];
                }
                else
                {
                    start_index++;
                }

                current_index = start_index;

                previous_row = current_row;

                cols[current_index] = current_col;
                data[current_index] = value;
                current_index += max_rows_to_check;
            }
        }

        fclose(file);

        vect = (cl_double*)malloc(sizeof(cl_double) * number_of_columns);
        for (i = 0; i < number_of_columns; ++i)
        {
            vect[i] = i;
        }

        output = (cl_double*)malloc(sizeof(cl_double) * (number_of_groups * max_rows_to_check));


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

        cl_mem buffer_data       = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_double) * elements_sum, NULL, &error);
        cl_mem buffer_indices    = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_int) * elements_sum, NULL, &error);
        cl_mem buffer_vect       = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_double) * number_of_columns, NULL, &error);
        cl_mem buffer_row_indices = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_int) * row_indices_size, NULL, &error);
        cl_mem buffer_output     = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_double) * (number_of_groups * max_rows_to_check), NULL, &error);

        if (error != CL_SUCCESS)
        {
            printf("clCreateBuffer error %d\n", error);
            return OpenCLProgramError;
        }

        size_t size_of_cl_file;
        char *source = read_source_from_cl_file("kernels/Sigma_C.cl", &size_of_cl_file);

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

        cl_kernel kernel = clCreateKernel(program, "sigma_c", &error);

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
        error |= clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&buffer_row_indices);
        error |= clSetKernelArg(kernel, 5, sizeof(int),    (void*)&max_rows_to_check);

        if (error != CL_SUCCESS)
        {
            printf("clSetKernelArg errror\n");
            return OpenCLProgramError;
        }

        error  = clEnqueueWriteBuffer(command_queue, buffer_data, CL_FALSE, 0, sizeof(cl_double) * elements_sum, data, 0, NULL, NULL);
        error |= clEnqueueWriteBuffer(command_queue, buffer_indices, CL_FALSE, 0, sizeof(cl_int) * elements_sum, cols, 0, NULL, NULL);
        error |= clEnqueueWriteBuffer(command_queue, buffer_vect, CL_FALSE, 0, sizeof(cl_double) * number_of_columns, vect, 0, NULL, NULL);
        error |= clEnqueueWriteBuffer(command_queue, buffer_row_indices, CL_FALSE, 0, sizeof(cl_int) * row_indices_size, row_indices, 0, NULL, NULL);

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

        error = clEnqueueReadBuffer(command_queue, buffer_output, CL_TRUE, 0, sizeof(cl_double) * (number_of_groups * max_rows_to_check), output, 0, NULL, NULL);
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
        clReleaseMemObject(buffer_row_indices);

        free(cols);
        free(data);
        free(row_indices);
        free(vect);
        free(output);
        free(source);

        clFlush(command_queue);
        clReleaseCommandQueue(command_queue);
        clFinish(command_queue);
        clReleaseKernel(kernel);
        clReleaseProgram(program);
        clReleaseContext(context);

        break;
    }

    return Success;
}
