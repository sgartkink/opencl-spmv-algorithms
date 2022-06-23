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
            cl_int *cols;
            cl_double *data_double;
            cl_int *data_int;
            
            if ((f = fopen("databases/cant.mtx-sorted", "r")) == NULL) 
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

            int longest_col = 0;
            int previous_row = 1;
            int current_col_len = 0;
            for (int i = 0; i < nonzeros_nr; i++)
            {
                int current_row;
                int a;
                double b;
                fscanf(f, "%d %d %lg\n", &current_row, &a, &b);
                
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
                    
                    current_col_len = 1;
                }
            }

            fseek(f, 0, SEEK_SET);

            if (f == NULL)
                return 3;
            
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
            cols = malloc(longest_col * rows_nr * sizeof(cl_int));
            data_int = malloc(longest_col * rows_nr * sizeof(cl_int));
            data_double = malloc(longest_col * rows_nr * sizeof(cl_double));
            
            previous_row = 1;
            int current_index = 0;
            int nonzeroes_in_row = 0;
            int m = 0;
            for (int i = 0; i < nonzeros_nr; i++)
            {
                int current_row;
                int current_col;
                fscanf(f, "%d %d %lg\n", &current_row, &current_col, &data_double[current_index]);
                
//                 printf("%d %d %lg\n", current_row, current_col, data_double[current_index]);
                
                current_col--;
                
                if (previous_row == current_row)
                {
                    data_int[current_index] = (int)data_double[current_index];
                    cols[current_index] = current_col;
                    current_index++;
                    nonzeroes_in_row++;
                }
                else 
                {
//                     printf("%d\n", current_row);
                    long i;
                    int diff = current_row - previous_row;
                    double read_value = data_double[current_index];
                    previous_row = current_row;
                    
                    if (diff > m) m = diff;
                    
                    for (i = nonzeroes_in_row; i < (long)longest_col * (long)diff; ++i)
                    {
                        cols[current_index] = -1;
                        current_index++;
                    }
                    
                    nonzeroes_in_row = 1;
                    cols[current_index] = current_col;
                    data_int[current_index] = (int)read_value;
                    current_index++;
                }
            }

//             for (int k = 0; k < rows_nr; ++k)
//             {
//                 printf("%d - ", k);
//                 for (int i = 0; i < longest_col; ++i)
//                 {
//                     printf("%d ", cols[i + longest_col *k]);
//                 }
//                 printf("\n");
//             }
            
            for (int i = nonzeroes_in_row; i < longest_col; ++i)
            {
                cols[current_index] = -1;
                current_index++;
            }
            
            if (f != stdin) 
            {
                fclose(f);
            }
            
            
            size_t vectorSize = rows_nr; // musi byc podzielne przez localWorkSize
            size_t localWorkSize = 1; // zmienia to ile leci na raz do jednego kernela?
    
            cl_int *vect = (cl_int*)malloc(sizeof(cl_int*) * cols_nr);
            for (int i = 0; i < cols_nr; ++i) {
                vect[i] = 1;
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
            
            cl_mem buffer_data = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_int) * longest_col * rows_nr, NULL, &error);
            cl_mem buffer_indices = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_int *) * longest_col * rows_nr, NULL, &error);
            cl_mem buffer_vect = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(cl_int) * cols_nr, NULL, &error);
            cl_mem buffer_output = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(cl_int) * rows_nr, NULL, &error);
            
            if (error != CL_SUCCESS)
            {
                printf("clCreateBuffer error %d\n", error);
                return 2;
            }
            
            size_t size;
            const char *source = read_source_from_cl_file("kernels/Ell.cl", &size);
            
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
            
            cl_kernel kernel = clCreateKernel(program, "ell", &error);
            
            if (error != CL_SUCCESS)
            {
                printf("clCreateKernel error %d\n", error);
                return 2;
            }
            
            error = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&buffer_data);
            error = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&buffer_indices);
            error = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&buffer_vect);
            error = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&buffer_output);
            error = clSetKernelArg(kernel, 4, sizeof(int), (void*)&rows_nr);
            error = clSetKernelArg(kernel, 5, sizeof(int), (void*)&longest_col);
            
            error = clEnqueueWriteBuffer(commandQueue, buffer_data, CL_FALSE, 0, sizeof(cl_int) * longest_col * rows_nr, data_int, 0, NULL, NULL);
            error = clEnqueueWriteBuffer(commandQueue, buffer_indices, CL_FALSE, 0, sizeof(cl_int) * longest_col * rows_nr, cols, 0, NULL, NULL);
            error = clEnqueueWriteBuffer(commandQueue, buffer_vect, CL_FALSE, 0, sizeof(cl_int) * cols_nr, vect, 0, NULL, NULL);
            
            if (error != CL_SUCCESS)
            {
                printf("clEnqueueWriteBuffer error %d\n", error);
                return 2;
            }

            error = clEnqueueNDRangeKernel(commandQueue, kernel, 1, NULL, &vectorSize, &localWorkSize, 0, NULL, NULL);

            if (error != CL_SUCCESS)
            {
                printf("clEnqueueNDRangeKernel error %d\n", error);
                return 2;
            }
            
            clock_t start = clock();
            error = clEnqueueReadBuffer(commandQueue, buffer_output, CL_TRUE, 0, sizeof(cl_int) * rows_nr, output, 0, NULL, NULL);
            
            if (error != CL_SUCCESS)
            {
                printf("clEnqueueReadBuffer error %d\n", error);
                return 2;
            }
            
            clock_t end = clock();
            float ms = (float)(end - start) / (CLOCKS_PER_SEC / 1000);
            printf("Your calculations took %.2lf ms to run.\n", ms);
            
//             for (size_t k = 0; k < rows_nr; ++k)
//             {
//                 printf("%ld: %d\n", k, output[k]);
//             }
            
            clFlush(commandQueue);
            clFinish(commandQueue);
            clReleaseKernel(kernel);
            clReleaseProgram(program);
            
            i = platformNumber;
            break;
        }
        
        free(deviceIds);
    }
    
    free(platformIds);
    
    return 0;
}
