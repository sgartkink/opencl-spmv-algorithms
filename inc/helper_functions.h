#define CL_TARGET_OPENCL_VERSION 220
#include <CL/cl.h>

#include "mmio.h"

char* read_source_from_cl_file(const char *file, size_t *size) 
{
    cl_int status;
    FILE *fp;
    char *source;

    fp = fopen(file, "rb");
    if (!fp) {
        printf("Could not open kernel file\n");
        exit(-1);
    }
    
    status = fseek(fp, 0, SEEK_END);
    if (status != 0) {
        printf("Error seeking to end of file\n");
        exit(-1);
    }
    
    *size = ftell(fp);
    if (*size < 0) {
        printf("Error getting file position\n");
        exit(-1);
    }

    rewind(fp);

    source = (char *)malloc(*size + 1);

    int i;
    for (i = 0; i < *size+1; i++) {
        source[i] = '\0';
    }

    if (source == NULL) {
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

    // Allocate memory for the log
    char *log = (char *)malloc(log_size);

    // Get the log
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, log_size, log, NULL);

    // Print the log
    printf("%s\n", log);
}
