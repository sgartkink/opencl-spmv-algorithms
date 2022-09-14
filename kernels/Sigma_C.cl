__kernel void sigma_c(__global const double* data, __global const int* indices, __global const double* vect, __global double *output,  __global const int *row_sizes, const int N, const int C)
{
    size_t i;
    
    for (i = get_group_id(0); i < N; i += get_num_groups(0))
    {
        __local int row_size;
        row_size = (row_sizes[i + 1] - row_sizes[i]) / C;
        size_t j;
        double sum = 0;
        
        for (j = get_local_id(0); j < C && ((i * C) + j) < N; j += get_local_size(0))
        {
            int k;
            int current_row_offset = j * row_size;
            
            for (k = 0; k < row_size; ++k)
            {
                int elem_idx = row_sizes[i] + k + current_row_offset;
                
                if (indices[elem_idx] != -1)
                {
                    sum += data[elem_idx] * vect[indices[elem_idx]];
                }
                else
                {
                    break;
                }
            }
            
            output[((i * C) + j)] = sum;
            sum = 0;
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
