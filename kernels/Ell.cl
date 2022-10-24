__kernel void ell(__global const double *data, __global const int *indices, __global const double *vect, __global double *output, const int N, const int row_size, __local double *partial_data)
{
    size_t i;
    
    for (i = get_group_id(0); i < N; i += get_num_groups(0))
    {
        double sum = 0;
        const int index = row_size * i;
        unsigned int local_id = get_local_id(0);
        unsigned int step;
        size_t j;
        
        for (j = local_id; j < row_size; j += get_local_size(0))
        {
            int elem_idx = index + j;

            sum += data[elem_idx] * vect[indices[elem_idx]];
        }
        
        partial_data[local_id] = sum;
        
        barrier(CLK_LOCAL_MEM_FENCE);

        for (step = get_local_size(0) / 2; step > 0; step >>= 1)
        {
            if (local_id < step)
            {
                partial_data[local_id] += partial_data[local_id + step];
            }

            barrier(CLK_LOCAL_MEM_FENCE);
        }
        
        if (local_id == 0)
        {
            output[i] = partial_data[0];
        }
    }
}
