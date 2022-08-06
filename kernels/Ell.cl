__kernel void ell(__global const int *data, __global const int *indices, __global const int *vect, __global int *output, const int N, const int row_size, __local int *partial_data)
{
    size_t i;
    
    for (i = get_group_id(0); i < N; i += get_num_groups(0))
    {
        int sum = 0;
        const int index = row_size * i;
        size_t j;
        
        for (j = get_local_id(0); j < row_size; j += get_local_size(0))
        {
            int elem_idx = index + j;
            
            if (indices[elem_idx] != -1)
            {
                sum += data[elem_idx] * vect[indices[elem_idx]];
            }
            else
            {
                break;
            }
        }
        
        partial_data[get_local_id(0)] = sum;
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        if (get_local_id(0) == 0)
        {
            int partial_data_sum = 0;
            for (int k = get_local_id(0); k < get_local_size(0); ++k)
            {
                partial_data_sum += partial_data[k];
            }
            partial_data[0] = partial_data_sum;
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        if (get_local_id(0) == 0)
        {
            output[i] = partial_data[0];
        }
    }
}
