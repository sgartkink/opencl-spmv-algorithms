__kernel void cmrs(__global const int *data, __global const int *indices, __global const int *strip_ptr, __global const int *row_in_strip, __global const int *vect, __global int *output, const int N, const int height, __local int *partial_data)
{
    size_t i;
    
    for (i = get_group_id(0); i < N; i += (get_num_groups(0) * height))
    {
        int sum = 0;
        int strip_start = strip_ptr[i];
        int strip_end = strip_ptr[i + 1];
        size_t j;
        
        for (j = get_local_id(0); j < strip_end - strip_start; j += get_local_size(0))
        {
            const int current_index = strip_start + j;
            const int strip_row = row_in_strip[current_index];
            int row_index = (i * height) + strip_row;
            
            partial_data[get_local_id(0) * height + strip_row] += data[current_index] * vect[row_index];
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        for (j = get_local_id(0); j < height; ++j)
        {
            if (get_local_id(0) == j)
            {
                int partial_data_sum = 0;
                for (; j < get_local_size(0) * height; j += height)
                {
                    partial_data_sum += partial_data[j];
                }
                partial_data[get_local_id(0)] = partial_data_sum;
            }
        }
        
        barrier(CLK_LOCAL_MEM_FENCE);
        
        if (get_local_id(0) < height)
        {
            output[(i * height) + get_local_id(0)] = partial_data[get_local_id(0)];
        }
    }
}
