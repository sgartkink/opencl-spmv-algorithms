__kernel void sigma_c(__global const int* data, __global const int* indices, __global const int* vect, __global int *output,  __global const int *rowSizes, const int N)
{
    int gid = get_global_id(0);

    if (gid < N)
    {
        int i;
        int sum = 0;
        int index = 0;
        
        for (i = 0; i < gid; ++i)
        {
            index += rowSizes[i / 5];
        }
        
        for (i = 0; i < rowSizes[gid / 5]; ++i)
        {
            int elem_idx = index + i;
            
            if (indices[elem_idx] != -1)
            {
                sum += data[elem_idx] * vect[indices[elem_idx]];
            }
            else
            {
                break;
            }
                
        }

        output[gid] = sum;
//         printf("%d %d %d\n", output[gid], rowSizes[gid / 5], gid);
    }
}
 
 
