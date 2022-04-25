__kernel void ell(__global const int* data, __global const int* indices, __global const int* vect, __global int *output, const int N, const int rowSize)
{
    int gid = get_global_id(0);

    if (gid < N)
    {
        int i;
        int sum = 0;
        int index = rowSize * gid;
//         printf("%d %d\t %d %d %d %d\t%d %d %d %d\n", gid, index, indices[index], indices[index+1], indices[index+2], indices[index+3], data[index], data[index+1], data[index+2], data[index+3]);
        for (i = 0; i < rowSize; ++i)
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
    }
}
 
