__kernel void csr(__global const int* ptr, __global const int* col, __global const int* data, __global const int* vect, __global int *output, const int N)
{
    int gid = get_global_id(0);
    
    if (gid < N)
    {
        int i;
        int sum = 0;
        
        for (i = ptr[gid]; i < ptr[gid+1]; ++i)
        {
            sum += data[i] * vect[col[i]];
        }
        
        output[gid] = sum;
    }
}

