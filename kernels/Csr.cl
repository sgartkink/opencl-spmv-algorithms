__kernel void csr(__global const int *ptr, __global const int *col, __global const int *data, __global const int *vect, __global int *output, const int N)
{
    size_t i;
    
    for (i = get_global_id(0); i < N; i += get_num_groups(0)) 
    {
        int sum = 0;
        int j;
        
        for (j = ptr[i]; j < ptr[i+1]; ++j)
        {
            sum += data[j] * vect[col[j]];
        }
        
        output[i] = sum;
    }
}
