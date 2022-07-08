__kernel void csr(__global const int *ptr, __global const int *col, __global const int *data, __global const int *vect, __global int *output, const int N)
{
    int gid = get_global_id(0);
    int i;
    
    for (i = gid; i < N; i += get_num_groups(0)) 
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
