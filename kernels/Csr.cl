__kernel void csr(__global const int *ptr, __global const int *col, __global const double *data, __global const double *vect, __global double *output, const int N)
{
    size_t i;
    
    for (i = get_global_id(0); i < N; i += get_global_size(0))
    {
        double sum = 0;
        int j;
        
        for (j = ptr[i]; j < ptr[i+1]; ++j)
        {
            sum += data[j] * vect[col[j]];
        }
        
        output[i] = sum;
    }
}
