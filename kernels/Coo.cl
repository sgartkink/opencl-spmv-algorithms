__kernel void coo(__global const int *row, __global const int *col, __global const int *data, __global const int *vect, __global int *output, const int N)
{
    size_t gid;
    
    for (gid = get_global_id(0); gid < N; gid += get_global_size(0))
    {
        atomic_add(&output[row[gid]], data[gid] * vect[col[gid]]);
    }
}
