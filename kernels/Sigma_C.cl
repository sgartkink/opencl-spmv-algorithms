__kernel void sigma_c(__global const double* data, __global const int* indices, __global const double* vect, __global double *output,  __global const int *row_indices, const int C)
{
    size_t i = get_group_id(0);

    const int index_offset = row_indices[i];
    const int row_size = row_indices[i + 1];

    size_t local_id = get_local_id(0);
    size_t j;
    double sum = 0;

    for (j = local_id + index_offset; j < row_size; j += C)
    {
        sum += data[j] * vect[indices[j]];
    }

    output[local_id + (i * C)] = sum;
}
