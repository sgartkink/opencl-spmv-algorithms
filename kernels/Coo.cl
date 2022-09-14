#pragma OPENCL EXTENSION cl_khr_fp64: enable
#pragma OPENCL EXTENSION cl_khr_int64_base_atomics: enable

double __attribute__((overloadable)) atomic_add(__global double *valq, double delta)
{
   union {
     double f;
     unsigned long i;
   } old_value;

   union {
     double f;
     unsigned long i;
   } new_value;

  do {
     old_value.f = *valq;
     new_value.f = old_value.f + delta;
   } while (atom_cmpxchg((volatile __global unsigned long *)valq, old_value.i, new_value.i) != old_value.i);

   return old_value.f;
}

__kernel void coo(__global const int *row, __global const int *col, __global const double *data, __global const double *vect, __global double *output, const int N)
{
    size_t i;

    for (i = get_global_id(0); i < N; i += get_global_size(0))
    {
        atomic_add(&output[row[i]], data[i] * vect[col[i]]);
    }
}
