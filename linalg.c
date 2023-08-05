#include <stdint.h>
#include "linalg.h"

void matrix_multiply(float *in1, float *in2, float *out,
                     const int64_t dim1, const int64_t dim2, const int64_t dim3)
{
  for (int64_t i = 0; i < dim1; i++)
  {
    for (int64_t j = 0; j < dim2; j++)
    {
      for (int64_t k = 0; k < dim3; k++)
      {
        out[i * dim3 + k] += in1[i * dim2 + j] * in2[j * dim3 + k];
      }
    }
  }
}

void scalar_vector_mult(const float scalar, float *vector, float *out, const int64_t dim)
{
    for (int64_t i = 0; i < dim; i++)
    {
        *(out + i) = scalar * *(vector + i);
    }
}

void vector_vector_sum(float *vector1, float *vector2, float *out, const int64_t dim)
{
    for (int64_t i = 0; i < dim; i++)
    {
        out[i] = vector1[i] + vector2[i];
    }
}


void vector_vector_mult(float *vector1, float *vector2, float *out, const int64_t dim)
{
    for (int64_t i = 0; i < dim; i++)
    {
        out[i] = vector1[i] * vector2[i];
    }
}

void broadcast_vectors(const float *in1, const float *in2, float *out,
    const int64_t dim1, const int64_t dim2, combinator_ptr combinator)
{
    for (int64_t i = 0; i < dim1; i++)
    {
        for (int64_t j = 0; j < dim2; j++)
        {
            out[i * dim2 + j] = combinator(in1[i], in2[j]);
        }

    }
}

static float multiply(float a, float b)
{
    return a * b;
}

combinator_ptr mult()
{
    return &multiply;
}

