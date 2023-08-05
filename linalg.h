void matrix_multiply(float *in1, float *in2, float *out,
                     int64_t dim1, int64_t dim2, int64_t dim3);

typedef float (*combinator_ptr)(float, float);

void broadcast_vectors(const float *in1, const float *in2, float *out,
                       int64_t dim1, int64_t dim2, combinator_ptr combinator);

void vector_vector_mult(float *vector1, float *vector2, float *out, int64_t dim);

void vector_vector_sum(float *vector1, float *vector2, float *out, int64_t dim);


void scalar_vector_mult(float scalar, float *vector, float *out, int64_t dim);

combinator_ptr mult();

