#include <stdio.h>
#include <immintrin.h>

static inline __m256d __attribute__((always_inline))
integracao_simd(__m256d vec_i, double width) {

    __m256d vec_width = _mm256_set1_pd(width);
    __m256d vec_half  = _mm256_set1_pd(0.5);
    __m256d vec_one   = _mm256_set1_pd(1.0);
    __m256d vec_four  = _mm256_set1_pd(4.0);

    __m256d x = _mm256_mul_pd(
        _mm256_add_pd(vec_i, vec_half),
        vec_width
    );

    __m256d x2  = _mm256_mul_pd(x, x);
    __m256d den = _mm256_add_pd(vec_one, x2);
    return _mm256_div_pd(vec_four, den);
}

int main() {
    int n = 1000000;
    double width = 1.0 / n;
    double sum   = 0.0;

    __m256d vec_sum = _mm256_setzero_pd();

    int i = 0;
    for (; i <= n - 4; i += 4) {

        __m256d vec_i = _mm256_set_pd(
            (double)(i + 3),
            (double)(i + 2),
            (double)(i + 1),
            (double)(i    )
        );

        __m256d fx = integracao_simd(vec_i, width);

        vec_sum = _mm256_add_pd(vec_sum, fx);
    }

    double lanes[4];
    _mm256_storeu_pd(lanes, vec_sum);
    sum = lanes[0] + lanes[1] + lanes[2] + lanes[3];

    for (; i < n; i++) {
        double x = (i + 0.5) * width;
        sum += 4.0 / (1.0 + x * x);
    }

    double pi = sum * width;
    printf("Estimated Pi = %.15f\n", pi);

    return 0;
}