#include <stdio.h>
#include <omp.h>

int main() {
    long n = 100000;
    double h = 1.0 / n;
    double pi = 0.0;

    #pragma omp parallel for reduction(+:pi)
    for (long i = 0; i < n; i++) {
        double x = (i + 0.5) * h;
        pi += 4.0 / (1.0 + x * x);
    }

    pi *= h;
    printf("π ≈ %.15f\n", pi);
    return 0;
}