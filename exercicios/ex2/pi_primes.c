#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <omp.h>
#include <string.h>

double calcular_pi(long long n) {
    double sum = 0.0;
    double dx  = 1.0 / (double)n;

    #pragma omp parallel for reduction(+:sum) schedule(static)
    for (long long i = 0; i < n; i++) {
        double x = (i + 0.5) * dx;
        sum += 4.0 / (1.0 + x * x);
    }

    return sum * dx;
}

#define LIMITE 1000

void crivo_eratostenes(char *eh_primo) {
    memset(eh_primo, 1, LIMITE + 1);
    eh_primo[0] = eh_primo[1] = 0;

    int raiz = (int)sqrt((double)LIMITE);

    for (int p = 2; p <= raiz; p++) {
        if (!eh_primo[p]) continue;

        #pragma omp parallel for schedule(dynamic) firstprivate(p)
        for (int j = p * p; j <= LIMITE; j += p) {
            eh_primo[j] = 0;
        }
    }
}

int main(void) {
    printf("Pi por integração numérica: \n\n");

    long long intervalos[] = {1000LL, 100000LL, 10000000LL, 1000000000LL};
    int n_casos = sizeof(intervalos) / sizeof(intervalos[0]);

    for (int k = 0; k < n_casos; k++) {
        long long n = intervalos[k];

        double t0  = omp_get_wtime();
        double pi  = calcular_pi(n);
        double t1  = omp_get_wtime();

        double erro = fabs(pi - M_PI);

        printf("  n = %12lld  →  π ≈ %.12f  |erro| = %.2e  (%.4f s)\n",
               n, pi, erro, t1 - t0);
    }

    printf("\n  Pi (referência Math.PI) = %.12f\n\n", M_PI);

    printf("Crivo de Eratóstenes (1 a %d) \n\n", LIMITE);

    char eh_primo[LIMITE + 1];

    double t0 = omp_get_wtime();
    crivo_eratostenes(eh_primo);
    double t1 = omp_get_wtime();

    int primos[200], qtd = 0;
    for (int i = 2; i <= LIMITE; i++)
        if (eh_primo[i]) primos[qtd++] = i;

    printf("  Total de primos encontrados: %d\n", qtd);
    printf("  Maior primo ≤ %d: %d\n", LIMITE, primos[qtd - 1]);
    printf("  Tempo do crivo: %.6f s\n\n", t1 - t0);

    printf("  Lista completa:\n  ");
    for (int i = 0; i < qtd; i++) {
        printf("%4d", primos[i]);
        if ((i + 1) % 20 == 0) printf("\n  ");
    }

    return 0;
}