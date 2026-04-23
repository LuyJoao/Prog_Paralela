#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

#define M 2147483647LL
#define A 48271LL
#define C 0LL
#define N_THREADS 8
#define N_SAMPLES 1000000000LL

long long mod_pow(long long base, long long exp, long long mod)
{
    long long result = 1;
    base %= mod;
    while (exp > 0)
    {
        if (exp & 1)
            result = (__int128)result * base % mod;
        base = (__int128)base * base % mod;
        exp >>= 1;
    }
    return result;
}

long long advance_seed(long long seed, int k)
{
    long long ak = mod_pow(A, k, M);
    return (__int128)ak * seed % M;
}

int main()
{
    long long seed = 123456789LL % M;
    if (seed == 0)
        seed = 1;

    long long A_leap = mod_pow(A, N_THREADS, M);

    long long A_mod[N_THREADS];
    for (int k = 0; k < N_THREADS; k++)
    {
        A_mod[k] = mod_pow(A, 1LL << k, M);
    }

    long long inside_std = 0, inside_mod = 0;

#pragma omp parallel num_threads(N_THREADS) reduction(+ : inside_std)
    {
        int tid = omp_get_thread_num();
        long long x = advance_seed(seed, tid);
        long long local = 0;
        long long total = N_SAMPLES / N_THREADS;

        for (long long i = 0; i < total; i++)
        {
            double u = (double)x / (double)M;
            x = (__int128)A_leap * x % M;
            double v = (double)x / (double)M;
            x = (__int128)A_leap * x % M;
            if (u * u + v * v <= 1.0)
                local++;
        }
        inside_std += local;
    }

#pragma omp parallel num_threads(N_THREADS) reduction(+ : inside_mod)
    {
        int tid = omp_get_thread_num();
        long long x = advance_seed(seed, tid);
        long long a_t = A_mod[tid];
        long long local = 0;
        long long total = N_SAMPLES / N_THREADS;

        for (long long i = 0; i < total; i++)
        {
            double u = (double)x / M;
            x = (__int128)a_t * x % M;
            double v = (double)x / M;
            x = (__int128)a_t * x % M;
            if (u * u + v * v <= 1.0)
                local++;
        }
        inside_mod += local;
    }

    long long used = (N_SAMPLES / N_THREADS) * N_THREADS;
    long long points = used;

    printf("Leapfrog padrão:     Pi ~= %.8f  (amostras: %lld)\n",
           4.0 * inside_std / points, points);
    printf("Leapfrog modificado: Pi ~= %.8f  (amostras: %lld)\n",
           4.0 * inside_mod / points, points);

    return 0;
}