// Parallel Quick Sort using OpenMP Tasks
#include <stdio.h>
#include <omp.h>

void swap(int* a, int* b) {
    int temp = *a;
    *a = *b;
    *b = temp;
}

int partition(int arr[], int low, int high) {
    int p = arr[low];
    int i = low;
    int j = high;

    while (i < j) {
        while (arr[i] <= p && i <= high - 1) {
            i++;
        }
        while (arr[j] > p && j >= low + 1) {
            j--;
        }
        if (i < j) {
            swap(&arr[i], &arr[j]);
        }
    }
    swap(&arr[low], &arr[j]);
    return j;
}

#define CUTOFF 1000

void quickSort(int arr[], int low, int high) {
    if (low < high) {
        int pi = partition(arr, low, high);

        if ((high - low) < CUTOFF) {
            quickSort(arr, low, pi - 1);
            quickSort(arr, pi + 1, high);
        } else {
            #pragma omp task shared(arr) firstprivate(low, pi)
            quickSort(arr, low, pi - 1);

            #pragma omp task shared(arr) firstprivate(pi, high)
            quickSort(arr, pi + 1, high);

            #pragma omp taskwait
        }
    }
}

int main() {
    int arr[] = { 4, 2, 5, 3, 1 };
    int n = sizeof(arr) / sizeof(arr[0]);

    #pragma omp parallel
    {
        #pragma omp single
        quickSort(arr, 0, n - 1);
    }

    for (int i = 0; i < n; i++)
        printf("%d ", arr[i]);
    printf("\n");

    return 0;
}