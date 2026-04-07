// Sudoku Solver Paralelo com OpenMP
// Estratégia: geração de tarefas paralelas para cada candidato
// na primeira célula vazia, cada tarefa opera em uma cópia independente do grid.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#define N 9

/* Imprime o grid */
void print(int arr[N][N])
{
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++)
            printf("%d ", arr[i][j]);
        printf("\n");
    }
}

/* Verifica se é seguro colocar num na posição (row, col) */
int isSafe(int grid[N][N], int row, int col, int num)
{
    // Verifica linha
    for (int x = 0; x < N; x++)
        if (grid[row][x] == num)
            return 0;

    // Verifica coluna
    for (int x = 0; x < N; x++)
        if (grid[x][col] == num)
            return 0;

    // Verifica bloco 3x3
    int startRow = row - row % 3, startCol = col - col % 3;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 3; j++)
            if (grid[i + startRow][j + startCol] == num)
                return 0;

    return 1;
}

/* Solver sequencial de backtracking (usado dentro de cada tarefa) */
int solveSudoku(int grid[N][N], int row, int col)
{
    if (row == N - 1 && col == N)
        return 1;

    if (col == N) {
        row++;
        col = 0;
    }

    if (grid[row][col] > 0)
        return solveSudoku(grid, row, col + 1);

    for (int num = 1; num <= N; num++) {
        if (isSafe(grid, row, col, num)) {
            grid[row][col] = num;
            if (solveSudoku(grid, row, col + 1) == 1)
                return 1;
            grid[row][col] = 0;
        }
    }
    return 0;
}

/*
 * solveSudokuParallel:
 * Localiza a PRIMEIRA célula vazia e lança uma tarefa OpenMP para cada
 * candidato válido (1-9). Cada tarefa recebe uma CÓPIA independente do
 * grid e executa o backtracking sequencial de forma isolada.
 * Uma variável compartilhada 'solved' sinaliza quando a solução é encontrada,
 * permitindo que as demais tarefas sejam descartadas (early exit).
 */
int solveSudokuParallel(int grid[N][N])
{
    // Localiza a primeira célula vazia
    int firstRow = -1, firstCol = -1;
    for (int i = 0; i < N && firstRow == -1; i++)
        for (int j = 0; j < N && firstRow == -1; j++)
            if (grid[i][j] == 0) {
                firstRow = i;
                firstCol = j;
            }

    // Sem células vazias → já resolvido
    if (firstRow == -1) {
        print(grid);
        return 1;
    }

    // Grid de saída compartilhado
    int solution[N][N];
    int solved = 0; // flag compartilhada

    #pragma omp parallel shared(solved, solution)
    {
        #pragma omp single nowait
        {
            #pragma omp taskgroup
            {
                for (int num = 1; num <= N; num++) {
                    if (isSafe(grid, firstRow, firstCol, num)) {

                        // Captura num por valor para a task
                        int candidate = num;

                        #pragma omp task firstprivate(candidate) shared(solved, solution)
                        {
                            // Early exit: se já foi resolvido por outra task, aborta
                            if (!solved) {
                                int localGrid[N][N];
                                memcpy(localGrid, grid, sizeof(localGrid));
                                localGrid[firstRow][firstCol] = candidate;

                                if (solveSudoku(localGrid, firstRow, firstCol + 1)) {
                                    #pragma omp critical
                                    {
                                        if (!solved) {
                                            solved = 1;
                                            memcpy(solution, localGrid, sizeof(solution));
                                        }
                                    }
                                }
                            }
                        } // fim task
                    }
                }
            } // fim taskgroup (sincroniza todas as tasks)
        } // fim single
    } // fim parallel

    if (solved) {
        print(solution);
        return 1;
    }
    return 0;
}

int main()
{
    // 0 representa células não preenchidas
    int grid[N][N] = {
        { 5, 3, 0, 0, 7, 0, 0, 0, 0 },
        { 6, 0, 0, 1, 9, 5, 0, 0, 0 },
        { 0, 9, 8, 0, 0, 0, 0, 6, 0 },
        { 8, 0, 0, 0, 6, 0, 0, 0, 3 },
        { 4, 0, 0, 8, 0, 3, 0, 0, 1 },
        { 7, 0, 0, 0, 2, 0, 0, 0, 6 },
        { 0, 6, 0, 0, 0, 0, 2, 8, 0 },
        { 0, 0, 0, 4, 1, 9, 0, 0, 5 },
        { 0, 0, 0, 0, 8, 0, 0, 7, 9 }
    };

    double start = omp_get_wtime();

    if (solveSudokuParallel(grid) != 1)
        printf("Sem solução.\n");

    double end = omp_get_wtime();
    printf("Tempo: %.6f segundos\n", end - start);

    return 0;
}