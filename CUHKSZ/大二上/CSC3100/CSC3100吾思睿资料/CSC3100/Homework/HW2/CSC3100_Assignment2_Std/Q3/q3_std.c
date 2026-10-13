#include <stdio.h>

int k, N;

int removed(int row, int col, int level, int len) 
{
    if(level == 0) return 0;
    len /= 3;
    if(len <= row && row < len + len
     && len <= col && col < len + len) {
        return 1;
    }
    return removed(row % len, col % len, level - 1, len);
}

int main(int argc, char **argv) 
{
    
    scanf("%d", &k);
    N = 1;
    for(int i = 0; i < k; ++i) N = N * 3;

    for(int i = 0; i < N; i++) {
        for(int j = 0; j < N; j++) {
            putchar(removed(i, j, k, N)?' ':'#');
        }
        putchar('\n');
    }
    return 0;
}
