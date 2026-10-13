#include <stdio.h>

int n;

void print_out(int number) {
    int p = 0, top = -1;
    int stack[32];
    while(number) {
        if(number % 2 == 1) {
            stack[++top] = p;
        } 
        number /= 2;
        p++;
    }
    
    while( ~ top ) {
        p = stack[top--];
        switch (p)
        {
        case 0:
            printf("2(0)");
            break;
        case 1:
            printf("2");
            break;
        default:
            printf("2(");
            print_out(p);
            printf(")");
            break;
        }
        if(~top)
            printf("+");
    }
    return ;
}

int main(int argc, char **argv) {
    scanf("%d", &n);
    print_out(n);
    return 0;
}
