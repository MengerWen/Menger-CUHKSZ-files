#include <stack>
#include <iostream>

int n;

void print_out(int number) {
    int p = 0;
    std::stack<int> S;
    while(number) {
        if(number % 2 == 1) {
            S.push(p);
        } 
        number /= 2;
        p++;
    }
    
    while(! S.empty()) {
        p = S.top();
        S.pop();
        switch (p)
        {
        case 0:
            std::cout<< "2(0)";
            break;
        case 1:
            std::cout<< "2";
            break;
        default:
            std::cout<< "2(";
            print_out(p);
            std::cout<< ")";
            break;
        }
        if(! S.empty())
            std::cout<< "+";
    }
    return ;
}

int main(int argc, char **argv) {
    std::cin>> n;
    print_out(n);
    return 0;
}
