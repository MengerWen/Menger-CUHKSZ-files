import java.util.*;
public class q1_std {
    public static void printAnswer(int n) {
        Stack<Integer> s = new Stack<Integer>();
        int p = 0;
        while(n != 0) {
            if(n % 2 == 1) {
                s.push(p);
            }
            p ++;
            n = n / 2;
        }
        while(!s.empty()) {
            p = s.pop();
            switch (p) {
                case 0:
                    System.out.print("2(0)");
                    break;
                case 1:
                    System.out.print("2");
                    break;
                default:
                    System.out.print("2(");
                    printAnswer(p);
                    System.out.print(")");
                    break;
            }
            if(!s.empty())
                System.out.print('+');
        }
    }
    public static void main(String[] args) {
        int n;
        Scanner jdr = new Scanner(System.in);
        n = jdr.nextInt();
        jdr.close();   
        printAnswer(n);
    }
}