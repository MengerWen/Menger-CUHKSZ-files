import java.util.*;
public class q3_std {
    public static int removed(int row, int col, int level, int len) {
        if(level == 0) return 0;
        len /= 3;
        if(len <= row && row < len + len
         && len <= col && col < len + len) {
            return 1;
        }
        return removed(row % len, col % len, level - 1, len);
    }
    public static void main(String[] args) {
        int k, N = 1;
        char tmpc;
        Scanner jdr = new Scanner(System.in);
        k = jdr.nextInt();
        jdr.close();
        for(int i = 0; i < k; ++i) N = N * 3;

        for(int i = 0; i < N; i++) {
            for(int j = 0; j < N; j++) {
                if(removed(i, j, k, N) == 1) 
                    tmpc = ' ';
                else 
                    tmpc = '#';
                System.out.printf("%c",tmpc);
            }
            System.out.println();
        }
    }
}
