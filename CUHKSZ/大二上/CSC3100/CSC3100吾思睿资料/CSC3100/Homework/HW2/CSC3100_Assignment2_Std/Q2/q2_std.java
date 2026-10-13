import java.util.*;

public class q2_std {
    public static long merge(int[] a, int left, int mid, int right) {
        long inv_num = 0;
        int i = left, j = mid + 1, k = 0;
        int[] b = new int[right - left + 1];
        while(i <= mid && j <= right) {
            if(a[i] <= a[j]) b[k++] = a[i++];
            else { 
                b[k++] = a[j++];
                inv_num += (mid - i + 1);
            }
        } 
        while(i <= mid) b[k++] = a[i++];
        while(j <= right) b[k++] = a[j++];
        for(i = 0; i < right - left + 1; i++) {
            a[i + left] = b[i];
        }
        return inv_num;
    }
    public static long merge_sort(int[] a, int left, int right) {
        if(left == right) return 0;
        int mid = (left + right) / 2;
        long inv_num = 0;
        inv_num += merge_sort(a, left, mid);
        inv_num += merge_sort(a, mid + 1, right);
        inv_num += merge(a, left, mid, right);
        return inv_num;
    }
    public static void main(String[] args) {
        int n;
        int[] a;
        long ans, same_num = 1;
        Scanner sc = new Scanner(System.in);
        n = sc.nextInt();
        a = new int[n];
        for(int i = 0; i < n; i++)  
            a[i] = sc.nextInt();
        sc.close();
        ans = merge_sort(a, 0, n - 1);

        for(int i = 1; i < n; i++) {
            if(a[i] != a[i - 1]) {
                ans += (same_num - 1) * (same_num) / 2;
                same_num = 1;
            } else {
                same_num += 1;
            }
        }
        ans += (same_num - 1) * same_num / 2;
        System.out.println(ans);
    }    
}