static void fun(int n){
    if (n==0) System.out.print("-");
    else{
    System.out.print(n);
    fun(n-1);
    System.out.print(n);
    }
}