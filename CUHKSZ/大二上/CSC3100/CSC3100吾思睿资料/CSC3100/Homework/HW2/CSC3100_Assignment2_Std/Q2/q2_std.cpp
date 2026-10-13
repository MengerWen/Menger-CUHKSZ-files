#include<bits/stdc++.h>
using namespace std;
const int N = 1e5 + 5;
int n, a[N], b[N];
long long ans = 0;
void merge_sort(int l,int r){
    if(l == r) return;
    int mid = (l + r) >> 1;
    merge_sort(l, mid);
    merge_sort(mid + 1, r);
    int p1 = l, p2 = mid + 1, pb = l;
    while((p1 != mid + 1 )&&(p2 != r + 1)){
        if(a[p1] < a[p2]) b[pb++] = a[p1++];
        else ans += (mid + 1 - p1), b[pb++] = a[p2++]; 
    }
    while(p1 != mid + 1) b[pb++] = a[p1++];
    while(p2 != r + 1) b[pb++] = a[p2++];
    for(int i = l; i <= r ; i++) a[i] = b[i];
}
int main(){
    scanf("%d",&n);
    for(int i = 1;i <= n; i++) scanf("%d",&a[i]);
    merge_sort(1,n);
    printf("%lld",ans);
}