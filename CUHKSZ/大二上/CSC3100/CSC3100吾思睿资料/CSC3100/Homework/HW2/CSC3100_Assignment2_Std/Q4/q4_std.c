#include <stdio.h>
#define re register
#define ll long long
int ch[200005][2],sz[200005],val[200005],root;
ll sum[200005];
int num;
void up(int tmp){
    sz[tmp] = 1, sum[tmp] = val[tmp];
    sz[tmp] += sz[ch[tmp][0]];
    sz[tmp] += sz[ch[tmp][1]];
    sum[tmp] += sum[ch[tmp][0]];
    sum[tmp] += sum[ch[tmp][1]];
}
int insert(int tmp,int k,int x){
    if(tmp == 0){
        num++;
        sz[num] = 1;
        val[num] = x;
        sum[num] = x;
        return num;
    }
    if(k <= sz[ch[tmp][0]]) ch[tmp][0] = insert(ch[tmp][0],k,x);
    else ch[tmp][1] = insert(ch[tmp][1],k-sz[ch[tmp][0]]-1,x);
    up(tmp);
    return tmp;
}
int delpre(int tmp){
    if(ch[ch[tmp][1]][1] == 0){
        int ans = val[ch[tmp][1]];
        ch[tmp][1] = ch[ch[tmp][1]][0];
        up(tmp);
        return ans;
    }
    int ans = delpre(ch[tmp][1]);
    up(tmp);
    return ans;
}
int remove_point(int tmp,int k){
    if(k == sz[ch[tmp][0]] + 1){
        if(ch[tmp][0] == 0) return ch[tmp][1];
        if(ch[tmp][1] == 0) return ch[tmp][0];
        if(ch[ch[tmp][0]][1] == 0){
            val[tmp] = val[ch[tmp][0]];
            ch[tmp][0] = ch[ch[tmp][0]][0];
            up(tmp);
            return tmp;
        }
        int newval = delpre(ch[tmp][0]);
        val[tmp] = newval;
        up(tmp);
        return tmp;
    }
    if(k <= sz[ch[tmp][0]]) ch[tmp][0] = remove_point(ch[tmp][0],k);
    else ch[tmp][1] = remove_point(ch[tmp][1],k-sz[ch[tmp][0]] - 1);
    up(tmp);
    return tmp;
}
ll getsum(int tmp,int k){
    if(tmp == 0) return 0;
    if(k <= sz[ch[tmp][0]]) return getsum(ch[tmp][0],k);
    else return sum[ch[tmp][0]] + val[tmp] + getsum(ch[tmp][1],k-sz[ch[tmp][0]]-1);
}
int n;
int main(){
    scanf("%d",&n);
    for(int i=1;i<=n;i++){
        int opt;
        scanf("%d",&opt);
        if(opt == 1){
            int k,x;
            scanf("%d%d",&k,&x);
            root = insert(root,k,x);
        }
        if(opt == 2){
            int k;
            scanf("%d",&k);
            root = remove_point(root,k);
        }
        if(opt == 3){
            int l,r;
            scanf("%d%d",&l,&r);
            printf("%lld\n",getsum(root,r) - getsum(root,l-1));
        }
    }
}