import java.util.*;

public class q4_std {
    public static void main(String[] args) {
        Tree tree = new Tree();
        Scanner kb = new Scanner(System.in);
        int num_q = kb.nextInt();
        for(int i = 0; i < num_q; i++) {
            int command = kb.nextInt();
            if(command == 1) {
                int k = kb.nextInt();   // k
                long x = kb.nextLong(); // x
                tree.insert(k, x);
            } else if (command == 2){
                int k = kb.nextInt();   // k
                tree.delete(k);
            } else {
                int l = kb.nextInt(), r = kb.nextInt();
                System.out.println(tree.sum(l, r));
            }
        }
        kb.close();
    }
}

class Tree {
    // introduce nullNode instead of using `null`
    Node root, nullNode;
    
    Tree() { 
        // initialize root & nullNode
        nullNode = new Node();
        nullNode.l_son = nullNode.r_son = nullNode;
        nullNode.weight = 0;
        nullNode.sum = nullNode.value = 0;
        root = nullNode;
    }

    private Node newNode(long key) {
        // create new node, note that nullNode is replacing null
        Node node = new Node();
        node.l_son = node.r_son = nullNode;
        node.weight = 1;
        node.sum = node.value = key;
        return node; 
    }
    
    public void insert(int rank, long value) {
        root = insert(root, rank, value);
    }

    public void delete(int rank) {
        root = delete(root, rank);
    }

    public long sum(int l, int r) {
        return sum(root, l, r);
    }

    private Node insert(Node t, int rank, long value) {
        // t is a tree node
        if(t == nullNode)
            return newNode(value);
        if(rank <= t.l_son.weight) // in left subtree, `0` is always in left
            t.l_son = insert(t.l_son, rank, value);
        else // in right subtree
            t.r_son = insert(t.r_son, rank - t.rank(), value);
        t.pushup();
        return t;
    }

    private Node delNode(Node t) {
        //case 1: has less than 2 kids
        if(t.l_son == nullNode) return t.r_son;
        if(t.r_son == nullNode) return t.l_son;

        // find the successor and swap value
        Node temp = t.r_son;
        while(temp.l_son != nullNode)
            temp = temp.l_son;
        t.value = temp.value;

        // the second deletion must be fall into case 1
        t.r_son = delete(t.r_son, 1);
        t.pushup();
        return t;
    }

    private Node delete(Node t, int rank) {
        if(rank <= t.l_son.weight) // in the left subtree
            t.l_son = delete(t.l_son, rank);
        else if(rank > t.rank()) // in right subtree
            t.r_son = delete(t.r_son, rank - t.rank());
        else // delete the root of current subtree
            t = delNode(t);

        // if t is nullNode, there is no need to update
        if(t != nullNode)
            t.pushup();
        return t;
    }

    private long sum(Node t, int l, int r) {

        //case 1: the range of tree has no intersection with [l, r]
        if(t == nullNode || r <= 0 || l > t.weight) return 0L;
        
        // case 2: the range of tree is inside [l, r]
        if(l <= 1 && t.weight <= r) return t.sum;

        // case 3: there is intersection between tree range and [l, r]
        long ans = (l <= t.rank() && t.rank() <= r) ? t.value : 0L;
        return sum(t.l_son, l, r) + sum(t.r_son, l - t.rank(), r - t.rank()) + ans;
    }
}

class Node {
    Node l_son, r_son;
    int weight;
    long sum, value;
    Node () { }
    int rank() { 
        return l_son.weight + 1;
    }
    void pushup() {
        weight = l_son.weight + r_son.weight + 1;
        sum = l_son.sum + r_son.sum + value;
    }
}