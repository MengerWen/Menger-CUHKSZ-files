import java.io.*;
import java.util.*;

public class Main {
    // Edge class to store destination, weight, and edge index
    static class Edge {
        int to;
        int weight;
        int edgeIdx;

        Edge(int to, int weight, int edgeIdx) {
            this.to = to;
            this.weight = weight;
            this.edgeIdx = edgeIdx;
        }
    }

    // State class for Dijkstra's algorithm with Comparable based on time
    static class State implements Comparable<State> {
        long time;
        int node;
        long mask;

        State(long time, int node, long mask) {
            this.time = time;
            this.node = node;
            this.mask = mask;
        }

        @Override
        public int compareTo(State other) {
            return Long.compare(this.time, other.time);
        }
    }

    // FastScanner class for efficient input reading
    static class FastScanner {
        BufferedReader br;
        StringTokenizer st;

        FastScanner() {
            br = new BufferedReader(new InputStreamReader(System.in));
        }

        // Method to fetch the next token
        String next() throws IOException {
            while (st == null || !st.hasMoreTokens()) {
                String line = br.readLine();
                if (line == null)
                    return null;
                st = new StringTokenizer(line);
            }
            return st.nextToken();
        }

        // Method to fetch the next integer
        int nextInt() throws IOException {
            String s = next();
            if (s == null)
                throw new EOFException();
            return Integer.parseInt(s);
        }
    }

    public static void main(String[] args) throws IOException {
        FastScanner sc = new FastScanner();
        BufferedWriter bw = new BufferedWriter(new OutputStreamWriter(System.out));

        // Read n, m, q
        int n = sc.nextInt();
        int m = sc.nextInt();
        int q = sc.nextInt();

        // Initialize graph with 1-based indexing
        ArrayList<ArrayList<Edge>> graph = new ArrayList<>(n + 1);
        for(int i = 0; i <= n; i++) {
            graph.add(new ArrayList<>());
        }

        // Read m edges
        for(int edgeIdx = 1; edgeIdx <= m; edgeIdx++) {
            int u = sc.nextInt();
            int v = sc.nextInt();
            int w = sc.nextInt();
            graph.get(u).add(new Edge(v, w, edgeIdx));
            graph.get(v).add(new Edge(u, w, edgeIdx));
        }

        // Read q queries' E_j lists
        ArrayList<ArrayList<Integer>> queriesE = new ArrayList<>(q);
        for(int i = 0; i < q; i++) {
            ArrayList<Integer> E_j = new ArrayList<>();
            int k_i = sc.nextInt();
            for(int j = 0; j < k_i; j++) {
                E_j.add(sc.nextInt());
            }
            queriesE.add(E_j);
        }

        // Read q queries' s_j and t_j
        ArrayList<int[]> queriesST = new ArrayList<>(q);
        for(int i = 0; i < q; i++) {
            int s_j = sc.nextInt();
            int t_j = sc.nextInt();
            queriesST.add(new int[]{s_j, t_j});
        }

        // Process each query
        for(int queryIdx = 0; queryIdx < q; queryIdx++) {
            ArrayList<Integer> E_j = queriesE.get(queryIdx);
            int s_j = queriesST.get(queryIdx)[0];
            int t_j = queriesST.get(queryIdx)[1];
            int k_i = E_j.size();

            if(k_i == 0){
                if(s_j == t_j){
                    bw.write("0\n");
                    continue;
                }
                // Perform standard Dijkstra's algorithm
                PriorityQueue<State> heap = new PriorityQueue<>();
                heap.add(new State(0, s_j, 0));
                boolean[] visited = new boolean[n + 1];
                boolean found = false;

                while(!heap.isEmpty()) {
                    State current = heap.poll();
                    long currentTime = current.time;
                    int currentNode = current.node;

                    if(currentNode == t_j){
                        bw.write(currentTime + "\n");
                        found = true;
                        break;
                    }

                    if(visited[currentNode]) continue;
                    visited[currentNode] = true;

                    for(Edge neighbor : graph.get(currentNode)){
                        int v = neighbor.to;
                        int w = neighbor.weight;
                        if(!visited[v]){
                            heap.add(new State(currentTime + w, v, 0));
                        }
                    }
                }

                if(!found){
                    bw.write("-1\n");
                }
                continue;
            }

            // Perform modified Dijkstra's algorithm with edge traversal requirements
            // Map edge index to bit position
            HashMap<Integer, Integer> E_j_map = new HashMap<>();
            for(int bitPos = 0; bitPos < k_i; bitPos++) {
                int edgeIdx = E_j.get(bitPos);
                E_j_map.put(edgeIdx, bitPos);
            }

            // Calculate target mask
            long targetMask;
            if(k_i >= 64){
                targetMask = -1L;
            }
            else{
                targetMask = (1L << k_i) - 1;
            }

            PriorityQueue<State> heap = new PriorityQueue<>();
            heap.add(new State(0, s_j, 0L));

            // Initialize visited sets for each node
            HashSet<Long>[] visited = new HashSet[n + 1];
            for(int i = 0; i <= n; i++) {
                visited[i] = new HashSet<>();
            }

            boolean found = false;

            while(!heap.isEmpty()){
                State current = heap.poll();
                long currentTime = current.time;
                int currentNode = current.node;
                long mask = current.mask;

                if(currentNode == t_j && mask == targetMask){
                    bw.write(currentTime + "\n");
                    found = true;
                    break;
                }

                if(visited[currentNode].contains(mask)){
                    continue;
                }

                visited[currentNode].add(mask);

                for(Edge neighbor : graph.get(currentNode)){
                    int v = neighbor.to;
                    int w = neighbor.weight;
                    int edgeIdx = neighbor.edgeIdx;
                    long newMask = mask;

                    if(E_j_map.containsKey(edgeIdx)){
                        int bitPos = E_j_map.get(edgeIdx);
                        newMask = mask | (1L << bitPos);
                    }

                    heap.add(new State(currentTime + w, v, newMask));
                }
            }

            if(!found){
                bw.write("-1\n");
            }
        }

        // Flush the output
        bw.flush();
    }
}