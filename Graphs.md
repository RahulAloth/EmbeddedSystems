# Graphs – Connecting Everything with Graphs  
A Complete Conceptual Guide for Data Structures in C

---

# 1. What Is a Graph?

A **graph** is a non-linear data structure consisting of:

- **Vertices (nodes)**  
- **Edges (connections between nodes)**  

A graph is formally written as:



\[
G = (V, E)
\]



Where:

- \(V\) = set of vertices  
- \(E\) = set of edges connecting pairs of vertices  

Graphs are used to represent **relationships**, not just data.

---

# 2. Why Graphs Matter

Graphs can represent almost any real-world system:

- Social networks  
- Maps and GPS routing  
- Airline routes  
- Internet connections  
- Dependencies in compilers  
- Neural networks  
- Game maps  
- File systems  
- Communication networks  

Graphs are the **most general-purpose** data structure.

---

# 3. Types of Graphs

## 3.1 Directed vs Undirected Graphs

- **Undirected graph:**  
  A—B means A is connected to B and B is connected to A.

- **Directed graph (digraph):**  
  A → B means a one-way relationship.

## 3.2 Weighted vs Unweighted Graphs

- **Weighted:** edges have costs (distance, time, weight)  
- **Unweighted:** all edges equal

## 3.3 Cyclic vs Acyclic Graphs

- **Cyclic:** contains loops  
- **Acyclic:** no loops (trees belong here)

## 3.4 Connected vs Disconnected Graphs

- **Connected:** every node reachable  
- **Disconnected:** some nodes isolated

---

# 4. What Is a Tree?

A **tree is a special type of graph** with strict rules.

A tree is:

- **Connected**  
- **Acyclic**  
- **Has exactly one root**  
- **Every node has exactly one parent (except root)**  

## 4.1 Properties of Trees

- If a tree has \(n\) nodes, it has exactly \(n - 1\) edges  
- There is exactly **one unique path** between any two nodes  
- Trees represent **hierarchical** data  

---

# 5. Graph vs Tree – Key Differences

| Feature | Graph | Tree |
|--------|-------|------|
| Structure | Network | Hierarchy |
| Cycles | Allowed | Not allowed |
| Root | No root | Exactly one root |
| Parent-child | Not required | Required |
| Path between nodes | Many possible | Exactly one |
| Edges | Any number | Exactly n−1 |
| Direction | Directed or undirected | Usually directed downward |
| Connectivity | May be disconnected | Always connected |
| Use cases | Maps, networks, routing, dependencies | File systems, BSTs, heaps |

### Key Insight  
**Every tree is a graph, but not every graph is a tree.**

---

# 6. Graph Representation in Data Structures

Graphs are represented in two main ways:

## 6.1 Adjacency Matrix

A 2D array where:

- `matrix[i][j] = 1` means an edge exists  
- Uses O(n²) space  
- Good for dense graphs  

## 6.2 Adjacency List

A list of linked lists:

- Each vertex stores a list of its neighbors  
- Uses O(V + E) space  
- Best for sparse graphs  
- Most common in algorithms (DFS, BFS, Dijkstra)

---

# 7. Graph Traversal

Traversal means visiting all nodes in a graph.

## 7.1 DFS – Depth First Search

- Uses recursion or stack  
- Explores deep paths first  
- Good for:
  - Cycle detection  
  - Topological sorting  
  - Maze solving  

## 7.2 BFS – Breadth First Search

- Uses queue  
- Explores level by level  
- Good for:
  - Shortest path in unweighted graphs  
  - Level-order processing  

---

# 8. When to Use Trees vs Graphs

## Use a **Tree** when:

- Data is hierarchical  
- You need fast search (BST, AVL, Red-Black)  
- You need sorted traversal  
- You want no cycles  
- You need a single root  

## Use a **Graph** when:

- Data is network-like  
- Multiple paths exist  
- Cycles are allowed  
- You need shortest path algorithms  
- You need to model relationships  

---

# 9. Summary – The Big Picture

- A **graph** is the most general structure for representing relationships.  
- A **tree** is a restricted graph with no cycles and a single root.  
- Trees are perfect for hierarchical data.  
- Graphs are perfect for networked data.  
- Understanding graphs unlocks advanced algorithms like:
  - DFS  
  - BFS  
  - Dijkstra  
  - Prim  
  - Kruskal  
  - Topological sorting  

Graphs connect everything — literally and conceptually — in data structures.

---

# Graph Concepts, Implementations, and Algorithms  
A Complete Guide for Data Structures

---

# 1. Graph Jargon (Essential Terminology)

Understanding graph terminology is the foundation for all graph algorithms.

## 1.1 Vertex (Node)
A fundamental unit of a graph.  
Example: a city, a person, a webpage.

## 1.2 Edge
A connection between two vertices.  
Example: a road, a friendship, a hyperlink.

## 1.3 Directed Edge
An edge with direction: A → B.

## 1.4 Undirected Edge
An edge without direction: A — B.

## 1.5 Weighted Edge
An edge with a cost (distance, time, weight).

## 1.6 Path
A sequence of edges connecting vertices.

## 1.7 Cycle
A path that starts and ends at the same vertex.

## 1.8 Degree
Number of edges connected to a vertex.

- **In-degree** (directed): edges coming in  
- **Out-degree** (directed): edges going out  

## 1.9 Connected Graph
Every vertex is reachable from every other vertex.

## 1.10 Component
A maximally connected subgraph.

## 1.11 Tree (in graph theory)
A connected, acyclic graph.

---

# 2. The Bare-Bones Graph Implementation (C Style)

Graphs are commonly represented in two ways:

## 2.1 Adjacency Matrix

A 2D array where:
```
matrix[i][j] = 1  → edge exists
matrix[i][j] = 0  → no edge
```

### Pros
- Simple  
- Fast edge lookup  

### Cons
- Uses O(n²) space  
- Not good for sparse graphs  

---

## 2.2 Adjacency List

A list of linked lists:
```

0 → 1 → 4
1 → 0 → 2
2 → 1 → 3
3 → 2
4 → 0
```

### Pros
- Space efficient  
- Ideal for sparse graphs  
- Used in BFS, DFS, Dijkstra  

### Cons
- Slower edge lookup than matrix  

---

# 3. Object-Oriented Graph Implementation (Conceptual)

Even in C (non-OOP), we can think in OOP terms:

## 3.1 Graph as an Object

- **Attributes:**
  - number of vertices  
  - adjacency list  
  - directed/undirected flag  

- **Methods:**
  - addVertex()  
  - addEdge()  
  - removeEdge()  
  - BFS()  
  - DFS()  

This mindset helps structure large graph algorithms cleanly.

---

# 4. Graph Search

Graph search means exploring all reachable nodes from a starting point.

Two fundamental algorithms:

- **DFS (Depth First Search)**  
- **BFS (Breadth First Search)**  

These are the foundation for:

- Cycle detection  
- Topological sorting  
- Shortest paths  
- Connected components  
- Maze solving  
- Network routing  

---

# 5. Depth First Search (DFS)

DFS explores **as deep as possible** before backtracking.

## 5.1 How DFS Works

1. Start at a node  
2. Visit it  
3. Recursively visit an unvisited neighbor  
4. Backtrack when no neighbors remain  

## 5.2 Characteristics

- Uses **stack** (explicit or recursion)  
- Good for:
  - Cycle detection  
  - Topological sorting  
  - Solving mazes  
  - Finding connected components  

## 5.3 Time Complexity



\[
O(V + E)
\]



---

# 6. Breadth First Search (BFS)

BFS explores **level by level**, like ripples in water.

## 6.1 How BFS Works

1. Start at a node  
2. Visit all neighbors  
3. Then neighbors of neighbors  
4. Continue outward  

## 6.2 Characteristics

- Uses **queue**  
- Finds **shortest path in unweighted graphs**  
- Good for:
  - Level-order traversal  
  - Shortest path  
  - Social network analysis  

## 6.3 Time Complexity



\[
O(V + E)
\]



---

# 7. Efficiency of Graph Search

Both BFS and DFS run in:



\[
O(V + E)
\]



Where:

- \(V\) = number of vertices  
- \(E\) = number of edges  

This is optimal because you must inspect every vertex and edge at least once.

---

# 8. Weighted Graphs

A **weighted graph** assigns a cost to each edge:
```
A --5--> B
A --2--> C
C --1--> B

```

Weights represent:

- Distance  
- Time  
- Cost  
- Capacity  

Weighted graphs require specialized algorithms for shortest paths.

---

# 9. The Shortest Path Problem

Given a weighted graph, find the minimum-cost path between two nodes.

## 9.1 Types of shortest path problems

### 1. **Single-source shortest path**
Find shortest path from one node to all others.

### 2. **Single-pair shortest path**
Find shortest path between two specific nodes.

### 3. **All-pairs shortest path**
Find shortest paths between all pairs of nodes.

---

# 10. Algorithms for Shortest Path

## 10.1 BFS (for unweighted graphs)
Shortest path = fewest edges.

## 10.2 Dijkstra’s Algorithm
For **positive weights only**.

- Uses priority queue  
- Greedy algorithm  
- Time:  
  - O(E log V) with heap  
  - O(V²) with matrix  

## 10.3 Bellman–Ford Algorithm
Handles **negative weights**.

## 10.4 Floyd–Warshall Algorithm
All-pairs shortest path.

---

# Summary

- Graphs represent relationships, not just data.  
- Trees are special graphs (connected + acyclic).  
- Graphs can be implemented using adjacency lists or matrices.  
- DFS explores deep; BFS explores wide.  
- Both run in O(V + E).  
- Weighted graphs require algorithms like Dijkstra or Bellman–Ford.  
- Shortest path problems are central to routing, navigation, and optimization.

Graphs are the backbone of modern computing — from Google Maps to compilers to social networks.


# 1. Graph Jargon (Basic Terminology)

Understanding graph terminology is essential before implementing or using graph algorithms.

## 1.1 Vertex (Node)
A fundamental unit of a graph.  
Example: a city, a person, a webpage.

## 1.2 Edge
A connection between two vertices.

## 1.3 Directed Edge
A one‑way connection: A → B.

## 1.4 Undirected Edge
A two‑way connection: A — B.

## 1.5 Weighted Edge
An edge with a cost (distance, time, weight).

## 1.6 Path
A sequence of edges connecting vertices.

## 1.7 Cycle
A path that starts and ends at the same vertex.

## 1.8 Degree
Number of edges connected to a vertex.

## 1.9 Connected Graph
Every vertex is reachable from every other vertex.

## 1.10 Component
A maximally connected subgraph.

---

# 2. The Bare‑Bones Graph Implementation (Adjacency List)

The adjacency list is the most common and efficient representation.

# 4. Graph Search

- Graph search means exploring all reachable nodes from a starting point.

- Two fundamental algorithms:
    - DFS (Depth First Search)
    - BFS (Breadth First Search)
 
# 5. Depth First Search (DFS)

- DFS explores as deep as possible before backtracking.

# - 6. Breadth First Search (BFS)

- BFS explores level by level, like ripples in water.

# 7. Efficiency of Graph Search

- Both DFS and BFS run in:
  ```
O(V+E)
Where:
    V = number of vertices
    E = number of edges
```

This is optimal because every vertex and edge must be inspected at least once.
# 8. Weighted Graphs

A weighted graph assigns a cost to each edge:
```
A --5--> B
A --2--> C
C --1--> B
Weights represent:
    - Distance
    - Time
    - Cost
```

# 9. The Shortest Path Problem

- Given a weighted graph, find the minimum‑cost path between two nodes.
## 9.1 Types of shortest path problems
    - Single‑source shortest path
    - Single‑pair shortest path
    - All‑pairs shortest path
# Dijkstra’s Algorithm (Shortest Path in Weighted Graphs)
- Works only when all weights are non‑negative.
- All Samples are given in Graph.c file. There are many things we can talk about graph but we are stopping here.



