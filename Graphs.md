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
