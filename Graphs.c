//1. Bare‑Bones Graph Implementation (Adjacency List)
#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
    int vertex;
    struct Node* next;
} Node;

typedef struct Graph {
    int numVertices;
    Node** adjList;
} Graph;

// Create a node
Node* createNode(int v) {
    Node* newNode = malloc(sizeof(Node));
    newNode->vertex = v;
    newNode->next = NULL;
    return newNode;
}

// Create a graph
Graph* createGraph(int vertices) {
    Graph* graph = malloc(sizeof(Graph));
    graph->numVertices = vertices;

    graph->adjList = malloc(vertices * sizeof(Node*));
    for (int i = 0; i < vertices; i++)
        graph->adjList[i] = NULL;

    return graph;
}

// Add edge (undirected)
void addEdge(Graph* graph, int src, int dest) {
    Node* newNode = createNode(dest);
    newNode->next = graph->adjList[src];
    graph->adjList[src] = newNode;

    newNode = createNode(src);
    newNode->next = graph->adjList[dest];
    graph->adjList[dest] = newNode;
}

// Print graph
void printGraph(Graph* graph) {
    for (int v = 0; v < graph->numVertices; v++) {
        Node* temp = graph->adjList[v];
        printf("Vertex %d: ", v);
        while (temp) {
            printf("%d -> ", temp->vertex);
            temp = temp->next;
        }
        printf("NULL\n");
    }
}

int main() {
    Graph* graph = createGraph(5);

    addEdge(graph, 0, 1);
    addEdge(graph, 0, 4);
    addEdge(graph, 1, 2);
    addEdge(graph, 1, 3);
    addEdge(graph, 1, 4);

    printGraph(graph);

    return 0;
}


// Graph in Object Oriented way:


#include <stdio.h>
#include <stdlib.h>

typedef struct Graph {
    int V;
    int** matrix;

    void (*addEdge)(struct Graph*, int, int);
    void (*print)(struct Graph*);
} Graph;

void addEdge(Graph* g, int src, int dest) {
    g->matrix[src][dest] = 1;
    g->matrix[dest][src] = 1;
}

void printGraph(Graph* g) {
    for (int i = 0; i < g->V; i++) {
        for (int j = 0; j < g->V; j++)
            printf("%d ", g->matrix[i][j]);
        printf("\n");
    }
}

Graph* createGraph(int V) {
    Graph* g = malloc(sizeof(Graph));
    g->V = V;

    g->matrix = malloc(V * sizeof(int*));
    for (int i = 0; i < V; i++) {
        g->matrix[i] = calloc(V, sizeof(int));
    }

    g->addEdge = addEdge;
    g->print = printGraph;

    return g;
}

int main() {
    Graph* g = createGraph(4);

    g->addEdge(g, 0, 1);
    g->addEdge(g, 0, 2);
    g->addEdge(g, 1, 3);

    g->print(g);

    return 0;
}
// Graph Depth first Search
#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
    int vertex;
    struct Node* next;
} Node;

typedef struct Graph {
    int V;
    Node** adjList;
    int* visited;
} Graph;

Node* createNode(int v) {
    Node* newNode = malloc(sizeof(Node));
    newNode->vertex = v;
    newNode->next = NULL;
    return newNode;
}

Graph* createGraph(int V) {
    Graph* graph = malloc(sizeof(Graph));
    graph->V = V;

    graph->adjList = malloc(V * sizeof(Node*));
    graph->visited = calloc(V, sizeof(int));

    for (int i = 0; i < V; i++)
        graph->adjList[i] = NULL;

    return graph;
}

void addEdge(Graph* graph, int src, int dest) {
    Node* newNode = createNode(dest);
    newNode->next = graph->adjList[src];
    graph->adjList[src] = newNode;

    newNode = createNode(src);
    newNode->next = graph->adjList[dest];
    graph->adjList[dest] = newNode;
}

void DFS(Graph* graph, int vertex) {
    graph->visited[vertex] = 1;
    printf("%d ", vertex);

    Node* temp = graph->adjList[vertex];
    while (temp) {
        if (!graph->visited[temp->vertex])
            DFS(graph, temp->vertex);
        temp = temp->next;
    }
}

int main() {
    Graph* graph = createGraph(5);

    addEdge(graph, 0, 1);
    addEdge(graph, 0, 2);
    addEdge(graph, 1, 3);
    addEdge(graph, 1, 4);

    printf("DFS: ");
    DFS(graph, 0);

    return 0;
}

// Graph Breadth first Search:
#include <stdio.h>
#include <stdlib.h>

#define MAX 100

typedef struct Node {
    int vertex;
    struct Node* next;
} Node;

typedef struct Graph {
    int V;
    Node** adjList;
    int visited[MAX];
} Graph;

Node* createNode(int v) {
    Node* newNode = malloc(sizeof(Node));
    newNode->vertex = v;
    newNode->next = NULL;
    return newNode;
}

Graph* createGraph(int V) {
    Graph* graph = malloc(sizeof(Graph));
    graph->V = V;

    graph->adjList = malloc(V * sizeof(Node*));
    for (int i = 0; i < V; i++) {
        graph->adjList[i] = NULL;
        graph->visited[i] = 0;
    }

    return graph;
}

void addEdge(Graph* graph, int src, int dest) {
    Node* newNode = createNode(dest);
    newNode->next = graph->adjList[src];
    graph->adjList[src] = newNode;

    newNode = createNode(src);
    newNode->next = graph->adjList[dest];
    graph->adjList[dest] = newNode;
}

void BFS(Graph* graph, int start) {
    int queue[MAX], front = 0, rear = 0;

    graph->visited[start] = 1;
    queue[rear++] = start;

    while (front < rear) {
        int vertex = queue[front++];
        printf("%d ", vertex);

        Node* temp = graph->adjList[vertex];
        while (temp) {
            if (!graph->visited[temp->vertex]) {
                graph->visited[temp->vertex] = 1;
                queue[rear++] = temp->vertex;
            }
            temp = temp->next;
        }
    }
}

int main() {
    Graph* graph = createGraph(5);

    addEdge(graph, 0, 1);
    addEdge(graph, 0, 2);
    addEdge(graph, 1, 3);
    addEdge(graph, 1, 4);

    printf("BFS: ");
    BFS(graph, 0);

    return 0;
}

// ✅ 5. Weighted Graph + Shortest Path (Dijkstra’s Algorithm)
#include <stdio.h>
#include <limits.h>

#define V 5

int minDistance(int dist[], int visited[]) {
    int min = INT_MAX, min_index;

    for (int v = 0; v < V; v++)
        if (!visited[v] && dist[v] <= min)
            min = dist[v], min_index = v;

    return min_index;
}

void dijkstra(int graph[V][V], int src) {
    int dist[V];
    int visited[V];

    for (int i = 0; i < V; i++)
        dist[i] = INT_MAX, visited[i] = 0;

    dist[src] = 0;

    for (int count = 0; count < V - 1; count++) {
        int u = minDistance(dist, visited);
        visited[u] = 1;

        for (int v = 0; v < V; v++)
            if (!visited[v] && graph[u][v] &&
                dist[u] + graph[u][v] < dist[v])
                dist[v] = dist[u] + graph[u][v];
    }

    printf("Vertex   Distance from Source\n");
    for (int i = 0; i < V; i++)
        printf("%d \t\t %d\n", i, dist[i]);
}

int main() {
    int graph[V][V] = {
        {0, 2, 0, 6, 0},
        {2, 0, 3, 8, 5},
        {0, 3, 0, 0, 7},
        {6, 8, 0, 0, 9},
        {0, 5, 7, 9, 0}
    };

    dijkstra(graph, 0);

    return 0;
}



// ## 2.1 C Implementation (Adjacency List)
#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
    int vertex;
    struct Node* next;
} Node;

typedef struct Graph {
    int V;
    Node** adjList;
} Graph;

Node* createNode(int v) {
    Node* newNode = malloc(sizeof(Node));
    newNode->vertex = v;
    newNode->next = NULL;
    return newNode;
}

Graph* createGraph(int V) {
    Graph* graph = malloc(sizeof(Graph));
    graph->V = V;

    graph->adjList = malloc(V * sizeof(Node*));
    for (int i = 0; i < V; i++)
        graph->adjList[i] = NULL;

    return graph;
}

void addEdge(Graph* graph, int src, int dest) {
    Node* newNode = createNode(dest);
    newNode->next = graph->adjList[src];
    graph->adjList[src] = newNode;

    newNode = createNode(src);
    newNode->next = graph->adjList[dest];
    graph->adjList[dest] = newNode;
}

void printGraph(Graph* graph) {
    for (int i = 0; i < graph->V; i++) {
        Node* temp = graph->adjList[i];
        printf("Vertex %d: ", i);
        while (temp) {
            printf("%d -> ", temp->vertex);
            temp = temp->next;
        }
        printf("NULL\n");
    }
}

int main() {
    Graph* graph = createGraph(5);

    addEdge(graph, 0, 1);
    addEdge(graph, 0, 4);
    addEdge(graph, 1, 2);

    printGraph(graph);
    return 0;
}

// Adjacency Matrix + Methods

#include <stdio.h>
#include <stdlib.h>

typedef struct Graph {
    int V;
    int** matrix;

    void (*addEdge)(struct Graph*, int, int);
    void (*print)(struct Graph*);
} Graph;

void addEdge(Graph* g, int src, int dest) {
    g->matrix[src][dest] = 1;
    g->matrix[dest][src] = 1;
}

void printGraph(Graph* g) {
    for (int i = 0; i < g->V; i++) {
        for (int j = 0; j < g->V; j++)
            printf("%d ", g->matrix[i][j]);
        printf("\n");
    }
}

Graph* createGraph(int V) {
    Graph* g = malloc(sizeof(Graph));
    g->V = V;

    g->matrix = malloc(V * sizeof(int*));
    for (int i = 0; i < V; i++)
        g->matrix[i] = calloc(V, sizeof(int));

    g->addEdge = addEdge;
    g->print = printGraph;

    return g;
}

int main() {
    Graph* g = createGraph(4);

    g->addEdge(g, 0, 1);
    g->addEdge(g, 0, 2);
    g->addEdge(g, 1, 3);

    g->print(g);
    return 0;
}
// DFS
void DFSUtil(Graph* graph, int v, int visited[]) {
    visited[v] = 1;
    printf("%d ", v);

    Node* temp = graph->adjList[v];
    while (temp) {
        if (!visited[temp->vertex])
            DFSUtil(graph, temp->vertex, visited);
        temp = temp->next;
    }
}

void DFS(Graph* graph, int start) {
    int visited[graph->V];
    for (int i = 0; i < graph->V; i++)
        visited[i] = 0;

    DFSUtil(graph, start, visited);
}

// 6.1 C Implementation (BFS)
void BFS(Graph* graph, int start) {
    int visited[graph->V];
    for (int i = 0; i < graph->V; i++)
        visited[i] = 0;

    int queue[100], front = 0, rear = 0;

    visited[start] = 1;
    queue[rear++] = start;

    while (front < rear) {
        int v = queue[front++];
        printf("%d ", v);

        Node* temp = graph->adjList[v];
        while (temp) {
            if (!visited[temp->vertex]) {
                visited[temp->vertex] = 1;
                queue[rear++] = temp->vertex;
            }
            temp = temp->next;
        }
    }
}
// 10. Dijkstra’s Algorithm (Shortest Path in Weighted Graphs)
#include <stdio.h>
#include <limits.h>

#define V 5

int minDistance(int dist[], int visited[]) {
    int min = INT_MAX, min_index = -1;

    for (int v = 0; v < V; v++)
        if (!visited[v] && dist[v] <= min)
            min = dist[v], min_index = v;

    return min_index;
}

void dijkstra(int graph[V][V], int src) {
    int dist[V];
    int visited[V];

    for (int i = 0; i < V; i++)
        dist[i] = INT_MAX, visited[i] = 0;

    dist[src] = 0;

    for (int count = 0; count < V - 1; count++) {
        int u = minDistance(dist, visited);
        visited[u] = 1;

        for (int v = 0; v < V; v++)
            if (!visited[v] && graph[u][v] &&
                dist[u] + graph[u][v] < dist[v])
                dist[v] = dist[u] + graph[u][v];
    }

    printf("Vertex   Distance from Source\n");
    for (int i = 0; i < V; i++)
        printf("%d \t\t %d\n", i, dist[i]);
}

int main() {
    int graph[V][V] = {
        {0, 2, 0, 6, 0},
        {2, 0, 3, 8, 5},
        {0, 3, 0, 0, 7},
        {6, 8, 0, 0, 9},
        {0, 5, 7, 9, 0}
    };

    dijkstra(graph, 0);
    return 0;
}



