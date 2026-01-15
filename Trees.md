# 1. Trees and binary search trees
## 1.1 Tree
- A tree is a hierarchical data structure:
    - Node: holds data and links to children.
    - Root: topmost node.
    - Edge: connection between parent and child.
    - Leaf: node with no children.
    - Subtree: a node plus all its descendants.
A binary tree is a tree where each node has at most two children: left and right.

## 1.2 Binary search tree (BST)

- A BST is a binary tree with an ordering rule:
    - For any node with key k:
        - All keys in the left subtree are < k
        - All keys in the right subtree are > k
    - No duplicates (in the classic definition).
    - Inorder traversal of a BST gives keys in sorted order
- This ordering is what speeds things up: each comparison lets you discard half of the remaining subtree—similar to binary search on arrays
## 2. BST node and basic structure in C
```
typedef struct Node {
    int data;
    struct Node *left;
    struct Node *right;
} Node;
```
You typically keep:
```
Node *root = NULL;
```

## 3. Searching in a binary search tree
### 3.1 Idea
- To search for key:
    - If root == NULL → not found.
    - If key == root->data → found.
    - If key < root->data → search in left subtree.
    - If key > root->data → search in right subtree.
 - This is logarithmic on average if the tree is balanced, O(h) where h is height

### 3.2 Recursive search in C
```
Node* search(Node *root, int key) {
    if (root == NULL || root->data == key)
        return root;

    if (key < root->data)
        return search(root->left, key);
    else
        return search(root->right, key);
}
```

## 4. Insertion in a BST

### 4.1 Idea
- To insert key:
    - If root == NULL → new node becomes root.
    - If key < root->data → insert into left subtree.
    - If key > root->data → insert into right subtree.
- Usually ignore duplicates or handle specially.

### 4.2 Order of insertion

- The shape of the BST depends heavily on the order of insertion:
    - Insert 1, 2, 3, 4, 5 in that order → tree becomes a right-skewed list (height n).
    - Insert 3, 1, 5, 2, 4 → more balanced tree (height ~log n).
- So:
    - Best/average case (random-ish insertion): height ≈ O(log n) → operations O(log n)
  ```
  Node* createNode(int value) {
    Node *newNode = (Node*)malloc(sizeof(Node));
    newNode->data = value;
    newNode->left = newNode->right = NULL;
    return newNode;
}

Node* insert(Node *root, int key) {
    if (root == NULL)
        return createNode(key);

    if (key < root->data)
        root->left = insert(root->left, key);
    else if (key > root->data)
        root->right = insert(root->right, key);
    // if equal, usually do nothing or handle duplicates

    return root;
}



## 5. Deletion in a BST
- Deletion is the trickiest part. There are three cases:
    - Node to delete has no children (leaf).
    - Node has one child.
    - Node has two children.
### 5.1 Case 1: Node with no children
    - Just free the node and set parent’s pointer to NULL.
### 5.2 Case 2: Node with one child
    - Replace the node with its single child.
    - Adjust parent’s pointer to point to that child.
    - Free the node.
### 5.3 Case 3: Node with two children
- This is the interesting one.
    - We need to delete node N with two children.
    -We must preserve BST ordering.
    - Standard trick: replace N’s value with its inorder successor (or predecessor), then delete that successor node.

##  6. Finding the successor node
### 6.1 Inorder successor
- The inorder successor of a node is the node with the smallest key greater than it.
-  For a node N:
    -  If N has a right child:
        - Successor is the minimum node in the right subtree.
        -  That is: go to N->right, then go left as far as possible.
##  7. Complete deletion algorithm (recursive)
7.1 High-level steps
- To delete key from BST rooted at root:
    - If root == NULL → nothing to delete.
    -  If key < root->data → delete from left subtree.
    - If key > root->data → delete from right subtree.
    - Else (key == root->data):
        - Case 1: no child → free node, return NULL.
        - Case 2: one child → return the non-NULL child after freeing node.
        - Case 3: two children:
            - Find inorder successor in right subtree.
            -  Copy successor’s data into current node.
            - Delete successor from right subtree.


