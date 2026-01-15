#include <stdio.h>
#include <stdlib.h>

typedef struct Node {
    int data;
    struct Node *left;
    struct Node *right;
} Node;

// Create a new node
Node* createNode(int value) {
    Node *newNode = (Node*)malloc(sizeof(Node));
    newNode->data = value;
    newNode->left = newNode->right = NULL;
    return newNode;
}

// Insert into BST
Node* insert(Node *root, int key) {
    if (root == NULL)
        return createNode(key);

    if (key < root->data)
        root->left = insert(root->left, key);
    else if (key > root->data)
        root->right = insert(root->right, key);

    return root;
}

// Search in BST
Node* search(Node *root, int key) {
    if (root == NULL || root->data == key)
        return root;

    if (key < root->data)
        return search(root->left, key);
    else
        return search(root->right, key);
}

// Find minimum node (used for successor)
Node* findMin(Node *root) {
    while (root != NULL && root->left != NULL)
        root = root->left;
    return root;
}

// Delete a node from BST
Node* deleteNode(Node *root, int key) {
    if (root == NULL)
        return NULL;

    if (key < root->data) {
        root->left = deleteNode(root->left, key);
    } else if (key > root->data) {
        root->right = deleteNode(root->right, key);
    } else {
        // Node found

        // Case 1: no child
        if (root->left == NULL && root->right == NULL) {
            free(root);
            return NULL;
        }
        // Case 2: one child (right only)
        else if (root->left == NULL) {
            Node *temp = root->right;
            free(root);
            return temp;
        }
        // Case 2: one child (left only)
        else if (root->right == NULL) {
            Node *temp = root->left;
            free(root);
            return temp;
        }
        // Case 3: two children
        else {
            Node *succ = findMin(root->right);
            root->data = succ->data;
            root->right = deleteNode(root->right, succ->data);
        }
    }
    return root;
}

// Inorder traversal (Left, Root, Right)
void inorder(Node *root) {
    if (root == NULL) return;
    inorder(root->left);
    printf("%d ", root->data);
    inorder(root->right);
}

// Preorder traversal (Root, Left, Right)
void preorder(Node *root) {
    if (root == NULL) return;
    printf("%d ", root->data);
    preorder(root->left);
    preorder(root->right);
}

// Postorder traversal (Left, Right, Root)
void postorder(Node *root) {
    if (root == NULL) return;
    postorder(root->left);
    postorder(root->right);
    printf("%d ", root->data);
}

int main() {
    Node *root = NULL;

    int values[] = {50, 30, 70, 20, 40, 60, 80};
    int n = sizeof(values) / sizeof(values[0]);

    for (int i = 0; i < n; i++)
        root = insert(root, values[i]);

    printf("Inorder (sorted): ");
    inorder(root);
    printf("\n");

    printf("Deleting 20 (leaf)...\n");
    root = deleteNode(root, 20);
    inorder(root);
    printf("\n");

    printf("Deleting 30 (one child)...\n");
    root = deleteNode(root, 30);
    inorder(root);
    printf("\n");

    printf("Deleting 50 (two children)...\n");
    root = deleteNode(root, 50);
    inorder(root);
    printf("\n");

    return 0;
}
