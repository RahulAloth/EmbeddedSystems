#include <stdio.h>
#include <stdlib.h>

// Node structure for singly linked list
typedef struct Node {
    int data;
    struct Node *next;
} Node;

// Create a new node
Node* createNode(int value) {
    Node *newNode = (Node*)malloc(sizeof(Node));
    newNode->data = value;
    newNode->next = NULL;
    return newNode;
}

// Insert at beginning
void insertAtHead(Node **head, int value) {
    Node *newNode = createNode(value);
    newNode->next = *head;
    *head = newNode;
}

// Insert at end
void insertAtEnd(Node **head, int value) {
    Node *newNode = createNode(value);

    if (*head == NULL) {
        *head = newNode;
        return;
    }

    Node *temp = *head;
    while (temp->next != NULL)
        temp = temp->next;

    temp->next = newNode;
}

// Search for a value
Node* search(Node *head, int key) {
    Node *temp = head;
    while (temp != NULL) {
        if (temp->data == key)
            return temp;
        temp = temp->next;
    }
    return NULL;
}

// Delete a node by value
void deleteValue(Node **head, int key) {
    if (*head == NULL) return;

    Node *temp = *head;

    // If head node is to be deleted
    if (temp->data == key) {
        *head = temp->next;
        free(temp);
        return;
    }

    Node *prev = NULL;
    while (temp != NULL && temp->data != key) {
        prev = temp;
        temp = temp->next;
    }

    if (temp == NULL) return; // Not found

    prev->next = temp->next;
    free(temp);
}

// Display list
void display(Node *head) {
    Node *temp = head;
    while (temp != NULL) {
        printf("%d → ", temp->data);
        temp = temp->next;
    }
    printf("NULL\n");
}

// Main function
int main() {
    Node *head = NULL;

    insertAtHead(&head, 10);
    insertAtHead(&head, 20);
    insertAtEnd(&head, 30);
    insertAtEnd(&head, 40);

    printf("Singly Linked List:\n");
    display(head);

    deleteValue(&head, 20);
    printf("After deleting 20:\n");
    display(head);

    Node *found = search(head, 30);
    if (found) printf("Found: %d\n", found->data);
    else printf("Not found\n");

    return 0;
}
/*
// Double Linked List.
#include <stdio.h>
#include <stdlib.h>

// Node structure for doubly linked list
typedef struct DNode {
    int data;
    struct DNode *prev;
    struct DNode *next;
} DNode;

// Create a new node
DNode* createDNode(int value) {
    DNode *newNode = (DNode*)malloc(sizeof(DNode));
    newNode->data = value;
    newNode->prev = NULL;
    newNode->next = NULL;
    return newNode;
}

// Insert at head
void insertAtHead(DNode **head, int value) {
    DNode *newNode = createDNode(value);
    newNode->next = *head;

    if (*head != NULL)
        (*head)->prev = newNode;

    *head = newNode;
}

// Insert at end
void insertAtEnd(DNode **head, int value) {
    DNode *newNode = createDNode(value);

    if (*head == NULL) {
        *head = newNode;
        return;
    }

    DNode *temp = *head;
    while (temp->next != NULL)
        temp = temp->next;

    temp->next = newNode;
    newNode->prev = temp;
}

// Delete a node by value
void deleteValue(DNode **head, int key) {
    if (*head == NULL) return;

    DNode *temp = *head;

    // If head is to be deleted
    if (temp->data == key) {
        *head = temp->next;
        if (*head != NULL)
            (*head)->prev = NULL;
        free(temp);
        return;
    }

    while (temp != NULL && temp->data != key)
        temp = temp->next;

    if (temp == NULL) return; // Not found

    if (temp->next != NULL)
        temp->next->prev = temp->prev;

    if (temp->prev != NULL)
        temp->prev->next = temp->next;

    free(temp);
}

// Display forward
void displayForward(DNode *head) {
    DNode *temp = head;
    printf("Forward: ");
    while (temp != NULL) {
        printf("%d ⇄ ", temp->data);
        temp = temp->next;
    }
    printf("NULL\n");
}

// Display backward
void displayBackward(DNode *head) {
    if (head == NULL) return;

    DNode *temp = head;
    while (temp->next != NULL)
        temp = temp->next;

    printf("Backward: ");
    while (temp != NULL) {
        printf("%d ⇄ ", temp->data);
        temp = temp->prev;
    }
    printf("NULL\n");
}

// Main function
int main() {
    DNode *head = NULL;

    insertAtHead(&head, 10);
    insertAtHead(&head, 20);
    insertAtEnd(&head, 30);
    insertAtEnd(&head, 40);

    printf("Doubly Linked List:\n");
    displayForward(head);
    displayBackward(head);

    deleteValue(&head, 30);
    printf("After deleting 30:\n");
    displayForward(head);
    displayBackward(head);

    return 0;
}


*/
