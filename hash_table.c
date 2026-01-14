#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define TABLE_SIZE 10

typedef struct Node {
    char key[50];
    int value;
    struct Node* next;
} Node;

Node* hashTable[TABLE_SIZE];

unsigned int hash(char* key) {
    unsigned int sum = 0;
    for (int i = 0; key[i] != '\0'; i++) {
        sum += key[i];
    }
    return sum % TABLE_SIZE;
}

void insert(char* key, int value) {
    unsigned int index = hash(key);

    Node* newNode = malloc(sizeof(Node));
    strcpy(newNode->key, key);
    newNode->value = value;
    newNode->next = hashTable[index];

    hashTable[index] = newNode;
}

int search(char* key) {
    unsigned int index = hash(key);
    Node* current = hashTable[index];

    while (current != NULL) {
        if (strcmp(current->key, key) == 0) {
            return current->value;
        }
        current = current->next;
    }

    return -1;  // not found
}

int main() {
    insert("burger", 5);
    insert("fries", 2);
    insert("pizza", 8);
    insert("cola", 1);

    printf("Price of pizza: %d\n", search("pizza"));
    printf("Price of cola: %d\n", search("cola"));
    printf("Price of sushi: %d\n", search("sushi")); // not found

    return 0;
}
