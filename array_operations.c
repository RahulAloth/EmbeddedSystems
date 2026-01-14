// C Example: Array Operations (Read, Search, Insert, Delete)

#include <stdio.h>

#define MAX 100

// Print array
void printArray(int arr[], int size) {
    for (int i = 0; i < size; i++) {
        printf("%d ", arr[i]);
    }
    printf("\n");
}

// Linear search — O(n)
int search(int arr[], int size, int target) {
    for (int i = 0; i < size; i++) {
        if (arr[i] == target)
            return i;   // return index
    }
    return -1;          // not found
}

// Insert at a given index — O(n)
void insert(int arr[], int *size, int index, int value) {
    if (*size >= MAX) return;

    for (int i = *size; i > index; i--) {
        arr[i] = arr[i - 1];   // shift right
    }
    arr[index] = value;
    (*size)++;
}

// Delete at a given index — O(n)
void delete(int arr[], int *size, int index) {
    for (int i = index; i < *size - 1; i++) {
        arr[i] = arr[i + 1];   // shift left
    }
    (*size)--;
}

int main() {
    int arr[MAX] = {10, 20, 30, 40, 50};
    int size = 5;

    printf("Initial array: ");
    printArray(arr, size);

    // Reading — O(1)
    printf("Read arr[2] = %d\n", arr[2]);

    // Searching — O(n)
    int idx = search(arr, size, 30);
    printf("Search 30 found at index: %d\n", idx);

    // Insertion — O(n)
    insert(arr, &size, 2, 99);
    printf("After inserting 99 at index 2: ");
    printArray(arr, size);

    // Deletion — O(n)
    delete(arr, &size, 3);
    printf("After deleting index 3: ");
    printArray(arr, size);

    return 0;
}
