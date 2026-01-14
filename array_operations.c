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
void print(int arr[], int size) {
    for (int i = 0; i < size; i++)
        printf("%d ", arr[i]);
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
/*
## C Answers for Ordered Array Exercises
Below are simple C programs that demonstrate the solutions to each exercise:  
binary search steps, insertion, deletion, and comparison of search methods.

### 1. Binary Search Steps for Finding 50  
Array: `10, 20, 30, 40, 50, 60`
*/ 
void binarySearchSteps( int arr[], int size, int target) {
    int left = 0, right = size - 1;
    int step = 1;

    while (left <= right) {
        int mid = (left + right) / 2;
        printf("Step %d: left=%d, right=%d, mid=%d, value=%d\n",
               step, left, right, mid, arr[mid]);

        if (arr[mid] == target) {
            printf("Found %d at index %d\n", target, mid);
            return;
        }
        else if (arr[mid] < target) {
            left = mid + 1;
        }
        else {
            right = mid - 1;
        }
        step++;
    }
}

int compute_binarySearchSteps() {
    int arr[] = {10, 20, 30, 40, 50, 60};
    int size = 6;

    binarySearchSteps(arr, size, 50);
    return 0;
}
// 2. Insert 35 into Ordered Array
void insertOrdered(int arr[], int *size, int value) {
    int i = *size - 1;

    // Shift elements to the right
    while (i >= 0 && arr[i] > value) {
        arr[i + 1] = arr[i];
        i--;
    }

    arr[i + 1] = value;
    (*size)++;
}


int compute_insertOrdered() {
    int arr[10] = {10, 20, 30, 40, 50, 60};
    int size = 6;

    printf("Before insertion: ");
    print(arr, size);

    insertOrdered(arr, &size, 35);

    printf("After inserting 35: ");
    print(arr, size);

    return 0;
}
// 3. Delete 20 from Ordered Array
void deleteValue(int arr[], int *size, int value) {
    int index = -1;

    // Find the value
    for (int i = 0; i < *size; i++) {
        if (arr[i] == value) {
            index = i;
            break;
        }
    }

    if (index == -1) return; // not found

    // Shift left
    for (int i = index; i < *size - 1; i++) {
        arr[i] = arr[i + 1];
    }

    (*size)--;
}

int compute_deleteValue() {
    int arr[10] = {10, 20, 30, 40, 50, 60};
    int size = 6;

    printf("Before deletion: ");
    print(arr, size);

    deleteValue(arr, &size, 20);

    printf("After deleting 20: ");
    print(arr, size);

    return 0;
}


int linearSteps(int arr[], int size, int target) {
    int steps = 0;
    for (int i = 0; i < size; i++) {
        steps++;
        if (arr[i] == target) break;
    }
    return steps;
}

// 4. Compare Linear Search vs Binary Search Steps
int binarySteps(int arr[], int size, int target) {
    int left = 0, right = size - 1;
    int steps = 0;

    while (left <= right) {
        steps++;
        int mid = (left + right) / 2;

        if (arr[mid] == target) break;
        else if (arr[mid] < target) left = mid + 1;
        else right = mid - 1;
    }
    return steps;
}
int compute_linear_binary_compare() {
    int arr[100];
    for (int i = 0; i < 100; i++) arr[i] = i + 1;

    int target = 60;

    printf("Linear search steps: %d\n", linearSteps(arr, 100, target));
    printf("Binary search steps: %d\n", binarySteps(arr, 100, target));

    return 0;
}


// Bubble Sort function
void bubbleSort(int arr[], int n) {
    for (int i = 0; i < n - 1; i++) {
        // After each pass, the largest element moves to the end
        for (int j = 0; j < n - i - 1; j++) {
            if (arr[j] > arr[j + 1]) {
                // Swap elements
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
}

int compute_bubble_sort() {
    int data[] = {64, 34, 25, 12, 22, 11, 90};
    int size = sizeof(data) / sizeof(data[0]);

    printf("Original array: ");
    printArray(data, size);

    bubbleSort(data, size);

    printf("Sorted array:   ");
    printArray(data, size);

    return 0;
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
