#include <stdio.h>
#define MAX 100
#include <stdio.h>
#define MAX 100

int stack[MAX];
int top = -1;

int queue[MAX];
int front = 0;
int rear = -1;


void push(int value) {
    if (top == MAX - 1) {
        printf("Stack Overflow\n");
        return;
    }
    stack[++top] = value;
}

int pop() {
    if (top == -1) {
        printf("Stack Underflow\n");
        return -1;
    }
    return stack[top--];
}

int peek() {
    if (top == -1) {
        printf("Stack is empty\n");
        return -1;
    }
    return stack[top];
}

void enqueue(int value) {
    if (rear == MAX - 1) {
        printf("Queue Overflow\n");
        return;
    }
    queue[++rear] = value;
}

int dequeue() {
    if (front > rear) {
        printf("Queue Underflow\n");
        return -1;
    }
    return queue[front++];
}

int peek_() {
    if (front > rear) {
        printf("Queue is empty\n");
        return -1;
    }
    return queue[front];
}

int fibonacci(int n) {
    if (n == 0) return 0;      // Base case 1
    if (n == 1) return 1;      // Base case 2
    return fibonacci(n - 1) + fibonacci(n - 2);  // Recursive step
}

int main() {
    int n = 6;
    
    push(10);
    push(20);
    push(30);

    printf("Top element: %d\n", peek());
    printf("Popped: %d\n", pop());
    printf("Top after pop: %d\n", peek());

   // Queue:
    enqueue(10);
    enqueue(20);
    enqueue(30);

    printf("Front element: %d\n", peek());
    printf("Dequeued: %d\n", dequeue());
    printf("Front after dequeue: %d\n", peek_());

    printf("Fibonacci(%d) = %d\n", n, fibonacci(n));
    return 0;
    return 0;
}



