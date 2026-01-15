#include <stdio.h>
#include <string.h>

//
// 1. NASA-Style Countdown (Repeated Execution)
//
void countdown(int n) {
    if (n == 0) {
        printf("Liftoff!\n");
        return;
    }
    printf("%d\n", n);
    countdown(n - 1);
}

//
// 2. Factorial (Top-Down Recursive Calculation)
//
int factorial(int n) {
    if (n == 0) return 1;
    return n * factorial(n - 1);
}

//
// 3. Array Sum (Process first n-1, recurse on rest)
//
int array_sum(int arr[], int n) {
    if (n == 0) return 0;
    return arr[n - 1] + array_sum(arr, n - 1);
}

//
// 4. String Reversal (Recursive)
//
void reverse_string(char *s, int start, int end) {
    if (start >= end) return;

    char temp = s[start];
    s[start] = s[end];
    s[end] = temp;

    reverse_string(s, start + 1, end - 1);
}

//
// 5. Counting occurrences of 'x'
//    Process first char, recurse on rest
//
int count_x(char *s) {
    if (*s == '\0') return 0;
    return (*s == 'x') + count_x(s + 1);
}

//
// 6. Staircase Problem (Ways to climb n steps)
//
int staircase(int n) {
    if (n == 0) return 1;
    if (n == 1) return 1;
    return staircase(n - 1) + staircase(n - 2);
}

//
// 7. Anagram Generation (Permutations)
//
void swap(char *a, char *b) {
    char temp = *a;
    *a = *b;
    *b = temp;
}

void generate_anagrams(char *s, int left, int right) {
    if (left == right) {
        printf("%s\n", s);
        return;
    }

    for (int i = left; i <= right; i++) {
        swap(&s[left], &s[i]);
        generate_anagrams(s, left + 1, right);
        swap(&s[left], &s[i]); // backtrack
    }
}

//
// MAIN PROGRAM
//
int main() {

    printf("=== 1. Countdown ===\n");
    countdown(5);

    printf("\n=== 2. Factorial ===\n");
    printf("Factorial(5) = %d\n", factorial(5));

    printf("\n=== 3. Array Sum ===\n");
    int arr[] = {1, 2, 3, 4, 5};
    printf("Array sum = %d\n", array_sum(arr, 5));

    printf("\n=== 4. String Reversal ===\n");
    char str[] = "recursion";
    reverse_string(str, 0, strlen(str) - 1);
    printf("Reversed string = %s\n", str);

    printf("\n=== 5. Count 'x' ===\n");
    char s2[] = "axbxcxdx";
    printf("Count of 'x' = %d\n", count_x(s2));

    printf("\n=== 6. Staircase Problem ===\n");
    printf("Ways to climb 5 steps = %d\n", staircase(5));

    printf("\n=== 7. Anagram Generation ===\n");
    char word[] = "abc";
    generate_anagrams(word, 0, strlen(word) - 1);

    return 0;
}
