# Learning to Write Recursive Functions

Recursion is a way of solving problems where a function **calls itself** with a smaller or simpler version of the same problem.  
To write good recursive code, you need to recognize patterns where a big problem can be broken into **subproblems of the same kind**.

This note focuses on:

- Recursive category: **Repeatedly execute**
- The **NASA spacecraft countdown** style example
- The **recursive trick: passing extra arguments**
- Two main areas where recursion shines
- Bottom‑up vs top‑down thinking
- Classic recursive problems:
  - Factorial  
  - Array sum  
  - String reversal  
  - Counting occurrences  
  - Staircase problem  
  - Anagram generation and its efficiency  

---
# Recursively Recurse with Recursion

## Introduction

Recursion is a key concept in computer science.  
It allows a function to **call itself** to solve a problem.  
This idea unlocks many advanced algorithms such as:

- Tree traversal  
- Graph search  
- Divide and conquer  
- Dynamic programming  
- Backtracking (mazes, Sudoku, N‑Queens)

Recursion is powerful because many problems can be broken into **smaller versions of themselves**.

---

# Recurse Instead of Loop

Many tasks that use loops can also be solved using recursion.  
The difference is in *how* repetition is expressed.

### Loop Mindset  
“Repeat this block of code until the condition becomes false.”

### Recursive Mindset  
“Break the problem into a smaller version of itself, and let the function handle the rest.”

---

# Example: Factorial (n!)

Factorial is defined as:



```
n! = n \cdot (n-1) \cdot (n-2) \cdots 1

```


---

## Factorial Using a Loop

```c
int factorial_loop(int n) {
    int result = 1;
    for (int i = 1; i <= n; i++) {
        result *= i;
    }
    return result;
}
```

## Factorial Using Recursion

```
int factorial_recursive(int n) {
    if (n == 0) {
        return 1;   // Base case
    }
    return n * factorial_recursive(n - 1);  // Recursive step
}
```
## Recursion in the Eyes of a Computer

- When a function calls itself, the computer uses the call stack.
- Each function call gets its own stack frame containing:
    - Parameters
    - Local variables
    - Return address
Example: factorial_recursive(3)
Call stack grows:

```
factorial(3)
  factorial(2)
    factorial(1)
      factorial(0)
```
## Infinite Recursion

- If a recursive function never reaches its base case, it calls itself forever.
Example:
```
void infinite() {
    infinite();   // No base case
}
```

## 1. Recursive Category: Repeatedly Execute

Some algorithms exist mainly to **repeat a task** until a condition is met.  
This is the same category where we normally use **loops**, but recursion can express the same idea in a more **declarative** way.

### Example Idea: NASA Spacecraft Countdown

Imagine a countdown before launching a spacecraft:

- Start at `10`
- Print the number
- Go to `9`, then `8`, … down to `0`
- At `0`, print “Liftoff!”

### Recursive Version (Conceptual Pseudocode)

```text
function countdown(n):
    if n == 0:
        print "Liftoff!"
        return
    print n
    countdown(n - 1)
```

Here:

    Base case: n == 0

    Recursive step: call countdown(n - 1)

This is a classic example of “repeatedly execute” using recursion instead of a loop.

### 12. Recursive Trick: Passing Extra Argument

- Sometimes recursion becomes much easier if we pass an extra argument that keeps track of progress.
- For example, instead of counting down from n to 0, we might want to count up from 1 to n:

```text
function count_up(current, n):
    if current > n:
        return
    print current
    count_up(current + 1, n)
```
- Here, current is an extra argument that tracks where we are.
- This trick is useful for:
    - Accumulating results
    - Tracking indices in arrays
    - Carrying partial answers (like a running sum or reversed string)

### 3. Two Areas Where Recursion Shines

- In Recursively Recurse with Recursion, we saw one major area:
    - Arbitrary depth structures
        - Trees, nested folders, HTML DOM, graphs
        - You don’t know how deep the structure goes
        - Recursion naturally “dives” into each level
- A second major area is:
    - Calculations based on subproblems
        - Factorial
        - Fibonacci
        - Array sum
        - String reversal
        - Counting occurrences
        - Staircase ways
        - Anagram generation
-In these problems, the answer for n depends on the answer for smaller inputs like n - 1, n - 2, etc.

### 4. Factorial: Classic Recursive Calculation
- Definition
```text
  n!=n⋅(n−1)!
  with
  0!=1
```
Recursive C Code

```text
int factorial(int n) {
    if (n == 0) {
        return 1;   // Base case
    }
    return n * factorial(n - 1);  // Recursive case
}
```
- Base case: n == 0
- Recursive step: n * factorial(n - 1)
- This is a perfect example of “calculate using a subproblem”.

### 5. Bottom‑Up vs Top‑Down

- There are two main ways to think about recursive calculations:
#### 5.1 Top‑Down (Classic Recursion)
- You start from the big problem and break it down:
```text
- To compute factorial(5), you say:
        5! = 5 * 4!
        4! = 4 * 3!
        and so on…
```
- This is top‑down: start big, go smaller.
#### 5.2 Bottom‑Up (Iterative or DP Style)
- You start from the smallest case and build up:
```text
    You know 0! = 1
    Then 1! = 1 * 0! = 1
    Then 2! = 2 * 1! = 2
```

- This is often implemented with loops or dynamic programming.
- Recursion is naturally top‑down, but you can convert many recursive solutions into bottom‑up ones for efficiency.
  
### 6. Array Sum (Recursive)

- Goal: compute the sum of all elements in an array.
- Idea
  
```text
    Sum of array [a0, a1, ..., an-1]  
    = a0 + sum of [a1, ..., an-1]
```

```text
int array_sum(int arr[], int n) {
    if (n == 0) {
        return 0;   // Empty array sum is 0
    }
    return arr[n - 1] + array_sum(arr, n - 1);
}
```

- Base case: n == 0 → sum is 0
- Recursive step: last element + sum of the rest

### 7. String Reversal (Recursive)
- Goal: reverse a string like "abcd" → "dcba".
- Idea
    - Reverse of "abcd":
         - Take first char 'a'
         - Reverse "bcd" → "dcb"
         - Append 'a' at the end → "dcba"
Recursive Pseudocode
```text
function reverse(s):
    if length(s) == 0 or length(s) == 1:
        return s
    return reverse(s[1:]) + s[0]
```
- Base case: empty or single‑character string
- Recursive step: reverse substring from index 1 onward, then add first char at the end

### 8. Counting X (Counting Occurrences)

- Goal: count how many times a character (say 'x') appears in a string.
- 
Idea
```text
    - Look at the first character:
        - If it is 'x', count 1 + count in rest
        - Else, just count in rest

- Recursive Pseudocode
```c
function count_x(s):
    if s is empty:
        return 0
    first = s[0]
    rest = s[1:]
    if first == 'x':
        return 1 + count_x(rest)
    else:
        return 0 + count_x(rest)
```
- This pattern—process first element, recurse on the rest—is extremely common.

### 9. Staircase Problem
- Problem: You have a staircase with n steps.
- You can climb either 1 step or 2 steps at a time.
- How many distinct ways can you reach the top?

```c
- Idea
- To reach step n, you must have come from:
    - Step n - 1 (then take 1 step), or
    - Step n - 2 (then take 2 steps)
- So:
  - ways(n)=ways(n−1)+ways(n−2)
- with base cases:
    - ways(0) = 1 (one way: stand still)
    - ways(1) = 1 (one way: single step)
```
```c
int ways(int n) {
    if (n == 0) return 1;
    if (n == 1) return 1;
    return ways(n - 1) + ways(n - 2);
}
```

- This is structurally similar to Fibonacci.

## 10. Anagram Generation

- Problem: Given a string, generate all possible rearrangements of its characters (anagrams or permutations).
- Example: "abc" → "abc", "acb", "bac", "bca", "cab", "cba".
- Idea
    - Fix one character at the front
    - Recursively generate all permutations of the remaining characters
    - Repeat for each character as the first
## Recursive Pseudocode
```c
function permute(s, prefix):
    if length(s) == 0:
        print prefix
        return

    for i from 0 to length(s) - 1:
        ch = s[i]
        remaining = s[0:i] + s[i+1:]
        permute(remaining, prefix + ch)
```

### 11. Efficiency of Anagram Generation
- If the string has n characters, the number of anagrams is:
- n!
- So:
    - For n = 3, there are 6 permutations
    - For n = 5, there are 120 permutations
    - For n = 10, there are 3,628,800 permutations
- This means:
    - Time complexity is O(n!)
    - The algorithm is inherently expensive because it must output every permutation
    - Recursion is a natural fit here because:
        - Each step fixes one character
        - Recursively permutes the rest
        - The structure of the problem is “take one, recurse on the remaining”
## Summary
    - Recursion is ideal for:
        - Repeated execution with a shrinking input
        - Problems defined in terms of smaller subproblems
        - Arbitrary depth structures
    - Key patterns:
        - Base case + recursive case
        - Process one part, recurse on the rest
        - Passing extra arguments to carry state
    - Classic examples:
        - Factorial
        - Array sum
        - String reversal
        - Counting occurrences
        - Staircase ways
        - Anagram generation (O(n!) complexity)




