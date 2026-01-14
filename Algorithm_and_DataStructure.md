# What Is an Algorithm?

An **algorithm** is simply a set of instructions for completing a specific task.  
It doesn’t have to be related to computers—any step‑by‑step process can be called an algorithm.

A fun example is preparing a bowl of cereal.  
This everyday activity can be written as an algorithm:

### Example: Cereal Preparation Algorithm

1. Grab a bowl  
2. Pour cereal into the bowl  
3. Pour milk into the bowl  
4. Dip the spoon into the bowl  
5. Eat the cereal  

Even simple tasks follow a sequence of steps.  
That’s all an algorithm really is: **a clear, ordered list of actions to achieve a goal**.


## Arrays and Measuring Speed

Writing efficient code requires understanding how data structures behave. Arrays are one of the most fundamental structures, and they give us a clear way to think about performance.

### What Is an Array?
An **array** is a collection of elements stored in a continuous block of memory.  
Each element can be accessed directly using its **index**, which makes some operations extremely fast.

Example:
Index:   0   1   2   3
Value:  10  20  30  40


---

## Measuring Speed: Big O Intuition
When we talk about “speed,” we don’t measure time in seconds.  
Instead, we measure **how the number of steps grows** as the data grows.

- **O(1)** → constant time (fast, does not depend on size)
- **O(n)** → linear time (slower, grows with number of elements)

Arrays give us a mix of both.

---

## Reading from an Array — **O(1)**
Reading an element by index is extremely fast.

```cpp
int x = arr[5];
- Because the computer can jump directly to the memory location.
- No searching, no scanning — just simple arithmetic.
```

## Searching in an Array — O(n)

If you want to find a value (not by index), the array must be scanned element by element.
```
for (int i = 0; i < n; i++) {
    if (arr[i] == target) { ... }
}
```
- Worst case: the value is at the end or not present at all.
- So searching grows linearly with the size of the array.
## Insertion in an Array

Insertion depends on where you insert.
### 1. Insert at the end — O(1) (amortized)
- If there is space, adding at the end is fast.
```
arr.push_back(50);
```
### 2. Insert at the beginning or middle — O(n)
- All elements must shift to make room.
- Example: inserting at index 0
```
Before: [10, 20, 30, 40]
Insert 5 at front
After:  [5, 10, 20, 30, 40]
```
## Deletion in an Array
- Just like insertion, deletion depends on position.
###  1. Delete from the end — O(1)
- No shifting required.
###  2. Delete from the beginning or middle — O(n)
- Elements must shift left to fill the gap.
Example: deleting index 1
```
Before: [10, 20, 30, 40]
After:  [10, 30, 40]
```
- Again, shifting → linear time.

## Summary Table

| Operation               | Speed          | Reason                     |
|-------------------------|----------------|-----------------------------|
| Read by index           | O(1)           | Direct memory access        |
| Search by value         | O(n)           | Must scan each element      |
| Insert at end           | O(1) amortized | Append without shifting     |
| Insert at front/middle  | O(n)           | Shift elements              |
| Delete at end           | O(1)           | Remove last element         |
| Delete at front/middle  | O(n)           | Shift elements              |



## Sets and How a Single Rule Affects Efficiency

A **set** is a data structure that stores values *without allowing duplicates*.  
This one rule — *no repeated values* — has a major impact on performance.

When we implement a set using an **array**, we get what is called an **array‑based set**.  
It behaves like a normal array, but with one additional constraint:

> **Before inserting a value, we must check whether it already exists.**

This extra step changes the efficiency of several operations.

---

## Why the “No Duplicates” Rule Matters

In a normal array:
- Inserting at the end is **O(1)** (fast)
- You don’t need to check anything before inserting

But in an **array‑based set**:
- You must **search the entire array** to ensure the value is not already present
- Searching is **O(n)**

So even if insertion itself is O(1),  
**the duplicate check makes insertion O(n)**.

This is the key idea:  
### A single rule can change the efficiency of an entire data structure.

---

## Operations in an Array‑Based Set

### 1. Reading — **O(1)**
Reading by index is still constant time, just like a normal array.

```c
int x = set[3];
```
No change here.
### 2. Searching — O(n)
To check if a value exists, we must scan the array.
```
for (int i = 0; i < size; i++) {
    if (set[i] == value) { ... }
}

```
This is identical to array searching.
### 3. Insertion — O(n)
- This is where the big change happens.
```
Steps:
    - Search entire array to ensure no duplicates → O(n)
    - If not found, insert at the end → O(1)
- Total = O(n) + O(1) = O(n)
```
- Even though insertion is normally fast,
- the duplicate check dominates the cost.
### 4. Deletion — O(n)
- Deletion works the same as arrays:
```
- Find the element → O(n)
- Shift elements left → O(n)
Total worst case: O(n)
```
- So in short, A single rule can completely change the efficiency of a data structure.


## Ordered Arrays

An **ordered array** is an array in which the elements are always kept in sorted order.  
This single rule — *the array must remain sorted* — changes the efficiency of several operations.

Keeping the array sorted gives us one major advantage:

> We can use **binary search**, which is much faster than linear search.

But it also introduces a disadvantage:

> Inserting new values becomes slower because we must place them in the correct position.

Let’s explore these effects in detail.

---

## Searching an Ordered Array

### Linear Search — **O(n)**  
Even though the array is sorted, we can still use a simple linear scan:

```c
for (int i = 0; i < size; i++) {
    if (arr[i] == target) return i;
    if (arr[i] > target) break;   // early stop because array is sorted
}
```
- The early stop helps, but in the worst case, it’s still O(n).

### Binary Search — O(log n)
- Binary search takes full advantage of the sorted order.
```
Steps:
    - Look at the middle element
    - If the target is smaller → search the left half
    - If the target is larger → search the right half
    - Repeat until found or the range becomes empty
```

- Each step cuts the search space in half.
- This gives binary search its famous efficiency:
- 1,000,000 elements → only ~20 steps
- 1,000 elements → only ~10 steps
- This is dramatically faster than linear search.

## Summary Table

| Operation       | Speed      | Reason                               |
|-----------------|------------|---------------------------------------|
| Read by index   | O(1)       | Direct memory access                  |
| Linear search   | O(n)       | Scan elements one by one              |
| Binary search   | O(log n)   | Halves the search space each step     |
| Insert          | O(n)       | Must shift elements to maintain order |
| Delete          | O(n)       | Must shift elements after removal     |

### Excercise:
- Solution is present in c code.
- 1. Given the ordered array:
    - 10, 20, 30, 40, 50, 60  
    - Show the steps of binary search when searching for 50.

- 2. Insert the value 35 into the ordered array above.
    - Show the shifting process.

- 3. Delete the value 20 from the array.
    - Show the shifting process.
-4. Compare the number of steps for linear search vs binary search
    - when searching for 60 in an array of size 100.


## Binary Search vs Linear Search (Conceptual Graph)

Below is a conceptual graph showing how the number of steps grows as the input size increases.
```
Steps
^
|                     Linear Search (O(n))
|                     *
|                   *
|                 *
|               *
|             *
|           *
|         *
|       *
|     *
|   *
| *
|-------------------------------------------------->  n
* Binary Search (O(log n))
*
*
*
*
*
```

### Interpretation
- **Linear Search** grows steadily as the array gets larger.  
  More elements → more steps → straight upward line.

- **Binary Search** grows very slowly.  
  Even huge arrays require only a few steps → curve that flattens quickly.

### Key Insight
Binary search becomes dramatically more efficient as data size increases,  
which is why **sorted arrays** are so powerful for fast lookup.


# Algorithm Efficiency and Big O Notation

Algorithm efficiency describes how the performance of an algorithm changes as the size of the input grows. Instead of using vague descriptions like “fast” or “slow,” computer scientists use a precise mathematical language called **Big O Notation**.

Big O gives us a consistent way to categorize and communicate the efficiency of algorithms and data structures, independent of hardware or programming language.

---

## Why Big O Notation Matters

Big O focuses on how an algorithm *scales* as the input size \(N\) becomes very large.  
It describes the **upper bound** of growth, ignoring constants and small variations.

Common Big O categories include:

- **O(1)** — Constant time  
- **O(N)** — Linear time  
- **O(N²)** — Quadratic time  
- **O(log N)** — Logarithmic time  
- **O(N log N)** — Linearithmic time  

Each category represents a different growth pattern.

---

## Logarithmic Time — The “Third Kind”

Logarithms have nothing to do with algorithms, even though the words sound similar.  
But logarithms appear naturally in many efficient algorithms.

### What Is a Logarithm?

A logarithm answers the question:

**How many times can you divide something in half until only 1 remains?**

This is why logarithmic time appears in algorithms that repeatedly cut the problem size down.

### Big O and Logarithms

When we write
```
O(log N)
```
it is shorthand for:
```
O(log₂ N)
```

The base of the logarithm does not matter in Big O notation because changing the base only multiplies by a constant, and Big O ignores constants.

So:

- log₂ N  
- log₁₀ N  
- ln N  

all simplify to **O(log N)** in Big O notation.

---

## Examples of O(log N) Algorithms

- **Binary search**  
- **Balanced binary search trees** (AVL, Red‑Black Trees)  
- **Heaps** (insert, delete-min)  
- **Divide-and-conquer steps** inside algorithms like merge sort  

These algorithms are efficient because they reduce the problem size very quickly.

---

## A Simple Analogy

Imagine a book with 1,000 pages.  
Instead of checking every page, you open the book in the middle:

- If the page is too high, go to the left half.  
- If it’s too low, go to the right half.  

Each step cuts the remaining pages in half.  
That’s **O(log N)** behavior.

---



