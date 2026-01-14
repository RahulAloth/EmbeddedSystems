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






