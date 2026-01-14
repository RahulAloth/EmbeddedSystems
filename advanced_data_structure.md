# Blazing Fast Lookup with Hash Tables  
### A Complete, Ordered Explanation

Hash tables are one of the most important data structures in computer science.  
They give programs the ability to look up information **instantly**, without scanning through long lists.

This chapter explains:

1. What a hash table is  
2. Why it is so fast  
3. How hash functions work  
4. How collisions happen  
5. How collisions are handled  
6. How to build an efficient hash table  
7. How hash tables help with organisation and speed  
8. How they compare to arrays of sub‑arrays (your fast‑food example)

---

# 1. What Is a Hash Table?

A **hash table** is a data structure that stores **key–value pairs**.

Examples of key–value pairs:
'''
- `"burger"` → `5`
- `"fries"` → `2`
- `"pizza"` → `8`
'''

A hash table lets you find the value **instantly** if you know the key.

Other names for hash tables:

- **hashes**
- **maps**
- **hash maps**
- **dictionaries**
- **associative arrays**

All of these mean the same thing:  
A structure that maps **keys** to **values** using a **hash function**.

---

# 2. Why Hash Tables Are Blazing Fast

Most data structures require scanning through items:

- Arrays → O(N)
- Linked lists → O(N)
- Trees → O(log N)

But hash tables give **average O(1)** lookup time.

This means:

> No matter how big the table gets, lookup time stays almost constant.

This is why hash tables are used everywhere:

- programming languages  
- databases  
- compilers  
- caches  
- operating systems  

---

# 3. Hashing with Hash Functions

A **hash function** converts a key (like `"burger"`) into a number.

Example idea:

'''
index = hash("burger") % table_size
'''

A good hash function must be:

- **Deterministic** → same key always gives same hash  
- **Fast** → quick to compute  
- **Uniform** → spreads keys evenly  
- **Low collision** → avoids clustering  

The hash function decides **where** in the internal array the value is stored.

---

# 4. Hash Table Lookups

To look up a value:

1. Take the key  
2. Hash it  
3. Jump directly to the index  
4. Read the value  

This is a **one-directional lookup**:

- You know the **key** → you want the **value**

Hash tables are not designed to search by value.

---

# 5. Dealing with Collisions

A **collision** happens when two different keys hash to the **same index**.

Example:
```cpp
hash("burger") % 10 → 3
hash("pizza") % 10 → 3
```

Both want to live at index 3.

Collisions are unavoidable because:

- There are infinite possible keys  
- But only a fixed number of array slots  

### Collision Handling Methods

#### 1. Separate Chaining  
Each array slot holds a **list** of key–value pairs.

```
index 3 → [ ("burger",5), ("pizza",8) ]
```

#### 2. Open Addressing  
If a slot is full, find another slot:

- linear probing  
- quadratic probing  
- double hashing  

Both methods keep hash tables fast even with collisions.

---

# 6. Making an Efficient Hash Table  
### The Great Balancing Act

A good hash table must balance:

- **Table size**  
- **Load factor** (how full the table is)  
- **Hash function quality**  
- **Collision strategy**  

If the table gets too full, performance drops.  
So many hash tables **resize** and **rehash** automatically.

---

# 7. Hash Tables for Organisation

Hash tables are perfect for organising information by **name**, **ID**, or **label**.

Examples:
```
- usernames → user profiles  
- product IDs → product details  
- words → definitions  
- food names → prices  
```
They give you a mental model like:
```
> “If I know the name, I can get the information instantly.”
```
---

# 8. Hash Tables for Speed

Hash tables provide:

- **Insert:** O(1) average  
- **Search:** O(1) average  
- **Delete:** O(1) average  

This is why they are considered **blazing fast**.

Worst case can degrade to **O(N)**,  
but with good hashing and resizing, this is rare.

---

# 9. Array Subsets vs Hash Tables  
### (Your Fast-Food Example)

You originally had:
```
[
  ["burger", 5],
  ["fries", 2],
  ["pizza", 8],
  ["cola", 1]
]
```
This is an array of sub‑arrays.
```
To find "pizza":
    You scan each sub‑array
    Compare the first element
    Stop when you find "pizza"
```

This is O(N) time.
With a Hash Table
```
"burger" → 5
"fries"  → 2
"pizza"  → 8
"cola"   → 1
```
### Summary
    - Hash tables store key–value pairs
    - They use hash functions to map keys to array indices
    - They provide blazing fast O(1) average lookup
    - Collisions are normal and handled with chaining or probing
    - Efficiency depends on load factor, hash function, and resizing
    - Hash tables are ideal for organisation and speed
    - They outperform arrays of sub‑arrays for lookups

