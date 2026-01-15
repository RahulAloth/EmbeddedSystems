# Linked Lists – Complete Notes  
## Linked Lists, Doubly Linked Lists, and Queues as DLL

---

# 1. Linked Lists

A **linked list** is a dynamic, node-based data structure where each node contains:

- `data` — the stored value  
- `next` — a pointer to the next node  
```
head
↓
[data | next] → [data | next] → [data | next] → NULL
```

## 1.1 Why Linked Lists?

- Dynamic size  
- No need for contiguous memory  
- Fast insertions/deletions at known positions  
- Ideal for queues, stacks, and adjacency lists  
- No shifting of elements like arrays  

---

# 2. Implementing Linked Lists

## 2.1 Node Structure (Conceptual)

A node typically contains:

- The data  
- A pointer to the next node  

You maintain a `head` pointer to track the start of the list.

---

# 3. Linked List Search

To search for a value:

1. Start at `head`  
2. Compare `current->data` with the target  
3. Move to `current->next`  
4. Stop when:  
   - Value is found  
   - Or `current == NULL`

**Time Complexity:** O(n)

Linked lists do not support random access like arrays.

---

# 4. Insertion in Linked Lists

Insertion depends on the position.

## 4.1 Insert at the Beginning (O(1))

1. Create a new node  
2. Point it to the current head  
3. Update head  

---

## 4.2 Insert at the End (O(n))

1. Traverse to the last node  
2. Link the new node at the end  

---

## 4.3 Insert After a Given Node (O(1) once node is known)

1. Link the new node to the next node  
2. Update the previous node’s pointer  

---

# 5. Deletion in Linked Lists

## 5.1 Delete from the Beginning (O(1))

1. Move head to the next node  
2. Free the old head  

---

## 5.2 Delete a Node by Value (O(n))

1. Traverse with `prev` and `current`  
2. Relink around the node  
3. Free the node  

---

# 6. Efficiency of Linked Lists

## Strengths

- Dynamic size  
- Fast insert/delete at head  
- No shifting elements  
- Good for queues, stacks, graphs  

## Weaknesses

- Slow search (O(n))  
- Extra memory for pointers  
- Poor cache locality  

---

# 7. Linked Lists in Action

Used in:

- Stacks  
- Queues  
- Graph adjacency lists  
- Undo/redo systems  
- Browser history  
- Music playlists  

Linked lists shine when you need **frequent insertions/deletions**.

---

# 8. Doubly Linked Lists (DLL)

A **doubly linked list** has two pointers:

- `next` → next node  
- `prev` → previous node  
```
NULL ← [prev | data | next] ⇄ [prev | data | next] ⇄ [prev | data | next] → NULL
```

## Advantages

- Traverse forward and backward  
- Delete a node in O(1) when pointer is known  
- Ideal for deques and queues  

## Disadvantages

- Extra memory for `prev`  
- More complex pointer updates  

---

# 9. Moving Forward and Backward

In a DLL:

- `node = node->next` moves forward  
- `node = node->prev` moves backward  

Used in:

- Browser back/forward  
- Music player next/previous  
- Text editor cursor movement  

---

# 10. Queues as Doubly Linked Lists

A **queue** is FIFO: First In, First Out.

To implement efficiently:

- **Enqueue at tail** → O(1)  
- **Dequeue from head** → O(1)  

A doubly linked list is ideal because:

- You maintain both `head` and `tail`  
- Insertions and deletions at both ends are O(1)  
- No shifting or resizing required  

This is why many queue implementations use DLLs internally.

---

# Summary

- Linked lists are dynamic, pointer-based structures  
- Doubly linked lists allow bidirectional traversal  
- Queues map naturally onto DLLs for efficient operations  
- Node-based structures are flexible and ideal for dynamic data  
