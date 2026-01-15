# Two New Data Structures: Stacks and Queues

## Introduction
Stacks and queues are not entirely new data structures. They are simply arrays with specific restrictions. These restrictions are what make them elegant and powerful. From operating system architecture to printing jobs and data traversal, stacks and queues act as temporary containers that help form efficient algorithms.

Think of temporary data like food orders in a restaurant. The order matters only until the meal is prepared and delivered. After that, the slip is discarded. Stacks and queues handle this kind of temporary data, with a special focus on the order in which data is processed.

---

# Stacks (LIFO — Last In, First Out)

A stack stores data like an array, but with three constraints:

- Data can be inserted only at the end of the stack.
- Data can be deleted only from the end of the stack.
- Only the last element of the stack can be read.

This is similar to a stack of dishes: you add and remove dishes only from the top.

## Abstract Stack (Pseudocode)

```ruby
class Stack
  def initialize
    @data = []
  end

  def push(element)
    @data << element
  end

  def pop
    @data.pop
  end

  def read
    @data.last
  end
end
```

## Stack in Action

- Stacks are excellent for handling temporary data in algorithms.
- A common example is a stack-based code linter that checks matching brackets.
- Importance of Constrained Data Structures
If a stack is just a restricted array, why use it?
    - It prevents potential bugs.
    - It enforces a LIFO order.
    - It makes code cleaner and more understandable.
    - It helps solve problems like undo operations in word processors.

## Queues (FIFO — First In, First Out)
- A queue is another structure for temporary data. It is similar to a stack but processes data in a different order.
- A queue is often depicted horizontally, with:
    - Front → where elements are removed
    - Back → where elements are added
# Queue — Concept and Pseudocode

A **Queue** is a linear data structure that follows the rule:

**FIFO — First In, First Out**

This means:
- The first element added is the first one removed  
- Like people standing in a line at a ticket counter  

Queues are used everywhere:
- Printer job scheduling  
- CPU task scheduling  
- Network packets  
- Breadth‑first search (BFS)  
- Real‑world waiting lines  

---

## How a Queue Works

A queue supports two main operations:

### **1. Enqueue (Insert)**
Add an element to the **back** of the queue.

### **2. Dequeue (Remove)**
Remove an element from the **front** of the queue.

Additional helpful operations:

- **Front / Peek** → Look at the first element without removing it  
- **IsEmpty** → Check if the queue has no elements  
- **IsFull** → For fixed‑size queues  

---


---

# Summary

- A **Queue** uses FIFO order  
- Enqueue adds to the **rear**  
- Dequeue removes from the **front**  
- Circular indexing avoids wasted space  
- Perfect for scheduling, buffering, and BFS  

---

```ruby
Queue:
    initialize:
        data = empty list
        front = 0
        back = -1

    enqueue(element):
        back = back + 1
        data[back] = element

    dequeue():
        if front > back:
            return "Queue Underflow"
        value = data[front]
        front = front + 1
        return value

    read():
        if front > back:
            return "Queue is empty"
        return data[front]
```

### Queue Rules
- Queues, like stacks, are arrays with restrictions:
    - Data can be inserted only at the back of the queue.
    - Data can be deleted only from the front of the queue.
    - Only the element at the front can be read.
- This behavior is the opposite of a stack.
### Queue in Action
- Queues are widely used in:
    - Printing jobs
    - Background workers in web applications
    - Task scheduling
    - Network packet processing

## Wrapping Up
- Stacks and queues are foundational data structures.
- Understanding them prepares you for recursion, which relies on a stack, and for many advanced algorithms that build on these concepts.
  




