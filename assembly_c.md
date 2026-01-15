# The Assembly Language of C  
### A Complete Study Guide for Low‑Level Embedded Programming

---

## 1. Binary, Hex, and Bit Positions

### Key Concepts
- Binary and hexadecimal representations
- Bit numbering (LSB = bit 0)
- Endianness (conceptual understanding only)

### Skills to Master
- Convert between binary ↔ hex
- Identify which bit is set in a hex value
- Quickly interpret bit patterns

### Practice
- Write binary for numbers 0–31
- Identify the set bit in: `0x20`, `0x8000`, `0x04`

---

## 2. Bitwise Operators in C

| Operator | Meaning | Typical Use |
|---------|---------|-------------|
| `&` | AND | Clear bits |
| `|` | OR | Set bits |
| `^` | XOR | Toggle bits |
| `~` | NOT | Invert masks |
| `<<` | Left shift | Position values |
| `>>` | Right shift | Extract values |

### Examples
```c
REG &= ~(1U << 3);   // clear bit 3
REG |=  (1U << 3);   // set bit 3
REG ^=  (1U << 3);   // toggle bit 3
```

## 3. Masks
-  Single‑Bit Mask
  - (1U << n)
- Multi‑Bit Field Mask
  - ((1U << width) - 1) << shift
  - Example: 5‑bit field at bit 20
  - #define FIELD_MASK   (0x1F << 20)

## 4. Memory‑Mapped Registers

- Concept
- A register is a fixed memory address mapped to hardware.
- #define GPIO_MODE   (*(volatile uint32_t*)0x40020000)
- Why volatile?
    - Prevents compiler optimizations
    - Ensures every read/write touches hardware
      
# 5. Universal Register Configuration Pattern
- Step 1 — Clear the field
  - REG &= ~FIELD_MASK;
- Step 2 — Write the new value
  - REG |= (VALUE << SHIFT);
  - Example: Configure GPIO pin mode
  - REG &= ~(0x1F << 20);      // clear bits 20–24
  - REG |=  (0x10 << 20);      // write 0b10000 (push‑pull output)
    
# 6. Reading Fields (Extracting Values)
- uint32_t val = (REG >> SHIFT) & FIELD_MASK;
- Example: Read 3‑bit field at bit 8
- uint32_t mode = (REG >> 8) & 0x7;
# 7. Struct‑Based Register Maps (CMSIS Style)
```
typedef struct {
    volatile uint32_t MODER;
    volatile uint32_t OTYPER;
    volatile uint32_t OSPEEDR;
} GPIO_TypeDef;

#define GPIOA ((GPIO_TypeDef*)0x40020000)

GPIOA->MODER &= ~(3U << (2 * pin));
GPIOA->MODER |=  (1U << (2 * pin));   // set pin as output

```
# 8. Inline Assembly (Optional)
```
 __asm__ volatile ("nop");
```
- Useful for
    - Reading special registers
    - Critical timing
    -  Debugging
# 9. Mini GPIO Driver (Putting It All Together)
- Set Mode
```
void gpio_set_mode(GPIO_TypeDef *port, uint8_t pin, uint8_t mode) {
    port->MODER &= ~(3U << (pin * 2));
    port->MODER |=  (mode << (pin * 2));
}
```
Write pin
```
void gpio_write(GPIO_TypeDef *port, uint8_t pin, uint8_t value) {
    if (value)
        port->ODR |=  (1U << pin);
    else
        port->ODR &= ~(1U << pin);
}
```
Toggle pin
```
void gpio_toggle(GPIO_TypeDef *port, uint8_t pin) {
    port->ODR ^= (1U << pin);
}
```







