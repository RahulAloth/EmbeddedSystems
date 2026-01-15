/*
 * bitwise_register_guide.c
 *
 * A compact playground for:
 *  - Bitwise operators
 *  - Masks and shifts
 *  - Memory-mapped registers
 *  - Struct-based register maps
 *  - Simple GPIO-style driver functions
 *
 * This is meant for study, not for a specific MCU.
 */

#include <stdint.h>
#include <stdio.h>

/* ------------------------------------------------------------------------- */
/* 1. Bitwise operators: basic examples                                      */
/* ------------------------------------------------------------------------- */

void bitwise_basic_examples(void)
{
    uint32_t REG = 0x00000000;

    /* Set bit 3 */
    REG |= (1U << 3);      /* REG = 0x00000008 */

    /* Clear bit 3 */
    REG &= ~(1U << 3);     /* REG = 0x00000000 */

    /* Toggle bit 3 */
    REG ^= (1U << 3);      /* REG = 0x00000008 */

    /* Print to observe behavior (for hosted environments) */
    printf("bitwise_basic_examples: REG = 0x%08lX\n", (unsigned long)REG);
}

/* ------------------------------------------------------------------------- */
/* 2. Masks and fields                                                       */
/* ------------------------------------------------------------------------- */

/* Example: 5-bit field at bit position 20 */
#define FIELD_SHIFT     20U
#define FIELD_WIDTH     5U
#define FIELD_MASK      (((1U << FIELD_WIDTH) - 1U) << FIELD_SHIFT)

/* Example values for the field */
#define FIELD_VALUE_PUSH_PULL   (0x10U)  /* 0b10000 */
#define FIELD_VALUE_ALT_FUNC    (0x02U)  /* 0b00010 */

void mask_and_field_examples(void)
{
    uint32_t REG = 0xFFFFFFFF;

    /* Clear the 5-bit field at bits 20–24 */
    REG &= ~FIELD_MASK;

    /* Write push-pull value into that field */
    REG |= (FIELD_VALUE_PUSH_PULL << FIELD_SHIFT);

    printf("mask_and_field_examples: REG = 0x%08lX\n", (unsigned long)REG);

    /* Read back the field */
    uint32_t field = (REG >> FIELD_SHIFT) & ((1U << FIELD_WIDTH) - 1U);
    printf("  Extracted field = 0x%lX\n", (unsigned long)field);
}

/* ------------------------------------------------------------------------- */
/* 3. Memory-mapped register style example                                   */
/* ------------------------------------------------------------------------- */

/*
 * In a real MCU, this would be something like:
 *   #define GPIO_MODE (*(volatile uint32_t*)0x40020000)
 *
 * Here we simulate it with a global variable.
 */

volatile uint32_t SIM_GPIO_MODE = 0x00000000;

#define GPIO_MODE   (SIM_GPIO_MODE)

void memory_mapped_register_example(void)
{
    /* Clear bits 20–24 */
    GPIO_MODE &= ~FIELD_MASK;

    /* Set them to push-pull (0b10000) */
    GPIO_MODE |= (FIELD_VALUE_PUSH_PULL << FIELD_SHIFT);

    printf("memory_mapped_register_example: GPIO_MODE = 0x%08lX\n",
           (unsigned long)GPIO_MODE);
}

/* ------------------------------------------------------------------------- */
/* 4. Struct-based register map (CMSIS style)                                */
/* ------------------------------------------------------------------------- */

typedef struct
{
    volatile uint32_t MODER;   /* mode register */
    volatile uint32_t OTYPER;  /* output type register */
    volatile uint32_t OSPEEDR; /* speed register */
    volatile uint32_t PUPDR;   /* pull-up/pull-down register */
    volatile uint32_t IDR;     /* input data register */
    volatile uint32_t ODR;     /* output data register */
} GPIO_TypeDef;

/* Simulated GPIOA instance */
GPIO_TypeDef SIM_GPIOA = {0};
#define GPIOA   (&SIM_GPIOA)

/*
 * For many MCUs (e.g., STM32), each pin mode is 2 bits in MODER:
 *   00 = input
 *   01 = output
 *   10 = alternate function
 *   11 = analog
 */

void gpio_set_mode(GPIO_TypeDef *port, uint8_t pin, uint8_t mode)
{
    /* Clear the 2-bit field for this pin */
    port->MODER &= ~(3U << (pin * 2U));

    /* Write the new mode */
    port->MODER |= ((uint32_t)mode << (pin * 2U));
}

void gpio_write(GPIO_TypeDef *port, uint8_t pin, uint8_t value)
{
    if (value)
    {
        port->ODR |= (1U << pin);
    }
    else
    {
        port->ODR &= ~(1U << pin);
    }
}

void gpio_toggle(GPIO_TypeDef *port, uint8_t pin)
{
    port->ODR ^= (1U << pin);
}

uint8_t gpio_read(GPIO_TypeDef *port, uint8_t pin)
{
    return (uint8_t)((port->IDR >> pin) & 0x1U);
}

void struct_based_register_example(void)
{
    /* Configure pin 5 as output (01) */
    gpio_set_mode(GPIOA, 5U, 0x1U);

    /* Write pin 5 high */
    gpio_write(GPIOA, 5U, 1U);

    /* Toggle pin 5 */
    gpio_toggle(GPIOA, 5U);

    printf("struct_based_register_example:\n");
    printf("  MODER = 0x%08lX\n", (unsigned long)GPIOA->MODER);
    printf("  ODR   = 0x%08lX\n", (unsigned long)GPIOA->ODR);
}

/* ------------------------------------------------------------------------- */
/* 5. Inline assembly (optional, illustrative only)                          */
/* ------------------------------------------------------------------------- */

static inline void do_nop(void)
{
#if defined(__GNUC__) || defined(__clang__)
    __asm__ volatile ("nop");
#endif
}

void inline_assembly_example(void)
{
    /* Just execute a NOP instruction (if supported by the toolchain) */
    do_nop();
    printf("inline_assembly_example: executed NOP (if supported).\n");
}

/* ------------------------------------------------------------------------- */
/* main(): run all examples                                                  */
/* ------------------------------------------------------------------------- */

int main(void)
{
    bitwise_basic_examples();
    mask_and_field_examples();
    memory_mapped_register_example();
    struct_based_register_example();
    inline_assembly_example();

    return 0;
}
