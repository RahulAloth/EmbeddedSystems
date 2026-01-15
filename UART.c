/*
 * uart_driver.c
 *
 * Interrupt‑driven UART driver with RX/TX ring buffers.
 * Hardware‑agnostic: you must implement the low‑level UART_* HAL hooks.
 *
 * Features:
 *  - Non‑blocking transmit and receive
 *  - Interrupt‑driven RX and TX
 *  - Simple ring buffers
 */

#include <stdint.h>
#include <stdbool.h>

/* -------------------------------------------------------------------------
 *  Configuration
 * ------------------------------------------------------------------------- */

#define UART_RX_BUFFER_SIZE   128
#define UART_TX_BUFFER_SIZE   128

/* -------------------------------------------------------------------------
 *  Ring buffer
 * ------------------------------------------------------------------------- */

typedef struct {
    volatile uint8_t buf[UART_RX_BUFFER_SIZE];
    volatile uint16_t head;
    volatile uint16_t tail;
} ringbuf_rx_t;

typedef struct {
    volatile uint8_t buf[UART_TX_BUFFER_SIZE];
    volatile uint16_t head;
    volatile uint16_t tail;
} ringbuf_tx_t;

static ringbuf_rx_t uart_rx;
static ringbuf_tx_t uart_tx;

/* Helpers */
static bool ringbuf_rx_is_empty(void)
{
    return uart_rx.head == uart_rx.tail;
}

static bool ringbuf_rx_is_full(void)
{
    return ((uart_rx.head + 1) % UART_RX_BUFFER_SIZE) == uart_rx.tail;
}

static bool ringbuf_tx_is_empty(void)
{
    return uart_tx.head == uart_tx.tail;
}

static bool ringbuf_tx_is_full(void)
{
    return ((uart_tx.head + 1) % UART_TX_BUFFER_SIZE) == uart_tx.tail;
}

/* -------------------------------------------------------------------------
 *  Low‑level hardware hooks (to be implemented per MCU)
 * ------------------------------------------------------------------------- */

/* Initialize UART peripheral: baud, frame format, enable IRQs, etc. */
void UART_HW_Init(uint32_t baud);

/* Enable/disable TX empty interrupt */
void UART_HW_EnableTxInterrupt(void);
void UART_HW_DisableTxInterrupt(void);

/* Enable/disable RX interrupt */
void UART_HW_EnableRxInterrupt(void);
void UART_HW_DisableRxInterrupt(void);

/* Check hardware flags */
bool UART_HW_IsTxEmpty(void);
bool UART_HW_IsRxNotEmpty(void);

/* Read/write data register */
void UART_HW_WriteDR(uint8_t data);
uint8_t UART_HW_ReadDR(void);

/* -------------------------------------------------------------------------
 *  Public API
 * ------------------------------------------------------------------------- */

void UART_Init(uint32_t baud)
{
    uart_rx.head = uart_rx.tail = 0;
    uart_tx.head = uart_tx.tail = 0;

    UART_HW_Init(baud);
    UART_HW_EnableRxInterrupt();
}

/* Non‑blocking: enqueue a byte for transmission */
bool UART_WriteByte(uint8_t data)
{
    if (ringbuf_tx_is_full())
        return false;

    uart_tx.buf[uart_tx.head] = data;
    uart_tx.head = (uart_tx.head + 1) % UART_TX_BUFFER_SIZE;

    UART_HW_EnableTxInterrupt();
    return true;
}

/* Blocking: send a byte (wait until enqueued) */
void UART_WriteByteBlocking(uint8_t data)
{
    while (!UART_WriteByte(data)) {
        /* wait or yield */
    }
}

/* Non‑blocking: read a byte if available */
bool UART_ReadByte(uint8_t *out)
{
    if (!out || ringbuf_rx_is_empty())
        return false;

    *out = uart_rx.buf[uart_rx.tail];
    uart_rx.tail = (uart_rx.tail + 1) % UART_RX_BUFFER_SIZE;
    return true;
}

/* Blocking: wait for a byte */
uint8_t UART_ReadByteBlocking(void)
{
    uint8_t ch;
    while (!UART_ReadByte(&ch)) {
        /* wait or yield */
    }
    return ch;
}

/* Convenience: write string */
void UART_WriteString(const char *s)
{
    while (*s) {
        UART_WriteByteBlocking((uint8_t)*s++);
    }
}

/* -------------------------------------------------------------------------
 *  Interrupt Service Routine (call from actual UART IRQ handler)
 * ------------------------------------------------------------------------- */

void UART_IRQHandler(void)
{
    /* RX: data received */
    if (UART_HW_IsRxNotEmpty()) {
        uint8_t data = UART_HW_ReadDR();

        if (!ringbuf_rx_is_full()) {
            uart_rx.buf[uart_rx.head] = data;
            uart_rx.head = (uart_rx.head + 1) % UART_RX_BUFFER_SIZE;
        }
        /* else: overflow, byte is dropped (could add error flag) */
    }

    /* TX: transmit buffer empty, send next byte if any */
    if (UART_HW_IsTxEmpty()) {
        if (!ringbuf_tx_is_empty()) {
            uint8_t data = uart_tx.buf[uart_tx.tail];
            uart_tx.tail = (uart_tx.tail + 1) % UART_TX_BUFFER_SIZE;
            UART_HW_WriteDR(data);
        } else {
            /* Nothing left to send, disable TX interrupt */
            UART_HW_DisableTxInterrupt();
        }
    }
}
