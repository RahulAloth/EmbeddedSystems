/*
 * i2c_driver.c
 *
 * A clean, hardware‑agnostic I²C master driver.
 * Replace the low‑level HAL functions with your MCU's register operations.
 *
 * Supports:
 *   - START / STOP
 *   - Repeated START
 *   - ACK / NACK
 *   - Byte write
 *   - Byte read
 *   - Register read/write
 *
 * Author: Rahul’s Embedded Systems Library
 */

#include <stdint.h>
#include <stdbool.h>

/* -------------------------------------------------------------------------
 *  Low‑level hardware hooks (replace with your MCU's I2C registers)
 * ------------------------------------------------------------------------- */

static void i2c_hw_start(void);
static void i2c_hw_stop(void);
static bool i2c_hw_send_byte(uint8_t byte);
static uint8_t i2c_hw_read_byte(bool ack);

/*
 * NOTE:
 * These functions must be implemented for your specific MCU.
 * For example, on STM32 you would use:
 *   I2C1->CR1 |= I2C_CR1_START;
 *   while (!(I2C1->SR1 & I2C_SR1_SB));
 *   ...
 */

/* -------------------------------------------------------------------------
 *  High‑level I2C API
 * ------------------------------------------------------------------------- */

/* Send START + address + R/W bit */
static bool i2c_send_address(uint8_t addr, bool read)
{
    i2c_hw_start();

    uint8_t byte = (addr << 1) | (read ? 1 : 0);
    return i2c_hw_send_byte(byte);
}

/* Write a single byte to a device register */
bool i2c_write_reg(uint8_t dev_addr, uint8_t reg, uint8_t data)
{
    if (!i2c_send_address(dev_addr, false))
        return false;

    if (!i2c_hw_send_byte(reg))
        return false;

    if (!i2c_hw_send_byte(data))
        return false;

    i2c_hw_stop();
    return true;
}

/* Read a single byte from a device register */
bool i2c_read_reg(uint8_t dev_addr, uint8_t reg, uint8_t *out)
{
    if (!out)
        return false;

    /* Write register address */
    if (!i2c_send_address(dev_addr, false))
        return false;

    if (!i2c_hw_send_byte(reg))
        return false;

    /* Repeated START */
    if (!i2c_send_address(dev_addr, true))
        return false;

    *out = i2c_hw_read_byte(false);  // NACK after last byte
    i2c_hw_stop();

    return true;
}

/* Write multiple bytes */
bool i2c_write_bytes(uint8_t dev_addr, uint8_t reg, const uint8_t *data, uint16_t len)
{
    if (!i2c_send_address(dev_addr, false))
        return false;

    if (!i2c_hw_send_byte(reg))
        return false;

    for (uint16_t i = 0; i < len; i++)
    {
        if (!i2c_hw_send_byte(data[i]))
            return false;
    }

    i2c_hw_stop();
    return true;
}

/* Read multiple bytes */
bool i2c_read_bytes(uint8_t dev_addr, uint8_t reg, uint8_t *buf, uint16_t len)
{
    if (!buf || len == 0)
        return false;

    /* Write register address */
    if (!i2c_send_address(dev_addr, false))
        return false;

    if (!i2c_hw_send_byte(reg))
        return false;

    /* Repeated START */
    if (!i2c_send_address(dev_addr, true))
        return false;

    for (uint16_t i = 0; i < len; i++)
    {
        bool ack = (i < (len - 1));
        buf[i] = i2c_hw_read_byte(ack);
    }

    i2c_hw_stop();
    return true;
}

/* -------------------------------------------------------------------------
 *  Example low‑level implementation (pseudo‑code)
 *  Replace with your MCU's I2C registers
 * ------------------------------------------------------------------------- */

static void i2c_hw_start(void)
{
    /* Example pseudo‑code:
     * I2C->CR1 |= START;
     * while (!(I2C->SR1 & SB));
     */
}

static void i2c_hw_stop(void)
{
    /* Example pseudo‑code:
     * I2C->CR1 |= STOP;
     */
}

static bool i2c_hw_send_byte(uint8_t byte)
{
    /* Example pseudo‑code:
     * I2C->DR = byte;
     * while (!(I2C->SR1 & TXE));
     * return !(I2C->SR1 & NACKF);
     */
    return true;
}

static uint8_t i2c_hw_read_byte(bool ack)
{
    /* Example pseudo‑code:
     * if (ack) I2C->CR1 |= ACK;
     * else     I2C->CR1 &= ~ACK;
     *
     * while (!(I2C->SR1 & RXNE));
     * return I2C->DR;
     */
    return 0xFF;
}
