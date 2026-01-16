/*
 * spi_tc49xx.c
 *
 * Minimal blocking SPI master driver for AURIX TC49xx (QSPI).
 * - Single QSPI channel
 * - Single chip select (CS)
 * - Blocking transfer (polling)
 *
 * NOTE:
 *  - Replace QSPIx, PORTx, and bitfields with actual TC49xx definitions.
 *  - This is a conceptual template, not production code.
 */

#include "Ifx_Types.h"
#include "IfxPort.h"
#include "IfxScuWdt.h"
#include "IfxQspi_reg.h"      /* Replace with actual QSPI register header */
#include "IfxQspi_bf.h"       /* Bitfield definitions, if available */

/* ------------------------------------------------------------------------- */
/* Configuration                                                             */
/* ------------------------------------------------------------------------- */

#define SPI_QSPI_MODULE      QSPI0          /* Example: use QSPI0 */
#define SPI_QSPI_CHANNEL     0             /* Channel 0 */

#define SPI_BAUDRATE         10000000U     /* 10 MHz example */

/* Example CS pin (adapt to your board) */
#define SPI_CS_PORT          &MODULE_P10
#define SPI_CS_PIN           3

/* ------------------------------------------------------------------------- */
/* Low-level helpers                                                         */
/* ------------------------------------------------------------------------- */

static void spi_cs_low(void)
{
    IfxPort_setPinLow(SPI_CS_PORT, SPI_CS_PIN);
}

static void spi_cs_high(void)
{
    IfxPort_setPinHigh(SPI_CS_PORT, SPI_CS_PIN);
}

/* ------------------------------------------------------------------------- */
/* QSPI basic init (very simplified)                                         */
/* ------------------------------------------------------------------------- */

void spi_init(void)
{
    /* Enable QSPI module clock in SCU (pseudo-code) */
    /* IfxScuWdt_clearCpuEndinit(...); */
    /* MODULE_QSPI0.CLC.B.DISR = 0; */
    /* IfxScuWdt_setCpuEndinit(...); */
    /* while (MODULE_QSPI0.CLC.B.DISS != 0) {} */

    /* Configure QSPI pins (SCLK, MOSI, MISO) via port control:
       - Use IfxPort_setPinModeOutput / Input
       - Or iLLD QSPI pin configuration helpers
    */

    /* Put QSPI into master mode, configure baudrate, CPOL/CPHA, etc.
       This is conceptual; use real bitfields from TC49xx manual.
    */

    /* Example pseudo-code: */
    /* SPI_QSPI_MODULE.GLOBALCON.B.MS = 1;        // master mode */
    /* SPI_QSPI_MODULE.GLOBALCON.B.EXPECT = 0;    // no external trigger */
    /* SPI_QSPI_MODULE.GLOBALCON1.B.BAUD = ...;   // set baudrate divider */

    /* Enable QSPI channel, configure frame format, data length, etc. */
    /* SPI_QSPI_MODULE.CH[ SPI_QSPI_CHANNEL ].BACON.B.BYTE = ...; */

    /* Configure CS pin as GPIO output, default high (inactive) */
    IfxPort_setPinModeOutput(SPI_CS_PORT, SPI_CS_PIN, IfxPort_OutputMode_pushPull, IfxPort_OutputIdx_general);
    spi_cs_high();
}

/* ------------------------------------------------------------------------- */
/* Blocking transfer: send one byte, receive one byte                        */
/* ------------------------------------------------------------------------- */

uint8 spi_transfer(uint8 data)
{
    /* Wait until transmit FIFO has space */
    while (SPI_QSPI_MODULE.STATUS.B.TXFIFOLEVEL == /* full level */ 0xF)
    {
        /* wait */
    }

    /* Write data to transmit FIFO (TBUF) */
    SPI_QSPI_MODULE.DATAENTRY[ SPI_QSPI_CHANNEL ].U = data;

    /* Wait until receive FIFO has data */
    while (SPI_QSPI_MODULE.STATUS.B.RXFIFOLEVEL == 0)
    {
        /* wait */
    }

    /* Read received data */
    uint32 rx = SPI_QSPI_MODULE.RXEXIT.U;
    return (uint8)(rx & 0xFF);
}

/* ------------------------------------------------------------------------- */
/* High-level helpers: write/read register                                   */
/* ------------------------------------------------------------------------- */

void spi_write_reg(uint8 reg, uint8 value)
{
    spi_cs_low();

    (void)spi_transfer(reg);      /* send register address/command */
    (void)spi_transfer(value);    /* send data */

    spi_cs_high();
}

uint8 spi_read_reg(uint8 reg)
{
    uint8 val;

    spi_cs_low();

    (void)spi_transfer(reg | 0x80U);  /* example: MSB=1 indicates read */
    val = spi_transfer(0xFFU);        /* dummy byte to clock data out */

    spi_cs_high();

    return val;
}

/* ------------------------------------------------------------------------- */
/* Example: write/read multiple bytes                                        */
/* ------------------------------------------------------------------------- */

void spi_write_bytes(uint8 reg, const uint8 *data, uint32 len)
{
    spi_cs_low();

    (void)spi_transfer(reg);
    for (uint32 i = 0; i < len; i++)
    {
        (void)spi_transfer(data[i]);
    }

    spi_cs_high();
}

void spi_read_bytes(uint8 reg, uint8 *buf, uint32 len)
{
    spi_cs_low();

    (void)spi_transfer(reg | 0x80U);
    for (uint32 i = 0; i < len; i++)
    {
        buf[i] = spi_transfer(0xFFU);
    }

    spi_cs_high();
}
