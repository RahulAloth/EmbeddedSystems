#include <stdio.h>
#include <stdint.h>

// We use a 12-bit Hamming(12,8) style layout in a uint16_t:
//
// Bit positions (1-based):
//  1: P1
//  2: P2
//  3: D1
//  4: P4
//  5: D2
//  6: D3
//  7: D4
//  8: P8
//  9: D5
// 10: D6
// 11: D7
// 12: D8
//
// We'll store them in bits 0..11 of a uint16_t (position-1).

static uint16_t set_bit(uint16_t x, int pos, int val) {
    if (val)
        x |=  (1u << (pos - 1));
    else
        x &= ~(1u << (pos - 1));
    return x;
}

static int get_bit(uint16_t x, int pos) {
    return (x >> (pos - 1)) & 1u;
}

// Encode 8-bit data into 12-bit Hamming codeword
uint16_t ecc_encode(uint8_t data) {
    uint16_t cw = 0;

    // Place data bits into their positions
    // D1..D8 = data bit 0..7
    cw = set_bit(cw,  3, (data >> 0) & 1);
    cw = set_bit(cw,  5, (data >> 1) & 1);
    cw = set_bit(cw,  6, (data >> 2) & 1);
    cw = set_bit(cw,  7, (data >> 3) & 1);
    cw = set_bit(cw,  9, (data >> 4) & 1);
    cw = set_bit(cw, 10, (data >> 5) & 1);
    cw = set_bit(cw, 11, (data >> 6) & 1);
    cw = set_bit(cw, 12, (data >> 7) & 1);

    // Compute parity bits (P1, P2, P4, P8)
    int p1 = get_bit(cw, 3) ^ get_bit(cw, 5) ^ get_bit(cw, 7) ^ get_bit(cw, 9) ^ get_bit(cw, 11);
    int p2 = get_bit(cw, 3) ^ get_bit(cw, 6) ^ get_bit(cw, 7) ^ get_bit(cw,10) ^ get_bit(cw, 11);
    int p4 = get_bit(cw, 5) ^ get_bit(cw, 6) ^ get_bit(cw, 7) ^ get_bit(cw,12);
    int p8 = get_bit(cw, 9) ^ get_bit(cw,10) ^ get_bit(cw,11) ^ get_bit(cw,12);

    cw = set_bit(cw, 1, p1);
    cw = set_bit(cw, 2, p2);
    cw = set_bit(cw, 4, p4);
    cw = set_bit(cw, 8, p8);

    return cw;
}

// Decode + correct single-bit error.
// Returns corrected data, and writes error position (0 = no error) via *err_pos.
uint8_t ecc_decode(uint16_t cw, int *err_pos) {
    // Recompute parity checks (syndrome)
    int s1 = get_bit(cw, 1) ^ get_bit(cw, 3) ^ get_bit(cw, 5) ^ get_bit(cw, 7) ^ get_bit(cw, 9) ^ get_bit(cw,11);
    int s2 = get_bit(cw, 2) ^ get_bit(cw, 3) ^ get_bit(cw, 6) ^ get_bit(cw, 7) ^ get_bit(cw,10) ^ get_bit(cw,11);
    int s4 = get_bit(cw, 4) ^ get_bit(cw, 5) ^ get_bit(cw, 6) ^ get_bit(cw, 7) ^ get_bit(cw,12);
    int s8 = get_bit(cw, 8) ^ get_bit(cw, 9) ^ get_bit(cw,10) ^ get_bit(cw,11) ^ get_bit(cw,12);

    int syndrome = (s8 << 3) | (s4 << 2) | (s2 << 1) | (s1 << 0);

    if (syndrome != 0) {
        // Single-bit error at position = syndrome
        *err_pos = syndrome;
        cw ^= (1u << (syndrome - 1)); // flip the bit to correct
    } else {
        *err_pos = 0; // no error detected
    }

    // Extract data bits back into 8-bit value
    uint8_t data = 0;
    data |= get_bit(cw,  3) << 0;
    data |= get_bit(cw,  5) << 1;
    data |= get_bit(cw,  6) << 2;
    data |= get_bit(cw,  7) << 3;
    data |= get_bit(cw,  9) << 4;
    data |= get_bit(cw, 10) << 5;
    data |= get_bit(cw, 11) << 6;
    data |= get_bit(cw, 12) << 7;

    return data;
}

static void print_bits16(uint16_t x) {
    for (int i = 11; i >= 0; --i) {
        printf("%d", (x >> i) & 1);
        if (i == 8 || i == 4) printf(" ");
    }
}

int main(void) {
    uint8_t data = 0xB5; // 0b10110101
    printf("Original data: 0x%02X\n", data);

    uint16_t cw = ecc_encode(data);
    printf("Encoded codeword (12 bits): ");
    print_bits16(cw);
    printf("\n");

    // Inject a single-bit error at position 6 (for example)
    int flip_pos = 6;
    cw ^= (1u << (flip_pos - 1));
    printf("Codeword with error at bit %d: ", flip_pos);
    print_bits16(cw);
    printf("\n");

    int err_pos = 0;
    uint8_t decoded = ecc_decode(cw, &err_pos);

    printf("Detected error position: %d\n", err_pos);
    printf("Decoded (corrected) data: 0x%02X\n", decoded);

    return 0;
}

