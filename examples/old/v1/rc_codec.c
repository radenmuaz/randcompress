/* Canonical (low, high) range coder.  RC_PREC = 16 (CDF sums to M = 65536).
 * 64-bit state; 128-bit temporaries for multiplications that can overflow 64 bits.
 * Emit a byte only when the top byte of low and high agree.
 * Build: cc -O2 -shared -fPIC rc_codec.c -o rc_codec.so
 *
 * API:
 *   rc_encode(cdfs, symbols, n_sym, V, out_buf, out_cap) -> bytes written, or (size_t)-1 on overflow
 *   rc_decode(in_buf, in_len, cdfs, n_sym, V, out_syms) -> 0
 *
 * CDF layout: cdfs[t*(V+1) .. t*(V+1)+V] = cumulative freqs for position t.
 * cdfs[t][0] = 0, cdfs[t][V] = M = 65536.  All values int32_t >= 0.
 */
#include <stdint.h>
#include <stddef.h>

#define RC_PREC 16
#define M       ((uint64_t)(1U << RC_PREC))   /* 65536 */

typedef unsigned __int128 u128;

/* Compute floor(range * f / M) as uint64_t.
 * range is u128 (up to 2^64+1), f is uint64_t [0..M], M=2^16.
 * Result fits in uint64_t since range*f/M <= range <= UINT64_MAX+1,
 * and in practice range <= UINT64_MAX so result <= UINT64_MAX.
 */
static inline uint64_t scale(u128 range, uint64_t f) {
    return (uint64_t)(range * f / M);
}

/* ── Encoder ────────────────────────────────────────────────────────────── */

size_t rc_encode(const int32_t *cdfs, const int32_t *symbols,
                 size_t n_sym, int32_t V, uint8_t *out_buf, size_t out_cap)
{
    uint64_t low  = 0;
    uint64_t high = UINT64_MAX;
    size_t   pos  = 0;
    int32_t  stride = V + 1;

    for (size_t t = 0; t < n_sym; ++t) {
        const int32_t *cf = cdfs + (size_t)t * stride;
        int32_t  sym    = symbols[t];
        u128     range  = (u128)high - (u128)low + 1;
        uint64_t cum_lo = (uint64_t)(uint32_t)cf[sym];
        uint64_t cum_hi = (uint64_t)(uint32_t)cf[sym + 1];

        high = low + scale(range, cum_hi) - 1;
        low  = low + scale(range, cum_lo);

        /* emit agreed top bytes */
        while ((low >> 56) == (high >> 56)) {
            if (pos >= out_cap) return (size_t)-1;
            out_buf[pos++] = (uint8_t)(low >> 56);
            low  <<= 8;
            high  = (high << 8) | 0xFFULL;
        }
    }

    /* flush 8 tail bytes */
    for (int i = 0; i < 8; ++i) {
        if (pos >= out_cap) return (size_t)-1;
        out_buf[pos++] = (uint8_t)(low >> 56);
        low <<= 8;
    }
    return pos;
}

/* ── Decoder ────────────────────────────────────────────────────────────── */

int rc_decode(const uint8_t *in_buf, size_t in_len,
              const int32_t *cdfs, size_t n_sym, int32_t V,
              int32_t *out_syms)
{
    uint64_t low  = 0;
    uint64_t high = UINT64_MAX;
    uint64_t code = 0;
    size_t   pos  = 0;
    int32_t  stride = V + 1;

    /* read first 8 bytes into code */
    for (int i = 0; i < 8; ++i)
        code = (code << 8) | (pos < in_len ? in_buf[pos++] : 0);

    for (size_t t = 0; t < n_sym; ++t) {
        const int32_t *cf = cdfs + (size_t)t * stride;
        u128     range  = (u128)high - (u128)low + 1;

        /* Find sym: largest sym where low + scale(range, cf[sym]) <= code.
         * Equivalently: code < low + scale(range, cf[sym+1]).
         * This matches the encoder's interval boundary exactly.
         */
        int32_t lo = 0, hi = V - 1;
        while (lo < hi) {
            int32_t mid = (lo + hi + 1) / 2;
            /* check if low + scale(range, cf[mid]) <= code */
            if (low + scale(range, (uint64_t)(uint32_t)cf[mid]) <= code)
                lo = mid;
            else
                hi = mid - 1;
        }
        out_syms[t] = lo;

        uint64_t cum_lo = (uint64_t)(uint32_t)cf[lo];
        uint64_t cum_hi = (uint64_t)(uint32_t)cf[lo + 1];
        high = low + scale(range, cum_hi) - 1;
        low  = low + scale(range, cum_lo);

        /* refill agreed top bytes */
        while ((low >> 56) == (high >> 56)) {
            low  <<= 8;
            high  = (high << 8) | 0xFFULL;
            code  = (code << 8) | (pos < in_len ? in_buf[pos++] : 0);
        }
    }
    return 0;
}
