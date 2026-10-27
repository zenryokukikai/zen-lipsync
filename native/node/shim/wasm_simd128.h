// The WebAssembly SIMD128 intrinsics used by cpu/lip8w.c, on x86 SSE4.1 (128-bit). Each maps to an instruction with
// the same result for the values the kernel uses (no NaN, in-range conversions), so the native portable kernel computes
// the same integers as cpu/lip8w.wasm. Compiled with -mavx2 the compiler uses VEX encodings (still 128-bit vectors).
#pragma once
#include <smmintrin.h>
#include <stdint.h>
#include <string.h>
typedef __m128i v128_t;
static inline v128_t wasm_v128_load(const void *p) { return _mm_loadu_si128((const __m128i *)p); }
static inline void wasm_v128_store(void *p, v128_t v) { _mm_storeu_si128((__m128i *)p, v); }
static inline v128_t wasm_v128_load32_splat(const void *p) { int32_t x; memcpy(&x, p, 4); return _mm_set1_epi32(x); }
static inline v128_t wasm_i32x4_splat(int32_t x) { return _mm_set1_epi32(x); }
static inline v128_t wasm_i32x4_add(v128_t a, v128_t b) { return _mm_add_epi32(a, b); }
static inline v128_t wasm_i32x4_dot_i16x8(v128_t a, v128_t b) { return _mm_madd_epi16(a, b); }
static inline v128_t wasm_f32x4_splat(float x) { return _mm_castps_si128(_mm_set1_ps(x)); }
static inline v128_t wasm_f32x4_add(v128_t a, v128_t b) { return _mm_castps_si128(_mm_add_ps(_mm_castsi128_ps(a), _mm_castsi128_ps(b))); }
static inline v128_t wasm_f32x4_mul(v128_t a, v128_t b) { return _mm_castps_si128(_mm_mul_ps(_mm_castsi128_ps(a), _mm_castsi128_ps(b))); }
static inline v128_t wasm_f32x4_max(v128_t a, v128_t b) { return _mm_castps_si128(_mm_max_ps(_mm_castsi128_ps(a), _mm_castsi128_ps(b))); }
static inline v128_t wasm_f32x4_min(v128_t a, v128_t b) { return _mm_castps_si128(_mm_min_ps(_mm_castsi128_ps(a), _mm_castsi128_ps(b))); }
static inline v128_t wasm_f32x4_convert_i32x4(v128_t a) { return _mm_castps_si128(_mm_cvtepi32_ps(a)); }
static inline v128_t wasm_f32x4_nearest(v128_t a) { return _mm_castps_si128(_mm_round_ps(_mm_castsi128_ps(a), _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC)); }
static inline v128_t wasm_i32x4_trunc_sat_f32x4(v128_t a) { return _mm_cvttps_epi32(_mm_castsi128_ps(a)); }
