// The portable native kernel: cpu/lip8w.c (the WebAssembly SIMD128 frame model) built for x86 through shim/, once with
// -mavx2 (lip8_avx2_*) and once with -msse4.1 (lip8_sse41_*) by build.sh. KERNEL names the exported functions.
#define ZCAT(a, b) a##b
#define ZXCAT(a, b) ZCAT(a, b)
#define lip_init ZXCAT(KERNEL, _init_impl)
#define lip_run ZXCAT(KERNEL, _run_impl)
#include "../../cpu/lip8w.c"
