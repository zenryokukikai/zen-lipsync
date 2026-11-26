#!/bin/sh
# Builds the CPU frame model for the browser (Emscripten): cpu/lip8w.{mjs,wasm} (SIMD128) and cpu/lip8r.{mjs,wasm}
# (relaxed SIMD; faster where supported). Single thread.
# Also cpu/sconvw.{mjs,wasm}: the sentence model's conv (native/x86/sconv.c, SIMD128) for Node without the native addon.
set -e
cd "$(dirname "$0")/../cpu"
for v in w r; do
  extra=""; [ "$v" = r ] && extra="-mrelaxed-simd"
  emcc -O3 -msimd128 $extra -ffp-contract=off lip8$v.c -o lip8$v.mjs \
    -sEXPORTED_FUNCTIONS=_lip_init,_lip_run,_malloc,_free -sEXPORTED_RUNTIME_METHODS=HEAPU8,HEAPF32 \
    -sMODULARIZE=1 -sEXPORT_ES6=1 -sEXPORT_NAME=createLip8 -sALLOW_MEMORY_GROWTH=1 -sINITIAL_MEMORY=33554432 \
    -sENVIRONMENT=web,worker,node -sFILESYSTEM=0 -sASSERTIONS=0
done
emcc -O3 -msimd128 -ffp-contract=off -DNAME=sconv ../native/x86/sconv.c -o sconvw.mjs \
  -sEXPORTED_FUNCTIONS=_sconv,_malloc,_free -sEXPORTED_RUNTIME_METHODS=HEAPF32 \
  -sMODULARIZE=1 -sEXPORT_ES6=1 -sEXPORT_NAME=createSconv -sALLOW_MEMORY_GROWTH=1 \
  -sENVIRONMENT=web,worker,node -sFILESYSTEM=0 -sASSERTIONS=0
ls -l lip8w.wasm lip8r.wasm sconvw.wasm
