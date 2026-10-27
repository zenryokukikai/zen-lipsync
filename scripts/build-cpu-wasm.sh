#!/bin/sh
# Builds the CPU frame model for the browser (Emscripten): cpu/lip8w.{mjs,wasm} (SIMD128) and cpu/lip8r.{mjs,wasm}
# (relaxed SIMD; faster where supported). Single thread.
set -e
cd "$(dirname "$0")/../cpu"
for v in w r; do
  extra=""; [ "$v" = r ] && extra="-mrelaxed-simd"
  emcc -O3 -msimd128 $extra -ffp-contract=off lip8$v.c -o lip8$v.mjs \
    -sEXPORTED_FUNCTIONS=_lip_init,_lip_run,_malloc,_free -sEXPORTED_RUNTIME_METHODS=HEAPU8,HEAPF32 \
    -sMODULARIZE=1 -sEXPORT_ES6=1 -sEXPORT_NAME=createLip8 -sALLOW_MEMORY_GROWTH=1 -sINITIAL_MEMORY=33554432 \
    -sENVIRONMENT=web,worker,node -sFILESYSTEM=0 -sASSERTIONS=0
done
ls -l lip8w.wasm lip8r.wasm
