#!/bin/sh
# Builds the native frame model for Node (linux-x64, gcc): native/node/build/zen_lipsync.node.
# Kernels in one file, picked at run time. float32 (native/x86/lipf.c): AVX-512F, AVX2+FMA.
# Sentence model conv (native/x86/sconv.c): AVX-512F, AVX2.
# int8 (frame.lip8): AVX-512 VNNI (native/x86/lip8.c), AVX2 and SSE4.1 (cpu/lip8w.c via shim/).
# NODE_INCLUDE: Node's include/node directory (default: next to the node binary on PATH).
set -e
cd "$(dirname "$0")"
NODE_INCLUDE=${NODE_INCLUDE:-$(dirname "$(dirname "$(command -v node)")")/include/node}
[ -f "$NODE_INCLUDE/node_api.h" ] || { echo "node_api.h not found in $NODE_INCLUDE (set NODE_INCLUDE)" >&2; exit 1; }
CC=${CC:-gcc}
mkdir -p build
F="-O3 -fPIC -ffp-contract=off -fvisibility=hidden"
$CC $F -march=icelake-server -DLIP8_LIB -fvisibility=default -c ../x86/lip8.c -o build/k512.o
$CC $F -mavx2 -mfma -Ishim -DKERNEL=lip8_avx2 -c portable.c -o build/k2.o
$CC $F -msse4.1 -Ishim -DKERNEL=lip8_sse41 -c portable.c -o build/k1.o
# AVX-512 float32: the Skylake-SP set by name (F CD BW DQ VL + FMA, the CPU check in addon.c is the same list), scheduled for it
$CC $F -mavx512f -mavx512cd -mavx512bw -mavx512dq -mavx512vl -mfma -mtune=skylake-avx512 -mprefer-vector-width=512 -c ../x86/lipf.c -o build/f512.o
$CC $F -mavx2 -mfma -c ../x86/lipf.c -o build/f2.o
$CC $F -mavx512f -DNAME=sconv_avx512 -c ../x86/sconv.c -o build/s512.o
$CC $F -mavx2 -mfma -DNAME=sconv_avx2 -c ../x86/sconv.c -o build/s2.o
$CC $F -I"$NODE_INCLUDE" -DNAPI_VERSION=8 -c addon.c -o build/addon.o
$CC -shared -o build/zen_lipsync.node build/addon.o build/k512.o build/k2.o build/k1.o build/f512.o build/f2.o build/s512.o build/s2.o -lm -lpthread
echo "built $(pwd)/build/zen_lipsync.node"
