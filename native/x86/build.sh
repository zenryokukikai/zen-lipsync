#!/bin/sh
# Native CPU frame model (x86-64 with AVX-512 VNNI: Intel Ice Lake / Sapphire Rapids and later, AMD Zen 4 and later).
# Single thread. Output: native/x86/lip8
set -e
cd "$(dirname "$0")"
gcc -O3 -march=icelake-server -o lip8 lip8.c -lm
echo "built $(pwd)/lip8"
