#!/bin/sh
# Times the native CPU frame model on one core: native/x86/bench.sh <avatar models dir> [reps]
# Input: 1 frame of zero features and latents (the timing does not depend on the values); prints JSON with ms per frame.
set -e
cd "$(dirname "$0")"
[ -x ./lip8 ] || ./build.sh
M=${1:?models dir (with frame.lip8)}
T=$(mktemp -d)
python3 -c "import sys; open('$T/feat.f32','wb').write(bytes(4*32*880)); open('$T/z.f32','wb').write(bytes(4*640))"
W8=$M/frame.lip8 FEAT=$T/feat.f32 ZIN=$T/z.f32 IN_N=1 taskset -c ${CORE:-0} ./lip8 ${2:-400}
rm -rf "$T"
