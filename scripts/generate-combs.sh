#!/usr/bin/env bash
# Generate fresh .comb files from local version of MicroHs.

set -eu

# Use gmhs for better performance
MHS="bin/gmhs"

root="$(git rev-parse --show-toplevel)"
cd "$root"

cd MicroHs || exit 1
make "$MHS"

compile() {
  module="$1"
  indir="$2"
  out="${3:-${module}.comb}"

  file="${indir}/${module/.//}.hs"
  echo -n "Compiling MicroHs/$file to $out... "

  rm -f "../combs/$out" "../combs/$out.mhserr"
  set +e
  if "$MHS" "-i$indir" "$module" "-o../combs/$out" >"../combs/$out.mhserr" 2>&1 ; then
    echo "OK"
    rm -f "../combs/$out.mhserr"
  else
    echo "ERR (see $out.mhserr)"
    rm -f "../combs/$out"
  fi
  set -e
}

mkdir -p ../combs

for test in tests/*.hs ; do
  hs="${test#*/}"
  module="${hs%.hs}"
  compile "$module" tests
done

compile Example .
compile MicroHs.Main src mhs.comb
