#!/usr/bin/env bash
# Run lice against MicroHs tests

set -eu

root="$(git rev-parse --show-toplevel)"
cd "$root"

for comb in combs/*.comb ; do
  test="${comb##*/}"
  test="${test%.comb}"
  result="$comb.result"
  ref="MicroHs/tests/$test.ref"

  case "$test" in mhs) continue ;; esac # don't attempt to run mhs

  printf "%-12s ... " "$test:"
  if ! cargo run -q --release --bin lice-run "$comb" >"$result" 2>&1 ; then
    echo -ne "\033[1;31m[ERR]\033[0;31m\t"
    if grep -q "not yet implemented: " < "$result" ; then
      grep "not yet implemented: " < "$result" | head -n 1
    elif grep -q "missing FFI symbol: " < "$result" ; then
      grep "missing FFI symbol: " < "$result" | head -n 1
    else
      echo "(some other kind of error)"
    fi
    # echo
    # cat $i.result | sed 's/^/    /' | grep -v RUST_BACKTRACE
    # echo
  else
    if ! [ -f "$ref" ] ; then
      echo -e "\033[1;32m[FINE]\033[0;32m\t(no reference output: $ref)"
      rm "$result"
    elif ! diff "$result" "$ref" >/dev/null ; then
      echo -e "\033[1;33m[MISMATCH]"
    else
      echo -e "\033[1;32m[OK]"
      rm "$result"
    fi
  fi
  echo -e -n "\033[0m"
done
