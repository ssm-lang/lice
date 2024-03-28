#!/usr/bin/env bash

for i in combs/*.comb ; do
  test="${i##*/}"
  test="${test%.comb}"
  case "$test" in mhs|Example) continue ;; esac
  printf "%-12s ... " "$test:"
  if ! cargo run -q --release --bin lice-run $i >$i.result 2>&1 ; then
    echo -ne "\033[1;31m[ERROR]\033[0;31m\t"
    if grep -q "not yet implemented: " < $i.result ; then
      grep "not yet implemented: " < "$i.result" | head -n 1
    elif grep -q "missing FFI symbol: " < "$i.result" ; then
      grep "missing FFI symbol: " < "$i.result" | head -n 1
    else
      echo "(some other kind of error)"
    fi
    # echo
    # cat $i.result | sed 's/^/    /' | grep -v RUST_BACKTRACE
    # echo
  else
    if ! diff $i.result MicroHs/tests/$test.ref >/dev/null ; then
      echo -e "\033[1;33m[MISMATCH]"
    else
      echo -e "\033[1;32m[OK]"
      rm $i.result
    fi
  fi
  echo -e -n "\033[0m"
done
