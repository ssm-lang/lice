set positional-arguments
just     := "just --justfile " + justfile()

MicroHs  := justfile_directory() / "MicroHs"
compiler := "gmhs"
runtime  := "lice"
scope    := "lice-scope"

profile  := env("PROFILE", "release")

out_dir  := justfile_directory() / "combs"
mhserr   := env("MHSERR", "show")
eg_dir   := justfile_directory() / "examples"

mhserr_check := if mhserr =~ "show|keep|hide" {
    "OK"
} else {
    error("Unknown MHSERR value: " + mhserr)
}
profile_check := if profile =~ "release|debug" {
    "OK"
} else {
    error("Unknown profile: " + profile)
}
log_check := if env("RUST_LOG", "warn")=~ "trace|debug|info|warn|error|all" {
    "OK"
} else {
    error("Unknown logging level: " + env("RUST_LOG"))
}
rustc_suppress := env("RUSTFLAGS", "") + \
    "-A dead_code " + \
    "-A unused_imports"

# Print this help menu
@help:
    just --list --unsorted
    echo
    echo "Environment variables"
    echo "    MHSERR   \033[0;34m# What to do if MHS encounters an error (show,keep,hide)\033[0m"
    echo "    PROFILE  \033[0;34m# Runtime build profile (release,debug)\033[0m"
    echo "    RUST_LOG \033[0;34m# Runtime log filter (trace,debug,info,warn,error,all)\033[0m"

# Clean the combs/ directory
@clean-comb:
    rm -rf {{out_dir}}/*.comb {{out_dir}}/*.result

# Run lice with given ARGS
@run *ARGS: ensure-runtime
    target/{{profile}}/{{runtime}} "$@"

# Run lice with given ARGS and info-level logging
@run-log *ARGS: ensure-runtime
    RUST_LOG=info target/{{profile}}/{{runtime}} "$@"

# Run lice with given ARGS and debug-level logging
@run-debug *ARGS: ensure-runtime
    RUST_LOG=debug target/{{profile}}/{{runtime}} "$@"

# Run lice with given ARGS and trace-level logging
@run-trace *ARGS: ensure-runtime
    RUST_LOG=trace target/{{profile}}/{{runtime}} "$@"

# Build MHS compiler built using GHC (faster)
build-gmhs:
    make -C {{MicroHs}}

# Build self-hosted MHS compiler (cooler)
build-mhs:
    make -C {{MicroHs}} bin/mhs

# Quietly ensure MHS compiler is built
[private]
@ensure-compiler:
    git submodule update --recursive --init
    make -C {{MicroHs}} {{"bin" / compiler}} --quiet

# Quietly ensure lice runtime is built
[private]
@ensure-runtime:
    RUSTFLAGS="{{rustc_suppress}}" cargo build --bin {{runtime}} --features=cli --quiet {{ if profile == "release" { "--release" } else { "" } }}

# Invoke mhs in the given directory
[private]
mhs INDIR MODULE OUT *FLAGS: ensure-compiler
    #!/usr/bin/env bash
    set -euo pipefail

    mkdir -p "{{out_dir}}"
    out="{{out_dir / OUT}}"

    cd {{INDIR}}

    echo -ne "Compiling {{MODULE}} to {{OUT}}... "

    if {{MicroHs / "bin" / compiler}} "{{MODULE}}" "-o$out" {{FLAGS}} >"$out.mhserr" 2>&1; then
        echo -e "\033[1;32mOK\033[0m"
        rm "$out.mhserr"
    else
        rm -f "$out"
        case "{{mhserr}}" in
            show)
                echo -e "\033[1;31mERROR\033[0m"
                echo -ne "\033[0;31m"
                cat "$out.mhserr"
                echo -ne "\033[0m"
                rm "$out.mhserr"
            ;;
            keep)
                echo -e "\033[1;31mERROR\033[0;31m (see $out.mhserr)\033[0m"
            ;;
            hide)
                echo -e "\033[1;31mERROR\033[0m"
                rm "$out.mhserr"
            ;;
        esac
    fi

# Compile a MicroHs test called NAME
@compile-test NAME:
    {{just}} mhs "{{MicroHs / "tests"}}" "{{NAME}}" "{{NAME}}.comb" "-i../lib"

# Compile example in the examples/ directory
@compile-example NAME:
    {{just}} mhs "{{MicroHs/""}}" "{{NAME}}" "{{NAME}}.comb" "-i{{eg_dir}}"

# Compile MicroHs itself to mhs.comb
@compile-mhs:
    {{just}} mhs "{{MicroHs/""}}" "MicroHs.Main" "mhs.comb" "-isrc"

# Compile all MicroHs tests
compile-tests:
    #!/usr/bin/env bash
    cd {{MicroHs}}/tests
    for test in *.hs ; do
        MHSERR=keep {{just}} compile-test "${test%.hs}"
    done

# Compile all examples in the examples/ directory
compile-examples:
    #!/usr/bin/env bash
    cd "{{eg_dir}}"
    for program in *.hs ; do
        MHSERR=keep {{just}} compile-example "${program%.hs}"
    done

# Compile everything known to man
compile-all: compile-tests compile-examples compile-mhs

# Run test of given name
test NAME: # (compile-test NAME)
    #!/usr/bin/env bash
    test="{{NAME}}"
    result="{{out_dir / NAME + ".comb.result"}}"
    ref="MicroHs/tests/$test.ref"
    test_makefile="MicroHs/tests/Makefile"

    printf "%-12s ... " "$test"
    if ! just run {{out_dir / NAME + ".comb"}} >"$result" 2>&1 ; then
        echo -ne "\033[1;31m[ERR]\033[0;31m\t"
        if grep -q "not yet implemented: " <"$result" ; then
            grep "not yet implemented: " <"$result" | head -n 1
        elif grep -q "missing FFI symbol: " <"$result" ; then
            grep "missing FFI symbol: " <"$result" | head -n 1
        elif grep -q "crash: " <"$result" ; then
            grep "crash: " <"$result" | head -n 1
        else
            echo "(some other kind of error)"
        fi
    else
        if ! [ -f "$ref" ] ; then
            echo -e "\033[1;34m[FINE]\033[0;34m\t(no reference output: $ref)"
            rm "$result"
        elif ! grep -q "^\t\\\$(TMHS) $test" < "$test_makefile" ; then
            echo -e "\033[1;34m[FINE]\033[0;34m\t(not tested, according to $test_makefile)"
        elif ! diff "$result" "$ref" >/dev/null ; then
            echo -e "\033[1;33m[MISMATCH]"
        else
            echo -e "\033[1;32m[PASS]"
            rm "$result"
        fi
    fi
    echo -e -n "\033[0m"

# Run all tests known to man
test-all:
    #!/usr/bin/env bash
    cd {{MicroHs}}/tests
    for test in *.hs ; do
        if [ -f "{{out_dir}}/${test%.hs}.comb" ]; then
            {{just}} test "${test%.hs}"
        else
            printf "\033[0;90m%-12s ... [SKIP] (.comb file missing)\033[0m\n" "${test%.hs}"
        fi
    done
    cd {{eg_dir}}
    for example in *.hs ; do
        if [ -f "{{out_dir}}/${example%.hs}.comb" ]; then
            {{just}} test "${example%.hs}"
        fi
    done

# Compile and run an example
@eg NAME: (compile-example NAME)
    {{just}} run {{"combs" / NAME + ".comb"}}

# Compile and run an example with trace-level logging and debug profile
@trace NAME: (compile-example NAME)
    PROFILE=debug {{just}} run-trace {{"combs" / NAME + ".comb"}}

@scope NAME:
    cargo run --bin "{{scope}}" --release -- graph {{ "combs" / NAME + ".comb" }}
