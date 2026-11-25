#!/usr/bin/env python3
"""Build linkcell with Cargo and copy its C ABI artifacts for Meson."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cargo", required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--target", type=Path, required=True)
    parser.add_argument("--profile", choices=("debug", "release"), required=True)
    parser.add_argument("--shared", required=True)
    parser.add_argument("--static", required=True)
    parser.add_argument("--out-shared", type=Path, required=True)
    parser.add_argument("--out-static", type=Path, required=True)
    parser.add_argument("--cc", help="C compiler for the cc crate (overrides CC)")
    parser.add_argument(
        "--library-path",
        action="append",
        default=[],
        help="library directory made visible to the cc crate",
    )
    parser.add_argument("--implib")
    parser.add_argument("--out-implib", type=Path)
    parser.add_argument("--depfile", type=Path)
    parser.add_argument(
        "--openmp",
        choices=("true", "false"),
        default="true",
        help="Compile the fused k-nearest OpenMP region (LINKCELL_OPENMP).",
    )
    parser.add_argument(
        "--sanitize",
        choices=("none", "thread"),
        default="none",
        help="Rebuild the crate and std with this sanitizer (LINKCELL_SANITIZE).",
    )
    args = parser.parse_args()
    if (args.implib is None) != (args.out_implib is None):
        parser.error("--implib and --out-implib must be supplied together")
    return args


def main() -> None:
    args = parse_args()
    # ThreadSanitizer does not see Rayon's futex join unless std and rayon
    # are rebuilt with the sanitizer. The final C++ link supplies one
    # compiler-rt via -Zexternal-clangrt, so the pool stays multithreaded.
    tsan = args.sanitize == "thread" or os.environ.get("LINKCELL_SANITIZE") == "thread"
    # Pixi's cargo is a real binary, not the rustup proxy, and it is ahead
    # of rustup on PATH. rustup run nightly still leaves that rustc first,
    # so the nightly compiler has to be named by its own path.
    cargo = os.path.abspath(args.cargo) if os.path.dirname(args.cargo) else args.cargo
    command = [cargo]
    if os.environ.get("LINKCELL_CARGO_VERBOSE") == "1":
        command.append("-vv")
    command += [
        "rustc",
        "--lib",
        "--manifest-path",
        str(args.source.absolute() / "Cargo.toml"),
        "--target-dir",
        str(args.target.absolute()),
        "--features",
        "capi",
    ]
    if tsan:
        command += ["-Zbuild-std", "--target", "x86_64-unknown-linux-gnu"]
    if (args.source / "Cargo.lock").is_file() and not tsan:
        command.append("--locked")
    if args.profile == "release":
        command.append("--release")

    rustc = ["--crate-type=cdylib", "--crate-type=staticlib"]
    if args.shared.endswith(".so"):
        rustc.append(f"-Clink-arg=-Wl,-soname,{args.shared}")
    elif args.shared.endswith(".dylib"):
        rustc.append(f"-Clink-arg=-Wl,-install_name,@rpath/{args.shared}")
    command.extend(["--", *rustc])
    env = os.environ.copy()
    build_target = env.get("CARGO_BUILD_TARGET")
    if build_target and build_target.endswith(".json"):
        env["CARGO_BUILD_TARGET"] = os.path.abspath(build_target)
    # Unset keeps the historical default inside build.rs. Meson always
    # passes the flag so a use_openmp=false build does not emit the region.
    env["LINKCELL_OPENMP"] = "1" if args.openmp == "true" else "0"
    # The OpenMP runtime Meson links follows its own C compiler. An
    # activated environment may export a different CC, so this wins.
    if args.cc:
        env["CC"] = args.cc
    if args.library_path:
        inherited = env.get("LIBRARY_PATH")
        paths = [*args.library_path]
        if inherited:
            paths.append(inherited)
        env["LIBRARY_PATH"] = os.pathsep.join(paths)
    if tsan:
        nightly = rustup_nightly()
        active = active_rustc(cargo, env)
        if nightly is not None and has_std_sources(nightly[1], env):
            nightly_cargo, nightly_rustc = nightly
            command[0] = nightly_cargo
            env["RUSTC"] = nightly_rustc
            nightly_bin = str(Path(nightly_rustc).parent)
            env["PATH"] = nightly_bin + os.pathsep + env.get("PATH", "")
        elif active is not None and has_std_sources(active, env):
            # -Zbuild-std only needs the toolchain's rust-src (conda-forge
            # ships it as a package); RUSTC_BOOTSTRAP lets a stable
            # toolchain take the -Z flags.
            env["RUSTC"] = active
            env["RUSTC_BOOTSTRAP"] = "1"
        else:
            raise SystemExit(
                "LINKCELL_SANITIZE=thread needs a toolchain whose sysroot has "
                "rust-src: rustup's nightly or the active one"
            )
        extra = "-Zsanitizer=thread -Zexternal-clangrt -Cforce-frame-pointers=yes"
        flags = env.get("RUSTFLAGS", "")
        if "-Zsanitizer=thread" not in flags:
            env["RUSTFLAGS"] = (flags + " " + extra).strip()
    subprocess.check_call(command, cwd=args.source, env=env)

    if tsan:
        build_target = "x86_64-unknown-linux-gnu"
    built = args.target
    if build_target:
        if build_target.endswith(".json"):
            build_target = Path(build_target).stem
        built /= build_target
    built /= args.profile
    shutil.copy2(built / args.shared, args.out_shared)
    shutil.copy2(built / args.static, args.out_static)
    if args.implib is not None:
        shutil.copy2(built / args.implib, args.out_implib)
    if args.depfile is not None:
        write_depfile(built / (Path(args.static).stem + ".d"), args.out_shared, args.depfile)


def rustup_nightly() -> tuple[str, str] | None:
    """rustup's installed nightly cargo and rustc, or None without one."""
    # rustup which installs a missing toolchain by default, and that
    # minimal nightly has no rust-src for -Zbuild-std.
    env = dict(os.environ, RUSTUP_AUTO_INSTALL="0")
    try:
        return tuple(
            subprocess.check_output(
                ["rustup", "which", "--toolchain", "nightly", tool],
                env=env,
                text=True,
                stderr=subprocess.DEVNULL,
            ).strip()
            for tool in ("cargo", "rustc")
        )
    except (OSError, subprocess.CalledProcessError):
        return None


def active_rustc(cargo: str, env: dict[str, str]) -> str | None:
    """The rustc Cargo would run: RUSTC, else the one beside cargo or on PATH."""
    if env.get("RUSTC"):
        return env["RUSTC"]
    beside = Path(cargo).parent / "rustc"
    if beside.is_file():
        return str(beside)
    return shutil.which("rustc", path=env.get("PATH"))


def has_std_sources(rustc: str, env: dict[str, str]) -> bool:
    """Whether rustc's sysroot holds the std sources -Zbuild-std compiles."""
    try:
        sysroot = subprocess.check_output(
            [rustc, "--print", "sysroot"], env=env, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return False
    return (Path(sysroot) / "lib" / "rustlib" / "src" / "rust" / "library" / "Cargo.lock").is_file()


def write_depfile(cargo_dep: Path, target: Path, depfile: Path) -> None:
    """Restate Cargo's dep-info for the first Meson output.

    Cargo names its own artifacts as the rule targets. Ninja wants the
    custom target's output, so only the prerequisites are kept. Cargo has
    already escaped spaces in them.
    """
    deps: list[str] = []
    for line in cargo_dep.read_text(encoding="utf-8").splitlines():
        _, sep, rest = line.partition(": ")
        if sep and rest.strip():
            for dep in re.split(r"(?<!\\) +", rest.strip()):
                if dep not in deps:
                    deps.append(dep)
    escaped = str(target).replace(" ", "\\ ")
    depfile.write_text(f"{escaped}: {' '.join(deps)}\n", encoding="utf-8")


if __name__ == "__main__":
    main()
