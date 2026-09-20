# VectorBoundary: learned RVV kernel-schedule dispatch

This branch (`vectorboundary`) is a research fork of
[ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp). It replaces the
RISC-V Vector (RVV) backend's fixed, VLEN-matched kernel-selection rule with
a small learned cost model, adds a legality mask that also checks numerical
correctness (not just legality), and fixes a real correctness bug the mask
found. Companion benchmarking/analysis tooling, generated figures, and the
paper draft live in a separate repo: **[vectorboundary-artifact](https://github.com/Fican1/vectorboundary-artifact)**
(link it here once pushed).

## What changed vs. upstream

Three commits on top of upstream `ggml-org/llama.cpp` (plus two merged
upstream PRs that widen the RVV kernel set first):

| Commit | What |
|---|---|
| `366c727e4` | Replace the `if`/`else` VLEN-matched kernel selector in `ggml/src/ggml-cpu/repack.cpp` with an enumerable schedule registry + legality mask (`min_vlenb`/`max_vlenb`/divisibility). `GGML_RVV_SCHEDULE=<name>` env var forces a schedule for benchmarking. |
| `449f9bde4` | **Correctness fix.** The legality mask's `max_vlenb` bound caught a real bug: the Q4\_0 8x8 gemv/gemm kernels' `vlenb >= 32` guard assumed a fixed register-grouping width and returns wrong numeric output at `vlenb=64/128` (VLEN 512/1024). Fixed to an exact `vlenb == 32` match. |
| `0a4cc6535` | **Learned dispatch.** Replaces the VLEN-matched default with a linear cost model (13 features, fit on exact QEMU instruction counts across VLEN 128/256/512/1024) that scores every legal schedule for a tensor, including "stay unrepacked" as a candidate. `GGML_RVV_DISPATCH=heuristic` reverts to the old rule for A/B comparison; `GGML_RVV_PROFILE={decode,prefill,interactive,throughput,balanced}` controls the decode/prefill cost blend (default `balanced`). Currently fitted for Q4\_0 only; other quant types fall back to the heuristic. |

Validated: bit-identical generated text vs. the non-repacked reference at
VLEN 128/256/512/1024 under QEMU, and a full `test-backend-ops MUL_MAT` pass
(1323/1323) at all four widths after the dispatch change.

## Build for x86_64 (native, for correctness testing only)

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DLLAMA_CURL=OFF
cmake --build build -j --target test-backend-ops test-quantize-fns llama-completion llama-bench
./build/bin/test-backend-ops test -b CPU -o MUL_MAT   # 1323/1323 expected
```

## Build for riscv64 + validate under QEMU

You need a `riscv64-linux-gnu-gcc` >= 14 (for `zvfh` intrinsics) and
`qemu-riscv64` >= 10 (QEMU 8.x lacks `zvfh` support and will SIGILL). The
easiest way is the Docker image below.

```sh
docker build -t vb-riscv -f Dockerfile.riscv .   # see Dockerfile.riscv, next section

docker run --rm -v "$PWD:/work" vb-riscv bash -c '
  cmake -S /work -B /work/build-riscv -DCMAKE_SYSTEM_NAME=Linux \
    -DCMAKE_SYSTEM_PROCESSOR=riscv64 -DCMAKE_C_COMPILER=riscv64-linux-gnu-gcc \
    -DCMAKE_CXX_COMPILER=riscv64-linux-gnu-g++ -DGGML_NATIVE=OFF -DLLAMA_CURL=OFF
  cmake --build /work/build-riscv -j --target llama-completion test-backend-ops
'

# correctness at all 4 VLENs (needs a GGUF model, e.g. any small Q4_0 model)
for vlen in 128 256 512 1024; do
  docker run --rm -v "$PWD:/work" -v "$PWD/models:/models" vb-riscv bash -c "
    qemu-riscv64 -L /usr/riscv64-linux-gnu -cpu max,vlen=$vlen,elen=64,vext_spec=v1.0 \
      -E LD_LIBRARY_PATH=/work/build-riscv/bin \
      /work/build-riscv/bin/llama-completion -m /models/YOUR_MODEL.gguf -n 8 'hi'
  "
done
```

A `Dockerfile.riscv` reproducing the exact toolchain used for all measurements
in the paper (Debian trixie, gcc-14.2, QEMU 10.0.13 built with plugin support)
is in the [vectorboundary-artifact](https://github.com/Fican1/vectorboundary-artifact)
repo, along with the full benchmarking harness (instruction-count QEMU plugin,
sweep scripts, the dataset, and the scripts that produced every number and
figure in the paper).

## Looking for real RVV hardware to test on?

All results in the paper so far are exact instruction counts under QEMU,
not real timing. If you have access to a real RVV 1.0 board (Banana Pi
BPI-F3, Milk-V Jupiter, or similar), running
[`vectorboundary-hw-sweep.sh`](vectorboundary-hw-sweep.sh) takes about 15
minutes and would genuinely help. Full instructions, what to send back, and
why it matters: **[HARDWARE-TESTING.md in the artifact repo](https://github.com/Fican1/vectorboundary-artifact/blob/main/HARDWARE-TESTING.md)**.

## Env vars this fork adds

| Var | Values | Effect |
|---|---|---|
| `GGML_RVV_SCHEDULE` | e.g. `q4_0_8x8`, comma-separated per type | Force a specific schedule; illegal/unlisted falls back safely. |
| `GGML_RVV_DISPATCH` | `heuristic` (else: learned model) | Revert to the old VLEN-matched rule. |
| `GGML_RVV_PROFILE` | `decode`/`prefill`/`interactive`/`throughput`/`balanced` | Decode/prefill cost-blend weight for the learned dispatcher. |

## AI-assistance disclosure

Development on this branch was AI-assisted (Claude). Every change is
understood and owned by the maintainer of this fork; see upstream's
[AGENTS.md](AGENTS.md) for this project's AI-contribution policy. This fork
has not been, and per upstream policy will not be, submitted upstream by an
agent — any upstreaming happens via the human maintainer, manually.
