# Fusion baselines

Per-device baselines for `test-fusion`, one CSV per backend (e.g. `ROCm.csv`). Rows are
`arch,moe,mode,label,count`. Regenerate a CSV whenever fusion patterns change.

## Update a baseline

```sh
cmake -B build -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=ON -DGPU_TARGETS=gfx908   # enable the target backend
cmake --build build --config Release --target test-llama-archs --target test-fusion -j

rm -rf build-ci-models && mkdir -p build-ci-models
./build/bin/test-llama-archs -o build-ci-models

./build/bin/test-fusion --models build-ci-models --device ROCm0 --record ROCm.csv
```

## Validate

```sh
./build/bin/test-fusion --models build-ci-models --device ROCm0 --check ROCm.csv
```

Non-zero exit means a row differs from the baseline. Use `--model FILE` to run a single
architecture. Note `--check` only sees present rows — a fusion that stops matching is not
reported, so diff the recorded CSV to catch removed patterns.
