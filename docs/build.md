# Build llama.cpp locally

The main product of this project is the `llama` library. Its C-style interface can be found in [include/llama.h](../include/llama.h).

The project also includes many example programs and tools using the `llama` library. The examples range from simple, minimal code snippets to sophisticated sub-projects such as an OpenAI-compatible HTTP server.

**To get the Code:**

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
```

The following sections describe how to build with different backends and options.

* [CPU Build](#cpu-build)
* [BLAS Build](#blas-build)
* [HIP (AMD MI100 / gfx908)](#hip)
* [Arm® KleidiAI™](#arm-kleidiai)
* [Android](#android-1)
* [Notes about GPU-accelerated backends](#notes-about-gpu-accelerated-backends)


## CPU Build

Build llama.cpp using `CMake`:

```bash
cmake -B build
cmake --build build --config Release
```

**Notes**:

- For faster compilation, add the `-j` argument to run multiple jobs in parallel, or use a generator that does this automatically such as Ninja. For example, `cmake --build build --config Release -j 8` will run 8 jobs in parallel.
- For faster repeated compilation, install [ccache](https://ccache.dev/)
- For debug builds, there are two cases:

    1. Single-config generators (e.g. default = `Unix Makefiles`; note that they just ignore the `--config` flag):

       ```bash
       cmake -B build -DCMAKE_BUILD_TYPE=Debug
       cmake --build build
       ```

    2. Multi-config generators (`-G` param set to Visual Studio, XCode...):

       ```bash
       cmake -B build -G "Xcode"
       cmake --build build --config Debug
       ```

    For more details and a list of supported generators, see the [CMake documentation](https://cmake.org/cmake/help/latest/manual/cmake-generators.7.html).
- For static builds, add `-DBUILD_SHARED_LIBS=OFF`:
  ```
  cmake -B build -DBUILD_SHARED_LIBS=OFF
  cmake --build build --config Release
  ```

- Building for Windows (x86, x64 and arm64) with MSVC or clang as compilers:
    - Install Visual Studio 2022, e.g. via the [Community Edition](https://visualstudio.microsoft.com/vs/community/). In the installer, select at least the following options (this also automatically installs the required additional tools like CMake,...):
    - Tab Workload: Desktop-development with C++
    - Tab Components (select quickly via search): C++-_CMake_ Tools for Windows, _Git_ for Windows, C++-_Clang_ Compiler for Windows, MS-Build Support for LLVM-Toolset (clang)
    - Please remember to always use a Developer Command Prompt / PowerShell for VS2022 for git, build, test
    - For Windows on ARM (arm64, WoA), build with:
      ```bash
      cmake --preset arm64-windows-llvm-release -D GGML_OPENMP_FETCH=ON
      cmake --build build-arm64-windows-llvm-release
      ```
      - Use `ARM64 Native Tools Command Prompt for VS 2022` if you are building on an ARM64 machine.
      - `GGML_OPENMP_FETCH` downloads the official LLVM OpenMP runtime and requires Clang, 7-Zip and network access during configuration. CMake selects the runtime from the target architecture, so this also works when cross-compiling for WoA from x64. The extracted header, import library, DLL and OpenMP license are placed under `build/_deps`. The build copies `libomp.dll` and `LICENSE-LLVM-OpenMP` to the runtime output directory and installs them together. Omit the option to use CMake's normal OpenMP detection, or pass `-D GGML_OPENMP=OFF` to disable OpenMP.
    - For building with ninja generator and clang compiler as default:
      - Set path:
        ```
        set LIB=C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\um\x64;C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Tools\MSVC\14.41.34120\lib\x64\uwp;C:\Program Files (x86)\Windows Kits\10\Lib\10.0.22621.0\ucrt\x64
        ```
      - Run:
        ```bash
        cmake --preset x64-windows-llvm-release
        cmake --build build-x64-windows-llvm-release
        ```
- If you want HTTPS/TLS features, you may install OpenSSL development libraries. If not installed, the project will build and run without SSL support.
  - **Debian / Ubuntu:** `sudo apt-get install libssl-dev`
  - **Fedora / RHEL / Rocky / Alma:** `sudo dnf install openssl-devel`
  - **Arch / Manjaro:** `sudo pacman -S openssl`

## BLAS Build

Building the program with BLAS support may lead to some performance improvements in prompt processing using batch sizes higher than 32 (the default is 512). Using BLAS doesn't affect the generation performance. There are currently several different BLAS implementations available for build and use:

### Accelerate Framework

This is only available on Mac PCs and it's enabled by default. You can just build using the normal instructions.

### OpenBLAS

This provides BLAS acceleration using only the CPU. Make sure to have OpenBLAS installed on your machine.

- Using `CMake` on Linux:

    ```bash
    cmake -B build -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS
    cmake --build build --config Release
    ```

### BLIS

Check [BLIS.md](./backend/BLIS.md) for more information.

### AMD AOCL-BLAS

AOCL-BLAS is available as a vendor option for the generic `GGML_BLAS` backend.

Source `amd-libs.cfg` from your AOCL install (MT tree by default), then build (CMake 3.27+ recommended for the `AOCL` / `AOCL_mt` vendors):

```bash
source /opt/aocl/<version>/aocc/MT/amd-libs.cfg   # adjust path; ST tree uses .../ST/amd-libs.cfg
cmake -B build -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=AOCL_mt -DBLAS_INCLUDE_DIRS="${AOCL_ROOT}/include" -DGGML_NATIVE=ON
cmake --build build --config Release
```

Full steps, threading notes, and a fallback for older CMake: [AOCL.md](./backend/AOCL.md).

### Intel oneMKL

Building through oneAPI compilers will make avx_vnni instruction set available for intel processors that do not support avx512 and avx512_vnni. Please note that this build config **does not support Intel GPU**.

- Using manual oneAPI installation:
  By default, `GGML_BLAS_VENDOR` is set to `Generic`, so if you already sourced intel environment script and assign `-DGGML_BLAS=ON` in cmake, the mkl version of Blas will automatically been selected. Otherwise please install oneAPI and follow the below steps:
    ```bash
    source /opt/intel/oneapi/setvars.sh # You can skip this step if  in oneapi-basekit docker image, only required for manual installation
    cmake -B build -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=Intel10_64lp -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DGGML_NATIVE=ON
    cmake --build build --config Release
    ```

- Using oneAPI docker image:
  If you do not want to source the environment vars and install oneAPI manually, you can also build the code using intel docker container: [oneAPI-basekit](https://hub.docker.com/r/intel/oneapi-basekit). Then, you can use the commands given above.

Check [Optimizing and Running LLaMA2 on Intel® CPU](https://builders.intel.com/solutionslibrary/optimizing-and-running-llama2-on-intel-cpu) for more information.

### Other BLAS libraries

Any other BLAS library can be used by setting the `GGML_BLAS_VENDOR` option. See the [CMake documentation](https://cmake.org/cmake/help/latest/module/FindBLAS.html#blas-lapack-vendors) for a list of supported vendors.

## HIP

This fork only targets the AMD Instinct MI100 (`gfx908`) through ROCm/HIP.
Make sure to have ROCm installed. You can get it from your Linux distro's package manager or from here: [ROCm Quick Start (Linux)](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/tutorial/quick-start.html#rocm-install-quick).

### Compilation

```bash
cmake -B build -DGGML_HIP=ON -DGPU_TARGETS=gfx908
cmake --build build -j
```

If CMake does not pick up the ROCm toolchain on its own, point it at the HIP clang explicitly:

```bash
HIPCXX="$(hipconfig -l)/clang" HIP_PATH="$(hipconfig -R)" \
    cmake -B build -DGGML_HIP=ON -DGPU_TARGETS=gfx908 -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

If you get `clang: error: cannot find ROCm device library`, search for a directory under `HIP_PATH` that contains `oclc_abi_version_400.bc` and prepend `HIP_DEVICE_LIB_PATH=<that-directory>` to the configure command.

### Nix

The flake provides a ROCm package built for gfx908:

```bash
nix build .#rocm
```

### Runtime environment variables

The environment variable [`HIP_VISIBLE_DEVICES`](https://rocm.docs.amd.com/en/latest/understand/gpu_isolation.html#hip-visible-devices) can be used to specify which GPU(s) will be used.

The HIP backend shares its implementation with the `ggml-cuda` sources, so its runtime variables keep the `GGML_CUDA_` prefix:

- `GGML_CUDA_ENABLE_UNIFIED_MEMORY=1` enables unified memory on Linux, allowing allocations to spill to system RAM instead of failing when VRAM is exhausted (this hurts performance on discrete GPUs).
- `GGML_CUDA_P2P` enables peer-to-peer access between multiple GPUs so they can transfer data directly rather than through system memory. May cause crashes or corrupted outputs on some motherboards and BIOS settings (e.g. IOMMU).

### Performance Tuning

The following compilation options are also available to tweak performance:

| Option                        | Legal values           | Default | Description                                                                                                                                                                                                                                                                                                                                                                      |
|-------------------------------|------------------------|---------|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| GGML_CUDA_FORCE_MMQ           | Boolean                | false   | Force the use of custom matrix multiplication kernels for quantized models instead of FP16 cuBLAS even if there is no int8 tensor core implementation available (affects V100, CDNA and RDNA3+). MMQ kernels are enabled by default on GPUs with int8 tensor core support. With MMQ force enabled, speed for large batch sizes will be worse but VRAM consumption will be lower. |
| GGML_CUDA_FORCE_CUBLAS        | Boolean                | false   | Force the use of FP16 cuBLAS instead of custom matrix multiplication kernels for quantized models. There may be issues with numerical overflows (except for V100, CDNA and RDNA4 which use FP32 compute type by default) and memory use will be higher. Prompt processing may become faster on recent datacenter GPUs (the custom kernels were tuned primarily for RTX 3000/4000).   |
| GGML_CUDA_FA_QUANTS           | `all` or `type_K-type_V` list | q4_0-q4_0;q8_0-q8_0;f16-f16;bf16-bf16 | Select which K/V type combinations to compile the FlashAttention CUDA kernels for. `all` compiles every combination, but compilation takes much longer. Otherwise a `;`-separated list of `type_K-type_V` pairs; f16-f16 is always compiled. Combinations that were not compiled fall back to f16-f16 kernel with a warning. Legal types: f16, bf16, q4_0, q4_1, q5_0, q5_1, q8_0. |
| GGML_CUDA_FA_ALL_QUANTS       | Boolean                | false   | Deprecated alias for `GGML_CUDA_FA_QUANTS=all`.                                                                                                                                                                                                                                                                                                                               |

## Arm® KleidiAI™
KleidiAI provides optimized Arm CPU microkernels used by the ggml CPU backend. Enabling it at build time makes those kernels available; it does not force every operation to use KleidiAI. At runtime, llama.cpp selects the best compatible CPU kernel from the detected CPU features, tensor type, operation shape, and active backend priority.

Supported targets:

| Platform | Supported ABI / architecture | Notes |
| --- | --- | --- |
| Linux | AArch64 / arm64 | Runtime CPU feature detection is automatic. |
| Android | `arm64-v8a` | Use the Android NDK command below for a portable build. |
| Apple | arm64 | Runtime CPU feature detection is automatic. Non-streaming SVE vector length is treated as unavailable. |
| Windows | arm64 | Runtime CPU feature detection is automatic. SMCU count is treated as unknown until a detection path is verified. |

`GGML_CPU_KLEIDIAI=ON` is valid only for AArch64/arm64 builds. Do not enable it for x86, 32-bit Arm, or Android ABIs other than `arm64-v8a`.

### Native AArch64/arm64 build

From the llama.cpp source directory:

```bash
cmake -S . -B build -DGGML_CPU_KLEIDIAI=ON
cmake --build build --config Release
```

### Android arm64-v8a NDK build

Set `ANDROID_NDK` to the Android NDK root, then run the following from the llama.cpp source directory. This command configures a portable Android `arm64-v8a` build with KleidiAI enabled and avoids Android dependencies that are not part of the NDK stable native API set.

```bash
cmake -S . -B build-android \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_TOOLCHAIN_FILE="$ANDROID_NDK/build/cmake/android.toolchain.cmake" \
  -DANDROID_ABI=arm64-v8a \
  -DANDROID_PLATFORM=android-28 \
  -DGGML_CPU_KLEIDIAI=ON \
  -DGGML_NATIVE=OFF \
  -DGGML_OPENMP=OFF \
  -DGGML_LLAMAFILE=OFF \
  -DLLAMA_OPENSSL=OFF
cmake --build build-android --config Release --parallel
cmake --install build-android --prefix {install-dir} --config Release
```

Important Android options:

- `GGML_CPU_KLEIDIAI=ON` enables KleidiAI for Android `arm64-v8a`.
- `GGML_NATIVE=OFF` is required for cross-compilation because the build host CPU is not the Android target CPU.
- `GGML_OPENMP=OFF` avoids adding an OpenMP runtime dependency to this NDK command-line build.
- `GGML_LLAMAFILE=OFF` avoids the llamafile backend, which is not supported on Android.
- `LLAMA_OPENSSL=OFF` avoids depending on OpenSSL, which is not part of the Android NDK stable native API set.

The Android Studio project under `examples/llama.android` enables KleidiAI automatically for `arm64-v8a`. For Android command-line CMake builds on `arm64-v8a`, pass `-DGGML_CPU_KLEIDIAI=ON` explicitly.

Global -march flags such as `-march=armv8.7a` flag are not required for a portable Android `arm64-v8a` build. Global `-march` flags raise the baseline instruction set for generic code. No manual architecture-specific source selection is required; llama.cpp selects compatible KleidiAI kernels at runtime. The KleidiAI libraries internal CMake handles the -march flags for each particular kernel.

### Verifying the build

Run an installed or in-tree binary:

```bash
./build/bin/llama-cli -m PATH_TO_MODEL -p "What is a car?"
```

If KleidiAI is enabled, the output contains a line similar to:

```
load_tensors: CPU_KLEIDIAI model buffer size =  3474.00 MiB
```

This confirms that the model has tensors allocated through the KleidiAI CPU buffer. It does not prove that every operation, or any specific SME-family operation, used a KleidiAI microkernel. Runtime CPU features, tensor type, operation shape, and backend priority still control dispatch.

Depending on the build target, another backend may have higher priority than the CPU backend. To force CPU execution for a run, disable higher priority backends at build time, for example `-DGGML_HIP=OFF`, or use a runtime device option such as `--device none` where supported.

### Runtime dispatch

KleidiAI microkernels use Arm CPU features such as dotprod, i8mm, SVE, and SME/SME2. Build-time configuration makes the kernels available. Runtime dispatch selects a compatible kernel for the detected CPU and operation. Older or lower-feature CPUs fall back automatically to compatible kernels.

KleidiAI accelerates selected `GGML_OP_MUL_MAT` paths for F32 and common quantized formats. Exact coverage depends on the bundled KleidiAI version and the llama.cpp runtime selector, so unsupported tensor types, unsupported operation shapes, or higher priority backends may bypass KleidiAI even when the CPU supports the required Arm feature. This is also why a model may not use SME-family kernels on SME-capable hardware.

The current llama.cpp KleidiAI SVE selector only enables SVE kernels when the runtime SVE vector length is known to be QK8_0 bytes, currently 32 bytes. Linux and Android query this at runtime. Apple reports SVE capability separately from userspace non-streaming SVE availability, so llama.cpp treats the SVE vector length as unknown there. Windows exposes SVE feature presence but not the runtime SVE vector length used by this selector, so that value is also treated as unknown. Windows arm64 also treats SMCU count as unknown until a detection mechanism is verified.

The set of available SME-family kernels depends on the bundled KleidiAI version and the detected CPU capabilities. Production configuration does not require any KleidiAI runtime environment variables.

### Diagnostics and debug overrides

KleidiAI runtime environment variables are diagnostics/debug overrides, not production configuration. Leave them unset for normal use.

`GGML_KLEIDIAI_SME` controls SME-family kernel selection and overrides the maximum number of threads assigned to selected quantized SME-family kernels:

- Not set: use automatic runtime detection.
- `0`: disable SME-family kernels.
- `<n> > 0`: enable compatible SME-family kernels and allow up to `<n>` threads for quantized SME-family kernels.

On Windows arm64, use `GGML_KLEIDIAI_SME=<n>` as the temporary diagnostics/debug override for SME thread-cap calibration until automatic SMCU count detection is verified.

If the CPU does not support the required SME-family capability for a bundled kernel, that kernel is disabled regardless of the environment variable.

## Android

To read documentation for how to build on Android, [click here](./android.md)

## IBM Z & LinuxONE

To read documentation for how to build on IBM Z & LinuxONE, [click here](./build-s390x.md)

---
## Notes about GPU-accelerated backends

The GPU may still be used to accelerate some parts of the computation even when using the `-ngl 0` option. You can fully disable GPU acceleration by using `--device none`.

In most cases, it is possible to build and use multiple backends at the same time. For example, you can build llama.cpp with both HIP and BLAS support by using the `-DGGML_HIP=ON -DGGML_BLAS=ON` options with CMake. At runtime, you can specify which backend devices to use with the `--device` option. To see a list of available devices, use the `--list-devices` option.

Backends can be built as dynamic libraries that can be loaded dynamically at runtime. This allows you to use the same llama.cpp binary on different machines with different GPUs. To enable this feature, use the `GGML_BACKEND_DL` option when building.
