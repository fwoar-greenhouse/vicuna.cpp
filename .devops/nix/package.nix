{
  lib,
  glibc,
  config,
  stdenv,
  stdenvNoCC,
  cmake,
  ninja,
  pkg-config,
  git,
  mpi,
  blas,
  darwin,
  rocmPackages,
  openssl,
  nodejs,
  importNpmLock,
  useBlas ? !useRocm && blas.meta.available,
  # Increases the runtime closure size by ~700M
  useMpi ? false,
  useRocm ? config.rocmSupport,
  # This fork targets AMD MI100 (CDNA1) only
  rocmGpuTargets ? "gfx908",
  useRpc ? false,
  llamaVersion ? "0.0.0", # Arbitrary version, substituted by the flake
  llamaCommit ? "unknown", # Git revision, substituted by the flake

  effectiveStdenv ? stdenv,
  enableStatic ? effectiveStdenv.hostPlatform.isStatic,
  useWebUi ? true,
}:

let
  inherit (lib)
    cmakeBool
    cmakeFeature
    optionalAttrs
    optionals
    strings
    ;

  stdenv = throw "Use effectiveStdenv instead";

  suffices =
    lib.optionals useBlas [ "BLAS" ]
    ++ lib.optionals useMpi [ "MPI" ]
    ++ lib.optionals useRocm [ "ROCm" ];

  pnameSuffix =
    strings.optionalString (suffices != [ ])
      "-${strings.concatMapStringsSep "-" strings.toLower suffices}";
  descriptionSuffix = strings.optionalString (
    suffices != [ ]
  ) ", accelerated with ${strings.concatStringsSep ", " suffices}";

  # apple_sdk is supposed to choose sane defaults, no need to handle isAarch64
  # separately
  darwinBuildInputs =
    with darwin.apple_sdk.frameworks;
    [
      Accelerate
      CoreVideo
      CoreGraphics
    ];

  rocmBuildInputs = with rocmPackages; [
    clr
    hipblas
    rocblas
  ];
in

effectiveStdenv.mkDerivation (finalAttrs: {
  pname = "llama-cpp${pnameSuffix}";
  version = llamaVersion;

  # Note: none of the files discarded here are visible in the sandbox or
  # affect the output hash. This also means they can be modified without
  # triggering a rebuild.
  src = lib.cleanSourceWith {
    filter =
      name: type:
      let
        noneOf = builtins.all (x: !x);
        baseName = baseNameOf name;
      in
      noneOf [
        (lib.hasSuffix ".nix" name) # Ignore *.nix files when computing outPaths
        (lib.hasSuffix ".md" name) # Ignore *.md changes whe computing outPaths
        (lib.hasPrefix "." baseName) # Skip hidden files and directories
        (baseName == "flake.lock")
      ];
    src = lib.cleanSource ../../.;
  };

  # Builds the webui locally, taking care not to require updating any sha256 hash.
  webui = stdenvNoCC.mkDerivation {
    pname = "webui";
    version = llamaVersion;
    src = lib.cleanSource ../../tools/ui;

    nativeBuildInputs = [
      nodejs
      importNpmLock.linkNodeModulesHook
    ];

    # no sha256 required when using buildNodeModules
    npmDeps = importNpmLock.buildNodeModules {
      npmRoot = ../../tools/ui;
      inherit nodejs;
    };

    installPhase = ''
      LLAMA_UI_OUT_DIR=$out npm run build --offline
    '';
  };

  postPatch = lib.optionalString useWebUi ''
    cp -r ${finalAttrs.webui} tools/ui/dist
    chmod -R u+w tools/ui/dist
  '';

  nativeBuildInputs =
    [
      cmake
      ninja
      pkg-config
      git
    ]
    ++ optionals (effectiveStdenv.hostPlatform.isGnu && enableStatic) [ glibc.static ];

  buildInputs =
    optionals effectiveStdenv.hostPlatform.isDarwin darwinBuildInputs
    ++ optionals useMpi [ mpi ]
    ++ optionals useRocm rocmBuildInputs
    ++ optionals useBlas [ blas ]
    ++ [ openssl ];

  cmakeFlags =
    [
      (cmakeBool "LLAMA_BUILD_SERVER" true)
      (cmakeBool "LLAMA_BUILD_WEBUI" useWebUi)
      (cmakeBool "BUILD_SHARED_LIBS" (!enableStatic))
      (cmakeBool "CMAKE_SKIP_BUILD_RPATH" true)
      (cmakeBool "GGML_NATIVE" false)
      (cmakeBool "GGML_BLAS" useBlas)
      (cmakeBool "GGML_HIP" useRocm)
      (cmakeBool "GGML_STATIC" enableStatic)
      (cmakeBool "GGML_RPC" useRpc)
      (cmakeFeature "LLAMA_BUILD_COMMIT" llamaCommit)
    ]
    ++ optionals useRocm [
      (cmakeFeature "CMAKE_HIP_COMPILER" "${rocmPackages.llvm.clang}/bin/clang")
      (cmakeFeature "CMAKE_HIP_ARCHITECTURES" rocmGpuTargets)
    ];

  # Environment variables needed for ROCm
  env = optionalAttrs useRocm {
    ROCM_PATH = "${rocmPackages.clr}";
    HIP_DEVICE_LIB_PATH = "${rocmPackages.rocm-device-libs}/amdgcn/bitcode";
  };

  # TODO(SomeoneSerge): It's better to add proper install targets at the CMake level,
  # if they haven't been added yet.
  postInstall = ''
    mkdir -p $out/include
    cp $src/include/llama.h $out/include/
  '';

  meta = {
    description = "Inference of LLaMA model in pure C/C++${descriptionSuffix}";
    homepage = "https://github.com/ggml-org/llama.cpp/";
    license = lib.licenses.mit;

    # Accommodates `nix run` and `lib.getExe`
    mainProgram = "llama-cli";

    # These people might respond, on the best effort basis, if you ping them
    # in case of Nix-specific regressions or for reviewing Nix-specific PRs.
    # Consider adding yourself to this list if you want to ensure this flake
    # stays maintained and you're willing to invest your time. Do not add
    # other people without their consent. Consider removing people after
    # they've been unreachable for long periods of time.

    # Note that lib.maintainers is defined in Nixpkgs, but you may just add
    # an attrset following the same format as in
    # https://github.com/NixOS/nixpkgs/blob/f36a80e54da29775c78d7eff0e628c2b4e34d1d7/maintainers/maintainer-list.nix
    maintainers = with lib.maintainers; [
      philiptaron
      SomeoneSerge
    ];

    # Extend `badPlatforms` instead
    platforms = lib.platforms.all;
  };
})
