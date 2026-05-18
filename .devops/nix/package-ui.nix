{
  lib,
  buildNpmPackage,
  llamaVersion ? "0.0.0",
}:

# Builds the llama.cpp server's embedded web UI (tools/ui, a SvelteKit app).
#
# `buildNpmPackage` resolves the npm dependency closure in a fixed-output
# derivation (`npm ci` against tools/ui/package-lock.json) — the only part of
# the build granted network access. `npmDepsHash` pins that FOD's contents.
#
# The result is consumed by package.nix, which stages these assets where
# tools/ui/CMakeLists.txt expects a "local" UI source, so the main build never
# needs npm or network access of its own.

buildNpmPackage {
  pname = "llama-cpp-ui";
  version = llamaVersion;

  src = lib.cleanSourceWith {
    name = "llama-cpp-ui-source";
    src = lib.cleanSource ../../tools/ui;
    # Drop build artifacts a developer may have left in the working tree, so
    # they neither bloat the source nor perturb the output hash.
    filter =
      name: _type:
      !builtins.elem (baseNameOf name) [
        "node_modules"
        ".svelte-kit"
        "build"
        "dist"
        "test-results"
        "storybook-static"
      ];
  };

  # Fixed-output derivation hash for the npm dependency closure.
  # Regenerate whenever tools/ui/package-lock.json changes:
  #   nix run nixpkgs#prefetch-npm-deps -- tools/ui/package-lock.json
  # Nix also prints the expected value when this hash goes stale.
  npmDepsHash = "sha256-WaEePrEZ7O/7deP2KJhe0AwiSKYA8HOqETmMHUkmBe0=";

  # The SvelteKit static adapter (svelte.config.js) and the project's custom
  # Vite plugin both hard-code an output directory of `../../build/tools/ui/dist`,
  # a path that only resolves inside a full llama.cpp checkout. Building this
  # package standalone (src = tools/ui), redirect both to a local `dist/`.
  postPatch = ''
    substituteInPlace svelte.config.js \
      --replace-fail "../../build/tools/ui/dist" "dist"
    substituteInPlace scripts/vite-plugin-llama-cpp-build.ts \
      --replace-fail "../../build/tools/ui/dist" "dist"
  '';

  # Playwright is a dev dependency pulled in by `npm ci`; its install script
  # downloads browser binaries over the network. The sandbox has no network and
  # the build only needs `vite build`, so skip that download.
  env.PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD = "1";

  # `npm run build` -> `vite build`
  npmBuildScript = "build";

  # Keep only the four assets that tools/ui/CMakeLists.txt embeds via xxd.cmake.
  installPhase = ''
    runHook preInstall

    for asset in index.html bundle.js bundle.css loading.html; do
      install -D -m644 "dist/$asset" "$out/$asset"
    done

    runHook postInstall
  '';

  meta = {
    description = "Embedded web UI assets for the llama.cpp server";
    homepage = "https://github.com/ggml-org/llama.cpp/";
    license = lib.licenses.mit;
  };
}
