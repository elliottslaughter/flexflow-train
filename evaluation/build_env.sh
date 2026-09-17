# Finds the FlexFlow build and defines how to build and run it. Sourced by the
# other scripts, which expect $REPO to be set.
#
# Two layouts exist. deploy/sapling.sh configures a plain CMake build under
# build/ whose binaries run directly. proj configures a Nix build under
# build/release/ whose binaries only run inside `nix develop`, and need nixGL to
# reach the host driver. The deploy build wins when both are present.
#
# Sets BIN and GL, and defines ffbuild and ffrun.

if [ -e "$REPO/build/CMakeCache.txt" ]; then
  FF_BUILD_KIND="deploy (no Nix)"
  BIN="build/bin"
  GL=""

  ffrun() { ( cd "$REPO" && bash -c "$*" ); }
  ffbuild() { ( cd "$REPO" && make -C build -j"$(nproc)" ); }
else
  FF_BUILD_KIND="Nix"
  BIN="build/release/bin"
  GL="nixGL --"

  ffrun() {
    ( cd "$REPO" && NIXPKGS_ALLOW_UNFREE=1 nix develop .#gpu \
        --accept-flake-config --impure --command bash -c "$*" )
  }
  ffbuild() { ffrun 'proj build --release'; }
fi
