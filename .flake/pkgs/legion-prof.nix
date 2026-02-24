{ lib
, fetchFromGitLab
, rustPlatform
}:

rustPlatform.buildRustPackage rec {
  pname = "legion-prof";
  version = "2026-02-24";

  src_root = fetchFromGitLab {
    owner = "StanfordLegion";
    repo = "legion";
    rev = "42abd7a5d16d26a802f733f59341414dbe14af72";
    sha256 = "sha256-g3IeYKuVV+gHoxcHSo9w33T1PqTmZ/v0IKSU1BVyOsI=";
  };
  src = src_root + "/tools/legion_prof_rs";

  cargoLock.lockFile = src + "/Cargo.lock";

  cargoHash = "fakeHash";
}
