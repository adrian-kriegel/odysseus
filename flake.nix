{
  description = "Development environment for Odysseus.";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-24.05";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
      in {
        devShell = pkgs.mkShell {
          packages = with pkgs; [
            poetry
            (python312.withPackages (ps: with ps; [ poetry-core ]))
          ];

          shellHook = ''
            # required for python to find libstdc++ etc.
            export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${pkgs.stdenv.cc.cc.lib}/lib/:${pkgs.zlib}/lib/:${pkgs.glib.dev}/lib/:${pkgs.glib}/lib/

            # ACADOS Extrawurst
            export ACADOS_SOURCE_DIR=${builtins.getEnv "PWD"}/submodules/acados
            export ACADOS_DIR=$ACADOS_SOURCE_DIR
            export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$ACADOS_DIR/lib
          '';
        };
      }
    );
}