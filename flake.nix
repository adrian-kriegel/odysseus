{
  description = "Development environment for Odysseus.";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs";

    flake-utils.url = "github:numtide/flake-utils";

    acados-overlay.url = "github:adrian-kriegel/acados-nix";
  };

  outputs = {
    self,
    nixpkgs,
    flake-utils,
    acados-overlay,
  }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { 
          inherit system; 

          overlays = [
            acados-overlay.overlays.default
          ];
        };
        
        python = pkgs.python312.withPackages (ps: with ps; [ 
          
          numpy
          sympy
          symengine
          casadi

          pytest

          # Only needed for the examples
          pkgs.acados_template
        ]);
      in {
        devShell = pkgs.mkShell {
          buildInputs = with pkgs; [ python acados ];

          shellHook = ''
            # required for python to find libstdc++ etc.
            export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${pkgs.stdenv.cc.cc.lib}/lib/:${pkgs.zlib}/lib/:${pkgs.glib.dev}/lib/:${pkgs.glib}/lib/

            # ACADOS Extrawurst
            export ACADOS_SOURCE_DIR=${pkgs.acados}
            
            # so we can import odysseus
            export PYTHONPATH=$PYTHONPATH:$(pwd)
          '';
        };
      }
    );
}