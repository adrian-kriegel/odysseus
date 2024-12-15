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

          # Test dependencies
          pytest
          # Only needed for the examples
          pkgs.acados_template
        ]);

      in {

        packages.default = pkgs.python3Packages.buildPythonPackage {
          
          name = "odysseus";
          version = "0.0.1";
          src = ./.;

          format = "setuptools";

          buildInputs = with pkgs.python3Packages; [
            setuptools
          ];

          propagatedBuildInputs = with pkgs.python3Packages; [
            numpy
            sympy
            symengine
            casadi
          ];

          meta = with pkgs.lib; {
            description = "";
            homepage = "https://github.com/adrian-kriegel/odysseus";
          };
        };

        devShell = pkgs.mkShell {
          buildInputs = with pkgs; [ python acados ];

          shellHook = ''
            # required for python to find libstdc++ etc.
            export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${pkgs.stdenv.cc.cc.lib}/lib/:${pkgs.zlib}/lib/:${pkgs.glib.dev}/lib/:${pkgs.glib}/lib/

            # so we can import odysseus
            export PYTHONPATH=$PYTHONPATH:$(pwd)
          '';
        };
      }
    );
}