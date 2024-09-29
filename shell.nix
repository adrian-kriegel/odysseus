{ pkgs ? import <nixpkgs> {} }:
pkgs.mkShell {
  packages = with pkgs; [
    poetry
    (python312.withPackages (ps: with ps; [ 
            poetry-core
        ]
    ))
  ];

  shellHook = ''
    # required for python to find libstdc++ etc.
    export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:${pkgs.stdenv.cc.cc.lib}/lib/:${pkgs.zlib}/lib/:${pkgs.glib.dev}/lib/:${pkgs.glib}/lib/
  '';
}