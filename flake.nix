{
  description = "Pallas Development Environment";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-25.05";
  };

  outputs = { self, nixpkgs, ... }: let
    system = "aarch64-darwin";
  in {
    devShells."${system}".default = let
      pkgs = import nixpkgs { inherit system; };
    in pkgs.mkShell {

      packages = with pkgs; [
        # core
        gcc 
        git

        # build utilities
        cmake
        doxygen
        pkg-config

        # library dependencies
        mpi
        zstd
      ];

      shellHook = ''
      '';

    };
  };
}
