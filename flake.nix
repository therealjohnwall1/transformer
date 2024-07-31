{
  description = "A very basic flake";

  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs?ref=nixos-unstable";
    vscode-server.url = "github:nix-community/nixos-vscode-server"
  };

  outputs = { self, nixpkgs, vscode-server }: {
    defaultPackage.x86_64-linux =
      # Notice the reference to nixpkgs here.
      with import nixpkgs { system = "x86_64-linux";
	modules = [
	   vscode-server.enable = true;
	]};
      let
        pkgs = import nixpkgs {
          inherit system;
          config = { allowUnfree = true; };
        };
      in 
        stdenvNoCC.mkDerivation {
          nativeBuildInputs = [
            pkgs.python311
            pkgs.cudatoolkit
	    pkgs.python311Packages.numpy
	    pkgs.python311Packages.torch-bin
          ];
          name = "hello";
          src = self;
	  #:${pkgs.linuxPackages.nvidia_x11}/lib
	  shellHook = ''
          echo hi
          export LD_LIBRARY_PATH=${stdenv.cc.cc.lib}/lib/
	  export CUDA_PATH=${pkgs.cudatoolkit}
        '';
        };
  };
}
