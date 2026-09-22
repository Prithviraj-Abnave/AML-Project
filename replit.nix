{ pkgs }: {
  deps = [
    pkgs.python311
    pkgs.python311Packages.pip
    pkgs.nss
    pkgs.nspr
    pkgs.dbus
    pkgs.atk
    pkgs.at-spi2-atk
    pkgs.cups
    pkgs.libdrm
    pkgs.expat
    pkgs.xorg.libXcomposite
    pkgs.xorg.libXdamage
    pkgs.xorg.libXfixes
    pkgs.mesa
    pkgs.xorg.libxcb
    pkgs.libxkbcommon
    pkgs.pango
    pkgs.cairo
    pkgs.alsa-lib
    pkgs.at-spi2-core
  ];
  env = {
    LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath [
      pkgs.nss
      pkgs.nspr
      pkgs.dbus
      pkgs.atk
      pkgs.at-spi2-atk
      pkgs.cups
      pkgs.libdrm
      pkgs.expat
      pkgs.xorg.libXcomposite
      pkgs.xorg.libXdamage
      pkgs.xorg.libXfixes
      pkgs.mesa
      pkgs.xorg.libxcb
      pkgs.libxkbcommon
      pkgs.pango
      pkgs.cairo
      pkgs.alsa-lib
      pkgs.at-spi2-core
    ];
  };
}
