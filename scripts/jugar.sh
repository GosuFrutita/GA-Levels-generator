#!/bin/sh
# Lanza el juego en Godot para probar los niveles generados por nivelga.
# Uso: sh scripts/jugar.sh                     (menú: modo infinito o evolución)
#      sh scripts/jugar.sh --nivel=RUTA.json   (un nivel concreto)
# El ejecutable de Godot se busca en la variable GODOT, en el PATH o en la
# carpeta de la versión anterior del proyecto.
set -eu
RAIZ="$(cd "$(dirname "$0")/.." && pwd)"
if [ -z "${GODOT:-}" ]; then
    if command -v godot >/dev/null 2>&1; then
        GODOT="$(command -v godot)"
    else
        GODOT="$RAIZ/../Algoritmos genéticos/godot/Godot_v4.7.2-stable_linux.x86_64"
    fi
fi
if [ ! -x "$GODOT" ]; then
    echo "No encuentro Godot 4; indica su ruta con GODOT=/ruta/a/godot" >&2
    exit 1
fi
if [ ! -x "$RAIZ/target/release/nivelga" ]; then
    (cd "$RAIZ" && cargo build --release)
fi
exec "$GODOT" --path "$RAIZ/godot" -- "$@"
