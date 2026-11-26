#!/bin/sh
# Descarga los niveles procesados de Super Mario Bros del Video Game Level
# Corpus (VGLC, licencia MIT) en data/vglc/smb. Los diseños originales son de
# Nintendo, así que se usan solo con fines académicos y no se versionan.
set -eu
BASE="https://raw.githubusercontent.com/TheVGLC/TheVGLC/master"
DEST="$(dirname "$0")/../data/vglc/smb"
mkdir -p "$DEST"
for n in 1-1 1-2 1-3 2-1 3-1 3-3 4-1 4-2 5-1 5-3 6-1 6-2 6-3 7-1 8-1; do
    curl -fsSL "$BASE/Super%20Mario%20Bros/Processed/mario-$n.txt" -o "$DEST/mario-$n.txt"
done
curl -fsSL "$BASE/Super%20Mario%20Bros/smb.json" -o "$DEST/smb.json"
curl -fsSL "$BASE/License.md" -o "$DEST/LICENSE.md"
echo "VGLC descargado en $DEST"
