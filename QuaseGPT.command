#!/bin/sh
# macOS double-click launcher for QuaseGPT.
# Uses its own file location — no hardcoded paths.
DIR="$(CDPATH='' cd -- "$(dirname -- "$0")" && pwd)"
cd "$DIR" && exec ./quasegpt
