#/bin/bash

BUILD=1
BROKEN=1
DUP=1
RELATED=1

BASE=/data/xukp

if [ $BUILD -eq 1 ]; then
    echo "Building"
    python build_index.py -f include_dirs.json
fi

if [ $BROKEN -eq 1 ]; then
    echo "Finding broken"
    python broken.py
fi

if [ $DUP -eq 1 ]; then
    echo "Finding duplicates"
    python duplicate.py
fi