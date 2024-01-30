#!/usr/bin/env bash
set -e

export GOOS=linux
export GOARCH=mips
export GOMIPS=softfloat
go build -o ./ml_mips

file ml_mips

if [[ ! -d venv ]]; then
    python3 -m venv venv
fi

../compile.py ml_mips
