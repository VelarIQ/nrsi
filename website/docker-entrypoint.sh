#!/bin/sh
set -eu

if [ -z "${PORT:-}" ]; then
  export PORT=8080
fi
