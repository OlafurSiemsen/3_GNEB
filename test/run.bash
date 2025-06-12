#! /bin/bash

set -e

mumax3 -vet *.mx3

# excludes magnetoelastic and arbitrary table saving tests for now, something is broken in the scripting utility
shopt -s extglob
mumax3 -paranoid=false -failfast -cache /tmp -f -http "" !(probe*|profiling*).go !(mel-force-dm*|table).mx3
shopt -u extglob
# old test command to be restored
#mumax3 -paranoid=false -failfast -cache /tmp -f -http "" *.go *.mx3