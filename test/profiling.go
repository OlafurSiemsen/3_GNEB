//go:build ignore
// +build ignore

package main

import (
	"fmt"
	"os"
	"runtime"
	"runtime/pprof"

	. "github.com/mumax/3/engine"
)

func main() {
	// Create a file to store the CPU profile
	cpuFile, err := os.Create("cpu.pprof")
	if err != nil {
		panic(err)
	}
	defer cpuFile.Close()

	// Start CPU profiling
	runtime.SetBlockProfileRate(1)
	if err := pprof.StartCPUProfile(cpuFile); err != nil {
		panic(err)
	}
	defer pprof.StopCPUProfile()

	// MuMax Section
	// Setting up standardproblem4
	defer InitAndClose()()

	SetGridSize(128, 32, 1)
	SetCellSize(500e-9/128, 125e-9/32, 3e-9)

	Msat.Set(800e3)
	Aex.Set(13e-12)
	Alpha.Set(0.02)
	M.Set(Uniform(1, .1, 0))

	AutoSave(&M, 100e-12)
	TableAdd(MaxTorque)
	TableAutoSave(5e-12)

	Relax()

	fmt.Printf("%+v \n", M.Buffer().MemType() == 1<<1)
	// reversal
	B_ext.Set(Vector(-24.6e-3, 4.3e-3, 0))
	Run(1e-9)
	TOL := 1e-3
	ExpectV("m", M.Average(), Vector(-0.9846124053001404, 0.12604089081287384, 0.04327124357223511), TOL)
}
