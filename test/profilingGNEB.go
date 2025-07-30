//go:build ignore
// +build ignore

package main

import (
	"os"
	"runtime"
	"runtime/pprof"
	"time"

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
	SetOutputFormat(OVF2_TEXT)
	AdvanceTime = true
	MinimizePath = true
	defer InitAndClose()()
	gridsize := [3]int{1, 1, 1}
	cellsize := [3]float64{1e-9, 1e-9, 0.4e-9}
	SetMesh(gridsize[0], gridsize[1], gridsize[2],
		cellsize[0], cellsize[1], cellsize[2],
		1, 1, 0,
		5)
	// SetGridSize(gridsize[0], gridsize[1], gridsize[2])
	// SetCellSize(cellsize[0], cellsize[1], cellsize[2])
	// n_images := 5
	// fmt.Println(M.N_images)
	// SetNImages(n_images)
	// fmt.Println(M.N_images)
	// // SetPBC(1, 1, 0)

	// Parameters
	Msat.Set(580e3)               // Saturation magnetization, A/m
	Aex.Set(15e-12)               // Exchange stiffness, J/m
	Alpha.Set(0.1)                // Landau-Lifshitz damping constant
	Ku1.Set(8e5)                  // 1st order uniaxial anisotropy constant J/m^3
	AnisU.Set([]float64{0, 0, 1}) // Uniaxial anisotropy direction
	// B_ext.Set(Vector(0, 1, 0))
	// Dind.Set(0.003) // Interfacial Dzyaloshinskii-Moriya strength, J/m^2
	// initial_config := NeelSkyrmion(1, -1).Scale(2, 2, 1).Transl(32e-9, 0, 0)
	// initial_config = initial_config.Add(1, NeelSkyrmion(1, -1).Scale(2, 2, 1).Transl(-32e-9, 0, 0))
	// M.SetInShape(nil, initial_config)
	M.SetInShape(nil, Uniform(0, 0, 1), 0)
	M.SetInShape(nil, Uniform(0, 0, -1), 4)
	AngularInterpolation(M.Buffer(), false)

	// Output
	// AutoSave(&M, Dt_si)
	Snapshot(&M)
	// Run
	start := time.Now()
	VPOMinimize()
	elapsed := time.Since(start)

	// Additional Output
	LogOut("Note: Bumping massVPO from 1.0 to 10.0")
	LogOut("Gridsize", Mesh().Size())
	LogOut("CellSize", Mesh().CellSize())
	LogOut("PBC", Mesh().PBC())
	LogOut("Msat", Msat.Average())
	LogOut("Aex", Aex.Average())
	LogOut("Alpha", Alpha.Average())
	LogOut("Ku1", Ku1.Average())
	LogOut("AnisU", AnisU.Average())
	LogOut("B_ext", B_ext.Average())
	LogOut("Dind", Dind.Average())
	LogOut("VPO stats:")
	LogOut("M_avg: ", M.Average())
	CalculateTotalImageEnergies(&M)
	LogOut("E_tot: ", M.E_img)
	LogOut("NSteps: ", NSteps)
	LogOut("Elapsed Time: ", elapsed)
	Snapshot(&M)
}
