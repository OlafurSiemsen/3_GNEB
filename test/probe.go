//go:build ignore
// +build ignore

/*
	This file is as a way to test various functions and snippets.
	Its contents should be treated as volatile
*/

package main

import (
	"time"

	"github.com/mumax/3/data"
	. "github.com/mumax/3/engine"
)

func main() {

	defer InitAndClose()()

	// gridsize := [3]int{4, 4, 4}
	// size := gridsize[0] * gridsize[1] * gridsize[2]
	// Is there really no non-script way to set anisU?
	// Eval(`
	// 	Msat  = 860E3
	// 	Aex   = 13E-12
	// 	Ku1   = 50
	// 	alpha = 3
	// 	anisU = vector(1,0,0)
	//  m = VortexWall(1, -1, 1, 1).scale(1/2, 1, 1)
	// 	`)

	// const t_eval_string = `
	// 		setgridsize(128, 32, 1)
	// 		setcellsize(500e-9/128, 125e-9/32, 3e-9)

	// 		Msat  = 1600e3
	// 		Aex   = 13e-12

	// 		Msat  = 800e3
	// 		alpha = 0.02

	// 	`
	// Eval(t_eval_string)

	// (Hyper)Geometry
	MinimizePath = false
	gridsize := [3]int{256, 256, 1}
	SetGridSize(gridsize[0], gridsize[1], gridsize[2])
	SetCellSize(1e-9, 1e-9, 0.4e-9)
	SetPBC(1, 1, 0)
	n_images := 1
	SetNImages(n_images)

	// Parameters

	Msat.Set(580e3) // Saturation magnetization, A/m
	Aex.Set(15e-12) // Exchange stiffness J/m
	Alpha.Set(0.1)  // Landau-Lifshitz damping constant
	Dind.Set(0.0034089785)
	initial_config := NeelSkyrmion(1, -1).Transl(-30e-9, 0e-9, 0)
	B_ext.Set(data.Vector{0, 0, 0.01}) // Externally applied field, T

	// Ku1.Set(0.59e6 + 4*math.Pi*1e-7*0.5*580e3*580e3) // 1st order uniaxial anisotropy constant J/m^3
	// AnisU.Set([]float64{0, 0, 1})                    // Uniaxial anisotropy direction
	// without compensating for in-plane tilts of the background this fails (corresponds to ext_backgroundtilt=0)
	// Dbulk.Set(1e-3)                                  // Bulk Dzyaloshinskii-Moriya strength J/m^2
	// Temp.Set(100)
	// Setup
	// SetOutputFormat(OVF2_TEXT)
	// TableAdd(E_total)
	// TableAutoSave(100 * Dt_si)
	// AutoSave(E_total, 10e-15)
	// t_seed := int(time.Now().UTC().UnixNano())
	// M.Set(RandomMagSeed(t_seed))
	// M.Set(Uniform(1, 0, 0))
	// t_config := VortexWall(1, -1, 1, 1).Scale(0.5, 1, 1)
	// t_config :=   BlochSkyrmion(-1, 1).Scale(6, 6, 6)
	M.Set(initial_config)
	Snapshot(&M)
	start := time.Now()
	VPOMinimize()
	elapsed := time.Since(start)
	LogOut("VPO stats:")
	LogOut("E_tot: ", GetTotalEnergy())
	LogOut("NSteps: ", NSteps)
	LogOut("Elapsed Time: ", elapsed)
	SnapshotAs(&M, "vpo.jpg")
}
