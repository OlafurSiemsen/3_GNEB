//go:build ignore
// +build ignore

package main

import (
	"math"

	. "github.com/mumax/3/engine"
)

func main() {
	defer InitAndClose()()
	Nx := 1024
	Ny := 16
	Nz := 16

	csX := 0.5e-9
	csY := 1e-9
	csZ := 2e-9

	SetGridSize(Nx, Ny, Nz)
	SetCellSize(csX, csY, csZ)

	Msat.Set(1000e3)
	B1.Set(1000e3)
	B2.Set(1000e3)
	EnableDemag = false
	M.Set(Uniform(1.0, 1.0, 1.0))

	mask := NewVectorMask(Nx, Ny, Nz)

	Period := 32e-9
	kx := 2 * math.Pi / Period

	pre := 1.0 / (2.0 * B1.Average() * kx)
	pre2 := 1.0 / (B2.Average() * kx)

	for ii := 0; ii < Nx; ii++ {
		for jj := 0; jj < Ny; jj++ {
			for kk := 0; kk < Nz; kk++ {
				r := Index2Coord(ii, jj, kk)
				x := r.X()
				mx := math.Sin(kx * x)
				my := math.Cos(kx * x)
				mask.SetVector(ii, jj, kk, Vector(mx, my, 0.0))
			}
		}
	}

	M.SetArray(mask)

	Fmel := F_mel.HostCopy()

	errx := -math.Inf(1)
	erry := -math.Inf(1)

	for ii := 0; ii < Nx; ii++ {
		for jj := 0; jj < Ny; jj++ {
			for kk := 0; kk < Nz; kk++ {
				r := Index2Coord(ii, jj, kk)
				x := r.X()

				ref := math.Sin(kx*x) * math.Cos(kx*x)
				ref2 := (math.Cos(kx*x)*math.Cos(kx*x) - math.Sin(kx*x)*math.Sin(kx*x))

				val := Fmel.Get(0, ii, jj, kk)
				val2 := Fmel.Get(1, ii, jj, kk)

				ex := math.Abs(val*pre - ref)
				ey := math.Abs(val2*pre2 - ref2)

				if ex > errx {
					errx = ex
				}
				if ey > erry {
					erry = ey
				}
			}
		}
	}

	ii := Nx / 2
	jj := Ny / 2
	kk := Nz / 2

	r := Index2Coord(ii, jj, kk)
	x := r.X()

	ref := math.Sin(kx*x) * math.Cos(kx*x)
	ref2 := (math.Cos(kx*x)*math.Cos(kx*x) - math.Sin(kx*x)*math.Sin(kx*x))

	val := Fmel.Get(0, ii, jj, kk)
	val2 := Fmel.Get(1, ii, jj, kk)

	ex := math.Abs(val*pre - ref)
	ey := math.Abs(val2*pre2 - ref2)

	ERRMAX := 0.004
	ERRMIN := 3e-6
	Expect("max((ΔFmel).x)", errx, 0., ERRMAX)
	Expect("max((ΔFmel).y)", erry, 0., ERRMAX)
	Expect("((ΔFmel).x)@center", ex, 0., ERRMIN)
	Expect("((ΔFmel).y)@center", ey, 0., ERRMIN)
}
