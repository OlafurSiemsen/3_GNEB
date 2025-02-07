//go:build ignore
// +build ignore

/*
	This file is as a way to test various functions and snippets.
	Its contents should be treated as volatile
*/

package main

import (
	. "github.com/mumax/3/engine"
)

func main() {

	defer InitAndClose()()

	SetGridSize(4, 4, 1)
	SetCellSize(4e-9, 4e-9, 2e-9)
	M.Set_N_Images(2)

	Aex.Set(13e-12)
	Alpha.Set(1)
	M.Set(RandomMag())

	Msat.Set(1100e3)
	K := 0.5e6
	u := ConstVector(1, 0, 0)

	prefactor := Const((2 * K) / (Msat.Average()))
	MyAnis := Mul(prefactor, Mul(Dot(u, &M), u))
	AddFieldTerm(MyAnis)
	AddEdensTerm(Mul(Const(-0.5), Dot(MyAnis, M_full)))

	// B_ext.Set(Vector(0, 0.00, 0))
	// Relax()

	Save(&M)
	Snapshot(&M)
}
