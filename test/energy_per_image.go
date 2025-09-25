//go:build ignore
// +build ignore

/*
	This file tests calculating energy on a per image basis.
	First it creates a system with physical parameters inspired by /test/energy.mx3
	and a bunch of images. Then it calculates the energy for each image using
	the new CalcTotalImageEnergies() function. It then substitutes each images into
	the magnetization buffer and uses the old function to calculate its energy.
	The figure of merit is:
		Automatic confirmation that the energy is the same (down to float64 precision)
		whether calculated the old or new way.
*/

package main

import (
	"strconv"
	"time"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	. "github.com/mumax/3/engine"
)

func main() {

	defer InitAndClose()()

	gridsize := [3]int{4, 4, 4}
	n_images := 16
	// TODO: Is there really no non-script way to set anisU?
	Eval(`
		Msat  = 860E3
		Aex   = 13E-12
		Ku1   = 50
		alpha = 3
		anisU = vector(1,0,0)
	`)

	SetGridSize(gridsize[0], gridsize[1], gridsize[2])
	SetCellSize(4e-9, 4e-9, 40e-9)
	SetNImages(n_images)
	M.Set(RandomMagSeed(int(time.Now().UTC().UnixNano())))
	slice_backup := cuda.NewSlice(3, gridsize, n_images)
	new_energies := make([]float64, n_images)
	old_energies := make([]float64, n_images)

	data.Copy(slice_backup, M.Buffer())
	CalculateTotalImageEnergies()
	copy(new_energies, M.E_img)

	SetNImages(1)
	for ind_img := range n_images {
		M.SetArray(slice_backup.SubSlice(ind_img))
		old_energies[ind_img] = GetTotalEnergy()
	}

	for ind_img := range n_images {
		t_msg := "Image Number " + strconv.Itoa(ind_img) + ": "
		Expect(t_msg, old_energies[ind_img], new_energies[ind_img], 1e-15)
	}
}
