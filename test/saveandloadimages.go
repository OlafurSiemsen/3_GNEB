//go:build ignore
// +build ignore

/*
	This file tests saving and loading multiple images.
	First, it creates 4 randomly initialized images and stores them all, renaming the saved jpgs for visual inspection
	Then it loads the second and third images into the first and fourth, overwriting them in the process
	The figures of merit are
		1. Visual inspection, confirming that
			m000001i1o.jpg == m000001i0.jpg == m000001i1.jpg
			m000001i2o.jpg == m000001i2.jpg == m000001i3.jpg
		2. Automated confirmation that
			The z components of each cell between images 0 and 1 are identical
			The z components of each cell between images 2 and 3 are identical

*/

package main

import (
	"os"
	"strconv"
	"time"

	. "github.com/mumax/3/engine"
)

func main() {

	defer InitAndClose()()

	Nx, Ny := 4, 4
	SetGridSize(Nx, Ny, 1)
	SetCellSize(4e-9, 4e-9, 2e-9)
	N_images := 4
	SetNImages(N_images)

	M.Set(RandomMagSeed(int(time.Now().UTC().UnixNano())))

	Save(&M)
	Snapshot(&M)
	DrainOutput()
	dir_str := "/home/olafur/go/src/github.com/mumax/3_GNEB/test/" + OD() // TODO: Make this not specific to my machine
	for it := range N_images {
		os.Rename(dir_str+"m000000i00"+strconv.Itoa(it)+".jpg", dir_str+"m000000i00"+strconv.Itoa(it)+"o.jpg")
	}
	M.LoadFiles(dir_str+"m000000i001.ovf", dir_str+"m000000i00"+strconv.Itoa(N_images-2)+".ovf")
	Save(&M)
	Snapshot(&M)
	host_slice := M.Buffer().HostCopy()
	// TODO: Swap this for the CompareSlices function
	for it_x := range Nx {
		for it_y := range Ny {
			Expect("Comparing images 0, 1 >> "+"Nx: "+strconv.Itoa(it_x)+" Ny: "+strconv.Itoa(it_y),
				host_slice.Get(2, it_x, it_y, 0, 0), host_slice.Get(2, it_x, it_y, 0, 1), 0)
			Expect("Comparing images 2, 3 >> "+"Nx: "+strconv.Itoa(it_x)+" Ny: "+strconv.Itoa(it_y),
				host_slice.Get(2, it_x, it_y, 0, 2), host_slice.Get(2, it_x, it_y, 0, 3), 0)
		}
	}
}
