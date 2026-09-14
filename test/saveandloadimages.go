//go:build ignore
// +build ignore

/*
	This file tests saving and loading multiple images.
	First, it creates 4 randomly initialized images and stores them all, renaming the saved jpgs for visual inspection
	Then it loads the second and third images into the first and fourth, overwriting them in the process
	The figures of merit are
		1. Visual inspection, confirming that
			gen_i001o.png == load_i000.png == load_i001.png
		2. Automated confirmation that
			All components are within eps(float32) for slices 0 and 1
			All components are within eps(float32) for slices 2 and 3

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
	SetOutputFormat(OVF2_TEXT)

	Nx, Ny := 4, 4
	SetGridSize(Nx, Ny, 1)
	SetCellSize(4e-9, 4e-9, 2e-9)
	N_images := 4
	SetNImages(N_images)

	M.Set(RandomMagSeed(int(time.Now().UTC().UnixNano())))

	SaveAs(&M, "gen")
	SnapshotAs(&M, "gen.png")
	DrainOutput()
	dir_str := OD() // TODO-olafur: Make this not specific to my machine, move to cache?
	for it := range N_images {
		os.Rename(dir_str+"gen_i00"+strconv.Itoa(it)+".png", dir_str+"gen_i00"+strconv.Itoa(it)+"o.png")
	}
	M.LoadFiles(dir_str+"gen_i001.ovf", dir_str+"gen_i00"+strconv.Itoa(N_images-2)+".ovf")
	SaveAs(&M, "load")
	SnapshotAs(&M, "load.png")
	CompareSlices2Log(M.Buffer().SubSlice(0), M.Buffer().SubSlice(1), true, false, true)
	CompareSlices2Log(M.Buffer().SubSlice(2), M.Buffer().SubSlice(3), true, false, true)
	// host_slice := M.Buffer().HostCopy()
	// // TODO: Swap this for the CompareSlices function
	// for it_x := range Nx {
	// 	for it_y := range Ny {
	// 		Expect("Comparing images 0, 1 >> "+"Nx: "+strconv.Itoa(it_x)+" Ny: "+strconv.Itoa(it_y),
	// 			host_slice.Get(2, it_x, it_y, 0, 0), host_slice.Get(2, it_x, it_y, 0, 1), 0)
	// 		Expect("Comparing images 2, 3 >> "+"Nx: "+strconv.Itoa(it_x)+" Ny: "+strconv.Itoa(it_y),
	// 			host_slice.Get(2, it_x, it_y, 0, 2), host_slice.Get(2, it_x, it_y, 0, 3), 0)
	// 	}
	// }
}
