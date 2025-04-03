//go:build ignore
// +build ignore

/*
	This file tests interpolating images in a slice.
	First, it creates an empty slice the first and last images of which it populates with a
	precalculated interpolated slice.
	Then it interpolates between the first and last image (default option) and compares to the
	precalculated slice.
	Finally it interpolates between the second and second to last images and compares again.
	The figures of merit are
		1. Automated confirmation that
			The interpolated slice is identical to the precalculated slice after the first interpolation
			The interpolated slice is identical to the precalculated slice after the second interpolation
*/

package main

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	. "github.com/mumax/3/engine"
)

func main() {

	defer InitAndClose()()

	gridsize := [3]int{2, 2, 1}
	SetGridSize(gridsize[0], gridsize[1], gridsize[2])
	SetCellSize(4e-9, 4e-9, 2e-9)
	n_images := 6
	SetNImages(n_images)

	premade_array := [][]float32{
		{-0.027566314, 0.038636565, 0.6063938, 0.85570455, -0.13594253, 0.19893019, 0.29921055, 0.70402884, -0.24431875, 0.3592238, -0.007972717, 0.55235314, -0.352695, 0.5195174, -0.31515598, 0.40067744, -0.4610712, 0.67981106, -0.62233925, 0.24900174, -0.5694474, 0.8401047, -0.9295225, 0.09732604},
		{-0.923128, -0.5152042, 0.36918235, -0.16218793, -0.6422961, -0.41522366, 0.36646545, -0.0063239634, -0.36146414, -0.3152431, 0.36374855, 0.14954, -0.08063221, -0.21526253, 0.36103165, 0.30540398, 0.20019972, -0.115282, 0.35831475, 0.46126795, 0.48103166, -0.015301466, 0.35559785, 0.61713195},
		{-0.08299148, 0.48376155, -0.3136264, 0.9833262, -0.06480186, 0.56930673, -0.11016849, 0.74791497, -0.04661224, 0.654852, 0.093289435, 0.51250374, -0.02842262, 0.74039716, 0.29674733, 0.27709246, -0.010233, 0.8259424, 0.5002053, 0.04168123, 0.007956624, 0.9114876, 0.70366323, -0.19373},
	}

	premade_slice := cuda.NewSlice(3, gridsize, n_images)
	data.Copy(premade_slice, data.SliceFromArray(premade_array, gridsize, n_images))

	test_slice := cuda.NewSlice(3, gridsize, n_images)
	data.Copy(test_slice.SubSlice(0), premade_slice.SubSlice(0))
	data.Copy(test_slice.SubSlice(n_images-1), premade_slice.SubSlice(n_images-1))

	PrintSlice(premade_slice)

	Interpolate(test_slice, 0, 5)
	CompareSlices(test_slice, premade_slice, false)

	Interpolate(test_slice, 1, 4)
	CompareSlices(test_slice, premade_slice, false)
}
