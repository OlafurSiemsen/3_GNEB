//go:build ignore
// +build ignore

/*
	This file tests interpolating images in a slice by rotation.
	First, it creates an empty slice the first and last images of which it populates with a
	precalculated interpolated slice.
	Then it interpolates between the first and last image (default option) and compares to the
	precalculated slice.
	The figure of merit for this stage is are
		1. Automated confirmation that the interpolated slice is identical to
		the precalculated slice after the interpolation
	Secondly it checks the case where the vectors in the initial and final images are
	antiparallel by creating a new empty slice and populating two images with
	random antiparallel vectors and then interpolating between them, also testing
	interpolation between images that are not the first and last images as well as
	the normalization in the interpolation function.
	The figure of merit for this stage is are
		1. Automated confirmation that the angle between the vectors
		in adjacent images is identical.
*/

package main

import (
	"math/rand"
	"time"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	. "github.com/mumax/3/engine"
)

func main() {

	defer InitAndClose()()

	gridsize := [3]int{2, 2, 2}
	n_images := 6

	// Hardcoded reference slice that is correctly interpolated for the first test
	premade_array := [][]float32{
		{0.254741894370544, 0.7111172722878139, 0.4905617114988439, -0.5762976374174821, -0.2970966915633388, 0.8877606795943617, -0.9329123883335332, 0.6755424048860443, 0.09556094573951673, 0.20772420560179686, 0.32915648933675085, -0.819204968315757, 0.008807932176528298, 0.9294577606258454, -0.9329120820442651, 0.7928750777737744, -0.06715679036090136, -0.3691821145288574, 0.1397528791580826, -0.9361713121275492, 0.3133279775576436, 0.9587606110482658, -0.9329117757540641, 0.85749462429335, -0.22738899963030232, -0.8154354937855273, -0.061538254019102, -0.909214774985882, 0.5685938815326883, 0.975278480145427, -0.9329114694629302, 0.8651049095087312, -0.3792053591156089, -0.9731075979387923, -0.257594880298485, -0.7424795369523519, 0.7344786722667833, 0.9787911037079874, -0.9329111631708634, 0.8151999748679322, -0.5169870232338172, -0.7863985296190226, -0.43174022230025016, -0.46159874390889244, 0.7849057989085754, 0.9692516412356883, -0.9329108568778637, 0.7110976761087082},
		{0.015827225817092803, 0.6647952412588111, -0.24687005189092884, 0.20591239618539844, -0.655269201260432, -0.2670463084737811, -0.3336972045502455, -0.735628925495511, 0.12179177900535963, 0.9684395905594939, -0.029336564981362515, 0.2899749073147507, -0.8473205380867017, -0.2534371909694829, -0.333698137524943, -0.5950099075355924, 0.2232487204854384, 0.929354767492578, 0.19069232018414198, 0.3294579450888403, -0.9061758356556335, -0.23644851269022935, -0.33369907049930675, -0.41483253627959904, 0.3164430472429431, 0.5613728264189545, 0.39450072133897174, 0.31829156005330184, -0.8225832321591858, -0.2163068160409108, -0.33370000347333684, -0.20707563765603254, 0.3979255622806746, -0.005278031191914221, 0.5647524871951211, 0.2581924233807415, -0.6096832136312244, -0.19328068862572498, -0.3337009364470332, 0.014448378337621217, 0.46468053217396116, -0.5700610022771092, 0.6869658282227753, 0.1583999133566073, -0.30094297218038246, -0.16767718165808615, -0.33370186942039587, 0.2350118152760417},
		{-0.9668795510173133, -0.22882200998500649, -0.8357059199806444, -0.7908736423754529, 0.6945184156965414, 0.3749229852872231, -0.1353538007193541, 0.04992537606367792, -0.9879447698207015, -0.13775127529014897, -0.9438195651090306, -0.49479063553646746, 0.5310087815349537, 0.26810046894007206, -0.1353536116556132, -0.13157629338036922, -0.9724453580073467, 0.0020694083256306084, -0.9716509516229658, -0.12264068153439774, 0.28402628989840223, 0.15770285840445153, -0.13535342259173688, -0.3043303076458893, -0.9209549612759198, 0.14115773175121574, -0.9168327132881502, 0.26836351418590293, -0.007604212920373757, 0.045202295289837977, -0.13535323352772527, -0.4568513717099221, -0.8353792806274353, 0.23029056693779582, -0.7840277455874465, 0.6181106775617139, -0.29803935814171645, -0.06790103612217945, -0.13535304446357826, -0.578999348305968, -0.7188855408382903, 0.237923950656208, -0.5845325750566231, 0.8728322101471463, -0.5416236833216668, -0.18009891369224285, -0.13535285539929592, -0.6626534099462962},
	}
	premade_slice := cuda.NewSlice(3, gridsize, n_images)
	data.Copy(premade_slice, data.SliceFromArray(premade_array, gridsize, n_images))
	cuda.Normalize(premade_slice, nil)

	// Creating a slice with just the first and last image for comparison
	test_slice := cuda.NewSlice(3, gridsize, n_images)
	data.Copy(test_slice.SubSlice(0), premade_slice.SubSlice(0))
	data.Copy(test_slice.SubSlice(n_images-1), premade_slice.SubSlice(n_images-1))

	// Interpolating
	AngularInterpolation(test_slice, false)
	// Comparing the reference slice to the interpolated slice
	CompareSlices(test_slice, premade_slice, true)

	// Randomly generating anti-parallel images for the second test
	n_images = 8
	start_slice_ind := 1
	end_slice_ind := 6
	antiparallel_test_slice := cuda.NewSlice(3, gridsize, n_images)
	t_slice := data.NewSlice(3, gridsize, 8)
	for ind_X := range gridsize[0] {
		for ind_Y := range gridsize[1] {
			for ind_Z := range gridsize[2] {
				rng := rand.New(rand.NewSource(int64(time.Now().UTC().UnixNano())))
				t_vec := data.Vector{rng.Float64() - 0.5, rng.Float64() - 0.5, rng.Float64() - 0.5}
				t_slice.SubSlice(start_slice_ind).SetVector(ind_X, ind_Y, ind_Z, t_vec)
				t_slice.SubSlice(end_slice_ind).SetVector(ind_X, ind_Y, ind_Z, t_vec.Mul(-1.0))
			}
		}
	}
	data.Copy(antiparallel_test_slice, t_slice)
	// Interpolating
	AngularInterpolation(antiparallel_test_slice, true, start_slice_ind, end_slice_ind)
	// Checking that angles are the same between adjacent images
	t_dotprod_slice := cuda.NewSlice(1, gridsize, end_slice_ind-start_slice_ind)
	for t_ind := range end_slice_ind - start_slice_ind {
		cuda.AddDotProduct(t_dotprod_slice.SubSlice(t_ind), 1, antiparallel_test_slice.SubSlice(start_slice_ind+t_ind), antiparallel_test_slice.SubSlice(start_slice_ind+t_ind+1))
	}
	for t_ind := range end_slice_ind - start_slice_ind - 1 {
		CompareSlices(t_dotprod_slice.SubSlice(t_ind), t_dotprod_slice.SubSlice(t_ind+1), true)
	}
}
