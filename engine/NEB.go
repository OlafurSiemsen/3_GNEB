package engine

import (
	"fmt"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

func InterpolateMagnetization(min_index, max_index int) {
	Interpolate(M.buffer_, min_index, max_index)
}

// Takes a data.Slice with multiple images and overwrites the images between
// two image indeces with linear interpolation. If no indeces are specified, the images
// between the first and last are interpolated.
func Interpolate(dst *data.Slice, ind_image_variadic ...int) {
	num_images_specified := len(ind_image_variadic)
	var min_index, max_index int
	switch num_images_specified {
	case 0:
		min_index = 0
		max_index = dst.N_images - 1
	case 2:
		min_index = ind_image_variadic[0]
		max_index = ind_image_variadic[1]
	default:
		panic("Please pass either 2 or 0 image indeces to interpolate")
	}
	num_intervals := max_index - min_index
	interval_scale := 1. / float32(num_intervals)

	diff_tot := cuda.Buffer(dst.NComp(), dst.Size())
	defer cuda.Recycle(diff_tot)
	cuda.Sub(diff_tot, dst.SubSlice(max_index), dst.SubSlice(min_index))

	interval := cuda.Buffer(dst.NComp(), dst.Size())
	defer cuda.Recycle(interval)

	for it := 1; it < num_intervals; it++ {
		cuda.Madd2(dst.SubSlice(min_index+it), dst.SubSlice(min_index), diff_tot, 1, interval_scale*float32(it))
	}
}

// Takes a data.Slice with multiple images and overwrites the images between
// two image indeces with angular interpolation. If no indeces are specified, the images
// between the first and last are interpolated. The vectors should be normalized,
// and if the aren't the parameter normalize should be set to true.
func AngularInterpolation(dst *data.Slice, normalize bool, ind_image_variadic ...int) {
	num_images_specified := len(ind_image_variadic)
	var min_index, max_index int
	switch num_images_specified {
	case 0:
		min_index = 0
		max_index = dst.N_images - 1
	case 2:
		min_index = ind_image_variadic[0]
		max_index = ind_image_variadic[1]
	default:
		panic("Please pass either 2 or 0 image indeces to interpolate")
	}
	num_intervals := max_index - min_index
	interval_scale := 1. / float32(num_intervals)

	if normalize {
		cuda.Normalize(dst, nil)
	}

	// Mise en place
	start_slice := dst.SubSlice(min_index)
	end_slice := dst.SubSlice(max_index)

	cross_prod_slice := cuda.Buffer(3, dst.Size())
	defer cuda.Recycle(cross_prod_slice)
	cuda.CrossProduct(cross_prod_slice, start_slice, end_slice)

	cross_prod_norm_slice := cuda.Buffer(1, dst.Size())
	defer cuda.Recycle(cross_prod_norm_slice)
	cuda.Veclen(cross_prod_norm_slice, cross_prod_slice)

	dot_prod_slice := cuda.Buffer(1, dst.Size())
	defer cuda.Recycle(dot_prod_slice)
	cuda.AddDotProduct(dot_prod_slice, 1, start_slice, end_slice)

	tot_angle_slice := cuda.Buffer(1, dst.Size())
	defer cuda.Recycle(tot_angle_slice)
	cuda.Atan2(tot_angle_slice, cross_prod_norm_slice, dot_prod_slice)
	angle_slice := cuda.Buffer(1, dst.Size())
	defer cuda.Recycle(angle_slice)
	cuda.Scale(angle_slice, tot_angle_slice, interval_scale)

	rot_axis_slice := cuda.Buffer(3, dst.Size())
	defer cuda.Recycle(rot_axis_slice)
	cuda.RotationVectors(rot_axis_slice, start_slice, cross_prod_slice, cross_prod_norm_slice)
	// cuda.VecScale(rot_axis_slice, cross_prod_slice, cross_prod_norm_slice)
	// data.Copy(rot_axis_slice, cross_prod_slice)
	// cuda.Normalize(rot_axis_slice, nil)

	cos_slice := cuda.Buffer(1, dst.Size())
	defer cuda.Recycle(cos_slice)
	cuda.Cos(cos_slice, angle_slice)

	// versine_slice := cuda.Buffer(1, dst.Size())
	// defer cuda.Recycle(versine_slice)
	// cuda.Constant(versine_slice, 1)
	// cuda.Sub(versine_slice, versine_slice, cos_slice)

	sin_slice := cuda.Buffer(1, dst.Size())
	defer cuda.Recycle(sin_slice)
	cuda.Sin(sin_slice, angle_slice)

	// TODO: recycle the unnecessary slices
	// Combining ingredients

	// fmt.Print("cos_slice \n")
	// fmt.Print("sin_slice \n")

	for iI := range num_intervals - 1 {
		RotateSlice(dst.SubSlice(min_index+iI+1), dst.SubSlice(min_index+iI), rot_axis_slice, cos_slice, sin_slice)
	}
}

// Implements Rodrigues' rotation formula.
// Takes in src and rotation_vector, 3d slices and slices containing the cos and sin
// of the rotation angles. Rotates the vectors in src around the vectors in rotation_vector
// by the angle represented by the cos and sin slices, sets to dst to the rotated vectors.
// Accepts an optional slice containing the versine of the angle, only applicable when
// the rotation vector is not orthogonal to the vector in src.
func RotateSlice(dst *data.Slice, src *data.Slice, rotation_vector *data.Slice, cos_angle *data.Slice, sin_angle *data.Slice, versine_angle_variadic ...*data.Slice) {
	var versine_term bool
	var versine_angle *data.Slice
	switch len(versine_angle_variadic) {
	case 0:
		versine_term = false
	case 1:
		versine_term = true
		versine_angle = versine_angle_variadic[0]
	default:
		panic("Please pass either 0 or 1 versine angle slices")
	}

	vec_term := cuda.Buffer(3, dst.Size())
	defer cuda.Recycle(vec_term)
	scalar_term := cuda.Buffer(1, dst.Size())
	defer cuda.Recycle(scalar_term)

	// dst = src * cos(angle)
	cuda.VecScale(dst, src, cos_angle)
	// dst = dst + (rotation_vector x src) * sin(angle)
	cuda.CrossProduct(vec_term, rotation_vector, src)
	cuda.VecScale(vec_term, vec_term, sin_angle)
	cuda.Add(dst, dst, vec_term)
	// dst = dst + rotation_vector * (rotation_vector · src) * (1 - cos(angle))
	// note: versine(angle) = 1 - cos(angle)
	// note: this term can be omitted iff the rotation vector is orthogonal to the
	// ending vectors
	if versine_term {
		cuda.AddDotProduct(scalar_term, 1, rotation_vector, src) // scalar_term = scalar_term {nil at this point} + (rotation_vector · src)
		cuda.Mul(scalar_term, scalar_term, versine_angle)        // scalar_term = scalar_term * (1-cos(angle))
		cuda.VecScale(vec_term, rotation_vector, scalar_term)    // vec_term = rotation_vector * (rotation_vector · src) * (1 - cos(angle))
		cuda.Add(dst, dst, vec_term)                             // dst = dst + rotation_vector * (rotation_vector · src) * (1 - cos(angle))
	}
}

func imageSub(dst *data.Slice, src *data.Slice, ind_img1 int, ind_img2 int) {
	cuda.Sub(dst, src.SubSlice(ind_img1), src.SubSlice(ind_img2))
}

func CalculateTangents(o_slice *data.Slice, i_slice *data.Slice, strategy string) {
	// Selecting a strategy
	switch strategy {
	case "central":
	default:
		fmt.Println("Invalid strategy. Please select one of: \"central\", ")
	}

	centralDifference(i_slice, o_slice)
}

// Applies forward differences to all but the last image in a slice
func centralDifference(i_slice *data.Slice, o_slice *data.Slice) {
	for it := 1; it < i_slice.N_images-1; it++ {
		cuda.Sub(o_slice.SubSlice(it), i_slice.SubSlice(it+1), i_slice.SubSlice(it))
	}
}
