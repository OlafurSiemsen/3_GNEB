package engine

import (
	"math"
	"slices"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

type GNEB_params struct {
	kappa    []float32
	max_iter int
}

func New_GNEB_params(max_iter int, kappa []float32, n_images int) GNEB_params {
	switch len(kappa) {
	case 1:
		kappa = make([]float32, n_images-1)
		for ind := 1; ind < n_images-1; ind++ {
			kappa[ind] = kappa[0]
		}
	case n_images - 1:
	default:
		panic("Please pass either 1 or n_images-1 kappas for geodesic elastic force calculation")
	}
	var o_params GNEB_params
	o_params.kappa = kappa
	o_params.max_iter = max_iter
	return o_params
}

func InterpolateMagnetization(min_index, max_index int) {
	AngularInterpolation(M.buffer_, false, min_index, max_index)
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
	cuda.CrossProduct(cross_prod_slice, start_slice, end_slice)

	cross_prod_norm_slice := cuda.Buffer(1, dst.Size())
	cuda.Veclen(cross_prod_norm_slice, cross_prod_slice)

	dot_prod_slice := cuda.Buffer(1, dst.Size())
	cuda.DotProduct(dot_prod_slice, 1, start_slice, end_slice)

	tot_angle_slice := cuda.Buffer(1, dst.Size())
	cuda.Atan2(tot_angle_slice, cross_prod_norm_slice, dot_prod_slice)
	angle_slice := cuda.Buffer(1, dst.Size())
	cuda.Scale(angle_slice, tot_angle_slice, interval_scale)

	rot_axis_slice := cuda.Buffer(3, dst.Size())
	cuda.RotationVectors(rot_axis_slice, start_slice, cross_prod_slice, cross_prod_norm_slice)

	cos_slice := cuda.Buffer(1, dst.Size())
	cuda.Cos(cos_slice, angle_slice)

	sin_slice := cuda.Buffer(1, dst.Size())
	cuda.Sin(sin_slice, angle_slice)

	// Assembly
	for iI := range num_intervals - 1 {
		RotateSlice(dst.SubSlice(min_index+iI+1), dst.SubSlice(min_index+iI), rot_axis_slice, cos_slice, sin_slice)
	}

	// Cleaning up
	// These should probably be 'defer' statements after the creation of each
	// buffer but this results in unexpected behavior which needs investigation.
	cuda.Recycle(cross_prod_slice)
	cuda.Recycle(cross_prod_norm_slice)
	cuda.Recycle(dot_prod_slice)
	cuda.Recycle(tot_angle_slice)
	cuda.Recycle(angle_slice)
	cuda.Recycle(rot_axis_slice)
	cuda.Recycle(cos_slice)
	cuda.Recycle(sin_slice)
}

// TODO?: Make this into an all-cuda function
// Implements Rodrigues' axis/angle rotation formula.
// Takes in src and rotation_axes, 3d slices and slices containing the cos and sin
// of the rotation angles. Rotates the vectors in src around the vectors in rotation_axes
// by the angle represented by the cos and sin slices, sets to dst to the rotated vectors.
// Accepts an optional slice containing the versine of the angle, only applicable when
// the rotation vector is not orthogonal to the vector in src.
func RotateSlice(dst *data.Slice, src *data.Slice, rotation_axes *data.Slice, cos_angle *data.Slice, sin_angle *data.Slice, versine_angle_variadic ...*data.Slice) {
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
	scalar_term := cuda.Buffer(1, dst.Size())

	// dst = src * cos(angle)
	cuda.VecScale(dst, src, cos_angle)
	// dst = dst + (rotation_axes x src) * sin(angle)
	cuda.CrossProduct(vec_term, rotation_axes, src)
	cuda.VecScale(vec_term, vec_term, sin_angle)
	cuda.Add(dst, dst, vec_term)
	// dst = dst + rotation_axes * (rotation_axes · src) * (1 - cos(angle))
	// note: versine(angle) = 1 - cos(angle)
	// note: this term can be omitted iff the rotation vector is orthogonal to the
	// ending vectors
	if versine_term {
		cuda.AddDotProduct(scalar_term, 1, rotation_axes, src) // scalar_term = scalar_term {nil at this point} + (rotation_axes · src)
		cuda.Mul(scalar_term, scalar_term, versine_angle)      // scalar_term = scalar_term * (1-cos(angle))
		cuda.VecScale(vec_term, rotation_axes, scalar_term)    // vec_term = rotation_axes * (rotation_axes · src) * (1 - cos(angle))
		cuda.Add(dst, dst, vec_term)                           // dst = dst + rotation_axes * (rotation_axes · src) * (1 - cos(angle))
	}

	defer cuda.Recycle(vec_term)
	defer cuda.Recycle(scalar_term)
}

func imageSub(dst *data.Slice, src *data.Slice, ind_img1 int, ind_img2 int) {
	cuda.Sub(dst, src.SubSlice(ind_img1), src.SubSlice(ind_img2))
}

// Calculates tangents between images as is used in NEB methods.
// Accepts a magnetization type struct and populates its tangent_buffer_ according
// to Appendix A of https://doi.org/10.1016/j.cpc.2015.07.001, selecting either forward,
// backwards, or a weighted average depending on the energy of neighbouring images.
func CalculateTangents(mag_variadic ...*magnetization) {
	var mag *magnetization
	switch len(mag_variadic) {
	case 0:
		mag = &M
	case 1:
		mag = mag_variadic[0]
	default:
		panic("Please pass either 0 or 1 magnetization for tangent calculation")
	}
	if !mag.E_img_calc {
		CalculateTotalImageEnergies(mag)
	}
	E_img := mag.E_img
	tangent_slice := mag.tangent_buffer_
	mag_slice := mag.buffer_
	n_images := mag_slice.N_images
	for ind_img := 1; ind_img < n_images-1; ind_img++ {
		E_np1 := E_img[ind_img+1]
		E_n := E_img[ind_img]
		E_nm1 := E_img[ind_img-1]
		switch {
		case E_np1 > E_n && E_n > E_nm1:
			// forward difference
			imageSub(tangent_slice.SubSlice(ind_img), mag_slice, ind_img+1, ind_img)
		case E_np1 < E_n && E_n < E_nm1:
			// backwards difference
			imageSub(tangent_slice.SubSlice(ind_img), mag_slice, ind_img, ind_img-1)
		default:
			delta_E := []float64{math.Abs(E_np1 - E_n), math.Abs(E_n - E_nm1)}
			max_delta_E := float32(slices.Max(delta_E))
			min_delta_E := float32(slices.Min(delta_E))
			fwd_diff_slice := cuda.Buffer(3, mag_slice.Size())
			imageSub(fwd_diff_slice, mag_slice, ind_img+1, ind_img)
			bkwd_diff_slice := cuda.Buffer(3, mag_slice.Size())
			imageSub(bkwd_diff_slice, mag_slice, ind_img, ind_img-1)
			switch {
			case E_np1 > E_nm1:
				// fwd diff * dE_max + bkwd diff * dE_min
				cuda.Madd2(tangent_slice.SubSlice(ind_img), fwd_diff_slice, bkwd_diff_slice, max_delta_E, min_delta_E)
			case E_np1 < E_nm1:
				// fwd diff * dE_min + bkwd diff * dE_max
				cuda.Madd2(tangent_slice.SubSlice(ind_img), fwd_diff_slice, bkwd_diff_slice, min_delta_E, max_delta_E)
			default:
				//something?
			}
			cuda.Recycle(fwd_diff_slice)
			cuda.Recycle(bkwd_diff_slice)
		}
	}
	mag.geodesic_tangents_calc = false
}

// Projects the tangents between images in the passed magnetization struct onto
// the geodesic tangent space of the magnetization according to section 3 of
// https://doi.org/10.1016/j.cpc.2015.07.001.
func ProjectTangents(mag_variadic ...*magnetization) {
	var mag *magnetization
	switch len(mag_variadic) {
	case 0:
		mag = &M
	case 1:
		mag = mag_variadic[0]
	default:
		panic("Please pass either 0 or 1 magnetization for tangent projection")
	}
	if mag.geodesic_tangents_calc {
		// Tangents are already projected
		return
	}
	tangent_slice := mag.tangent_buffer_
	mag_slice := mag.buffer_
	// n_images := mag_slice.N_images
	cuda.Orthogonalize(tangent_slice, tangent_slice, mag_slice)
	mag.geodesic_tangents_calc = true
}

func CalculateGeodesicDistances(mag_variadic ...*magnetization) {
	var mag *magnetization
	switch len(mag_variadic) {
	case 0:
		mag = &M
	case 1:
		mag = mag_variadic[0]
	default:
		panic("Please pass either 0 or 1 magnetization for tangent projection")
	}
	if mag.Geodesic_distances_calc {
		// Geodesic distances are already calculated
		return
	}
	mag_slice := mag.buffer_
	n_images := mag_slice.N_images
	cross_prod_slice := cuda.Buffer(3, mag_slice.Size())
	cross_prod_norm_slice := cuda.Buffer(1, mag_slice.Size())
	dot_prod_slice := cuda.Buffer(1, mag_slice.Size())
	tot_angle_slice := cuda.Buffer(1, mag_slice.Size())
	for ind_img := 0; ind_img < n_images-1; ind_img++ {
		img_n := mag_slice.SubSlice(ind_img)
		img_np1 := mag_slice.SubSlice(ind_img + 1)

		cuda.CrossProduct(cross_prod_slice, img_n, img_np1)

		cuda.Veclen(cross_prod_norm_slice, cross_prod_slice)

		cuda.DotProduct(dot_prod_slice, 1, img_n, img_np1)

		cuda.Atan2(tot_angle_slice, cross_prod_norm_slice, dot_prod_slice)

		mag.Geodesic_distances[ind_img] = math.Sqrt(float64(cuda.ReduceSquareSum(tot_angle_slice)))
	}
	cuda.Recycle(cross_prod_slice)
	cuda.Recycle(cross_prod_norm_slice)
	cuda.Recycle(dot_prod_slice)
	cuda.Recycle(tot_angle_slice)
	mag.Geodesic_distances_calc = true
}

// Transforms the real force (energy gradient) to GNEB force according to eq. 14 of https://doi.org/10.1016/j.cpc.2015.07.001.
func GNEBForceTransformation(energy_gradient *data.Slice, kappa []float32, mag_variadic ...*magnetization) {
	var mag *magnetization
	switch len(mag_variadic) {
	case 0:
		mag = &M
	case 1:
		mag = mag_variadic[0]
	default:
		panic("Please pass either 0 or 1 magnetization for GNEB calculation")
	}
	mag_slice := mag.buffer_
	tangent_slice := mag.tangent_buffer_
	n_images := mag.N_images

	// Project energy real force orthogonal to
	cuda.Global_Orthogonalize(energy_gradient, energy_gradient, tangent_slice)
	// Generate elastic forces
	elastic_force_slice := cuda.Buffer(3, mag_slice.Size(), n_images)
	CalculateGeodesicElasticForces(elastic_force_slice, energy_gradient, kappa, mag)
	cuda.Add(energy_gradient, energy_gradient, elastic_force_slice)
	// Cleanup
	cuda.Recycle(elastic_force_slice)
}

// Calculates the total GNEB force according to eq. 12 of https://doi.org/10.1016/j.cpc.2015.07.001.
func CalculateGeodesicElasticForces(geodesic_elastic_force *data.Slice, energy_gradient *data.Slice, kappa []float32, mag *magnetization) {
	mag_slice := mag.buffer_
	tangent_slice := mag.tangent_buffer_
	n_images := mag_slice.N_images
	switch len(kappa) {
	case 1:
		kappa = make([]float32, n_images-1)
		for ind := 1; ind < n_images-1; ind++ {
			kappa[ind] = kappa[0]
		}
	case n_images - 1:
	default:
		panic("Please pass either 1 or n_images-1 kappas for geodesic elastic force calculation")
	}

	elastic_force_slice := cuda.Buffer(3, mag_slice.Size(), n_images)
	for ind_img := 1; ind_img < n_images-1; ind_img++ {
		elastic_force_n := elastic_force_slice.SubSlice(ind_img)
		tangent_n := tangent_slice.SubSlice(ind_img)
		coeff := kappa[ind_img] * float32(mag.Geodesic_distances[ind_img]-mag.Geodesic_distances[ind_img-1])
		cuda.Scale(elastic_force_n, tangent_n, coeff)
	}
	cuda.Add(geodesic_elastic_force, geodesic_elastic_force, elastic_force_slice)

	cuda.Recycle(elastic_force_slice)
}
