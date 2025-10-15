package engine

import (
	"fmt"
	"math"
	"slices"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

// TODO: Remove?
type GNEB_params struct {
	kappa    []float32
	max_iter int
}

// TODO: Remove?
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
func Interpolation(dst *data.Slice, ind_image_variadic ...int) {
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
	cuda.VecNorm(cross_prod_norm_slice, cross_prod_slice)

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

// TODO: Make this into an all-cuda function, rework doc string?
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
	cuda.VecVecScale(dst, src, cos_angle)
	// dst = dst + (rotation_axes x src) * sin(angle)
	cuda.CrossProduct(vec_term, rotation_axes, src)
	cuda.VecVecScale(vec_term, vec_term, sin_angle)
	cuda.Add(dst, dst, vec_term)
	// dst = dst + rotation_axes * (rotation_axes · src) * (1 - cos(angle))
	// note: versine(angle) = 1 - cos(angle)
	// note: this term can be omitted iff the rotation axis is orthogonal to the
	// vectors to be rotated
	if versine_term {
		cuda.AddDotProduct(scalar_term, 1, rotation_axes, src) // scalar_term = scalar_term {nil at this point} + (rotation_axes · src)
		cuda.Mul(scalar_term, scalar_term, versine_angle)      // scalar_term = scalar_term * (1-cos(angle))
		cuda.VecVecScale(vec_term, rotation_axes, scalar_term) // vec_term = rotation_axes * (rotation_axes · src) * (1 - cos(angle))
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
	if mag.tangent_calc {
		// Tangents are already calculated
		return
	}
	if !mag.E_img_calc {
		// Energies are required to calculate tangents
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
		case (E_nm1 < E_n && E_n <= E_np1) || (E_nm1 <= E_n && E_n < E_np1):
			// forward difference
			imageSub(tangent_slice.SubSlice(ind_img), mag_slice, ind_img+1, ind_img)
		case (E_nm1 >= E_n && E_n > E_np1) || (E_nm1 > E_n && E_n >= E_np1):
			// backwards difference
			imageSub(tangent_slice.SubSlice(ind_img), mag_slice, ind_img, ind_img-1)
		default:
			epsilon := float32(1e-36)
			delta_E := []float64{math.Abs(E_np1 - E_n), math.Abs(E_n - E_nm1)}
			max_delta_E := float32(slices.Max(delta_E))
			min_delta_E := float32(slices.Min(delta_E))
			fwd_diff_slice := cuda.Buffer(3, mag_slice.Size())
			imageSub(fwd_diff_slice, mag_slice, ind_img+1, ind_img)
			bkwd_diff_slice := cuda.Buffer(3, mag_slice.Size())
			imageSub(bkwd_diff_slice, mag_slice, ind_img, ind_img-1)
			// If the delta_E are close to bottom of float32, we use the central difference
			if max_delta_E < epsilon {
				// fwd diff + bkwd diff
				cuda.Madd2(tangent_slice.SubSlice(ind_img), fwd_diff_slice, bkwd_diff_slice, 1, 1)
			}
			switch {
			case E_np1 > E_nm1:
				// fwd diff * dE_max + bkwd diff * dE_min
				recip_max_delta_E := 1 / max_delta_E
				cuda.Madd2(tangent_slice.SubSlice(ind_img), fwd_diff_slice, bkwd_diff_slice, 1, min_delta_E*recip_max_delta_E)
			case E_np1 < E_nm1:
				// fwd diff * dE_min + bkwd diff * dE_max
				recip_max_delta_E := 1 / max_delta_E
				cuda.Madd2(tangent_slice.SubSlice(ind_img), fwd_diff_slice, bkwd_diff_slice, min_delta_E*recip_max_delta_E, 1)
			default:
				// fwd diff + bkwd diff
				cuda.Madd2(tangent_slice.SubSlice(ind_img), fwd_diff_slice, bkwd_diff_slice, 1, 1)
			}
			cuda.Recycle(fwd_diff_slice)
			cuda.Recycle(bkwd_diff_slice)
		}
	}
	mag.tangent_calc = true
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
	if !mag.tangent_calc {
		CalculateTangents(mag)
	}
	tangent_slice := mag.tangent_buffer_
	mag_slice := mag.buffer_
	n_images := mag_slice.N_images
	// n_images := mag_slice.N_images
	cuda.Orthogonalize(tangent_slice, tangent_slice, mag_slice)
	// TODO: Implement cases for last and first image
	for ind_img := 1; ind_img < n_images-1; ind_img++ {
		norm := cuda.Dot(tangent_slice.SubSlice(ind_img), tangent_slice.SubSlice(ind_img))
		norm = float32(math.Sqrt(float64(norm))) // Ugh
		cuda.Scale(tangent_slice.SubSlice(ind_img), tangent_slice.SubSlice(ind_img), 1/norm)
	}
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
		cuda.VecNorm(cross_prod_norm_slice, cross_prod_slice)
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

// Calculates the total GNEB force according to eq. 12 of https://doi.org/10.1016/j.cpc.2015.07.001.
func CalculateGeodesicElasticForces(geodesic_elastic_force *data.Slice, kappa []float32, mag *magnetization) {
	if !mag.tangent_calc {
		CalculateTangents(mag)
	}
	if !mag.geodesic_tangents_calc {
		ProjectTangents(mag)
	}
	if !mag.Geodesic_distances_calc {
		CalculateGeodesicDistances(mag)
	}
	mag_slice := mag.buffer_
	tangent_slice := mag.tangent_buffer_
	n_images := mag_slice.N_images
	switch len(kappa) {
	case 1:
		// kappa = make([]float32, n_images-1)
		for ind := 1; ind < n_images-1; ind++ {
			kappa = append(kappa, kappa[0])
		}
	case n_images - 1:
	default:
		panic("Please pass either 1 or n_images-1 kappas for geodesic elastic force calculation")
	}

	elastic_force_slice := cuda.Buffer(3, mag_slice.Size(), n_images)
	cuda.Zero(elastic_force_slice)
	for ind_img := 1; ind_img < n_images-1; ind_img++ {
		if FixEndImages && (ind_img == 0 || ind_img == n_images-1) {
			continue
		}
		elastic_force_n := elastic_force_slice.SubSlice(ind_img)
		tangent_n := tangent_slice.SubSlice(ind_img)
		coeff := kappa[ind_img] * float32(mag.Geodesic_distances[ind_img]-mag.Geodesic_distances[ind_img-1])
		cuda.Scale(elastic_force_n, tangent_n, coeff)
		// cuda.Orthogonalize(elastic_force_slice, elastic_force_slice, mag_slice)
	}
	// cuda.Add(geodesic_elastic_force, geodesic_elastic_force, elastic_force_slice)
	data.Copy(geodesic_elastic_force, elastic_force_slice)
	cuda.Recycle(elastic_force_slice)
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
	n_images := mag.GetNImages()

	// Project energy real force orthogonal to path
	// LogSlice(energy_gradient, "B_eff⟂m", NSteps)
	CalculateTangents(mag)
	ProjectTangents(mag)
	for ind_img := 1; ind_img < n_images-1; ind_img++ {
		if ClimbingImage == true && ind_img == *mag.climbing_image_index {
			continue
		}
		cuda.Global_Orthogonalize(energy_gradient.SubSlice(ind_img), energy_gradient.SubSlice(ind_img), tangent_slice.SubSlice(ind_img))
	}
	// LogSlice(M.tangent_buffer_, "τ", NSteps)
	// LogSlice(energy_gradient, "(B_eff⟂m)⟂τ", NSteps)
	// Generate elastic forces
	elastic_force_slice := cuda.Buffer(3, mag_slice.Size(), n_images)
	CalculateGeodesicElasticForces(elastic_force_slice, kappa, mag)
	// LogSlice(elastic_force_slice, "F_s", NSteps)
	cuda.Add(energy_gradient, energy_gradient, elastic_force_slice)
	// LogSlice(energy_gradient, "F_GNEB", NSteps)
	// Cleanup
	cuda.Recycle(elastic_force_slice)
}

// TODO-olafur: Where do I put this?
// Implementation of the Cubic Hermite Interpolating Polynomial for
// interpolating the energy between images in the context of GNEB.
// See https://en.wikipedia.org/wiki/Cubic_Hermite_spline
type CHIP struct {
	d     []float64
	c     []float64
	b     []float64
	a     []float64
	x_arr []float64
	x_min float64
	x_max float64
}

// Constructor for CHIP struct
// Inputs:
// x_diffs: A list of (n-1) distances between points
// y: y coordinates at each point
// dy/dx: derivative at each point
// Output:
// CHIP object that can then be evaluated on the interval [0, sum(x_diffs)]
func New_CHIP(x_diffs []float64, y []float64, dydx []float64) *CHIP {
	o_CHIP := new(CHIP)
	n_intervals := len(x_diffs)
	n_points := len(y)
	// x[n] = x_diffs[n] + x_diffs[n-1]; x[0] = 0
	o_CHIP.x_arr = make([]float64, n_points)
	o_CHIP.x_arr[0] = 0
	for ind := 1; ind < len(o_CHIP.x_arr); ind++ {
		o_CHIP.x_arr[ind] = x_diffs[ind-1] + o_CHIP.x_arr[ind-1]
	}
	o_CHIP.x_min = o_CHIP.x_arr[0]
	o_CHIP.x_max = o_CHIP.x_arr[len(o_CHIP.x_arr)-1]
	// a[n] = (dydx[n+1] + dydx[n]) / (x[n+1] - x[n+1])**2
	//		-2*(y[n+1]-y[n]) / (x[n+1] - x[n+1])**3
	y_diffs := make([]float64, n_intervals)
	for ind := range y_diffs {
		y_diffs[ind] = y[ind+1] - y[ind]
	}
	slope := make([]float64, n_intervals)
	for ind := range slope {
		slope[ind] = y_diffs[ind] / x_diffs[ind]
	}
	dydx_term := make([]float64, n_intervals)
	for ind := range slope {
		dydx_term[ind] = (dydx[ind+1] + dydx[ind] - 2*slope[ind]) / x_diffs[ind]
	}
	o_CHIP.a = make([]float64, n_intervals)
	for ind := range o_CHIP.a {
		o_CHIP.a[ind] = dydx_term[ind] / x_diffs[ind]
	}
	// b[n] =-(dydx[n+1] + 2dydx[n]) / (x[n+1] - x[n])
	// 		+3*(y[n+1] - y[n]) / (x[n+1] - x[n])**2
	o_CHIP.b = make([]float64, n_intervals)
	for ind := range o_CHIP.b {
		o_CHIP.b[ind] = (slope[ind]-dydx[ind])/x_diffs[ind] - dydx_term[ind]
	}
	o_CHIP.c = dydx[:len(dydx)-1]
	o_CHIP.d = y
	return o_CHIP
}

func (chip *CHIP) evaluate(xs_in []float64, add_nodes bool) []float64 {
	if add_nodes == true {
		xs_in = append(xs_in, chip.x_arr[1:len(chip.x_arr)-1]...)
		slices.Sort(xs_in)
	}
	o_y := make([]float64, len(xs_in))
	for ind_x_in, x_in := range xs_in {
		if x_in < chip.x_min || chip.x_max < x_in {
			panic(fmt.Sprintf("Input value %E outside domain of CHIP [%E, %E]", x_in, chip.x_min, chip.x_max))
		}
		for ind_x_val, x_val := range chip.x_arr {
			if x_in < x_val {
				x_n := chip.x_arr[ind_x_val-1]
				// # print(f"{self.x_arr[t_ind-1]}< {x} < {self.x_arr[t_ind]}")
				x_trans := x_in - x_n
				a := chip.a[ind_x_val-1]
				b := chip.b[ind_x_val-1]
				c := chip.c[ind_x_val-1]
				d := chip.d[ind_x_val-1]
				o_y[ind_x_in] = a*x_trans*x_trans*x_trans + b*x_trans*x_trans + c*x_trans + d
				break
			}
			if x_val == x_in {
				// # print(f"{self.x_arr[t_ind]-x} = {0}")
				o_y[ind_x_in] = chip.d[ind_x_val]
			}
		}
	}
	return o_y
}

func (chip *CHIP) evaluate_on_domain(n_steps int, add_nodes bool) ([]float64, []float64) {
	xs := linspace(chip.x_min, chip.x_max, n_steps)
	if add_nodes == true {
		xs = append(xs, chip.x_arr[1:len(chip.x_arr)-1]...)
		slices.Sort(xs)
	}
	return xs, chip.evaluate(xs, false)
}

func Interpolate_energy_path(mag *magnetization, cell_volume float64, M_sat float64, n_points int) ([]float64, []float64) {
	mag_slice := mag.Buffer()
	size := mag_slice.Size()
	n_images := mag.GetNImages()
	grad_slice := cuda.Buffer(3, size, n_images)
	defer cuda.Recycle(grad_slice)
	SetEffectiveField(grad_slice, mag)
	gradDOTtau := make([]float64, n_images)
	for ind_img := 0; ind_img < n_images; ind_img++ {
		gradDOTtau[ind_img] = cell_volume * M_sat * (-float64(cuda.Dot(grad_slice.SubSlice(ind_img), M.tangent_buffer_.SubSlice(ind_img))))
	}
	chip := New_CHIP(M.Geodesic_distances, M.E_img, gradDOTtau)
	o_xs, o_ys := chip.evaluate_on_domain(n_points, true)
	return o_xs, o_ys
}

// Inverts the gradient along the direction tangental to the path
// B_eff = -grad(Energy)
// B_eff = B_eff - 2*dot(force, tangent)*tangent
func climbing_force(Beff *data.Slice, mag *magnetization, climbing_image_index int) {
	image_Beff := Beff.SubSlice(climbing_image_index)
	image_tangent := mag.GetTangentBuffer().SubSlice(climbing_image_index)

	Beff_dot_tau := cuda.Dot(image_Beff, image_tangent)
	cuda.Madd2(image_Beff, image_Beff, image_tangent, 1.0, -2*Beff_dot_tau)
}
