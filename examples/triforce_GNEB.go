//go:build ignore
// +build ignore

package main

import (
	"math"
	"math/rand"
	"os"
	"runtime"
	"runtime/pprof"
	"time"

	. "github.com/mumax/3/engine"
)

func main() {
	// Create a file to store the CPU profile
	cpuFile, err := os.Create("cpu.pprof")
	if err != nil {
		panic(err)
	}
	defer cpuFile.Close()

	// Start CPU profiling
	runtime.SetBlockProfileRate(1)
	if err := pprof.StartCPUProfile(cpuFile); err != nil {
		panic(err)
	}
	defer pprof.StopCPUProfile()

	// Physics
	Msat.Set(2 * 700e3) // Saturation magnetization, A/m
	Aex.Set(2 * 12e-12) // Exchange stiffness, J/m
	Ku_Angle := 90.0 * math.Pi / 180.0
	Ku1.Set(0)
	AnisU.Set([]float64{math.Cos(Ku_Angle), math.Sin(Ku_Angle), 0}) // Uniaxial anisotropy direction

	material := "permalloy"
	switch material {
	case "permalloy":
		Msat.Set(860e3)
		Aex.Set(13e-12)
	}

	// MuMax Section
	SetOutputFormat(OVF2_TEXT)
	SnapshotFormat = "png"
	defer InitAndClose()()

	// sample_dim := [3]float64{604.0e-9, 512.0e-9, 5.0e-9} // m
	// resolution := [3]float64{4e-9, 4e-9, 5.0e-9}         // m per cell

	NPointsEnergyCHIP = 100

	gridsize := [3]int{
		300,
		250,
		1,
	}

	cellsize := [3]float64{
		2.0e-9,
		2.0e-9,
		5.0e-9,
	}

	sample_dim := [3]float64{
		cellsize[0] * float64(gridsize[0]),
		cellsize[1] * float64(gridsize[1]),
		cellsize[2] * float64(gridsize[2]),
	}

	sample_origin := [2]float64{
		-0.5 * sample_dim[0],
		-0.5 * sample_dim[1],
	}
	n_images := 21

	NPointsEnergyCHIP = 100

	middle_image_index := (n_images - 1) / 2
	image_index_slice := make([]int, n_images)
	for i := range image_index_slice {
		image_index_slice[i] = i
	}
	middle_index_slice := image_index_slice[1 : len(image_index_slice)-1]

	SetMesh(gridsize[0], gridsize[1], gridsize[2],
		cellsize[0], cellsize[1], cellsize[2],
		0, 0, 0,
		n_images)

	hex_lattice_length := 300e-9
	motif_scale := 0.6
	hex_lattice_offset := [3]float64{0, 82e-9, 0}

	a1 := [2]float64{hex_lattice_length, 0}
	a2 := [2]float64{hex_lattice_length * math.Cos(math.Pi/3), hex_lattice_length * math.Sin(math.Pi/3)}

	sample_origin = [2]float64{
		sample_origin[0] + hex_lattice_offset[0], //math.Mod(hex_lattice_offset[0], a1[0]),
		sample_origin[1] + hex_lattice_offset[1], //math.Mod(hex_lattice_offset[1], a2[1]),
	}

	basis_motifs_loc := [][2]int{
		{0, 0},
	}

	recip_basis_motifs_loc := [][2]int{
		{0, 0},
		{1, 1},
		{0, 1},
	}

	var motif Shape
	var hex_triangles_geometry Shape
	var shape_slice []Shape

	motif_scale = 0.8
	t_triangle_height := hex_lattice_length * math.Sin(math.Pi/3)
	motif = Triangle(
		-0.5*hex_lattice_length, -(1.0/3.0)*t_triangle_height,
		0.5*hex_lattice_length, -(1.0/3.0)*t_triangle_height,
		0, (2.0/3.0)*t_triangle_height,
	).Scale(motif_scale, motif_scale, motif_scale)

	id_counter := 1
	for _, ind_hex := range basis_motifs_loc {
		ind_hex_1 := ind_hex[0]
		ind_hex_2 := ind_hex[1]
		t_a1 := [2]float64{1 * a1[0], 1 * a1[1]}
		t_a2 := [2]float64{1 * a2[0], 1 * a2[1]}

		// id := ind_hex_1*n_basis[0] + ind_hex_2 + 1
		t_translation_vector := [2]float64{
			hex_lattice_offset[0] + float64(ind_hex_1)*t_a1[0] + float64(ind_hex_2)*t_a2[0],
			hex_lattice_offset[1] + float64(ind_hex_1)*t_a1[1] + float64(ind_hex_2)*t_a2[1],
		}
		a1_shove := math.Floor(float64(ind_hex_2+1) / 2)
		t_translation_vector = [2]float64{
			t_translation_vector[0] - a1_shove*t_a1[0],
			t_translation_vector[1] - a1_shove*t_a1[1],
		}
		t_shape := motif.Transl(
			t_translation_vector[0],
			t_translation_vector[1],
			0,
		)
		DefRegion(id_counter, t_shape)
		shape_slice = append(shape_slice, t_shape)
		id_counter++
		if hex_triangles_geometry == nil {
			hex_triangles_geometry = t_shape
		} else {
			hex_triangles_geometry = hex_triangles_geometry.Add(t_shape)
		}
	}

	for _, ind_hex := range recip_basis_motifs_loc {

		ind_hex_1 := ind_hex[0]

		ind_hex_2 := ind_hex[1]
		t_a1 := [2]float64{1 * a1[0], 1 * a1[1]}
		t_a2 := [2]float64{1 * a2[0], 1 * a2[1]}

		// id := ind_hex_1*n_basis[0] + ind_hex_2 + 1
		t_translation_vector := [2]float64{
			hex_lattice_offset[0] + float64(ind_hex_1)*t_a1[0] + float64(ind_hex_2)*t_a2[0] - 0,
			hex_lattice_offset[1] + float64(ind_hex_1)*t_a1[1] + float64(ind_hex_2)*t_a2[1] - 2*t_triangle_height/3,
		}
		a1_shove := math.Floor(float64(ind_hex_2+1) / 2)
		t_translation_vector = [2]float64{
			t_translation_vector[0] - a1_shove*t_a1[0],
			t_translation_vector[1] - a1_shove*t_a1[1],
		}
		t_shape := motif.RotZ(math.Pi/3.0).Transl(
			t_translation_vector[0],
			t_translation_vector[1],
			0,
		)
		DefRegion(id_counter, t_shape)
		shape_slice = append(shape_slice, t_shape)
		id_counter++
		if hex_triangles_geometry == nil {
			hex_triangles_geometry = t_shape
		} else {
			hex_triangles_geometry = hex_triangles_geometry.Add(t_shape)
		}
	}

	SetGeom(hex_triangles_geometry)
	SnapshotAs(&M, "geometry.png")
	SaveAs(&Universe_regions, "regions.ovf")
	SaveAs(&Universe_geometry, "geometry.ovf")

	MinimizePath = true
	FixEndImages = true
	{ // GNEB
		state_list := []string{
			"/home/olafur/go/src/github.com/mumax/3_GNEB/examples/triforce_minima/CCWvortex_minus.ovf",
			"/home/olafur/go/src/github.com/mumax/3_GNEB/examples/triforce_minima/9_scale0.8_minimized_6.3709e-18",
		}
		interpolation_modes := []string{
			"direct",
			// "random_midpoint",
		}
		start := time.Now()
		{
			for ind_state_initial, path_state_initial := range state_list {
				for ind_state_final, path_state_final := range state_list {
					for _, interpolation_mode := range interpolation_modes {
						if ind_state_initial >= ind_state_final { // GNEB commutes
							continue
						}
						ClimbingImage = false
						_, _ = path_state_final, path_state_initial
						M.LoadFile(path_state_initial, 0)
						M.LoadFile(path_state_final, n_images-1)

						if interpolation_mode == "direct" {
							InterpolateMagnetization(M, false, 0, n_images-1)
						}
						if interpolation_mode == "random_midpoint" {
							M.SetInShape(nil, RandomMagSeed(rand.Int()), middle_image_index)
							InterpolateMagnetization(M, false, 0, middle_image_index)
							InterpolateMagnetization(M, false, middle_image_index, n_images-1)
						}

						M.AddRandomNoise(nil, 0.1, middle_index_slice...)

						// Short run to make sure the climbing image splits the path equalishly
						MaxIterVPO = 10000
						GNEB_kappa = []float32{1.0}
						ClimbingImage = false
						FixEndImages = true
						VPOMinimize()

						MaxIterVPO = 200000
						GNEB_kappa = []float32{1.0}
						ClimbingImage = true
						FixEndImages = true
						VPOMinimize()

					}
				}
			}
		}
		LogSystem()
		elapsed := time.Since(start)
		LogOut("Elapsed compute time, number of steps: ", elapsed, NSteps)
	}
}

func arbitrary_shapes(lattice_coord_slice [][2]int, sample_origin [2]float64, a1 [2]float64, a2 [2]float64, motif_shape Shape, geometry Shape) Shape {
	for ind, iter_lattice_coord := range lattice_coord_slice {
		id := ind + 1
		t_translation_vector := [2]float64{
			sample_origin[0] + float64(iter_lattice_coord[0])*a1[0] + float64(iter_lattice_coord[1])*a2[0],
			sample_origin[1] + float64(iter_lattice_coord[0])*a1[1] + float64(iter_lattice_coord[1])*a2[1],
		}
		t_shape := motif_shape.Transl(
			t_translation_vector[0],
			t_translation_vector[1],
			0,
		)
		DefRegion(id, t_shape)
		if geometry == nil {
			geometry = t_shape
		} else {
			geometry = geometry.Add(t_shape)
		}
	}
	return geometry
}

func hex_grid(n_basis [2]int, sample_origin [2]float64, a1 [2]float64, a2 [2]float64, motif_shape Shape, geometry Shape) (Shape, [][2]float64) {
	lattice_site_slice := make([][2]float64, n_basis[0]*n_basis[1])
	for ind_hex_1 := range n_basis[0] {
		for ind_hex_2 := range n_basis[1] {
			id := ind_hex_1*n_basis[0] + ind_hex_2 + 1
			t_translation_vector := [2]float64{
				sample_origin[0] + float64(ind_hex_1)*a1[0] + float64(ind_hex_2)*a2[0],
				sample_origin[1] + float64(ind_hex_1)*a1[1] + float64(ind_hex_2)*a2[1],
			}
			a1_shove := math.Floor(float64(ind_hex_2+1) / 2)
			t_translation_vector = [2]float64{
				t_translation_vector[0] - a1_shove*a1[0],
				t_translation_vector[1] - a1_shove*a1[1],
			}
			t_shape := motif_shape.Transl(
				t_translation_vector[0],
				t_translation_vector[1],
				0,
			)
			DefRegion(id, t_shape)
			if geometry == nil {
				geometry = t_shape
			} else {
				geometry = geometry.Add(t_shape)
			}
			lattice_site_slice[id-1] = t_translation_vector
		}
	}
	return geometry, lattice_site_slice
}

// 2D triangle with given vertices.
func Triangle(x0, y0, x1, y1, x2, y2 float64) Shape {
	double_area := x0*(y1-y2) + x1*(y2-y0) + x2*(y0-y1) // 2 * area
	if double_area == 0 {
		return func(x, y, z float64) bool { return false }
	}
	recip_double_area := 1 / double_area

	Sc := recip_double_area * (y0*x2 - x0*y2)
	Sx := recip_double_area * (y2 - y0)
	Sy := recip_double_area * (x0 - x2)

	Tc := recip_double_area * (x0*y1 - y0*x1)
	Tx := recip_double_area * (y0 - y1)
	Ty := recip_double_area * (x1 - x0)

	return func(x, y, z float64) bool {
		// barycentric coordinates
		s := Sc + Sx*x + Sy*y
		t := Tc + Tx*x + Ty*y
		return ((0 <= s) && (0 <= t) && (s+t <= 1))
	}
}
