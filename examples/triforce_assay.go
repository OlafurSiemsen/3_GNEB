//go:build ignore
// +build ignore

package main

import (
	"fmt"
	"math"
	"math/rand"
	"os"
	"runtime"
	"runtime/pprof"

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

	N_images := 1
	SetMesh(gridsize[0], gridsize[1], gridsize[2],
		cellsize[0], cellsize[1], cellsize[2],
		0, 0, 0,
		N_images)

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
	var experiment_name string
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

	// regions: 1=middle, 2=bottom, 3=top right, 4=top left
	M.SetInShape(nil, Vortex(1, -1).Transl(hex_lattice_offset[0], hex_lattice_offset[1], hex_lattice_offset[2]))
	SnapshotAs(&M, "0_init.png")
	SaveAs(&M, "0_init.ovf")

	Relax()
	Minimize()
	SnapshotAs(&M, "0_final.png")
	SaveAs(&M, "0_final.ovf")
	DrainOutput()
	var N_experiments int

	N_experiments = 10
	experiment_name = "random_init"
	os.Mkdir(fmt.Sprint(OD(), experiment_name), 0777)
	for ind_experiment := range N_experiments {
		M.SetRegion(1, RandomMagSeed(rand.Int()))
		SnapshotAs(&M, fmt.Sprint(experiment_name, "/", ind_experiment, "_inital.png"))

		Relax()
		Minimize()
		t_energy := GetTotalEnergy()
		SnapshotAs(&M, fmt.Sprint(experiment_name, "/", ind_experiment, "_minimized", fmt.Sprintf("_%.4e", t_energy), ".png"))
		SaveAs(&M, fmt.Sprint(experiment_name, "/", ind_experiment, "_minimized", fmt.Sprintf("_%.4e", t_energy)))
		M.LoadFile(OD() + "0_final.ovf")
	}

	LogSystem()
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
