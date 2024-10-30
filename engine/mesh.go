package engine

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

var globalmesh_ data.Mesh // mesh for m and everything that has the same size

func init() {
	DeclFunc("SetGridSize", SetGridSize, `Sets the number of cells for X,Y,Z`)
	DeclFunc("SetCellSize", SetCellSize, `Sets the X,Y,Z cell size in meters`)
	DeclFunc("SetMesh", SetMeshWrapped, `Sets GridSize, CellSize and PBC at the same time`)
	DeclFunc("SetPBC", SetPBC, "Sets the number of repetitions in X,Y,Z to create periodic boundary "+
		"conditions. The number of repetitions determines the cutoff range for the demagnetization.")
}

func Mesh() *data.Mesh {
	checkMesh()
	return &globalmesh_
}

func arg(msg string, test bool) {
	if !test {
		panic(UserErr(msg + ": illegal argument"))
	}
}

// Set the simulation mesh to Nx x Ny x Nz cells of given size.
// Can be set only once at the beginning of the simulation.
// TODO: dedup arguments from globals

// The below is a struct that contains the parameters for the mesh.
// With this we can make each function default to the globalmesh_ unless pointed at a specific mesh.
type MeshParams struct {
	Nx, Ny, Nz                      int
	cellSizeX, cellSizeY, cellSizeZ float64
	pbcx, pbcy, pbcz                int
	mesh                            data.Mesh
}

// This is a *partial* constructor for the MeshParams struct.
// It has the same signature as the old functions that interact with
// data.Mesh and returns a MeshParams struct with an empty mesh field
func NewPartialMeshParams(Nx, Ny, Nz int, cellSizeX, cellSizeY, cellSizeZ float64, pbcx, pbcy, pbcz int) MeshParams {
	return MeshParams{
		Nx: Nx, Ny: Ny, Nz: Nz,
		cellSizeX: cellSizeX, cellSizeY: cellSizeY, cellSizeZ: cellSizeZ,
		pbcx: pbcx, pbcy: pbcy, pbcz: pbcz,
	}
}

// Check whether mesh field is intialized and substitute globalmesh_ if not
func SetMeshDefault(mesh data.Mesh) data.Mesh {
	t_nilPrm := data.Mesh{}
	if mesh != t_nilPrm {
		return mesh
	} else {
		return globalmesh_
	}
}

// Wrapper for the new SetMeshWrapped function for compatibility with scripting
func SetMeshWrapped(Nx, Ny, Nz int, cellSizeX, cellSizeY, cellSizeZ float64, pbcx, pbcy, pbcz int) {
	prm := NewPartialMeshParams(Nx, Ny, Nz, cellSizeX, cellSizeY, cellSizeZ, pbcx, pbcy, pbcz)
	SetMesh(prm)
}
func SetMesh(prm MeshParams) {
	SetBusy(true)
	defer SetBusy(false)

	// Unpack MeshParams fields into seperate parameters for ease of use
	Nx, Ny, Nz := prm.Nx, prm.Ny, prm.Nz
	cellSizeX, cellSizeY, cellSizeZ := prm.cellSizeX, prm.cellSizeY, prm.cellSizeZ
	pbcx, pbcy, pbcz := prm.pbcx, prm.pbcy, prm.pbcz
	// Check whether mesh field is intialized and substitute globalmesh_ if not
	var mesh data.Mesh
	mesh = SetMeshDefault(prm.mesh)

	arg("GridSize", Nx > 0 && Ny > 0 && Nz > 0)
	arg("CellSize", cellSizeX > 0 && cellSizeY > 0 && cellSizeZ > 0)
	arg("PBC", pbcx >= 0 && pbcy >= 0 && pbcz >= 0)

	sizeChanged := globalmesh_.Size() != [3]int{Nx, Ny, Nz}
	cellSizeChanged := globalmesh_.CellSize() != [3]float64{cellSizeX, cellSizeY, cellSizeZ}
	pbc := []int{pbcx, pbcy, pbcz}

	if mesh.Size() == [3]int{0, 0, 0} {
		// first time mesh is set
		globalmesh_ = *data.NewMesh(Nx, Ny, Nz, cellSizeX, cellSizeY, cellSizeZ, pbc...)
		M.alloc()
		regions.alloc()
	} else {
		// here be dragons
		LogOut("resizing...")

		// free everything to trigger kernel recalculation, etc
		conv_.Free()
		conv_ = nil
		mfmconv_.Free()
		mfmconv_ = nil
		cuda.FreeBuffers()

		// resize everything
		globalmesh_ = *data.NewMesh(Nx, Ny, Nz, cellSizeX, cellSizeY, cellSizeZ, pbc...)
		if sizeChanged || cellSizeChanged {
			M.resize()
			regions.resize()
			geometry.buffer.Free()
			geometry.buffer = data.NilSlice(1, Mesh().Size())
			geometry.setGeom(geometry.shape)

			// remove excitation extra terms if they don't fit anymore
			// up to the user to add them again
			B_ext.RemoveExtraTerms()
			J.RemoveExtraTerms()

			B_therm.noise.Free()
			B_therm.noise = nil
		}
	}
	lazy_gridsize = []int{Nx, Ny, Nz}
	lazy_cellsize = []float64{cellSizeX, cellSizeY, cellSizeZ}
	lazy_pbc = []int{pbcx, pbcy, pbcz}
}

func printf(f float64) float32 {
	return float32(f)
}

// for lazy setmesh: set gridsize and cellsize in separate calls
var (
	lazy_gridsize []int
	lazy_cellsize []float64
	lazy_pbc      = []int{0, 0, 0}
)

func SetGridSize(Nx, Ny, Nz int) {
	lazy_gridsize = []int{Nx, Ny, Nz}
	if lazy_cellsize != nil {
		prm := MeshParams{
			Nx: Nx, Ny: Ny, Nz: Nz,
			cellSizeX: lazy_cellsize[X], cellSizeY: lazy_cellsize[Y], cellSizeZ: lazy_cellsize[Z],
			pbcx: lazy_pbc[X], pbcy: lazy_pbc[Y], pbcz: lazy_pbc[Z],
		}
		SetMesh(prm)
	}
}

func SetCellSize(cx, cy, cz float64) {
	lazy_cellsize = []float64{cx, cy, cz}
	if lazy_gridsize != nil {
		prm := MeshParams{
			Nx: lazy_gridsize[X], Ny: lazy_gridsize[Y], Nz: lazy_gridsize[Z],
			cellSizeX: cx, cellSizeY: cy, cellSizeZ: cz,
			pbcx: lazy_pbc[X], pbcy: lazy_pbc[Y], pbcz: lazy_pbc[Z],
		}
		SetMesh(prm)
	}
}

func SetPBC(pbcx, pbcy, pbcz int) {
	lazy_pbc = []int{pbcx, pbcy, pbcz}
	if lazy_gridsize != nil && lazy_cellsize != nil {
		prm := MeshParams{
			Nx: lazy_gridsize[X], Ny: lazy_gridsize[Y], Nz: lazy_gridsize[Z],
			cellSizeX: lazy_cellsize[X], cellSizeY: lazy_cellsize[Y], cellSizeZ: lazy_cellsize[Z],
			pbcx: pbcx, pbcy: pbcy, pbcz: pbcz,
		}
		SetMesh(prm)
	}
}

// check if mesh is set
func checkMesh() {
	if globalmesh_.Size() == [3]int{0, 0, 0} {
		panic("need to set mesh first")
	}
}
