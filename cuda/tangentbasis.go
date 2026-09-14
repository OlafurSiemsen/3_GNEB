package cuda

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

type Tangent_space_basis struct {
	Magnetization_slice *data.Slice // Pointer back to the magnetization data.Slice to which this basis corresponds
	Tangent_basis0      *data.Slice // 3D basis vector for each magnetic moment TODO-olafur: Make private
	Tangent_basis1      *data.Slice // 3D basis vector for each magnetic moment TODO-olafur: Make private
	Fresh               bool        // Whether the basis corresponds to the current configuration, set to 0 when magnetization is updated
}

func NewTangentSpace(magnetization_slice *data.Slice) *Tangent_space_basis {
	o_tangent_space := new(Tangent_space_basis)

	o_tangent_space.Magnetization_slice = magnetization_slice
	o_tangent_space.Fresh = false
	return o_tangent_space
}

// see file://./generate_tangent_basis.cu
func (tangent_space *Tangent_space_basis) generate() {
	mag_slice := tangent_space.Magnetization_slice
	util.Argument(mag_slice.NComp() == 3)
	tangent_space.Tangent_basis0 = NewSlice(3, mag_slice.Size(), mag_slice.N_images)
	tangent_space.Tangent_basis1 = NewSlice(3, mag_slice.Size(), mag_slice.N_images)

	N := mag_slice.Len()
	cfg := make1DConf(N)
	k_generate_tangent_basis_async(
		tangent_space.Tangent_basis0.DevPtr(0),
		tangent_space.Tangent_basis0.DevPtr(1),
		tangent_space.Tangent_basis0.DevPtr(2),
		tangent_space.Tangent_basis1.DevPtr(0),
		tangent_space.Tangent_basis1.DevPtr(1),
		tangent_space.Tangent_basis1.DevPtr(2),
		mag_slice.DevPtr(X),
		mag_slice.DevPtr(Y),
		mag_slice.DevPtr(Z),
		N, cfg,
	)
	tangent_space.Fresh = true
}

// see file://./project_into_tangent_basis.cu
func (tangent_space *Tangent_space_basis) Project_into(o_slice, i_slice *data.Slice) {
	util.Argument(i_slice.NComp() == 3)
	util.Argument(o_slice.NComp() == 2)

	if tangent_space.Fresh == false {
		tangent_space.generate()
	}

	N := i_slice.Len()
	cfg := make1DConf(N)
	k_project_into_tangent_basis_async(
		tangent_space.Tangent_basis0.DevPtr(0),
		tangent_space.Tangent_basis0.DevPtr(1),
		tangent_space.Tangent_basis0.DevPtr(2),
		tangent_space.Tangent_basis1.DevPtr(0),
		tangent_space.Tangent_basis1.DevPtr(1),
		tangent_space.Tangent_basis1.DevPtr(2),
		i_slice.DevPtr(X),
		i_slice.DevPtr(Y),
		i_slice.DevPtr(Z),
		o_slice.DevPtr(0),
		o_slice.DevPtr(1),
		N, cfg,
	)
}

// see file://./project_out_of_tangent_basis.cu
func (tangent_space *Tangent_space_basis) Project_out_of(o_slice, i_slice *data.Slice) {
	util.Argument(i_slice.NComp() == 2)
	util.Argument(o_slice.NComp() == 3)

	if tangent_space.Fresh == false {
		tangent_space.generate()
	}

	N := i_slice.Len()
	cfg := make1DConf(N)
	k_project_out_of_tangent_basis_async(
		tangent_space.Tangent_basis0.DevPtr(0),
		tangent_space.Tangent_basis0.DevPtr(1),
		tangent_space.Tangent_basis0.DevPtr(2),
		tangent_space.Tangent_basis1.DevPtr(0),
		tangent_space.Tangent_basis1.DevPtr(1),
		tangent_space.Tangent_basis1.DevPtr(2),
		i_slice.DevPtr(0),
		i_slice.DevPtr(1),
		o_slice.DevPtr(X),
		o_slice.DevPtr(Y),
		o_slice.DevPtr(Z),
		N, cfg,
	)
}

func (tangent_space *Tangent_space_basis) Free() {
	tangent_space.Tangent_basis0.Free()
	tangent_space.Tangent_basis1.Free()
	tangent_space.Fresh = false
}
