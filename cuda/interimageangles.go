package cuda

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

// Calculates the angle between corresponging vectors in neighbouring images in
// src, copying them into dst
// see file://./interimageangles.cu
func InterimageAngles(dst *data.Slice, src *data.Slice, vol *data.Slice) {
	util.Argument(vol == nil || vol.NComp() == 1)
	util.Argument(dst.NComp() == 1)
	util.Argument(src.NComp() == 3)
	util.Assert(dst.N_images == src.N_images-1)
	N_cells := prod(src.Size())
	// Note: we skip the last image, forward difference
	N := src.Len() - N_cells
	cfg := make1DConf(N)

	k_interimageangles_async(dst.DevPtr(0),
		src.DevPtr(X), src.DevPtr(Y), src.DevPtr(Z),
		vol.DevPtr(0), N, N_cells, cfg)

}
