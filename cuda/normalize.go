package cuda

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

// Normalize vec to unit length, unless length or vol are zero.
func Normalize(vec, vol *data.Slice) {
	n_images := vec.N_images
	if n_images != 1 {
		for it_image := 0; it_image < n_images; it_image++ {
			Normalize(vec.SubSlice(it_image), vol)
		}
		return
	}
	util.Argument(vol == nil || vol.NComp() == 1)
	N := vec.Len()
	cfg := make1DConf(N)

	k_normalize_async(vec.DevPtr(X), vec.DevPtr(Y), vec.DevPtr(Z), vol.DevPtr(0), N, cfg)

}
