package cuda

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

// dst[i] = atan2(a[i],b[i])
func Atan2(dst *data.Slice, a *data.Slice, b *data.Slice) {
	util.Argument(dst.NComp() == 1 && a.NComp() == 1 && b.NComp() == 1)
	util.Argument(dst.Len() == a.Len() && dst.Len() == b.Len())

	N := dst.Len()
	cfg := make1DConf(N)
	k_cuatan2_async(dst.DevPtr(0),
		a.DevPtr(0), b.DevPtr(0),
		N, cfg)
}

// dst[i] = sin(a[i])
func Sin(dst *data.Slice, a *data.Slice) {
	util.Argument(dst.NComp() == 1 && a.NComp() == 1)
	util.Argument(dst.Len() == a.Len())

	N := dst.Len()
	cfg := make1DConf(N)
	k_cusin_async(dst.DevPtr(0),
		a.DevPtr(0),
		N, cfg)
}

// dst[i] = cos(a[i])
func Cos(dst *data.Slice, a *data.Slice) {
	util.Argument(dst.NComp() == 1 && a.NComp() == 1)
	util.Argument(dst.Len() == a.Len())

	N := dst.Len()
	cfg := make1DConf(N)
	k_cucos_async(dst.DevPtr(0),
		a.DevPtr(0),
		N, cfg)
}
