package cuda

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

// dst[i] = |a[i]|
func Veclen(dst *data.Slice, a *data.Slice) {
	util.Argument(dst.NComp() == 1 && a.NComp() == 3)
	util.Argument(dst.Len() == a.Len())

	N := dst.Len()
	cfg := make1DConf(N)
	k_veclen_async(dst.DevPtr(0),
		a.DevPtr(X), a.DevPtr(Y), a.DevPtr(Z),
		N, cfg)
}
