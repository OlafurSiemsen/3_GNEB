package cuda

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

func RotateVectors(a, b *data.Slice, dt float32) {

	util.Argument(a.NComp() == 3 && b.NComp() == 3)
	util.Argument(a.Len() == b.Len())

	N := b.Len()
	cfg := make1DConf(N)

	k_rotatevectors_async(
		a.DevPtr(X), a.DevPtr(Y), a.DevPtr(Z),
		b.DevPtr(X), b.DevPtr(Y), b.DevPtr(Z),
		dt, N, cfg)
}
