package cuda

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

// Projects a onto the tangent space of b
// Assumes vectors in b are normalized
// dst[ith cell] = a[ith cell] - (dot(a[ith cell],b[ith cell]) * b[ith cell]
func ProjectOntoTangent(dst, a, b *data.Slice) {
	util.Argument(dst.NComp() == 3 && a.NComp() == 3 && b.NComp() == 3)
	util.Argument(dst.Len() == a.Len() && dst.Len() == b.Len())

	N := dst.Len()
	cfg := make1DConf(N)
	k_projectontotangent_async(dst.DevPtr(X), dst.DevPtr(Y), dst.DevPtr(Z),
		a.DevPtr(X), a.DevPtr(Y), a.DevPtr(Z),
		b.DevPtr(X), b.DevPtr(Y), b.DevPtr(Z),
		N, cfg)
}

// Projects a onto b
// Assumes vectors in b are normalized
// dst[ith cell] = (dot(a[ith cell],b[ith cell]) * b[ith cell]
func ProjectOnto(dst, a, b *data.Slice) {
	util.Argument(dst.NComp() == 3 && a.NComp() == 3 && b.NComp() == 3)
	util.Argument(dst.Len() == a.Len() && dst.Len() == b.Len())

	N := dst.Len()
	cfg := make1DConf(N)
	k_projectonto_async(dst.DevPtr(X), dst.DevPtr(Y), dst.DevPtr(Z),
		a.DevPtr(X), a.DevPtr(Y), a.DevPtr(Z),
		b.DevPtr(X), b.DevPtr(Y), b.DevPtr(Z),
		N, cfg)
}
