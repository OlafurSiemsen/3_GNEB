package cuda

import (
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

// Projects each vector in a to be orthogonal to corresponding vector in b
// Assumes vectors in b are normalized
// dst[ith cell] = a[ith cell] - (dot(a[ith cell],b[ith cell]) * b[ith cell]
// see file://./orthogonalize.cu
func Orthogonalize(dst, a, b *data.Slice) {
	util.Argument(dst.NComp() == 3 && a.NComp() == 3 && b.NComp() == 3)
	util.Argument(dst.Len() == a.Len() && dst.Len() == b.Len())

	N := dst.Len()
	cfg := make1DConf(N)
	k_orthogonalize_async(dst.DevPtr(X), dst.DevPtr(Y), dst.DevPtr(Z),
		a.DevPtr(X), a.DevPtr(Y), a.DevPtr(Z),
		b.DevPtr(X), b.DevPtr(Y), b.DevPtr(Z),
		N, cfg)
}

// Projects each vector in a onto corresponding vector in b
// Assumes vectors in b are normalized
// dst[ith cell] = (dot(a[ith cell],b[ith cell])) * b[ith cell]
// see file://./projectonto.cu
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

// Projects each vector in a to be orthogonal to corresponsing vector b
// Note that this is the dot product on the configuration space of a and b
// Assumes b in normalized
// dst[ith cell] = a[ith cell] - (sum(dot(a[ith cell],b[ith cell])))*b[ith cell]
func Global_Orthogonalize(dst, a, b *data.Slice) {
	util.Argument(dst.NComp() == 3 && a.NComp() == 3 && b.NComp() == 3)
	util.Argument(dst.Len() == a.Len() && dst.Len() == b.Len())

	adotb := Dot(a, b)
	Madd2(dst, a, b, 1, -adotb) // TODO: Is this performant or should I write a seperate kernel? Also, doesn't fit in with other /cuda functions.
}
