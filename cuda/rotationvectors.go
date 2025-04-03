package cuda

import (
	"math"
	"math/rand"
	"time"

	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

func RotationVectors(dst *data.Slice, start_vec *data.Slice, cross_prod_slice *data.Slice, cross_prod_norm_slice *data.Slice) {
	util.Argument(dst.NComp() == 3 && start_vec.NComp() == 3 && cross_prod_slice.NComp() == 3)
	util.Argument(dst.Len() == start_vec.Len() && dst.Len() == cross_prod_slice.Len())
	N := dst.Len()

	// Creating 2 orthonormal random vectors to create rotation vectors when
	// starting and ending vectors are anti-parralel
	rng := rand.New(rand.NewSource(int64(time.Now().UTC().UnixNano())))
	R := data.Vector{rng.Float64() - 0.5, rng.Float64() - 0.5, rng.Float64() - 0.5}
	R.Mul(1 / R.Len()) // normalize r1
	costheta := rng.Float64()*2 - 1
	sintheta := math.Sqrt(1 - costheta*costheta)
	versinetheta := (1 - costheta)
	// r1 is (1,0,0) rotated by theta around R
	r1 := data.Vector{
		costheta + R.X()*R.X()*versinetheta,
		R.Z()*sintheta + R.Y()*R.X()*versinetheta,
		-R.Y()*sintheta + R.Z()*R.X()*versinetheta,
	}
	// r2 is (0,1,0) rotated by theta around R
	r2 := data.Vector{
		-R.Z()*sintheta + R.X()*R.Y()*versinetheta,
		costheta + R.Y()*R.Y()*versinetheta,
		R.X()*sintheta + R.Z()*R.Y()*versinetheta,
	}

	r := Buffer(3, [3]int{2, 1, 1})
	defer Recycle(r)
	t_slice := data.NewSlice(3, [3]int{2, 1, 1})
	t_slice.SetVector(0, 0, 0, r1)
	t_slice.SetVector(1, 0, 0, r2)
	data.Copy(r, t_slice)
	cfg := make1DConf(N)
	k_rotationvectors_async(dst.DevPtr(X), dst.DevPtr(Y), dst.DevPtr(Z),
		start_vec.DevPtr(X), start_vec.DevPtr(Y), start_vec.DevPtr(Z),
		cross_prod_slice.DevPtr(X), cross_prod_slice.DevPtr(Y), cross_prod_slice.DevPtr(Z),
		cross_prod_norm_slice.DevPtr(X),
		r.DevPtr(X), r.DevPtr(Y), r.DevPtr(Z),
		N, cfg)
}
