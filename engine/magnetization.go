package engine

import (
	"math"
	"reflect"
	"slices"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

var M magnetization // reduced magnetization (unit length)

func init() { DeclLValue("m", &M, `Reduced magnetization (unit length)`) }

// Special buffered quantity to store magnetization
// makes sure it's normalized etc.
type magnetization struct {
	buffer_                 *data.Slice
	N_images                int
	E_img                   []float64   // Energy, one for each image
	E_img_calc              bool        // True if path E_img has been calculated
	tangent_buffer_         *data.Slice // Contains the tangents pointing from one image to the next
	tangent_calc            bool        // True if path tangent has been calculated
	geodesic_tangents_calc  bool        // True if the tangents are orthogonal to m
	Geodesic_distances      []float64   // Geodesic distance between neighbouring images
	Geodesic_distances_calc bool        // True if path Geodesic_distances has been calculated
}

func (m *magnetization) Mesh() *data.Mesh    { return Mesh() }
func (m *magnetization) NComp() int          { return 3 }
func (m *magnetization) Name() string        { return "m" }
func (m *magnetization) Unit() string        { return "" }
func (m *magnetization) Buffer() *data.Slice { return m.buffer_ } // todo: rename Gpu()?

func (m *magnetization) Comp(c int) ScalarField  { return Comp(m, c) }
func (m *magnetization) SetValue(v interface{})  { m.SetInShape(nil, v.(Config)) }
func (m *magnetization) InputType() reflect.Type { return reflect.TypeOf(Config(nil)) }
func (m *magnetization) Type() reflect.Type      { return reflect.TypeOf(new(magnetization)) }
func (m *magnetization) Eval() interface{}       { return m }
func (m *magnetization) average() []float64      { return sAverageMagnet(M.Buffer()) }
func (m *magnetization) Average() data.Vector    { return unslice(m.average()) }
func (m *magnetization) normalize()              { cuda.Normalize(m.Buffer(), geometry.Gpu()) }

// allocate storage (not done by init, as mesh size may not yet be known then)
func (m *magnetization) alloc() {
	m.buffer_ = cuda.NewSlice(3, m.Mesh().Size(), m.N_images)
	if m.N_images != 0 {
		m.tangent_buffer_ = cuda.NewSlice(3, m.Mesh().Size(), m.N_images)
		m.Geodesic_distances = make([]float64, m.N_images-1)
	}
	m.reset_calc_flags()
	m.Set(RandomMag()) // sane starting config
}

// Resets the quantities to do with the path, this should be invoked whenever
// the magnetization is changed
func (m *magnetization) reset_calc_flags() {
	m.E_img_calc = false
	m.tangent_calc = false
	m.geodesic_tangents_calc = false
	m.Geodesic_distances_calc = false
}

func (m *magnetization) Set_N_Images(n_images int) {
	m.N_images = n_images
	// m.buffer_.N_images = n_images
}

// TODO: Perhaps also change the tangent slice to a subslice
func (i_magnetization *magnetization) SubMagnetization(ind_image int) *magnetization {
	o_magnetization := i_magnetization
	o_magnetization.buffer_ = o_magnetization.buffer_.SubSlice(ind_image)
	o_magnetization.N_images = 1
	return o_magnetization
}

func (b *magnetization) SetArray(src *data.Slice, ind_image_variadic ...int) {
	ind_image := data.ImageIndex(ind_image_variadic)
	if src.Size() != b.Mesh().Size() {
		src = data.Resample(src, b.Mesh().Size())
	}
	if len(ind_image_variadic) == 0 {
		data.Copy(b.Buffer(), src)
	} else {
		data.Copy(b.Buffer().SubSlice(ind_image), src)
	}
	b.normalize()
}

func (m *magnetization) Set(c Config) {
	checkMesh()
	m.SetInShape(nil, c)
}

func (m *magnetization) LoadFile(fname string, ind_image_variadic ...int) {
	if m.N_images == 1 { // Just the normal MuMax way
		m.SetArray(LoadFile(fname))
	} else { // Load to a specific image
		ind_image := data.ImageIndex(ind_image_variadic)
		data.Copy(m.buffer_.SubSlice(ind_image), LoadFile(fname)) // Is this slow?
		// // More detailed assignment, in case the one liner fails
		// stored_magnetization_ptr := m.buffer_ // We store a pointer to the original magnetization...
		// m.buffer_ = stored_magnetization_ptr.SubSlice(ind_image)
		// m.SetArray(LoadFile(fname))
		// m.buffer_ = stored_magnetization_ptr //...and then restore the original magnetization pointer
	}
}

func (m *magnetization) LoadFiles(fname ...string) {
	n_files := len(fname)
	var it_indeces []int
	if n_files > m.N_images {
		panic("Loading more images than there are in the path not supported (yet)")
	} else {
		it_indeces = SpreadIndex(n_files, m.N_images)
	}
	for ind_fname, ind_image := range it_indeces {
		m.LoadFile(fname[ind_fname], ind_image)
	}
}

// Returns a slice of length n with its members being equally(ish) spaced indeces in
// the range [0, max_index], inclusive
func SpreadIndex(n_indeces int, n_images int) []int {
	max_index := n_images - 1
	if n_indeces == 2 {
		return []int{0, max_index}
	}
	o_indeces := make([]int, n_indeces)
	interval := float64(max_index) / float64(n_indeces-1)
	for it := 0; it < n_indeces; it++ {
		o_indeces[it] = int(math.Round(float64(it) * interval))
	}
	return o_indeces
}
func (m *magnetization) Slice() (s *data.Slice, recycle bool) {
	return m.Buffer(), false
}

func (m *magnetization) EvalTo(dst *data.Slice) {
	data.Copy(dst, m.buffer_)
}

func (m *magnetization) Region(r int) *vOneReg { return vOneRegion(m, r) }

func (m *magnetization) String() string { return util.Sprint(m.Buffer().HostCopy()) }

// Set the value of one cell.
func (m *magnetization) SetCell(ix, iy, iz int, v data.Vector) {
	r := Index2Coord(ix, iy, iz)
	if geometry.shape != nil && !geometry.shape(r[X], r[Y], r[Z]) {
		return
	}
	vNorm := v.Len()
	for c := 0; c < 3; c++ {
		cuda.SetCell(m.Buffer(), c, ix, iy, iz, float32(v[c]/vNorm))
	}
}

// Get the value of one cell.
func (m *magnetization) GetCell(ix, iy, iz int) data.Vector {
	mx := float64(cuda.GetCell(m.Buffer(), X, ix, iy, iz))
	my := float64(cuda.GetCell(m.Buffer(), Y, ix, iy, iz))
	mz := float64(cuda.GetCell(m.Buffer(), Z, ix, iy, iz))
	return Vector(mx, my, mz)
}

func (m *magnetization) Quantity() []float64 { return slice(m.Average()) }

// Sets the magnetization inside the shape
func (m *magnetization) SetInShape(region Shape, conf Config, ind_image_variadic ...int) {
	checkMesh()

	if region == nil {
		region = universe
	}
	images_specified := len(ind_image_variadic) != 0 // Checks if the user specified images to save
	host := m.Buffer().HostCopy()
	stored_host_ptr := host // We store a pointer to the original magnetization...
	n := m.Mesh().Size()
	var h [3][][][]float32
	for it_image := 0; it_image < m.N_images; it_image++ {
		if images_specified && !slices.Contains(ind_image_variadic, it_image) { // Skips images that weren't specified by user
			continue
		}
		host = stored_host_ptr.SubSlice(it_image) // ...iterate over the images...
		h = host.Vectors()
		for iz := 0; iz < n[Z]; iz++ {
			for iy := 0; iy < n[Y]; iy++ {
				for ix := 0; ix < n[X]; ix++ {
					r := Index2Coord(ix, iy, iz)
					x, y, z := r[X], r[Y], r[Z]
					if region(x, y, z) { // inside
						m := conf(x, y, z)
						h[X][iz][iy][ix] = float32(m[X])
						h[Y][iz][iy][ix] = float32(m[Y])
						h[Z][iz][iy][ix] = float32(m[Z])
					}
				}
			}
		}
	}
	host = stored_host_ptr // ...and then restore the original magnetization pointer
	m.SetArray(host)
}

// set m to config in region
func (m *magnetization) SetRegion(region int, conf Config) {
	host := m.Buffer().HostCopy()
	h := host.Vectors()
	n := m.Mesh().Size()
	r := byte(region)

	regionsArr := regions.HostArray()

	for iz := 0; iz < n[Z]; iz++ {
		for iy := 0; iy < n[Y]; iy++ {
			for ix := 0; ix < n[X]; ix++ {
				pos := Index2Coord(ix, iy, iz)
				x, y, z := pos[X], pos[Y], pos[Z]
				if regionsArr[iz][iy][ix] == r {
					m := conf(x, y, z)
					h[X][iz][iy][ix] = float32(m[X])
					h[Y][iz][iy][ix] = float32(m[Y])
					h[Z][iz][iy][ix] = float32(m[Z])
				}
			}
		}
	}
	m.SetArray(host)
}

func (m *magnetization) resize() {
	backup := m.Buffer().HostCopy()
	s2 := Mesh().Size()
	resized := data.Resample(backup, s2)
	m.buffer_.Free()
	m.buffer_ = cuda.NewSlice(VECTOR, s2)
	data.Copy(m.buffer_, resized)
}
