package engine

import (
	"math"
	"math/rand"
	"reflect"
	"slices"
	"time"

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
	n_images                int         // Number of images, used to describe the calculation currently in progress
	E_img                   []float64   // Energy, one for each image
	E_img_calc              bool        // True if path E_img has been calculated
	E_derivative            []float64   // Tangential derivatives, i.e. inner product of B_eff and tangent
	E_derivative_calc       bool        // True if tangential derivatives has been calculated
	tangent_buffer_         *data.Slice // Contains the tangents pointing from one image to the next
	tangent_calc            bool        // True if path tangent has been calculated
	geodesic_tangents_calc  bool        // True if the tangents are orthogonal to m
	Geodesic_distances      []float64   // Geodesic distance between neighbouring images
	Geodesic_distances_calc bool        // True if path Geodesic_distances has been calculated
	climbing_image_index    *int        // Indicates which image is currently climbing, nil means outdated
}

func (m *magnetization) Mesh() *data.Mesh              { return Mesh() }
func (m *magnetization) NComp() int                    { return 3 }
func (m *magnetization) Name() string                  { return "m" }
func (m *magnetization) Unit() string                  { return "" }
func (m *magnetization) Buffer() *data.Slice           { return m.buffer_ } // todo: rename Gpu()?
func (m *magnetization) GetTangentBuffer() *data.Slice { return m.tangent_buffer_ }

func (m *magnetization) Comp(c int) ScalarField  { return Comp(m, c) }
func (m *magnetization) SetValue(v interface{})  { m.SetInShape(nil, v.(Config)) }
func (m *magnetization) InputType() reflect.Type { return reflect.TypeOf(Config(nil)) }
func (m *magnetization) Type() reflect.Type      { return reflect.TypeOf(new(magnetization)) }
func (m *magnetization) Eval() interface{}       { return m }
func (m *magnetization) average() []float64      { return sAverageMagnet(M.Buffer()) }
func (m *magnetization) Average() data.Vector    { return unslice(m.average()) }
func (m *magnetization) normalize()              { cuda.Normalize(m.Buffer(), Universe_geometry.Gpu()) }

// allocate storage (not done by init, as mesh size may not yet be known then)
func (m *magnetization) alloc() {
	n_images := m.GetNImages()
	m.buffer_ = cuda.NewSlice(3, m.Mesh().Size(), n_images)
	if n_images != 1 {
		m.tangent_buffer_ = cuda.NewSlice(3, m.Mesh().Size(), n_images)
		m.Geodesic_distances = make([]float64, n_images-1)
	}
	m.Reset_calc_flags()
	m.Set(RandomMag()) // sane starting config
}

// TODO-olafur: Refactor redundancy
func (m *magnetization) GetNImages() int {
	if m.buffer_ == nil {
		return m.n_images
	} else {
		return m.n_images
	}
}

// Resets the quantities to do with the path, this should be invoked whenever
// the magnetization is changed
func (m *magnetization) Reset_calc_flags() {
	m.E_img_calc = false
	m.E_derivative_calc = false
	m.tangent_calc = false
	m.geodesic_tangents_calc = false
	m.Geodesic_distances_calc = false
	// m.climbing_image_index = nil
}

func (m *magnetization) SetNImages(n_images int) {
	if m.buffer_ == nil {
		m.n_images = n_images
	} else {
		m.n_images = n_images
		m.buffer_.N_images = n_images
	}
	// m.buffer_.N_images = n_images
}

// TODO-olafur: Perhaps also change the tangent slice to a subslice.
// TODO-olafur: Consider removing
func (i_magnetization *magnetization) SubMagnetization(ind_image int) *magnetization {
	o_magnetization := *i_magnetization
	o_magnetization.buffer_ = o_magnetization.buffer_.SubSlice(ind_image)
	o_magnetization.SetNImages(1)
	return &o_magnetization
}

// TODO-olafur: Add guard clause to check number of passed images
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
	b.Reset_calc_flags()
}

func (m *magnetization) Set(c Config) {
	checkMesh()
	m.SetInShape(nil, c)
}

func (m *magnetization) LoadFile(fname string, ind_image_variadic ...int) {
	n_images := m.GetNImages()
	if n_images == 1 { // Just the normal MuMax way
		m.SetArray(LoadFile(fname))
	} else {
		stored_magnetization := m.buffer_ // We store a pointer to the original magnetization...
		ind_image := data.ImageIndex(ind_image_variadic)
		m.buffer_ = stored_magnetization.SubSlice(ind_image) // replace the original magnetization with the right image
		m.SetArray(LoadFile(fname))
		m.buffer_ = stored_magnetization //...and then restore the original magnetization pointer

		// Load to a specific image

		// data.Copy(m.buffer_.SubSlice(ind_image), LoadFile(fname)) // TODO-olafur: rework to use SetArray
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
	n_images := m.GetNImages()
	if n_files > n_images {
		panic("Loading more images than there are in the path not supported (yet)")
	} else {
		it_indeces = SpreadIndex(n_files, n_images)
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
	if Universe_geometry.shape != nil && !Universe_geometry.shape(r[X], r[Y], r[Z]) {
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
	host := m.Buffer().HostCopy()                    // TODO-olafur: Doesn't this need to be recycled at the end?
	stored_host_ptr := host                          // We store a pointer to the original magnetization...
	n := m.Mesh().Size()
	var h [3][][][]float32
	n_images := m.GetNImages()
	for it_image := 0; it_image < n_images; it_image++ {
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
	m.Reset_calc_flags()
}

// set m to config in region
func (m *magnetization) SetRegion(region int, conf Config, ind_image_variadic ...int) {
	host := m.Buffer().HostCopy()
	n := m.Mesh().Size()
	r := byte(region)
	regionsArr := Universe_regions.HostArray()
	images_specified := len(ind_image_variadic) != 0 // Checks if the user specified images to save
	stored_host_ptr := host                          // We store a pointer to the original magnetization...
	var h [3][][][]float32
	n_images := m.GetNImages()

	for it_image := 0; it_image < n_images; it_image++ {
		if images_specified && !slices.Contains(ind_image_variadic, it_image) { // Skips images that weren't specified by user
			continue
		}
		host = stored_host_ptr.SubSlice(it_image) // ...iterate over the images...
		h = host.Vectors()
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
	}
	host = stored_host_ptr // ...and then restore the original magnetization pointer
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

// Adds random noise to the magnetization in the form of a field of scaled random vectors.
// TODO-olafur: Change to use e.g. Uniform(1, 0, 0).Add(0.2, RandomMag())
func (m *magnetization) AddRandomNoise(region Shape, scale float64, ind_image_variadic ...int) {
	checkMesh()
	scale32 := float32(scale)
	if region == nil {
		region = universe
	}
	images_specified := len(ind_image_variadic) != 0 // Checks if the user specified images to randomize
	host := m.Buffer().HostCopy()
	stored_host_ptr := host // We store a pointer to the original magnetization...
	n := m.Mesh().Size()
	var h [3][][][]float32
	random_generator := rand.New(rand.NewSource(time.Now().UnixNano()))
	n_images := m.GetNImages()
	for ind_image := 0; ind_image < n_images; ind_image++ {
		if images_specified && !slices.Contains(ind_image_variadic, ind_image) { // Skips images that weren't specified by user
			continue
		}
		host = stored_host_ptr.SubSlice(ind_image) // ...iterate over the images...
		h = host.Vectors()
		for iz := 0; iz < n[Z]; iz++ {
			for iy := 0; iy < n[Y]; iy++ {
				for ix := 0; ix < n[X]; ix++ {
					r := Index2Coord(ix, iy, iz)
					x, y, z := r[X], r[Y], r[Z]
					if region(x, y, z) { // inside
						for icomp := 0; icomp < 3; icomp++ {
							h[icomp][iz][iy][ix] += scale32 * 2 * (random_generator.Float32() - 0.5)
						}

					}
				}
			}
		}
	}
	host = stored_host_ptr // ...and then restore the original magnetization pointer
	m.SetArray(host)
}
