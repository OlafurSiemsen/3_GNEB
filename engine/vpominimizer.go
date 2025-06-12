package engine

// TODO: Introductory line what this code does and reference to paper where this was introduced.
//   - see minimizer.go for reference

import (
	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

// Some generic parameter values which can be overwritten bei init()
// TODO: explain what they are.
// TODO: Check if they are actually good.
var (
	massVPO     = 1.0 // TODO: massVPO of the ??? What units? What is reasonable here?
	recip2mVPO  = 0.5 / massVPO
	stepsizeVPO = 0.01 // dt effective time step
	//MaxForce          = 100.0	// TODO: maximal allowed force???
	DmSamplesVPO int     = 10    // number of dm to keep for convergence check
	StopMaxDmVPO float64 = 1e-6  // stop minimizer if sampled dm is smaller than this
	MinimizePath bool    = true  // True if the path should be minimized, false if each image is minimized seperately
	FixEndImages bool    = true  // True if the first and last images are not to be modified
	AdvanceTime  bool    = false // Whether to increment time globally - time is reset at the end
)

// Initialization of the function. Can override variables.
func init() {
	DeclFunc("VPOMinimize", VPOMinimize, "Use gradient velocity projection optimization to zero the forces (or energy gradients)")
	DeclVar("massVPO", &massVPO, "Mass used in the VPO minimizer")
	DeclVar("stepsizeVPO", &stepsizeVPO, "stepsizeVPO used in the VPO minimizer")
	//DeclVar("MaxForce", &MaxForce, "MaxForce")
	DeclVar("VPOMinimizerStop", &StopMaxDmVPO, "Stopping max dM for VPOMinimize")
	DeclVar("VPOMinimizerSamples", &DmSamplesVPO, "Number of max dM to collect for VPOMinimize convergence check.")
}

// fixed length FIFO. Items can be added but not removed. Copied from minimizer.
type fifoRingVPO struct {
	count int
	tail  int // index to put next item. Will loop to 0 after exceeding length
	data  []float64
}

func FifoRingVPO(length int) fifoRingVPO {
	return fifoRingVPO{data: make([]float64, length)}
}

func (r *fifoRingVPO) Add(item float64) {
	r.data[r.tail] = item
	r.count++
	r.tail = (r.tail + 1) % len(r.data)
	if r.count > len(r.data) {
		r.count = len(r.data)
	}
}

func (r *fifoRingVPO) Max() float64 {
	max := r.data[0]
	for i := 1; i < r.count; i++ {
		if r.data[i] > max {
			max = r.data[i]
		}
	}
	return max
}

// objects that need to be stored for next iteration step
type VPOMinimizer struct {
	GNEB_kappa []float32   // Spring constants for the inter-image elastic forces
	f          *data.Slice // force
	v          *data.Slice // velocity
	lastDm     fifoRingVPO
}

// VPOMinimizer step
func (mini *VPOMinimizer) Step() {

	mag_slice := M.Buffer()
	size := mag_slice.Size()
	n_images := mag_slice.N_images

	// Initialize force to -\nabla B_eff
	if mini.f == nil { // make sure this is not empty upon first usage
		mini.f = cuda.Buffer(3, size, n_images)
		SetEffectiveField(mini.f, &M)
		if n_images > 1 && MinimizePath {
			GNEBForceTransformation(mini.f, mini.GNEB_kappa, &M)
		}
		cuda.Orthogonalize(mini.f, mini.f, mag_slice)
	}

	// Initialize velocity to 0
	if mini.v == nil {
		mini.v = cuda.Buffer(3, size, n_images)
		cuda.Zero3(mini.v)
	}

	// Update and increment index on force
	SetEffectiveField(mini.f, &M)
	if n_images > 1 && MinimizePath {
		GNEBForceTransformation(mini.f, mini.GNEB_kappa, &M)
	}
	cuda.Orthogonalize(mini.f, mini.f, mag_slice)

	// Convergence check
	m_nm1 := cuda.Buffer(3, size, n_images) // allocate memory
	defer cuda.Recycle(m_nm1)               // purge once this function ends
	data.Copy(m_nm1, mag_slice)             // copy the current magnetization to m0

	// Update and increment index on velocity
	cuda.Madd2(mini.v, mini.v, mini.f, 1, float32(recip2mVPO))

	// Factor for projection of velocity on force,
	// sets velocity to 0 if v.f < 0
	vDOTf := cuda.Dot(mini.v, mini.f) / cuda.Dot(mini.f, mini.f)
	if vDOTf <= 0.0 {
		vDOTf = 0.0
	}
	cuda.Madd2(mini.v, mini.f, mini.f, vDOTf, float32(recip2mVPO*stepsizeVPO))

	// Update and increment index on m
	// Note that this is a rotation since the length of each m is fixed
	cuda.RotateVectors(mag_slice, mini.v, float32(stepsizeVPO))
	// Since the magnetization of each image is changed, the quantities related to
	// the path need to be recalculated
	M.reset_calc_flags()

	// Rotate the velocity to the cotangent space of the new magnetization
	cuda.CotangentSpaceRotation(mini.v, mini.v, m_nm1, mag_slice)

	// End of this iteration step
	NSteps++

	dm := m_nm1 // this is just for readability
	cuda.Madd2(dm, mag_slice, m_nm1, 1., -1.)
	max_dm := cuda.MaxVecNorm(dm)
	mini.lastDm.Add(max_dm)
	err := mini.lastDm.Max()
	setLastErr(err) // report maxDm to user as LastErr

	if AdvanceTime {
		Time += Dt_si
	}
}

// Free
func (mini *VPOMinimizer) Free() {
	mini.f.Free()
	mini.v.Free()
}

// The main part of this function: Based on engine/minimizer.go
func VPOMinimize() {
	// Refer("TODO")
	SanityCheck()
	// Save the settings we are changing...
	prevType := solvertype
	prevFixDt := FixDt
	prevPrecess := Precess
	t0 := Time

	relaxing = true // disable temperature noise

	// ...to restore them later. Read as "defer ..." = "when function ends, do ..."
	// defer func() {
	// 	SetSolver(prevType)
	// 	FixDt = prevFixDt
	// 	Precess = prevPrecess
	// 	Time = t0

	// 	relaxing = false
	// }()

	// disable precession for torque calculation
	Precess = false

	// remove previous stepper
	if stepper != nil {
		stepper.Free()
	}

	// set stepper to the VPOMinimizer
	mini := VPOMinimizer{
		GNEB_kappa: []float32{0.1},
		f:          nil,
		v:          nil,
		lastDm:     FifoRingVPO(DmSamplesVPO)}
	stepper = &mini

	// break condition: change of magnetization is below a reasonable threshold
	cond := func() bool {
		return (mini.lastDm.count < DmSamplesVPO || mini.lastDm.Max() > StopMaxDmVPO)
	}

	RunWhile(cond)
	pause = true
	SetSolver(prevType)
	FixDt = prevFixDt
	Precess = prevPrecess
	Time = t0

	relaxing = false
}
