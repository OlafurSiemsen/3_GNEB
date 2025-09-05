package engine

// TODO: Introductory line what this code does and reference to paper where this was introduced.
//   - see minimizer.go for reference

import (
	"math"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
)

// Some generic parameter values which can be overwritten by init()
// TODO: explain what they are.
// TODO: Check if they are actually good.
var (
	massVPO     = 1e-0 // TODO: massVPO of the ??? What units? What is reasonable here?
	recip2mVPO  = 0.5 / massVPO
	stepsizeVPO = 0.05 // dt effective time step
	//MaxForce          = 100.0	// TODO: maximal allowed force???
	DmSamplesVPO int       = 10             // number of dm to keep for convergence check
	StopMaxDmVPO float64   = 1e-6           // stop minimizer if sampled dm is smaller than this
	FSamplesVPO  int       = 1              // Number of max F to keep for convergence check
	StopMaxFVPO  float64   = 1e-6           // stop minimizer if sampled maximum force is smaller than this
	MaxIterVPO   int       = 10000          // Maximum number of iterations
	MinimizePath bool      = true           // True if the path should be minimized, false if each image is minimized seperately
	FixEndImages bool      = true           // True if the first and last images are not to be modified
	AdvanceTime  bool      = false          // Whether to increment time globally - time is reset at the end
	GNEB_kappa   []float32 = []float32{0.5} // Spring constants for the inter-image elastic forces

	//Diagnostic outputs, TODO:Remove, or at least disable for performance
	LastMaxVPOForce     float64
	MaxVPOForce         = NewScalarValue("maxVPOForce", "T", "Maximum VPO force, over all cells", GetMaxVPOForce)
	LastVPOForceNorm    float64
	VPOForceNorm        = NewScalarValue("VPOForceNorm", "T", "Norm of VPO force, over all cells", GetVPOForceNorm)
	LastVPOVelocity     *data.Slice
	VPOVelocity         = NewVectorField("VPOv", "", "Velocity as accumulated internally to VPO", GetVPOVelocity)
	LastVPOVelocityNorm float64
	VPOVelocityNorm     = NewScalarValue("VPOvNorm", "", "Norm Velocity as accumulated internally to VPO", GetVPOVelocityNorm)
	LastBeffperp        *data.Slice
	B_eff_perp          = NewVectorField("B_eff_perp", "T", "Effective field that is cotangent to m", GetBeffperp)
	Lastvdotf           float64
	Vdotf               = NewScalarValue("vdotf", "", "Dot(v,f) before rotation between cotangent spaces", GetVdotF)
	Lastvdotfrot        *data.Slice
	Vdotfrot            = NewScalarField("vdotfrot", "", "Dot(v,f) after rotation between cotangent spaces", GetVdotFrot)
	LastMaxDm           float64
	MaxDm               = NewScalarValue("maxDm", "T", "Maximum magnetization displacement", GetMaxDm)
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
	f_n    *data.Slice // force n
	f_np1  *data.Slice // force n+1
	v      *data.Slice // velocity
	lastDm fifoRingVPO
	lastF  fifoRingVPO
}

// VPOMinimizer step
func (mini *VPOMinimizer) Step() {

	mag_slice := M.Buffer()
	size := mag_slice.Size()
	n_images := mag_slice.N_images

	// Diagnostic block TODO: remove
	log_mode := false
	print_mode := false
	panic_mode := false
	diagnostic_mode := log_mode || print_mode || panic_mode

	var (
		diag_scalar_slice1     *data.Slice
		diag_scalar_slice2     *data.Slice
		diag_vector_slice      *data.Slice
		diag_scalar_zero_slice *data.Slice
		diag_scalar_one_slice  *data.Slice
		diag_vector_zero_slice *data.Slice
	)

	// LastDiagVec1 := cuda.Buffer(3, size, n_images)
	// defer cuda.Recycle(LastDiagVec1)
	// LastDiagVec2 := cuda.Buffer(3, size, n_images)
	// defer cuda.Recycle(LastDiagVec2)
	// LastDiagVec3 := cuda.Buffer(3, size, n_images)
	// defer cuda.Recycle(LastDiagVec3)
	// LastDiagVec4 := cuda.Buffer(3, size, n_images)
	// defer cuda.Recycle(LastDiagVec4)

	if diagnostic_mode {
		diag_scalar_slice1 := cuda.Buffer(1, size, n_images) // Diagnostic scalar slice
		defer cuda.Recycle(diag_scalar_slice1)
		diag_scalar_slice2 := cuda.Buffer(1, size, n_images) // Diagnostic scalar slice
		defer cuda.Recycle(diag_scalar_slice2)
		diag_vector_slice := cuda.Buffer(3, size, n_images) // Diagnostic vector slice
		defer cuda.Recycle(diag_vector_slice)
		diag_scalar_zero_slice := cuda.Buffer(1, size, n_images) // Diagnostic scalar slice with all zeros
		cuda.Constant(diag_scalar_zero_slice, float32(0.0))
		defer cuda.Recycle(diag_scalar_zero_slice)
		diag_scalar_one_slice := cuda.Buffer(1, size, n_images) // Diagnostic scalar slice with all ones
		cuda.Constant(diag_scalar_one_slice, float32(1.0))
		defer cuda.Recycle(diag_scalar_one_slice)
		diag_vector_zero_slice := cuda.Buffer(3, size, n_images) // Diagnostic vector slice with all zeros
		cuda.Constant(diag_vector_zero_slice, float32(0.0))
		defer cuda.Recycle(diag_vector_zero_slice)
	}

	// Initialize force to -\nabla B_eff
	if mini.f_n == nil { // make sure this is not empty upon first usage
		mini.f_n = cuda.Buffer(3, size, n_images)
		// Update and increment index on force
		SetEffectiveField(mini.f_n, &M)                   // f_n arbitrary
		cuda.Orthogonalize(mini.f_n, mini.f_n, mag_slice) // f_n ⟂ m_n
	}

	// Initialize velocity to 0
	if mini.v == nil {
		mini.v = cuda.Buffer(3, size, n_images)
		cuda.Zero3(mini.v)
	}

	maxF := cuda.MaxVecNorm(mini.f_n)
	mini.lastF.Add(maxF)
	SetBeffperp(mini.f_n)
	SetMaxVPOForce(maxF)
	SetVPOVelocity(mini.v)
	SetVPOVelocityNorm(float64(cuda.Dot(mini.v, mini.v)))

	// Diagnostic block TODO: remove
	if diagnostic_mode {
		cuda.DotProduct(diag_scalar_slice1, 1, mini.f_n, mag_slice)
		CompareSlices2Log(diag_scalar_slice1, diag_scalar_zero_slice, log_mode, print_mode, panic_mode, NSteps, ": m_n ⟂ f_n at beginning of step")
	}

	if n_images > 1 && MinimizePath {
		// GNEBForceTransformation(mini.f_n, GNEB_kappa, &M)
		// TODO: Refold the following into the above function
		// Project energy real force orthogonal to path
		// cuda.Orthogonalize(energy_gradient, energy_gradient, mag_slice) //TODO: Wrong order? unnecessary?
		LogSlice(mini.f_n, "B_eff⟂m", NSteps)
		CalculateTangents(&M)
		ProjectTangents(&M)
		// cuda.Global_Orthogonalize(mini.f_n, mini.f_n, M.tangent_buffer_)
		for ind_img := 1; ind_img < n_images-1; ind_img++ {
			cuda.Global_Orthogonalize(mini.f_n.SubSlice(ind_img), mini.f_n.SubSlice(ind_img), M.tangent_buffer_.SubSlice(ind_img))
		}
		LogSlice(M.tangent_buffer_, "τ", NSteps)
		LogSlice(mini.f_n, "(B_eff⟂m)⟂τ", NSteps)
		// Generate elastic forces
		elastic_force_slice := cuda.Buffer(3, mag_slice.Size(), n_images)
		CalculateGeodesicElasticForces(elastic_force_slice, GNEB_kappa, &M)
		LogSlice(elastic_force_slice, "F_s", NSteps)
		cuda.Add(mini.f_n, mini.f_n, elastic_force_slice)
		// Cleanup
		cuda.Recycle(elastic_force_slice)
	}
	SetVPOForceNorm(math.Sqrt(float64(cuda.Dot(mini.f_n, mini.f_n))))

	// Convergence check
	// Store a copy of the magnetization for comparison and convergence check
	// TODO: Possibly remove to save on memory, since we want to use VPO force instead
	m_n := cuda.Buffer(3, size, n_images)
	defer cuda.Recycle(m_n)
	data.Copy(m_n, mag_slice)
	// Diagnostic block TODO: remove
	if diagnostic_mode {
		cuda.DotProduct(diag_scalar_slice1, 1, m_n, mini.v)
		CompareSlices2Log(diag_scalar_slice1, diag_scalar_zero_slice, log_mode, print_mode, panic_mode, NSteps, ": m_n ⟂ v_n at beginning of step")
		cuda.DotProduct(diag_scalar_slice1, 1, m_n, mini.f_n)
		CompareSlices2Log(diag_scalar_slice1, diag_scalar_zero_slice, log_mode, print_mode, panic_mode, NSteps, ": m_n ⟂ f_n at beginning of step")
		cuda.CrossProduct(diag_vector_slice, mini.v, mini.f_n)
		CompareSlices2Log(diag_vector_slice, diag_vector_zero_slice, log_mode, print_mode, panic_mode, NSteps, ": v_n ∥ f_n before m update")
	}

	// Update m_n to m_{n+1}
	vtilde := cuda.Buffer(3, size, n_images)
	defer cuda.Recycle(vtilde)
	cuda.Madd2(vtilde, mini.v, mini.f_n, 1, float32(recip2mVPO*stepsizeVPO))
	cuda.RotateVectors(mag_slice, vtilde, float32(stepsizeVPO))
	// TODO: this rotates each vector accoring to the length of the corresponding v vector - should this be the 'global norm instead?
	// Since the magnetization of each image is changed, the quantities related to
	// the path need to be recalculated
	M.reset_calc_flags()

	// Diagnostic block TODO: remove
	if diagnostic_mode {
		cuda.CrossProduct(diag_vector_slice, mini.v, mini.f_n)
		CompareSlices2Log(diag_vector_slice, diag_vector_zero_slice, log_mode, print_mode, panic_mode, NSteps, ": v_n ∥ f_n")
		cuda.VecNorm(diag_scalar_slice1, mag_slice)
		CompareSlices2Log(diag_scalar_slice1, diag_scalar_one_slice, log_mode, print_mode, panic_mode, NSteps, ": |m|=1")
		cuda.CrossProduct(diag_vector_slice, m_n, mag_slice)
		cuda.DotProduct(diag_scalar_slice1, 1, diag_vector_slice, mini.f_n)
		CompareSlices2Log(diag_scalar_slice1, diag_scalar_zero_slice, log_mode, print_mode, panic_mode, NSteps, ": f_n ∈ span(m_n, m_n+1)")
		cuda.DotProduct(diag_scalar_slice1, 1, diag_vector_slice, mini.v)
		CompareSlices2Log(diag_scalar_slice1, diag_scalar_zero_slice, log_mode, print_mode, panic_mode, NSteps, ": f_n ∈ span(m_n, m_n+1)")
	}

	// Rotate the velocity to the cotangent space of the new magnetization
	// Diagnostic block TODO: remove
	if diagnostic_mode {
		cuda.DotProduct(diag_scalar_slice1, 1, mini.v, mini.v)
	}

	cuda.CotangentSpaceRotation(mini.v, mini.v, mag_slice, m_n)

	// Diagnostic block TODO: remove
	if diagnostic_mode {
		cuda.DotProduct(diag_scalar_slice2, 1, mini.v, mini.v)
		CompareSlices2Log(diag_scalar_slice1, diag_scalar_slice2, log_mode, print_mode, panic_mode, NSteps, ": |v_n|=|Rotate(v_n)|")
		cuda.DotProduct(diag_scalar_slice1, 1, mini.f_n, mini.f_n)
	}

	cuda.CotangentSpaceRotation(mini.f_n, mini.f_n, mag_slice, m_n) // f_n ⟂ m_n+1,

	// Diagnostic block TODO: remove
	if diagnostic_mode {
		cuda.DotProduct(diag_scalar_slice2, 1, mini.f_n, mini.f_n)
		CompareSlices2Log(diag_scalar_slice1, diag_scalar_slice2, log_mode, print_mode, panic_mode, NSteps, ": |f_n|=|Rotate(f_n)|")
		cuda.DotProduct(diag_scalar_slice1, 1, mag_slice, mini.v)
		CompareSlices2Log(diag_scalar_slice1, diag_scalar_zero_slice, log_mode, print_mode, panic_mode, NSteps, ": m_n+1 ⟂ v_n after cotangent space rotation")
		cuda.DotProduct(diag_scalar_slice1, 1, mag_slice, mini.f_n)
		CompareSlices2Log(diag_scalar_slice1, diag_scalar_zero_slice, log_mode, print_mode, panic_mode, NSteps, ": m_n+1 ⟂ f_n after cotangent space rotation")
		cuda.DotProduct(diag_scalar_slice1, 1, diag_vector_slice, mini.f_n)
		CompareSlices2Log(diag_scalar_slice1, diag_scalar_zero_slice, log_mode, print_mode, panic_mode, NSteps, ": f_n ∈ span(m_n, m_n+1) after cotangent space rotation")
		cuda.DotProduct(diag_scalar_slice1, 1, diag_vector_slice, mini.v)
		CompareSlices2Log(diag_scalar_slice1, diag_scalar_zero_slice, log_mode, print_mode, panic_mode, NSteps, ": v_n ∈ span(m_n, m_n+1) after cotangent space rotation")
	}
	// Update v_n to v_{n+1}
	mini.f_np1 = cuda.Buffer(3, size, n_images)
	defer cuda.Recycle(mini.f_np1)
	SetEffectiveField(mini.f_np1, &M)
	cuda.Orthogonalize(mini.f_np1, mini.f_np1, mag_slice) // f_n+1 ⟂ m_n+1
	fac := float32(stepsizeVPO * recip2mVPO)
	cuda.Madd3(mini.v, mini.v, mini.f_n, mini.f_np1, 1, fac, fac)

	// Diagnostic block TODO: remove
	// cuda.DotProduct(diag_scalar_slice1, 1, mag_slice, mini.fnp1)
	// CompareSlices2Log(diag_scalar_slice1, diag_scalar_zero_slice, print_only, NSteps, ": m_n+1 ⟂  f_n+1")

	// Diagnostic block TODO: remove
	if diagnostic_mode {
		cuda.DotProduct(diag_scalar_slice1, 1, mag_slice, mini.v)
		CompareSlices2Log(diag_scalar_slice1, diag_scalar_zero_slice, log_mode, print_mode, panic_mode, NSteps, ": m_n+1 DOT v_n+1 after addition")
	}

	// Factor for projection of velocity on force,
	// sets velocity to 0 if v.f < 0
	vDOTf := cuda.Dot(mini.v, mini.f_np1)
	// Diagnostic block TODO: remove
	SetVdotF(float64(vDOTf))
	if vDOTf <= 0.0 {
		cuda.Zero(mini.v)
	} else {
		cuda.Scale(mini.v, mini.f_np1, vDOTf/cuda.Dot(mini.f_np1, mini.f_np1))
	}
	data.Copy(mini.f_n, mini.f_np1)
	// cuda.Orthogonalize(mini.v, mini.v, mag_slice) //check if necessary
	// Diagnostic block TODO: remove
	if diagnostic_mode {
		cuda.CrossProduct(diag_vector_slice, mini.v, mini.f_np1)
		CompareSlices2Log(diag_vector_slice, diag_vector_zero_slice, log_mode, print_mode, panic_mode, NSteps, ": v_n+1 ∥ f_n+1 after velocity projection")
	}

	// End of this iteration step
	NSteps++

	cuda.Madd2(m_n, mag_slice, m_n, 1., -1.)
	max_dm := cuda.MaxVecNorm(m_n)
	mini.lastDm.Add(max_dm)
	err := mini.lastDm.Max()
	SetMaxDm(max_dm)
	setLastErr(err) // report maxDm to user as LastErr

	if AdvanceTime {
		Time += Dt_si
	}
}

// Free
func (mini *VPOMinimizer) Free() {
	mini.f_n.Free()
	mini.f_np1.Free()
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
		f_n:    nil,
		v:      nil,
		lastDm: FifoRingVPO(DmSamplesVPO),
		lastF:  FifoRingVPO(FSamplesVPO)}
	stepper = &mini

	// TODO: Reconsider which break condition to use
	// break condition: change of magnetization is below a reasonable threshold
	cond := func() bool {
		// return (((mini.lastDm.count < DmSamplesVPO) || (mini.lastF.Max() > StopMaxFVPO)) && NSteps < MaxIterVPO)
		return (((mini.lastDm.count < DmSamplesVPO) || (mini.lastDm.Max() > StopMaxDmVPO)) && NSteps < MaxIterVPO)
	}

	RunWhile(cond)
	pause = true
	SetSolver(prevType)
	FixDt = prevFixDt
	Precess = prevPrecess
	Time = t0

	relaxing = false
}

func SetMaxVPOForce(val float64) { LastMaxVPOForce = val }

func GetMaxVPOForce() float64 { return LastMaxVPOForce }

func SetVPOForceNorm(val float64) { LastVPOForceNorm = val }

func GetVPOForceNorm() float64 { return LastVPOForceNorm }

func SetVPOVelocity(val *data.Slice) {
	if LastVPOVelocity == nil {
		LastVPOVelocity = cuda.Buffer(3, M.Buffer().Size(), M.Buffer().N_images)
	}
	data.Copy(LastVPOVelocity, val)
}

func GetVPOVelocity(dst *data.Slice) {
	if LastVPOVelocity == nil {
		LastVPOVelocity = cuda.Buffer(3, M.Buffer().Size(), M.Buffer().N_images)
	}
	data.Copy(dst, LastVPOVelocity)
}

func SetVPOVelocityNorm(val float64) {
	LastVPOVelocityNorm = val
}

func GetVPOVelocityNorm() float64 {
	return LastVPOVelocityNorm
}

func SetBeffperp(val *data.Slice) {
	if LastBeffperp == nil {
		LastBeffperp = cuda.Buffer(3, M.Buffer().Size(), M.Buffer().N_images)
	}
	data.Copy(LastBeffperp, val)
}

func GetBeffperp(dst *data.Slice) {
	if LastBeffperp == nil {
		LastBeffperp = cuda.Buffer(3, M.Buffer().Size(), M.Buffer().N_images)
	}
	data.Copy(dst, LastBeffperp)
}

func SetVdotF(val float64) {
	Lastvdotf = val
}

func GetVdotF() float64 {
	return Lastvdotf
}

func SetVdotFrot(val *data.Slice) {
	if Lastvdotfrot == nil {
		Lastvdotfrot = cuda.Buffer(1, M.Buffer().Size(), M.Buffer().N_images)
	}
	data.Copy(Lastvdotfrot, val)
}

func GetVdotFrot(dst *data.Slice) {
	if Lastvdotfrot == nil {
		Lastvdotfrot = cuda.Buffer(1, M.Buffer().Size(), M.Buffer().N_images)
	}
	data.Copy(dst, Lastvdotfrot)
}

func SetMaxDm(val float64) { LastMaxDm = val }

func GetMaxDm() float64 { return LastMaxDm }
