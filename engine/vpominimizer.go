package engine

// TODO: Introductory line what this code does and reference to paper where this was introduced.
//   - see minimizer.go for reference

import (
	"bufio"
	"fmt"
	"math"
	"os"
	"strings"
	"time"

	"github.com/mumax/3/cuda"
	"github.com/mumax/3/data"
	"github.com/mumax/3/util"
)

// Some generic parameter values which can be overwritten by init()
// TODO: explain what they are.
// TODO: Check if they are actually good.
var (
	massVPO     = 1e-0 // TODO: massVPO of the ??? What units? What is reasonable here?
	recip2mVPO  = 0.5 / massVPO
	stepsizeVPO = 0.05 // dt effective time step
	//MaxForce          = 100.0	// TODO: maximal allowed force???
	DmSamplesVPO      int       = 10             // number of dm to keep for convergence check
	StopMaxDmVPO      float64   = 1e-6           // stop minimizer if sampled dm is smaller than this
	FSamplesVPO       int       = 1              // Number of max F to keep for convergence check
	StopMaxFVPO       float64   = 1e-6           // stop minimizer if sampled maximum force is smaller than this
	MaxIterVPO        int       = 200000         // Maximum number of iterations
	MinimizePath      bool      = true           // True if the path should be minimized, false if each image is minimized seperately
	FixEndImages      bool      = true           // True if the first and last images are not to be modified
	ClimbingImage     bool      = true           // True if one image is to climb towards a saddle point
	AdvanceTime       bool      = false          // Whether to increment time globally - time is reset at the end
	GNEB_kappa        []float32 = []float32{2.0} // Spring constants for the inter-image elastic forces
	NPointsEnergyCHIP int       = 0

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
	f_n         *data.Slice // force n
	f_np1       *data.Slice // force n+1
	v           *data.Slice // velocity
	vtilde      *data.Slice // Modified velocity, vtilde = v + f_n * recip2mVPO*stepsizeVPO
	lastDm      fifoRingVPO
	lastF       fifoRingVPO
	iter        int // current number of iterations
	file_map    map[string]*os.File
	writer_map  map[string]*bufio.Writer
	energy_chip *CHIP // Cubic hermite interpolation polynomial for the energy
	OutputDir   string
}

// VPOMinimizer step
func (mini *VPOMinimizer) Step() {

	mag_slice := M.Buffer()
	size := mag_slice.Size()
	n_images := mag_slice.N_images

	// Initialize force to -\nabla B_eff
	if mini.iter == 0 { // make sure this is not empty upon first usage
		// mini.f_n = cuda.Buffer(3, size, n_images)
		// Update and increment index on force
		SetEffectiveField(mini.f_n, &M)                   // f_n arbitrary
		cuda.Orthogonalize(mini.f_n, mini.f_n, mag_slice) // f_n ⟂ m_n

		if n_images > 1 && MinimizePath {
			GNEBForceTransformation(mini.f_n, GNEB_kappa, &M)
		}
	}

	// Initialize velocity to 0
	if mini.v == nil {
		// mini.v = cuda.Buffer(3, size, n_images)
		cuda.Zero3(mini.v)
	}
	// TODO-olafur: Rework, preferrably should use the side standard side process async output
	if mini.iter%1000 == 0 {
		CalculateTangents(&M)
		ProjectTangents(&M)
		CalculateTotalImageEnergies(&M)
		// SetEffectiveField(diag_vector_slice, &M)
		gradDOTtau := make([]float64, n_images)
		for ind_img := 0; ind_img < n_images; ind_img++ {
			gradDOTtau[ind_img] = cellVolume() * Msat.Average() * (-float64(cuda.Dot(mini.f_n.SubSlice(ind_img), M.tangent_buffer_.SubSlice(ind_img))))
		}

		t_xs, t_ys, t_chip := Interpolate_energy_path(&M, cellVolume(), Msat.Average(), NPointsEnergyCHIP)
		mini.energy_chip = t_chip
		// Janky output
		csv_lines := fmt.Sprintf("%v,", mini.iter) + go_slice_to_csv_line(append(t_xs, t_ys...))
		csv_lines = strings.ReplaceAll(csv_lines, " ", ",")
		csv_lines = strings.ReplaceAll(csv_lines, "[", "")
		csv_lines = strings.ReplaceAll(csv_lines, "]", "")
		mini.writer_map["EnergyPathCHIP.csv"].WriteString(csv_lines)
	}
	if mini.iter%1000 == 0 && mini.iter != 0 {
		SnapshotAs(&M, OD()+mini.OutputDir+fmt.Sprintf("mag_iter%06d.png", mini.iter))
	}

	maxF := cuda.MaxVecNorm(mini.f_n)
	mini.lastF.Add(maxF)
	SetBeffperp(mini.f_n)
	SetMaxVPOForce(maxF)
	SetVPOVelocity(mini.v)
	SetVPOVelocityNorm(float64(cuda.Dot(mini.v, mini.v)))

	SetVPOForceNorm(math.Sqrt(float64(cuda.Dot(mini.f_n, mini.f_n))))

	// Convergence check
	// Store a copy of the magnetization for comparison and convergence check
	// TODO: Possibly remove to save on memory, since we want to use VPO force instead
	m_n := cuda.Buffer(3, size, n_images)
	defer cuda.Recycle(m_n)
	data.Copy(m_n, mag_slice)

	// Update m_n to m_{n+1}
	// vtilde := cuda.Buffer(3, size, n_images)
	cuda.Madd2(mini.vtilde, mini.v, mini.f_n, 1, float32(recip2mVPO*stepsizeVPO))
	cuda.RotateVectors(mag_slice, mini.vtilde, float32(stepsizeVPO))
	// Since the magnetization of each image is changed, the quantities related to
	// the path need to be recalculated
	M.Reset_calc_flags()

	// Rotate the velocity to the cotangent space of the new magnetization

	cuda.CotangentSpaceRotation(mini.v, mini.v, mag_slice, m_n)

	cuda.CotangentSpaceRotation(mini.f_n, mini.f_n, mag_slice, m_n) // f_n ⟂ m_n+1,

	// Update v_n to v_{n+1}
	// mini.f_np1 = cuda.Buffer(3, size, n_images)
	SetEffectiveField(mini.f_np1, &M)
	cuda.Orthogonalize(mini.f_np1, mini.f_np1, mag_slice) // f_n+1 ⟂ m_n+1

	if n_images > 1 && MinimizePath {
		GNEBForceTransformation(mini.f_np1, GNEB_kappa, &M)
	}

	fac := float32(stepsizeVPO * recip2mVPO)
	cuda.Madd3(mini.v, mini.v, mini.f_n, mini.f_np1, 1, fac, fac)

	// Factor for projection of velocity on force,
	// sets velocity to 0 if v.f < 0
	vDOTf := cuda.Dot(mini.v, mini.f_np1)
	SetVdotF(float64(vDOTf))
	if vDOTf <= 0.0 {
		cuda.Zero(mini.v)
	} else {
		cuda.Scale(mini.v, mini.f_np1, vDOTf/cuda.Dot(mini.f_np1, mini.f_np1))
	}
	data.Copy(mini.f_n, mini.f_np1)
	// cuda.Orthogonalize(mini.v, mini.v, mag_slice) //check if necessary
	// Diagnostic block TODO: remove

	// End of this iteration step
	mini.iter++

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
	// mini.f_n.Free()
	// mini.f_np1.Free()
	// mini.v.Free()
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
	defer func() {
		SetSolver(prevType)
		FixDt = prevFixDt
		Precess = prevPrecess
		Time = t0

		relaxing = false
	}()

	// disable precession for torque calculation
	Precess = false

	// remove previous stepper
	if stepper != nil {
		stepper.Free()
	}

	// set stepper to the VPOMinimizer
	mini := VPOMinimizer{
		f_n:        cuda.Buffer((&M).NComp(), (&M).buffer_.Size(), (&M).GetNImages()),
		f_np1:      cuda.Buffer((&M).NComp(), (&M).buffer_.Size(), (&M).GetNImages()),
		v:          cuda.Buffer((&M).NComp(), (&M).buffer_.Size(), (&M).GetNImages()),
		vtilde:     cuda.Buffer((&M).NComp(), (&M).buffer_.Size(), (&M).GetNImages()),
		lastDm:     FifoRingVPO(DmSamplesVPO),
		lastF:      FifoRingVPO(FSamplesVPO),
		iter:       0,
		file_map:   make(map[string]*os.File, 0),
		writer_map: make(map[string]*bufio.Writer, 0)}
	fname_slice := []string{"EnergyPathCHIP.csv"}
	t_time := time.Now()
	mini.OutputDir = fmt.Sprintf("VPO%02v%02v%02v.out/", t_time.Hour(), t_time.Minute(), t_time.Second())
	for _, it_fname := range fname_slice {
		// TODO-olafur: Abstract and funcitonalize
		err := os.Mkdir(OD()+mini.OutputDir, 0755)
		util.FatalErr(err)
		t_file, err := os.Create(OD() + mini.OutputDir + it_fname)
		util.FatalErr(err)
		mini.file_map[it_fname] = t_file
		t_writer := bufio.NewWriter(t_file)
		mini.writer_map[it_fname] = t_writer
	}
	EnergyPathCHIP_header := GenerateCSVHeader([]string{"x", "y"}, NPointsEnergyCHIP, true)
	mini.writer_map["EnergyPathCHIP.csv"].WriteString(EnergyPathCHIP_header)
	stepper = &mini
	FixDt = 1

	// TODO-olafur: Reconsider which break condition to use
	// break condition: change of magnetization is below a reasonable threshold
	cond := func() bool {
		// return (((mini.lastDm.count < DmSamplesVPO) || (mini.lastF.Max() > StopMaxFVPO)) && NSteps < MaxIterVPO)
		return (((mini.lastDm.count < DmSamplesVPO) || (mini.lastDm.Max() > StopMaxDmVPO)) && mini.iter < MaxIterVPO)
	}

	SaveAs(&M, mini.OutputDir+"M_initial")
	SnapshotAs(&M, OD()+mini.OutputDir+"M_initial.png")
	RunWhile(cond)
	if mini.iter == MaxIterVPO {
		LogOut("Warning! Maximum iterations reached in VPO")
	}
	SaveAs(&M, mini.OutputDir+"M_final")
	SnapshotAs(&M, mini.OutputDir+"M_final.png")
	pause = true
	for it_fname, _ := range mini.writer_map {
		mini.writer_map[it_fname].Flush()
		mini.file_map[it_fname].Close()
	}

	// Cleanup
	if LastVPOVelocity != nil {
		cuda.Recycle(LastVPOVelocity)
		LastVPOVelocity = nil
	}
	if LastBeffperp != nil {
		cuda.Recycle(LastBeffperp)
		LastBeffperp = nil
	}
	if Lastvdotfrot != nil {
		cuda.Recycle(Lastvdotfrot)
		Lastvdotfrot = nil
	}
	cuda.Recycle(mini.f_n)
	cuda.Recycle(mini.f_np1)
	cuda.Recycle(mini.v)
	cuda.Recycle(mini.vtilde)

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
