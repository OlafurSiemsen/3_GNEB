package engine

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
	"gonum.org/v1/gonum/lapack/gonum"
)

// TODO-olafur: Decide which methods and fields are private and public
// Struct representing a run of the Lanczos method to decompose the Hessian into
// matrices T, V s.t. T = V*@Hess(m)@V.
// V is an orthogonal 2*N_cells x N_modes matrix
// T is a tridiagonal matrix
// Note that V and T are in a tangent space basis, using basis vectors in mag_slice.tangent_basis
// See https://en.wikipedia.org/wiki/Lanczos_algorithm
type Lanczos_run struct {
	N_modes_max              int // Maximum number of modes to expand to
	reorthogonalization_freq int
	tol                      float64       // Tolerance where algorithm is terminated
	initial_lanczos_vecs     []*data.Slice // Optional slice of data.slice pointers for initial guesses for Lanczos vectors, in Euclidean space

	magnetization_ptr *magnetization           // Pointer to the magnetization
	file_map          map[string]*os.File      // Map of files for output
	writer_map        map[string]*bufio.Writer // Map of writers for output
	OutputDir         string                   // Directory for output

	Lanczos_vecs []*data.Slice // Slice of data.slice pointers that store the Lanczos vectors
	Alphas       []float64     // Diagonal of tridiagonal matrix
	Betas        []float64     // Off-diagonal of tridiagonal matrix
	Eigvals      []float64     // Eigenvalues of tridiagonal matrix (and Hessian)

	n_iter int         // Current iteration
	omega  *data.Slice // Variable used during iteration
}

func (lanczos_run *Lanczos_run) iterate() {
	lanczos_run.Lanczos_vecs[lanczos_run.n_iter] = cuda.NewSlice(2, lanczos_run.magnetization_ptr.buffer_.Size())

	// Are these allocations slow to do each iteration?
	omega_prime := cuda.Buffer(2, lanczos_run.magnetization_ptr.Buffer().Size())

	omega_prime_3N := cuda.Buffer(3, lanczos_run.magnetization_ptr.Buffer().Size())
	lanczos_vec_3N := cuda.Buffer(3, lanczos_run.magnetization_ptr.Buffer().Size())

	var (
		alpha  float32
		beta   float32
		t_norm float32
	)
	// Priming step
	if lanczos_run.n_iter == 0 {
		if len(lanczos_run.initial_lanczos_vecs) != 0 {
			lanczos_run.magnetization_ptr.
				tangent_space_basis.Project_into(lanczos_run.Lanczos_vecs[0], lanczos_run.initial_lanczos_vecs[0])
		} else {
			lanczos_run.Lanczos_vecs[0].Randomize(1)
		}
		t_norm = float32(math.Sqrt(float64(cuda.Dot(lanczos_run.Lanczos_vecs[0], lanczos_run.Lanczos_vecs[0]))))
		cuda.Scale(lanczos_run.Lanczos_vecs[0], lanczos_run.Lanczos_vecs[0], 1/t_norm)

		lanczos_run.magnetization_ptr.tangent_space_basis.Project_out_of(lanczos_vec_3N, lanczos_run.Lanczos_vecs[0])
		Hessian_findiff(omega_prime_3N, lanczos_run.magnetization_ptr, lanczos_vec_3N, 1e-3) // TODO-olafur: Find good displacement value
		lanczos_run.magnetization_ptr.tangent_space_basis.Project_into(omega_prime, omega_prime_3N)

		alpha = cuda.Dot(omega_prime, lanczos_run.Lanczos_vecs[0])
		lanczos_run.Alphas[0] = float64(alpha)
		cuda.Madd2(lanczos_run.omega, omega_prime, lanczos_run.Lanczos_vecs[0], 1, -alpha)
	} else {
		if lanczos_run.reorthogonalization_freq != 0 && lanczos_run.n_iter%lanczos_run.reorthogonalization_freq == 0 {
			for ind_mode := 0; ind_mode < lanczos_run.n_iter; ind_mode++ {
				cuda.Global_Orthogonalize(lanczos_run.omega, lanczos_run.omega, lanczos_run.Lanczos_vecs[ind_mode])
			}
		}
		if lanczos_run.n_iter%10 == 0 {
			t_max := 0.0
			for ind_mode := 0; ind_mode < lanczos_run.n_iter; ind_mode++ {
				t_dot := float64(cuda.Dot(lanczos_run.omega, lanczos_run.Lanczos_vecs[ind_mode]))
				t_max = math.Max(t_max, t_dot)
			}
			LogOut(lanczos_run.n_iter, " maxdot: ", t_max)
		}
		beta = float32(math.Sqrt(float64(cuda.Dot(lanczos_run.omega, lanczos_run.omega)))) // Thanks Go
		if beta != 0.0 {
			cuda.Scale(
				lanczos_run.Lanczos_vecs[lanczos_run.n_iter],
				lanczos_run.omega,
				1/beta,
			)
		} else {
			LogOut("Warning! Beta becomes zero at iteration ", lanczos_run.n_iter)
			lanczos_run.Lanczos_vecs[lanczos_run.n_iter].Randomize(10)
			for ind_mode := 0; ind_mode < lanczos_run.n_iter; ind_mode++ {
				cuda.Global_Orthogonalize(
					lanczos_run.Lanczos_vecs[lanczos_run.n_iter],
					lanczos_run.Lanczos_vecs[lanczos_run.n_iter],
					lanczos_run.Lanczos_vecs[ind_mode],
				)
			}

			t_norm = float32(math.Sqrt(float64(cuda.Dot(lanczos_run.Lanczos_vecs[lanczos_run.n_iter], lanczos_run.Lanczos_vecs[lanczos_run.n_iter]))))
			cuda.Scale(lanczos_run.Lanczos_vecs[lanczos_run.n_iter], lanczos_run.Lanczos_vecs[lanczos_run.n_iter], 1/t_norm)
		}

		lanczos_run.magnetization_ptr.tangent_space_basis.Project_out_of(lanczos_vec_3N, lanczos_run.Lanczos_vecs[lanczos_run.n_iter])
		Hessian_findiff(omega_prime_3N, lanczos_run.magnetization_ptr, lanczos_vec_3N, 1e-3) // TODO-olafur: Determine good displacement parameter
		lanczos_run.magnetization_ptr.tangent_space_basis.Project_into(omega_prime, omega_prime_3N)

		cuda.Madd2(omega_prime,
			omega_prime,
			lanczos_run.Lanczos_vecs[lanczos_run.n_iter-1],
			1.0,
			-beta,
		)
		alpha = cuda.Dot(omega_prime, lanczos_run.Lanczos_vecs[lanczos_run.n_iter])

		cuda.Madd2(lanczos_run.omega, omega_prime, lanczos_run.Lanczos_vecs[lanczos_run.n_iter], 1.0, -alpha)

		lanczos_run.Alphas[lanczos_run.n_iter] = float64(alpha)
		lanczos_run.Betas[lanczos_run.n_iter-1] = float64(beta)
	}

	lanczos_run.n_iter = lanczos_run.n_iter + 1
	cuda.Recycle(omega_prime)
	cuda.Recycle(omega_prime_3N)
	cuda.Recycle(lanczos_vec_3N)

}

func (lanczos_run *Lanczos_run) Run() {
	if lanczos_run.OutputDir == "" {
		t_time := time.Now()
		lanczos_run.OutputDir = fmt.Sprintf("Lanczos%02v%02v%02v.out/", t_time.Hour(), t_time.Minute(), t_time.Second())
	}
	lanczos_run.OutputDir = OD() + lanczos_run.OutputDir
	err := os.Mkdir(lanczos_run.OutputDir, 0755)
	util.FatalErr(err)

	fname_slice := []string{"EigVal.csv"}
	for _, it_fname := range fname_slice {
		t_file, err := os.Create(lanczos_run.OutputDir + it_fname)
		util.FatalErr(err)
		lanczos_run.file_map[it_fname] = t_file
		t_writer := bufio.NewWriter(t_file)
		lanczos_run.writer_map[it_fname] = t_writer
	}
	EigVal_header := GenerateCSVHeader([]string{"eval"}, lanczos_run.N_modes_max, true)
	lanczos_run.writer_map["EigVal.csv"].WriteString(EigVal_header)
	t_eigs := make([]float64, lanczos_run.N_modes_max)

	t_alphas := make([]float64, len(lanczos_run.Alphas))
	t_betas := make([]float64, len(lanczos_run.Betas))

	var t_csv_line string

	for lanczos_run.n_iter < lanczos_run.N_modes_max {
		lanczos_run.iterate()
		if lanczos_run.n_iter == 0 {
			t_eigs[0] = lanczos_run.Alphas[0]
		} else {
			_ = copy(t_alphas, lanczos_run.Alphas)
			_ = copy(t_betas, lanczos_run.Betas)
			t_eigs = TridiagEigval(
				t_alphas[0:lanczos_run.n_iter],
				t_betas[0:lanczos_run.n_iter-1],
			)
		}
		tesla_joule := cellVolume() * Msat.Average()
		for ind := 0; ind < len(t_eigs); ind++ {
			t_eigs[ind] = tesla_joule * t_eigs[ind]
		}
		t_csv_line = fmt.Sprintf("%v,", lanczos_run.n_iter) + go_slice_to_csv_line(t_eigs)
		t_csv_line = strings.ReplaceAll(t_csv_line, " ", ",")
		t_csv_line = strings.ReplaceAll(t_csv_line, "[", "")
		t_csv_line = strings.ReplaceAll(t_csv_line, "]", "")
		lanczos_run.writer_map["EigVal.csv"].WriteString(t_csv_line)
		lanczos_run.Eigvals = t_eigs
	}
	SnapshotAs(&M, lanczos_run.OutputDir+"mag.png")
	SaveAs(&M, lanczos_run.OutputDir+"mag")

	for it_fname, _ := range lanczos_run.writer_map {
		lanczos_run.writer_map[it_fname].Flush()
		lanczos_run.file_map[it_fname].Close()
	}
}

func (lanczos_run *Lanczos_run) SetOutputName(outputdir string) {
	strings.CutSuffix(outputdir, "/")
	lanczos_run.OutputDir = outputdir + "/"
}

func New_Lanczos_Run(mag_slice *magnetization, N_modes_max int, reorthogonalization_freq int, tol float64, initial_lanczos_vecs ...*data.Slice) *Lanczos_run {
	if len(initial_lanczos_vecs) != 0 && len(initial_lanczos_vecs) != 1 {
		panic("Only 0 or 1 initial Lanczos vectors currently supported.")
	}

	o_lanczos_run := new(Lanczos_run)

	o_lanczos_run.N_modes_max = N_modes_max
	// Cannot have more modes than degrees of freedom
	o_lanczos_run.N_modes_max = int(math.Min(
		float64(N_modes_max),
		2.0*float64(prod(mag_slice.buffer_.Size())),
	))
	o_lanczos_run.tol = tol
	o_lanczos_run.reorthogonalization_freq = reorthogonalization_freq
	o_lanczos_run.initial_lanczos_vecs = initial_lanczos_vecs

	o_lanczos_run.Lanczos_vecs = make([]*data.Slice, o_lanczos_run.N_modes_max)
	o_lanczos_run.omega = cuda.NewSlice(2, mag_slice.Buffer().Size())
	o_lanczos_run.Alphas = make([]float64, o_lanczos_run.N_modes_max)
	o_lanczos_run.Betas = make([]float64, o_lanczos_run.N_modes_max-1)
	o_lanczos_run.Eigvals = make([]float64, o_lanczos_run.N_modes_max)
	o_lanczos_run.magnetization_ptr = mag_slice
	o_lanczos_run.n_iter = 0

	o_lanczos_run.file_map = make(map[string]*os.File, 0)
	o_lanczos_run.writer_map = make(map[string]*bufio.Writer, 0)

	return o_lanczos_run
}

func (lanczos_run Lanczos_run) SaveOutput(dir ...string) {
}

// TODO-olafur: Move somewhere else. Perhaps create new .go for numerical methods
// TODO-olafur: Rewrite for regional Msat
// Estimates the action of the Hessian along some magnetic displacement v using
// finite difference of B_eff (-∇E)
// Note: The output is in Tesla, in order to scale to Joule, it must be scaled
// by a factor tesla_joule := float32(cellVolume() * Msat.Average())
func Hessian_findiff(
	Hv *data.Slice,
	mag *magnetization,
	v_displacement *data.Slice,
	stepsize float32,
) {
	Beff_nm1 := cuda.Buffer(3, mag.Buffer().Size())
	Beff_np1 := cuda.Buffer(3, mag.Buffer().Size())
	t_m := cuda.Buffer(3, Mesh().Size())

	mag_slice := mag.Buffer()
	data.Copy(t_m, mag_slice)

	cuda.RotateVectors(mag_slice, v_displacement, -stepsize)
	SetEffectiveField(Beff_nm1, mag)
	cuda.CotangentSpaceRotation(Beff_nm1, Beff_nm1, t_m, mag_slice)
	data.Copy(mag_slice, t_m)

	cuda.RotateVectors(mag_slice, v_displacement, stepsize)
	SetEffectiveField(Beff_np1, mag)
	cuda.CotangentSpaceRotation(Beff_np1, Beff_np1, t_m, mag_slice)

	coeff := 1 / (2 * stepsize)
	cuda.Madd2(Hv, Beff_nm1, Beff_np1, -(-coeff), -(coeff)) // Extra minus sign, since B_eff = -∇E
	// tesla_joule := float32(cellVolume() * Msat.Average())
	// cuda.Scale(Hv, Hv, tesla_joule) // convert to normal units

	data.Copy(mag_slice, t_m)

	cuda.Recycle(Beff_nm1)
	cuda.Recycle(Beff_np1)
	cuda.Recycle(t_m)
}

// TODO-olafur: Change this function to not require copying the alphas and betas beforehand
func TridiagEigval(alphas, betas []float64) []float64 {
	impl := gonum.Implementation{}
	impl.Dsterf(len(alphas), alphas, betas)
	return alphas
}
