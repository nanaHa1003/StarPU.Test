// Standard library headers
#include <cstdio>
#include <cstdlib>
#include <cmath>

// Boost headers
#include <boost/timer.hpp>

// CUDA headers
#include <cuda.h>
#include <cuda_runtime.h>

// StarPU headers
#include <starpu.h>
#include <starpu_cuda.h>

// Other include files
// GPU IPLM headers
#define  PCBAND_GPU
#include <PC_internal.h>
#include <get_blochWaves.h>
#include <getMtx_PCmtx_gpu.cuh>
#include <getMtx_PCmtx2_gpu.cuh>
#include <evSolver_IPLM_gpu.h>
#include <PC_Create.cuh>
#include <PC_Destroy.cuh>
#include <get_PCegval_gpu.h>
#include <set_PCpara.h>
#include <set_LSEVinfo.h>

// CPU IPLM headers
extern "C" {
#include <mkl_dfti.h>
#include <mkl.h>
#include <petscksp.h>
#include <petscvec.h>
#include <petscmat.h>
#include <slepceps.h>
#include <mpi.h>

#include <SCinternal.h>
#include <find_bandgap.h>
#include <generate_mtx.h>
#include <mtx_operation.h>
#include <generate_B.h>
#include <MDgenerate_B.h>
#include <IPLM.h>
#include <IPLM_sing.h>
}

#define PC_MKL_THREADS 1

// Shared parameters for CPU kernel
struct IPLM_CPU_Params {
    int           *mesh;   // int[3]
    int           mesh_k;
    int           num_ew;
    double        *length; // double[3]
    MKL_Complex16 *SC_B;
};

// Shared parameters for GPU kernel
struct IPLM_GPU_Params {
    uint      Nwant;       // mesh_k
    uint      Nwaves;      // Number of total jobs
    realCPU   *blochWaves;
    LSEV_INFO LSEV_info;
    PC_PARA   PC_para;
};

// Common parameters for IPLM
static struct IPLM_CPU_Params cpu_args;
static struct IPLM_GPU_Params gpu_args;

double *eigs;

// Common computation kernel
static void IPLM_Common(int wid, int type)
{
    // Generate arguments & call kernel w.r.t type
    if(type == STARPU_CPU)
    {
        int mesh_k   = cpu_args.mesh_k;
        int k_length = 0.5 / mesh_k;
        double vec_k[3];

        if(wid < mesh_k)
        {
            vec_k[0] = wid * k_length;
            vec_k[1] = 0.0;
            vec_k[2] = 0.0;
        }
        else if(wid < 2 * mesh_k)
        {
            vec_k[0] = 0.5;
            vec_k[1] = (wid - mesh_k) * k_length;
            vec_k[2] = 0;
        }
        else if(wid < 3 * mesh_k)
        {
            vec_k[0] = 0.5;
            vec_k[1] = 0.5;
            vec_k[2] = (wid - 2 * mesh_k) * k_length;
        }
        else {
            vec_k[0] = 0.5 - (wid - 3 * mesh_k) * k_length;
            vec_k[1] = vec_k[0];
            vec_k[2] = vec_k[0];
        }

        if(wid == 0) vec_k[0] = 1e-4;

        PetscScalar *EW = (PetscScalar *) malloc(cpu_args.num_ew * mesh_k * sizeof(PetscScalar));

        // Invoke IPLM CPU kernel
        IPLM(cpu_args.mesh,
             cpu_args.length,
             vec_k,
             cpu_args.SC_B,
             cpu_args.num_ew,
             EW);

        // Get results from EW
        MKL_Complex16 *temp = (MKL_Complex16 *) EW;
        for(int i = 0; i < cpu_args.num_ew; i++)
        {
            eigs[wid * cpu_args.num_ew + i] = 1 / temp[i].real;
        }

        free(EW);
    }
    else if(type == STARPU_CUDA)
    {
        dPC_MTX        PC_mtx;
        LANCZOS_BUFFER lBuffer;
        CG_BUFFER      cgBuffer;
        CULIB_HANDLES  cuHandles;
        realCPU        *PC_egval;
        
        PC_egval = (realCPU *) malloc(gpu_args.Nwant * sizeof(realCPU));

        PC_Create(gpu_args.PC_para.MeshSize,
                  gpu_args.LSEV_info.EV_info.Nstep,
                  &PC_mtx,
                  &lBuffer,
                  &cgBuffer,
                  &cuHandles);

        getMtx_PCmtx2_gpu(&PC_mtx,
                          gpu_args.PC_para,
                          gpu_args.blochWaves + wid * 3);

        evSolver_IPLM_gpu(PC_egval,
                          PC_mtx,
                          gpu_args.LSEV_info,
                          lBuffer,
                          cgBuffer,
                          cuHandles);

        for(int i = 0; i < gpu_args.Nwant; i++)
        {
            eigs[wid * gpu_args.Nwant + i] = PC_egval[i];
        }

        free(PC_egval);
        PC_Destroy(PC_mtx, lBuffer, cgBuffer, cuHandles);
    }

    return;
}

// StarPU CPU function
void IPLM_CPU(void **buffer, void *cl_arg)
{
    int wid = starpu_worker_get_id();
    IPLM_Common(wid, STARPU_CPU);
}

// StarPU GPU function
void IPLM_GPU(void **buffer, void *cl_arg)
{
    int wid = starpu_worker_get_id();
    IPLM_Common(wid, STARPU_CUDA);
}

static int can_execute(unsigned wid, struct starpu_task *task, unsigned nimpl)
{
    const struct cudaDeviceProp *props;
    if(starpu_worker_get_type(wid) == STARPU_CPU_WORKER) return 1;
    if(nimpl == 0) return 1;

    props = starpu_cuda_get_device_properties(wid);
    if(props->major >= 2 || props->minor >= 0) return 1;

    return 0;
}

int main(int argc, char **argv)
{
    // Set up MKL
    mkl_set_dynamic(1);
    mkl_domain_set_num_threads(PC_MKL_THREADS, MKL_BLAS);
    mkl_domain_set_num_threads(PC_MKL_THREADS, MKL_VML);
    mkl_domain_set_num_threads(PC_MKL_THREADS, MKL_FFT);

    // Initialize Slepc
    SlepcInitialize(&argc, &argv, NULL, NULL);

    // Prepare to generate SC matrix
    int cpu_mesh[3]  = { atoi(argv[1]), atoi(argv[2]), atoi(argv[3]) };
    int mesh_k       = 10;
    int num_ew       = atoi(argv[4]);
    double length[3] = { 1.0 / cpu_mesh[0], 1.0 / cpu_mesh[1], 1.0 / cpu_mesh[2] };
    double aver;
    MKL_Complex16 *SC_B;

    MDgenerate_B(argc, argv, &SC_B, &aver);

    // Set CPU IPLM parameters
    cpu_args.mesh   = cpu_mesh;
    cpu_args.mesh_k = mesh_k;
    cpu_args.num_ew = num_ew;
    cpu_args.length = length;
    cpu_args.SC_B   = SC_B;

    // Set up GPU IPLM
    // Nwaves = meah_k or num_ew
    set_PCpara  (&(gpu_args.PC_para)  , "PC_para.txt"  );
    set_LSEVinfo(&(gpu_args.LSEV_info), "LSEV_info.txt");

    get_blochWaves(gpu_args.PC_para.lattice,
                   gpu_args.PC_para.ltcCnst,
                   gpu_args.PC_para.Npts,
                   &(gpu_args.blochWaves),
                   &(gpu_args.Nwaves));    

    // Set up StarPU codelet
    struct starpu_codelet cl;
    starpu_codelet_init(&cl);
    cl.where       = STARPU_CPU | STARPU_CUDA;
    cl.can_execute = can_execute;
    cl.cpu_func    = IPLM_CPU; // Original : cl.cpu_funcs  = { IPLM_CPU, NULL };
    cl.cuda_func   = IPLM_GPU; //          : cl.cuda_funcs = { IPLM_GPU, NULL };
    cl.nbuffers    = 0;

    // Set up common parameters 
    eigs = (double *) malloc(num_ew * mesh_k * 4 * sizeof(double));

    // Start StarPU
    starpu_init(NULL);

    // Start timer
    boost::timer clock;

    // Submit tasks
    int tasks = 4 * mesh_k;
    for(int id = 0; id < tasks; id++)
    {
        struct starpu_task *task = starpu_task_create();
        task->cl = &cl;
        starpu_task_submit(task); printf("Task ID %d submitted\n", id);
    }

    starpu_task_wait_for_all();

    // Get time
    double calcTime = clock.elapsed();

    // Terminate StarPU
    starpu_shutdown();

    // Collect results
    FILE *Output = fopen("PC.out", "w");
    for(int i = 0; i < tasks; i++)
    {
        fprintf(Output, "%lf\n", eigs[i]);
    }

    fclose(Output);
    free(eigs);

    printf("Calculation time : %lf\n", calcTime);

    return 0;
}
