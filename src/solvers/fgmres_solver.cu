// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

//#include <amgx_cublas.h>
#include <blas.h>
#include <chrono>
#include <cutil.h>
#include <norm.h>
#include <util.h>
#include <cusp/blas.h>
// #include "solvers/fgmres_utils.h"
#include <solvers/fgmres_solver.h>

//TODO remove synchronization from this module by moving host operations to the device


namespace amgx
{

template<class T_Config>
void
FGMRES_Solver<T_Config>::printSolverParameters() const
{
    std::cout << "gmres_n_restart=" << this->m_restart << std::endl;

    if (use_preconditioner)
    {
        std::cout << "preconditioner: " << this->m_preconditioner->getName() << " with scope name: " << this->m_preconditioner->getScope() << std::endl;
    }
}

//init the frist vector
template <class TConfig>
void KrylovSubspaceBuffer<TConfig>::setup(int N_dim, int restart_iters)
{
    // printf("\n Setting up Krylov Subspace Buffer\n");
    new_basis = new thrust::device_vector<float>(N_dim);
    V_matrix = new CudaMatrix(N_dim, restart_iters);
    Z_matrix = new CudaMatrix(N_dim, restart_iters);
    H_Matrix = new CudaMatrix(restart_iters+1, restart_iters);

    this->N_dim = N_dim;
    this->restart_iters = restart_iters;
    this->iteration = -1;
}


template< class T_Config>
FGMRES_Solver<T_Config>::FGMRES_Solver( AMG_Config &cfg, const std::string &cfg_scope ) :
    Solver<T_Config>(cfg, cfg_scope), m_preconditioner(0)

{
    // printf("\nMaking New GMRES Solver\n");
    std::string solverName, new_scope, tmp_scope;
    cfg.getParameter<std::string>("preconditioner", solverName, cfg_scope, new_scope);

    if (solverName.compare("NOSOLVER") == 0)
    {
        use_preconditioner = false;
        m_preconditioner = NULL;
    }
    else
    {
        use_preconditioner = true;
        m_preconditioner = SolverFactory<T_Config>::allocate( cfg, cfg_scope, "preconditioner" );
    }

    m_restart = cfg.AMG_Config::template getParameter<int>("gmres_n_restart", cfg_scope);

    e_vect = new thrust::device_vector<float>(m_restart+1);

    // Init least squares solver
    cublasHandle_t handle_cublas = Cublas::get_handle();
    cusolverDnHandle_t hanlde_cusolver = nullptr;
    cusolverDnCreate(&hanlde_cusolver);
    lstsq_solver = new LeastSquaresSolver(hanlde_cusolver, handle_cublas, m_restart+1, m_restart);

    // Gram schmidt solver
    auto gs_params = cfg.getParameter<std::string>("gram_schmidt_options", cfg_scope);
    GS_solver = new GramSchmidtSolver(m_restart+2, gs_params);

    CUDA_CHECK(cudaMalloc((void**)&d_norm_tmp, sizeof(float)));
}

template<class T_Config>
FGMRES_Solver<T_Config>::~FGMRES_Solver()
{
    if (use_preconditioner) { delete m_preconditioner; }
    delete lstsq_solver;
    delete e_vect;
    delete v_m_vvect;
    delete p_inv_v_m;
    cudaFree(d_norm_tmp);
    delete GS_solver;
}


template<class T_Config>
void FGMRES_Solver<T_Config>::solver_setup(bool reuse_matrix_structure)
{   /* This runs every time the solver is called */
    if (use_preconditioner)
    {
        m_preconditioner->setup( *this->m_A, reuse_matrix_structure );
    }

    ViewType oldView = this->m_A->currentView();
    this->m_A->setViewExterior();

    int dim = this->m_A->get_num_cols();
    this->m_A->setView(oldView);

    // Setup subspace if it isn't already setup
    if (m_dim != dim)
    {
        if (m_dim != 0)
        {
            // Need to delete old objects
            subspace = KrylovSubspaceBuffer<T_Config>();
            if (use_preconditioner){delete v_m_vvect; delete p_inv_v_m;}
        }
        m_dim = dim;
        subspace.setup(this->m_A->get_num_cols(), this->m_restart);

        // Initialize temp vvects
        if (use_preconditioner)
        {
            v_m_vvect = new VVector(m_dim);
            p_inv_v_m = new VVector(m_dim);
        }
    }
    // TODO: Zero matrices?
}

//Run preconditioned GMRES
template<class T_Config>
AMGX_STATUS
FGMRES_Solver<T_Config>::solve_iteration( VVector &b, VVector &x, bool xIsZero )
{
    /*using Clock = std::chrono::steady_clock;
    using TimePoint = std::chrono::time_point<Clock>;
    using Duration = std::chrono::duration<double, std::milli>; // milliseconds
    TimePoint start, end;
    Duration duration;


    cudaDeviceSynchronize();
    start = Clock::now();
    cudaDeviceSynchronize();
    end = Clock::now();
    duration = end - start;
    std::cout << "main time: " << duration.count() << " ms\n";*/

    // AMGX_STATUS conv_stat = AMGX_ST_CONVERGED;
    cublasHandle_t cublas_handle = Cublas::get_handle();

    int m = this->m_curr_iter % m_restart;  //Iterations between restarts, do we still need restart?

    auto& new_basis = *subspace.new_basis;
    float* new_basis_ptr = thrust::raw_pointer_cast(new_basis.data());
    float* e_vect_ptr = thrust::raw_pointer_cast(e_vect->data());
    auto* x_ptr = (float*)thrust::raw_pointer_cast(x.data());
    auto& V = *subspace.V_matrix;
    auto& Z = *subspace.Z_matrix;
    auto& H = *subspace.H_Matrix;

    // A matrix
    auto& A = dynamic_cast<Matrix<T_Config>&>(*this->m_A);

    if (m == 0){
        //initialize gmres
        // A never ever changes, but set once per iteration anyway.
        sp_axpy.set_matrix(A);

        subspace.iteration = 0;
        // compute initial residual r0 = b - Ax
        thrust::copy(b.begin(), b.end(), new_basis.begin());
        sp_axpy.axpy(x_ptr, new_basis_ptr, -1.0f, 1.0f);

        // normalize initial residual. d_norm_tmp = beta = ||r0||
        compute_L2_norm(new_basis, d_norm_tmp);
        scale_vector(new_basis, d_norm_tmp);
        V.setColumn(0, new_basis_ptr);

        // e = [beta, 0, 0, ...]
        thrust::fill(e_vect->begin(), e_vect->end(), 0);
        cudaMemcpy(e_vect_ptr, d_norm_tmp, sizeof(float), cudaMemcpyDeviceToDevice);
    }

    subspace.iteration = m;

    // Compute current vector from V[m].
    if (use_preconditioner)
{       // Run one iteration of preconditioner with zero initial guess and v_m as rhs, i.e. solve Az_m=v_m
        // Copy cuda array into VVector
        cudaMemcpy(v_m_vvect->raw(), new_basis_ptr, m_dim * sizeof(float), cudaMemcpyDeviceToDevice);
        thrust::fill((*p_inv_v_m).begin(), (*p_inv_v_m).end(), 0);
        // Solve for p_inv_v_m = P^-1 * v_m
        m_preconditioner->solve( *v_m_vvect, *p_inv_v_m, true ); //TODO: check if using zero as initial solution when solving for residual inside subspace is correct

        Z.setColumn(m, (float*) p_inv_v_m->raw());

    } else
    {   // Do nothing to get V[m]
        Z.setColumn(m, new_basis_ptr);
    }

    // new_basis is now V_m
    //obtain v_m+1 := A*z_m
    sp_axpy.axpy(Z.get_col_ptr(m), new_basis_ptr, 1.0f, 0.0f);

    // Compute entry in Hessenberg matrix and new residual vector.
    // gram_schmidt_step(V.get_col_ptr(0), m_dim, m, H.getDevicePointer(), m_restart+1, new_basis_ptr);
    GS_solver->gram_schmidt(V.get_col_ptr(0), m_dim, m, H.getDevicePointer(), m_restart+1, new_basis_ptr);

    //H(m+1,m) = || v_m+1 ||
    compute_L2_norm(new_basis, d_norm_tmp);
    H.set_element_device(m+1, m, d_norm_tmp);
    // //normalize v_m+1
    scale_vector(new_basis, d_norm_tmp);

    if (m < m_restart-1)
    {
        V.setColumn(m+1, new_basis_ptr);
    }

    // // If reached restart limit or last iteration or if converged, compute x vector
    //    if ( !update_x_every_iteration && (m == m_R - 1 || this->is_last_iter() || isDone(conv_stat) ))
    if (this->is_last_iter() || m == m_restart - 1 )
    {
        // H u = e
        lstsq_solver->lstsq_solve(H.getDevicePointer(), e_vect_ptr);
        // printvec(e_vect_ptr, m_restart+1, "\ne_vect");

        // x = x + Z * u
        constexpr float one = 1.0f;
        cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST);
        cublasSgemv(cublas_handle,
                    CUBLAS_OP_N, // No transpose
                    Z.getRows(),           // Number of rows of A
                    Z.getCols(),           // Number of columns of A
                    &one,      // alpha
                    Z.getDevicePointer(),         // A
                    Z.getRows(),           // leading dimension of A
                    e_vect_ptr,         // y
                    1,           // stride of y
                    &one,       // beta
                    x_ptr,         // x
                    1            // stride of x
        );

    }

    cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST);
    return AMGX_ST_NOT_CONVERGED;
    //return Base::m_monitor_convergence ? conv_stat : AMGX_ST_CONVERGED;
}

template<class T_Config>
void
FGMRES_Solver<T_Config>::solve_finalize( VVector &b, VVector &x )
{
    // residual.resize(0);
}

/****************************************
* Explict instantiations
***************************************/
#define AMGX_CASE_LINE(CASE) template class FGMRES_Solver<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace amgx
