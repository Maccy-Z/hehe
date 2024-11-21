// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#include <amgx_cublas.h>
#include <blas.h>
#include <chrono>
#include <cutil.h>
#include <norm.h>
#include <util.h>
#include <cusp/blas.h>
#include <solvers/fgmres_solver.h>
#include <thrust/transform.h>
#include <thrust/functional.h>
#include "solvers/fgmres_utils.h"

//TODO remove synchronization from this module by moving host operations to the device


namespace amgx
{

//init the frist vector
template <class TConfig>
void KrylovSubspaceBuffer<TConfig>::setup(int N, int blockdim, int tag)
{
    this->N = N;
    this->blockdim = blockdim;
    this->tag = tag;
    this->iteration = -1;
}


template< class T_Config>
FGMRES_Solver<T_Config>::FGMRES_Solver( AMG_Config &cfg, const std::string &cfg_scope ) :
    Solver<T_Config>(cfg, cfg_scope), m_preconditioner(0), e_vect(251)

{
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
        m_preconditioner = SolverFactory<T_Config>::allocate(cfg, cfg_scope, "preconditioner");
    }

    m_R = cfg.AMG_Config::template getParameter<int>("gmres_n_restart", cfg_scope);
    m_krylov_size = std::min(this->m_max_iters, m_R);
    int krylov_param = cfg.AMG_Config::template getParameter<int>("gmres_krylov_dim", cfg_scope);

    if (krylov_param > 0)
    {
        m_krylov_size = std::min(m_krylov_size, krylov_param);
    }

    // Init least squares solver
    cublasHandle_t handle_cublas = Cublas::get_handle();
    cusolverDnHandle_t hanlde_cusolver = nullptr;
    cusolverDnCreate(&hanlde_cusolver);
    lstsq_solver = new LeastSquaresSolver(hanlde_cusolver, handle_cublas, 251, 250);
}

template<class T_Config>
FGMRES_Solver<T_Config>::~FGMRES_Solver()
{
    if (use_preconditioner) { delete m_preconditioner; }
    delete lstsq_solver;
}

template<class T_Config>
void
FGMRES_Solver<T_Config>::printSolverParameters() const
{
    std::cout << "gmres_n_restart=" << this->m_R << std::endl;

    if (use_preconditioner)
    {
        std::cout << "preconditioner: " << this->m_preconditioner->getName() << " with scope name: " << this->m_preconditioner->getScope() << std::endl;
    }
}

template<class T_Config>
void
FGMRES_Solver<T_Config>::solver_setup(bool reuse_matrix_structure)
{
    if (use_preconditioner)
    {
        m_preconditioner->setup( *this->m_A, reuse_matrix_structure );
    }

    ViewType oldView = this->m_A->currentView();
    this->m_A->setViewExterior();
    //should we warn the user about the extra computational work?
    // printf("m_nrm.size() = %d, m_use_scalar_norm = %d, m_norm_type = %d\n", this->m_nrm.size(), this->m_use_scalar_norm, this->m_norm_type);
    use_scalar_L2_norm = (this->m_nrm.size() == 1 || this->m_use_scalar_norm) && this->m_norm_type == L2;
    subspace.setup(this->m_A->get_num_cols()*this->m_A->get_block_dimy(), this->m_A->get_block_dimy(), this->tag);
    // residual.tag = (this->tag + 1) * 100 - 2;

    this->m_A->setView(oldView);
}

template<class T_Config>
void
FGMRES_Solver<T_Config>::solve_init( VVector &b, VVector &x, bool xIsZero )
{
    //init residual, even if we don't plan to use it, we might need it, so make sure we have enough memory to store it now
    // residual.resize( b.size() );
    // residual.set_block_dimx( 1 );
    // residual.set_block_dimy( this->m_A->get_block_dimy() );
    // residual.dirtybit = 1;
    // residual.delayed_send = 1;
}




//Run preconditioned GMRES
template<class T_Config>
AMGX_STATUS
FGMRES_Solver<T_Config>::solve_iteration( VVector &b, VVector &x, bool xIsZero )
{
    using Clock = std::chrono::steady_clock;
    using TimePoint = std::chrono::time_point<Clock>;
    using Duration = std::chrono::duration<double, std::milli>; // milliseconds
    TimePoint start, end;
    Duration duration;

    // AMGX_STATUS conv_stat = AMGX_ST_CONVERGED;
    cublasHandle_t cublas_handle = Cublas::get_handle();

    // Operator<T_Config> &A = *this->m_A;
    int m = this->m_curr_iter % m_R;

    auto new_basis = subspace.new_basis;
    float* new_basis_ptr = thrust::raw_pointer_cast(new_basis.data());
    auto& V = subspace.V_matrix;
    auto& H = subspace.H_Matrix;
    float* e_vect_ptr = thrust::raw_pointer_cast(e_vect.data());
    float* x_ptr = (float*) thrust::raw_pointer_cast(x.data());

    // A matrix
    auto& A = dynamic_cast<Matrix<T_Config>&>(*this->m_A);
    if (m == 0)
    {
        sp_axpy.set_matrix(A);
    }

    if (m == 0){
        //initialize gmres
        //subspace.set_iteration(0);
        subspace.iteration = 0;
        // compute initial residual r0 = b - Ax
        float* x_ptr = (float*) x.raw();

        thrust::copy(b.begin(), b.end(), new_basis.begin());
        sp_axpy.axpy(x_ptr, new_basis_ptr, -1.0f, 1.0f);

        // normalize initial residual
        float* beta = compute_L2_norm(new_basis);
        scale_vector(new_basis, beta);
        V.setColumn(m, new_basis_ptr);

        // e = [beta, 0, 0, ...]
        thrust::fill(e_vect.begin(), e_vect.end(), 0);
        cudaMemcpy(e_vect_ptr, beta, sizeof(float), cudaMemcpyDeviceToDevice);

    }
    // Copy new_basis into V
    subspace.iteration = m;

    // Run one iteration of preconditioner with zero initial guess and v_m as rhs, i.e. solve Az_m=v_m
    // copy(subspace.V(m), subspace.Z(m), offset, size);

    //obtain v_m+1 := A*z_m
    sp_axpy.axpy(V.getColPtr(m), new_basis_ptr, 1.0f, 0.0f);

    // start = Clock::now();

    gram_schmidt_step(V.getColPtr(0), 10200, m, H.getDevicePointer(), 251, new_basis_ptr);
    // cudaDeviceSynchronize() ;
    // end = Clock::now();
    // duration = end - start;
    // std::cout << "Block 1 took " << duration.count() << " ms\n";
    //H(m+1,m) = || v_m+1 ||
    float* norm = compute_L2_norm(new_basis);
    H.set_element_device(m+1, m, norm);
    // //normalize v_m+1
    scale_vector(new_basis, norm);

    if (m < 249)
    {
        V.setColumn(m+1, new_basis_ptr);
    }



    // // If reached restart limit or last iteration or if converged, compute x vector
    //    if ( !update_x_every_iteration && (m == m_R - 1 || this->is_last_iter() || isDone(conv_stat) ))
    if (this->is_last_iter() || m == m_R - 1 )
    {   // printf("m = %d, m_R = %d, is_last_iter = %d, isDone = %d\n", m, m_R, this->is_last_iter(), isDone(conv_stat));
        // Solve Hessenberg system
        // printf("\nm = %d, current iter = %d\n", m, this->m_curr_iter);
        // //    Update the solution
        // // This is dense M-V,
        // printf("\nLast Iteration\n");
        // printf("m = %d\n", m);

        lstsq_solver->lstsq_solve(H.getDevicePointer(), e_vect_ptr);

        const float one = 1.0f;
        // x = x + A e_vect
        cublasSetPointerMode(cublas_handle, CUBLAS_POINTER_MODE_HOST);
        cublasSgemv(cublas_handle,
                    CUBLAS_OP_N, // No transpose
                    V.getRows(),           // Number of rows of A
                    V.getCols(),           // Number of columns of A
                    &one,      // alpha
                    V.getDevicePointer(),         // A
                    V.getRows(),           // leading dimension of A
                    e_vect_ptr,         // y
                    1,           // stride of y
                    &one,       // beta
                    x_ptr,         // x
                    1            // stride of x
        );


        // printvec(x_ptr, 10);


    }
    //
    //
    // A.setView(oldView);
    //
    //
    // std::exit(69);
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
