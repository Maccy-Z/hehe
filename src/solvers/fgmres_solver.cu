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
    Solver<T_Config>( cfg, cfg_scope ), m_preconditioner(0)

{
    std::string solverName, new_scope, tmp_scope;
    cfg.getParameter<std::string>( "preconditioner", solverName, cfg_scope, new_scope );

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

    m_R = cfg.AMG_Config::template getParameter<int>("gmres_n_restart", cfg_scope);
    m_krylov_size = std::min( this->m_max_iters, m_R );
    int krylov_param = cfg.AMG_Config::template getParameter<int>( "gmres_krylov_dim", cfg_scope );

    if ( krylov_param > 0 )
    {
        m_krylov_size = std::min( m_krylov_size, krylov_param );
    }
}

template<class T_Config>
FGMRES_Solver<T_Config>::~FGMRES_Solver()
{
    if (use_preconditioner) { delete m_preconditioner; }
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
    residual.tag = (this->tag + 1) * 100 - 2;


    if ( this->m_R == 1 || this->m_max_iters == 1 )
    {   std::exit(69);
        update_x_every_iteration = true;
        update_r_every_iteration = false;
    }
    else
    {
        // printf("krylov_size = %d, R = %d\n", m_krylov_size, m_R);
        // printf("use_scalar_L2_norm = %d\n", use_scalar_L2_norm);
        // The update of x is needed only if running the truncated gmres
        update_x_every_iteration = (m_krylov_size < m_R);
        update_r_every_iteration = false; // (!use_scalar_L2_norm || (m_krylov_size < m_R)) && Base::m_monitor_convergence;
    }

    this->m_A->setView(oldView);
}

template<class T_Config>
void
FGMRES_Solver<T_Config>::solve_init( VVector &b, VVector &x, bool xIsZero )
{
    //init residual, even if we don't plan to use it, we might need it, so make sure we have enough memory to store it now
    residual.resize( b.size() );
    residual.set_block_dimx( 1 );
    residual.set_block_dimy( this->m_A->get_block_dimy() );
    residual.dirtybit = 1;
    residual.delayed_send = 1;
}


template <typename T>
void gram_schmidt_step(T* V, int n, int m, T* H, int ldH, T* Vm1) {
    /*  V: pointer to V, Matrix of previous vectors. size n x (m+1), columns V(0) to V(m), stored column-major
        H: pointer to m_H. Matrix Hessenberg with leading dimension ldH
        Vm1: pointer to V(m+1), vector to be orthogonalized, size n*/

    float one = 1.0f;
    float zero = 0.0f;
    float minus_one = -1.0f;

    // Compute H(i, m) = <V(i), V(m+1)> for i = 0 to m
    // H_col points to H(:, m)
    float* H_col = &H[m * ldH];  // H(:, m)

    // Compute H(:, m) = V^T * Vm1
    Cublas::gemv(true,     // Transpose operation
                n,                       // Number of rows in V
                m + 1,                   // Number of columns in V (from V(0) to V(m))
                &one,                    // Scalar alpha
                V,                       // Matrix V
                n,                       // Leading dimension of V
                Vm1,                     // Vector V(m+1)
                1,                       // Increment for Vm1
                &zero,                   // Scalar beta
                H_col,                   // Resulting vector H(:, m)
                1);                      // Increment for H_col

    // Update Vm1 = Vm1 - V * H(:, m)
    Cublas::gemv(false,     // No transpose operation
                n,                       // Number of rows in V
                m + 1,                   // Number of columns in V
                &minus_one,              // Scalar alpha (-1.0)
                V,                       // Matrix V
                n,                       // Leading dimension of V
                H_col,                   // Vector H(:, m)
                1,                       // Increment for H_col
                &one,                    // Scalar beta
                Vm1,                     // Vector Vm1 (updated in-place)
                1);                      // Increment for Vm1
}



//Run preconditioned GMRES
template<class T_Config>
AMGX_STATUS
FGMRES_Solver<T_Config>::solve_iteration( VVector &b, VVector &x, bool xIsZero )
{
    // AMGX_STATUS conv_stat = AMGX_ST_CONVERGED;
    cublasHandle_t handle = Cublas::get_handle();
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);

    Operator<T_Config> &A = *this->m_A;
    int m = this->m_curr_iter % m_R;

    printf("\nm = %d\n", m);

    auto new_basis = subspace.new_basis;
    float* new_basis_ptr = thrust::raw_pointer_cast(new_basis.data());
    auto& V = subspace.V_matrix;
    auto& H = subspace.H_Matrix;

    // A matrix
    auto & A_ptr = dynamic_cast<Matrix<T_Config>&>(*this->m_A);

    if (m == 0){
        //initialize gmres
        //subspace.set_iteration(0);
        subspace.iteration = 0;
        // compute initial residual r0 = b - Ax
        float* x_ptr = (float*) x.raw();

        thrust::copy(b.begin(), b.end(), new_basis.begin());
        spmv_axpy(A_ptr, x_ptr, new_basis_ptr, -1.0f, 1.0f);

        // normalize initial residual
        // float beta = computeL2Norm(new_basis);
        // thrust::transform(new_basis.begin(), new_basis.end(), new_basis.begin(), scale_by_norm(beta));
        normaliseL2(new_basis);
        V.setColumn(m, new_basis_ptr);

        // printvec(y, 10);

        // compute initial residual r0 = b - Ax
        // axmb( A, x, b, subspace.V(0), offset, size );
        // // compute initial residual norm
        // this->beta = get_norm(A, subspace.V(0), L2);
        //
        // // normalize initial residual
        // scal( subspace.V(0), ValueTypeB(1.0 / this->beta), offset, size );
        // //set reduced system rhs = beta*e1
        // thrust_wrapper::fill<AMGX_host>( m_s.begin(), m_s.end(), ValueTypeB(0.0) );
        // m_s[0] = this->beta;
    }

    // Copy new_basis into V
    subspace.iteration = m;

    // Run one iteration of preconditioner with zero initial guess and v_m as rhs, i.e. solve Az_m=v_m
    // copy(subspace.V(m), subspace.Z(m), offset, size);

    //obtain v_m+1 := A*z_m
    // A.apply( subspace.Z(m), subspace.V(m + 1) );
    spmv_axpy(A_ptr, V.getColPtr(m), new_basis_ptr, 1.0f, 0.0f);

    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
    gram_schmidt_step(V.getColPtr(0), 10200, m, H.getDevicePointer(), 251, new_basis_ptr);


    // Modified Gram-Schmidt
    // for ( int i = 0; i <= m; i++ )
    // {
    //     m_H(i, m) = dotc(V.getColThrust(i+1), new_basis, 0, 10200);
    //
    //
    //     // // H(i,m) = <V(i),V(m+1)>
    //     // m_H(i, m) = dot(A, subspace.V(i), subspace.V(m + 1));
    //     // // V(m+1) -= H(i, m) * V(i)
    //     // axpy( subspace.V(i), subspace.V(m + 1), -m_H(i, m), offset, size );
    //
    // }
    //
    //H(m+1,m) = || v_m+1 ||
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);

    float* norm = compute_L2_norm(new_basis);
    H.set_element_device(m+1, m, norm);
    // float norm = computeL2Norm(new_basis);
    // H.set_element(m + 1, m, norm);
    // // m_H(m + 1, m) = get_norm(A, subspace.V(m + 1), L2);
    // //normalize v_m+1

    scale_vector(new_basis, norm);
    //thrust::transform(new_basis.begin(), new_basis.end(), new_basis.begin(), scale_by_norm(norm));
    // // scal( subspace.V(m + 1), ValueTypeB(1.0) / m_H(m + 1, m), offset, size );
    // scal( V_test, ValueTypeB(1.0) / m_H(m + 1, m), offset, size );

    V.setColumn(m+1, new_basis_ptr);

    printvec(V.getColPtr(m), 10);
    printvec(H.getColPtr(m), 10);
    // copy(V_test, subspace.V(m + 1), offset, size);
    //
    //
    // this->gamma[m] = m_s[m];
    // PlaneRotation( m_H, m_cs, m_sn, m_s, m );
    //
    //
    // // If reached restart limit or last iteration or if converged, compute x vector
    if (this->is_last_iter() || m==2 )
    {   // printf("m = %d, m_R = %d, is_last_iter = %d, isDone = %d\n", m, m_R, this->is_last_iter(), isDone(conv_stat));
        // Solve upper triangular system in place

        //    Update the solution
        // This is dense M-V, x += [Z]*m_s
        std::exit(69);
        // for (int j = m; j >= 0; j--)
        // {
        //     m_s[j] /= m_H(j, j);
        //
        //     //S(0:j) = s(0:j) - s[j] H(0:j,j)
        //     for (int k = j - 1; k >= 0; k--)
        //     {
        //         m_s[k] -= m_H(k, j) * m_s[j];
        //     }
        // }
        // for (int j = 0; j <= m; j++)
        // {
        //     axpy( subspace.V(j), x, m_s[j], offset, size ); // m_s[j]+=a*x
        // }
    }
    //
    //
    // A.setView(oldView);
    //
    //
    // std::exit(69);

    return AMGX_ST_NOT_CONVERGED;
    //return Base::m_monitor_convergence ? conv_stat : AMGX_ST_CONVERGED;
}

template<class T_Config>
void
FGMRES_Solver<T_Config>::solve_finalize( VVector &b, VVector &x )
{
    residual.resize(0);
}

/****************************************
* Explict instantiations
***************************************/
#define AMGX_CASE_LINE(CASE) template class FGMRES_Solver<TemplateMode<CASE>::Type>;
AMGX_FORALL_BUILDS(AMGX_CASE_LINE)
#undef AMGX_CASE_LINE

} // namespace amgx
