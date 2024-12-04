// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include<solvers/solver.h>
// #include<cusp/array1d.h>
// #include<cusp/array2d.h>
#include<cusolverDn.h>
#include <amgx_cublas.h>

#include "solvers/fgmres_utils.h"


namespace amgx
{
//data structure that manages the krylov vectors and takes care of all the modulo calculation etc.
template <class TConfig>
class KrylovSubspaceBuffer
{
    public:
        typedef Vector<TConfig> VVector;

        KrylovSubspaceBuffer(){}

        ~KrylovSubspaceBuffer() {
            printf("-Deleting KrylovSubspaceBuffer-");
            delete new_basis;
            delete V_matrix;
            delete Z_matrix;
            delete H_Matrix;

        };
        thrust::device_vector<float>* new_basis = nullptr; //new basis vector
        CudaMatrix* V_matrix = nullptr;     //Matrix of Krylov vectors
        CudaMatrix* Z_matrix = nullptr;     //Matrix of Z vectors
        CudaMatrix* H_Matrix = nullptr;     //Matrix of Hessenberg matrix
        int iteration;

        void setup(int N_dim, int restart_iters);

    private:
        int N_dim, restart_iters;
};


template<class T_Config>
class FGMRES_Solver : public Solver<T_Config>
{

    public:

        typedef Solver<T_Config> Base;
        typedef typename Base::VVector VVector;

        FGMRES_Solver( AMG_Config &cfg, const std::string &cfg_scope );
        ~FGMRES_Solver();

        bool is_residual_needed() const { return false; }
        void printSolverParameters() const;
        void solver_setup(bool reuse_matrix_structure);

        bool isColoringNeeded( ) const { if (m_preconditioner != NULL) return m_preconditioner->isColoringNeeded(); else return false; }
        void getColoringScope( std::string &cfg_scope_for_coloring) const { if (m_preconditioner != NULL) m_preconditioner->getColoringScope(cfg_scope_for_coloring); }
        bool getReorderColsByColorDesired() const { if (m_preconditioner != NULL) return m_preconditioner->getReorderColsByColorDesired(); return false; }

        bool getInsertDiagonalDesired() const
        {
            if (m_preconditioner != NULL)
            {
                return m_preconditioner->getInsertDiagonalDesired();
            }

            return false;
        }

        void solve_init( VVector &b, VVector &x, bool xIsZero ){};
        // Run a single iteration. Compute the residual and its norm and decide convergence.
        bool solve_one_iteration( VVector &b, VVector &x );
        // Run a single iteration. Compute the residual and its norm and decide convergence.
        AMGX_STATUS solve_iteration( VVector &b, VVector &x, bool xIsZero );
        void solve_finalize( VVector &b, VVector &x );

    private:

        int m_restart = 0;  //Iterations between restarts, do we still need restart?
        int m_dim = 0;      //Dimension of problem to solve
        bool use_preconditioner;

        Solver<T_Config> *m_preconditioner;

        SpmvAxpy<T_Config> sp_axpy = SpmvAxpy<T_Config>();
        GramSchmidtSolver* GS_solver = nullptr;

        //DEVICE WORKSPACE
        KrylovSubspaceBuffer<T_Config> subspace;
        LeastSquaresSolver* lstsq_solver = nullptr;
        thrust::device_vector<float>* e_vect;
        float* d_norm_tmp;

        // VVector for calling preconditioner
        VVector* v_m_vvect = nullptr;
        // Zero VVector for calling preconditioner
        VVector* p_inv_v_m = nullptr;

        AMGX_STATUS checkConvergenceGMRES(bool check_V_0);

};

template<class T_Config>
class FGMRES_SolverFactory : public SolverFactory<T_Config>
{
    public:
        Solver<T_Config> *create( AMG_Config &cfg, const std::string &cfg_scope, ThreadManager *tmng ) { return new FGMRES_Solver<T_Config>( cfg, cfg_scope ); }
};


} // namespace amgx
