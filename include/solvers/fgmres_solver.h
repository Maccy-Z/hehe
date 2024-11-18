// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include<solvers/solver.h>
#include<cusp/array1d.h>
#include<cusp/array2d.h>

namespace amgx
{
class CudaMatrix {
    public:
        /**
         * @brief Constructor that initializes the matrix to zeros.
         *
         * @param rows Number of rows in the matrix.
         * @param cols Number of columns in the matrix.
         */
        CudaMatrix(int rows, int cols)
            : rows_(rows), cols_(cols), d_matrix_(nullptr)
        {   printf("Making cuda matrix \n");
            size_t size = static_cast<size_t>(rows_) * cols_ * sizeof(float);
            // Allocate device memory
            cudaMalloc(reinterpret_cast<void**>(&d_matrix_), size);
            // Initialize to zero
            cudaMemset(d_matrix_, 0, size);
        }

        /**
         * @brief Destructor that frees the device memory.
         */
        ~CudaMatrix()
        {   printf("Deleting cuda matrix\n");
            if (d_matrix_) {
                cudaFree(d_matrix_);
                d_matrix_ = nullptr;
            }
        }

        /**
         * @brief Sets a specific column of the matrix with data from the host.
         *
         * @param col_idx The index of the column to set (0-based).
         * @param host_data Pointer to the host data array of size `rows_`.
         */
        /**
         * @brief Sets a specific column of the matrix with data from device memory.
         *
         * @param col_idx The index of the column to set (0-based).
         * @param device_data Pointer to the device data array of size `rows_`.
         */
        void setColumn(int col_idx, const float* device_data){
            if (col_idx < 0 || col_idx >= cols_) {
                std::cerr << "Error: Column index out of bounds in setColumn()." << std::endl;
                exit(EXIT_FAILURE);
            }
            // printf("rows: %d, cols: %d\n", rows_, cols_);
            // std::exit(9);
            // Calculate the device pointer to the start of the specified column
            float* d_col_ptr = d_matrix_ + static_cast<size_t>(col_idx) * rows_;

            // Copy data from device to device
            cudaMemcpy(d_col_ptr, device_data, rows_ * sizeof(float), cudaMemcpyDeviceToDevice);
        }

        /**
         * @brief Retrieves the device pointer to a specific column.
         *
         * @param col_idx The index of the column to retrieve (0-based).
         * @return float* Device pointer to the start of the specified column.
         */
        float* getColPtr(int col_idx) const{
            if (col_idx < 0 || col_idx >= cols_) {
                std::cerr << "Error: Column index out of bounds in getColumnPtr()." << std::endl;
                exit(EXIT_FAILURE);
            }

            return d_matrix_ + static_cast<size_t>(col_idx) * rows_;
        }

        const thrust::device_vector<float> getColThrust(int col_idx) const {
            // Calculate the starting pointer of the desired column
            float* column_start_ptr = getColPtr(col_idx);

            // Wrap the raw device pointer with a thrust::device_ptr
            thrust::device_ptr<float> dev_ptr(column_start_ptr);

            // Construct and return the device_vector using iterators
            return thrust::device_vector<float>(dev_ptr, dev_ptr + rows_);
        }

        void set_element(int row, int col, float value) {
            float* d_ptr = d_matrix_ + static_cast<size_t>(col) * rows_ + row;
            cudaMemcpy(d_ptr, &value, sizeof(float), cudaMemcpyHostToDevice);
        }

    void set_element_device(int row, int col, const float* device_value) {
            // Calculate the device pointer to the target matrix element
            float* d_ptr = d_matrix_ + static_cast<size_t>(col) * rows_ + row;

            // Perform a device-to-device memory copy
            cudaError_t err = cudaMemcpy(d_ptr, device_value, sizeof(float), cudaMemcpyDeviceToDevice);

        }

        float* get_element_ptr(int row, int col) {
            auto d_ptr = d_matrix_ + static_cast<size_t>(col) * rows_ + row;
            return d_ptr;
        }

        /**
         * @brief Retrieves the number of rows in the matrix.
         *
         * @return int Number of rows.
         */
        int getRows() const { return rows_; }

        /**
         * @brief Retrieves the number of columns in the matrix.
         *
         * @return int Number of columns.
         */
        int getCols() const { return cols_; }

        /**
         * @brief Retrieves the device pointer to the entire matrix.
         *
         * @return float* Device pointer to the matrix.
         */
        float* getDevicePointer() const { return d_matrix_; }

        // Delete copy constructor and copy assignment operator
        CudaMatrix(const CudaMatrix&) = delete;
        CudaMatrix& operator=(const CudaMatrix&) = delete;

    private:
        int rows_;      // Number of rows
        int cols_;      // Number of columns
        float* d_matrix_; // Device pointer to the matrix
};


//data structure that manages the krylov vectors and takes care of all the modulo calculation etc.
template <class TConfig>
class KrylovSubspaceBuffer
{
    public:
        typedef Vector<TConfig> VVector;

        KrylovSubspaceBuffer(): V_matrix(10200, 251), H_Matrix(251, 251), new_basis(10200){
            this->N = 0;
            printf("Making Krylov Buffer\n");
        }

        ~KrylovSubspaceBuffer() {
            printf("FREEING \n");
        };
        thrust::device_vector<float> new_basis;
        CudaMatrix V_matrix;
        CudaMatrix H_Matrix;
        int iteration;

        void setup(int N, int blockdim, int tag);

    private:
        // std::vector<VVector *> m_V_vector;
        //std::vector<VVector *> m_Z_vector;

        // int dimension;
        // int max_dimension;
        int N, tag, blockdim;

        bool increase_dimension();
};



template<class T_Config>
class FGMRES_Solver : public Solver<T_Config>
{

    public:

        typedef Solver<T_Config> Base;
        typedef typename Base::VVector VVector;
        typedef typename Base::Vector_h Vector_h;
        typedef typename Base::ValueTypeA ValueTypeA;
        typedef typename Base::ValueTypeB ValueTypeB;

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

        void solve_init( VVector &b, VVector &x, bool xIsZero );
        // Run a single iteration. Compute the residual and its norm and decide convergence.
        bool solve_one_iteration( VVector &b, VVector &x );
        // Run a single iteration. Compute the residual and its norm and decide convergence.
        AMGX_STATUS solve_iteration( VVector &b, VVector &x, bool xIsZero );
        void solve_finalize( VVector &b, VVector &x );

    private:

        int m_R;  //Iterations between restarts, do we still need restart?
        int m_krylov_size;
        bool use_preconditioner;
        bool use_scalar_L2_norm;
        bool update_x_every_iteration; // x is solution
        bool update_r_every_iteration; // r is residual
        Solver<T_Config> *m_preconditioner;

        //DEVICE WORKSPACE
        KrylovSubspaceBuffer<T_Config> subspace;

        //HOST WORKSPACE
        //TODO: move those to device
        // cusp::array2d<ValueTypeB, cusp::host_memory, cusp::column_major> m_H; //Hessenberg matrix
        // cusp::array1d<ValueTypeB, cusp::host_memory> m_s; // rotated right-hand side vector, size=m+1
        // cusp::array1d<ValueTypeB, cusp::host_memory> m_cs; // Givens rotation cosine
        // cusp::array1d<ValueTypeB, cusp::host_memory> m_sn; // Givens rotation sine
        // cusp::array1d<ValueTypeB, cusp::host_memory> gamma; // recursion for residual calculation

        ValueTypeA beta;

        VVector residual; //compute the whole residual recursively

        AMGX_STATUS checkConvergenceGMRES(bool check_V_0);

};

template<class T_Config>
class FGMRES_SolverFactory : public SolverFactory<T_Config>
{
    public:
        Solver<T_Config> *create( AMG_Config &cfg, const std::string &cfg_scope, ThreadManager *tmng ) { return new FGMRES_Solver<T_Config>( cfg, cfg_scope ); }
};

} // namespace amgx
