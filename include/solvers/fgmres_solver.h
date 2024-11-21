// SPDX-FileCopyrightText: 2011 - 2024 NVIDIA CORPORATION. All Rights Reserved.
//
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include<solvers/solver.h>
#include<cusp/array1d.h>
#include<cusp/array2d.h>
#include<cusolverDn.h>
// #include "fgmres_utils.h"

// Error checking macro for cuBLAS calls
#define CUBLAS_CHECK(status) \
if (status != CUBLAS_STATUS_SUCCESS) { \
std::cerr << "cuBLAS Error: " << status \
<< " at line " << __LINE__ << std::endl; \
exit(EXIT_FAILURE); \
}

#define CHECK_CUSPARSE(func) { \
cusparseStatus_t status = (func); \
if (status != CUSPARSE_STATUS_SUCCESS) { \
fprintf(stderr, "cuSPARSE API failed at line %d with error: %d\n", \
__LINE__, status); \
exit(EXIT_FAILURE); \
} \
}


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
            if (row < 0 || row >= rows_ || col < 0 || col >= cols_) {
                std::cerr << "Error: Row or column index out of bounds in setElement()." << std::endl;
                exit(EXIT_FAILURE);
            }
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

class LeastSquaresSolver {
public:
    LeastSquaresSolver(cusolverDnHandle_t cusolverH, cublasHandle_t cublasH,
                       int m, int n)
        : cusolverH_(cusolverH), cublasH_(cublasH), m_(m), n_(n),
          d_tau_(nullptr), devInfo_(nullptr), d_work_(nullptr), lwork_(0) {
        allocate_cuda_memory();
    }

    ~LeastSquaresSolver() {
        cudaFree(d_tau_);
        cudaFree(devInfo_);
        cudaFree(d_work_);
    }

    void lstsq_solve(float* d_A, float* d_b) {
    // Compute QR factorization
    cusolverDnSgeqrf(cusolverH_, m_, n_, d_A, m_, d_tau_,
                     d_work_, lwork_, devInfo_);

    // Compute Q^T * b
    cusolverDnSormqr(cusolverH_, CUBLAS_SIDE_LEFT, CUBLAS_OP_T,
                     m_, 1, n_, d_A, m_, d_tau_,
                     d_b, m_, d_work_, lwork_, devInfo_);
    //
    // Solve R * x = c
    constexpr int nrhs = 1;
    constexpr float alpha = 1.0f;
    //

    cublasSetPointerMode(cublasH_, CUBLAS_POINTER_MODE_HOST);
    CUBLAS_CHECK(cublasStrsm(cublasH_, CUBLAS_SIDE_LEFT, CUBLAS_FILL_MODE_UPPER,
                CUBLAS_OP_N, CUBLAS_DIAG_NON_UNIT,
                n_, nrhs, &alpha, d_A, m_, d_b, m_));
    }

private:
    void allocate_cuda_memory() {
        // Allocate device memory for d_tau and devInfo
        cudaMalloc((void**)&d_tau_, n_ * sizeof(float));
        cudaMalloc((void**)&devInfo_, sizeof(int));

        // Query working space of geqrf and ormqr
        int lwork_geqrf = 0;
        int lwork_ormqr = 0;

        float* d_A_2 = nullptr;
        float* d_b_2 = nullptr;

        cusolverDnSgeqrf_bufferSize(cusolverH_, m_, n_, d_A_2, m_, &lwork_geqrf);
        cusolverDnSormqr_bufferSize(cusolverH_, CUBLAS_SIDE_LEFT, CUBLAS_OP_T,
                                    m_, 1, n_, d_A_2, m_, d_tau_, d_b_2, n_, &lwork_ormqr);

        lwork_ = (lwork_geqrf > lwork_ormqr) ? lwork_geqrf : lwork_ormqr;
        cudaMalloc((void**)&d_work_, lwork_ * sizeof(float));
    }

    // Handles
    cusolverDnHandle_t cusolverH_;
    cublasHandle_t cublasH_;

    // Device pointers
    float* d_tau_;
    int* devInfo_;
    float* d_work_;
    int lwork_;

    // Dimensions
    int m_;
    int n_;
};

template<typename T_Config>
class SpmvAxpy {
public:
    SpmvAxpy()
                : matA(nullptr), vecX(nullptr), vecY(nullptr),
                  dBuffer(nullptr), bufferSize(0),
                  num_rows(0), num_cols(0), nnz(0),
                  matrix_set(false)
    {
        // Initialize the cuSPARSE handle
        handle = Cusparse::get_instance().get_handle();
    }

    ~SpmvAxpy() {
        // Destroy descriptors if they were created
        if (matA) {
            cusparseDestroySpMat(matA);
            matA = nullptr;
        }
        if (vecX) {
            cusparseDestroyDnVec(vecX);
            vecX = nullptr;
        }
        if (vecY) {
            cusparseDestroyDnVec(vecY);
            vecY = nullptr;
        }
        // Free buffer if it was allocated
        if (dBuffer) {
            cudaFree(dBuffer);
            dBuffer = nullptr;
        }
    }

    void set_matrix(Matrix<T_Config> &A) {
        // If a matrix was previously set, clean up existing resources
        if (matrix_set) {
            if (matA) {
                cusparseDestroySpMat(matA);
                matA = nullptr;
            }
            if (dBuffer) {
                cudaFree(dBuffer);
                dBuffer = nullptr;
                bufferSize = 0;
            }
        }

        // Store matrix dimensions
        num_rows = A.get_num_rows();
        num_cols = A.get_num_cols();
        nnz = A.get_num_nz();

        // Get raw pointers to the matrix data
        int* row_offsets = A.row_offsets.raw();
        int* col_indices = A.col_indices.raw();
        auto values = (float*)A.values.raw();

        // Create a sparse matrix descriptor in CSR format
        CHECK_CUSPARSE(cusparseCreateCsr(&matA,
            num_rows, num_cols, nnz,
            row_offsets,         // row offsets
            col_indices,         // column indices
            values,              // non-zero values
            CUSPARSE_INDEX_32I,  // index type for row offsets
            CUSPARSE_INDEX_32I,  // index type for column indices
            CUSPARSE_INDEX_BASE_ZERO, // base index (0 or 1)
            CUDA_R_32F           // data type of values
        ));

        // Create dense vector descriptors if not already created
        if (!vecX) {
            CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, num_cols, nullptr, CUDA_R_32F));
        }
        if (!vecY) {
            CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, num_rows, nullptr, CUDA_R_32F));
        }

        // Determine the size of the temporary buffer
        float alpha = 1.0f;
        float beta = 0.0f;

        CHECK_CUSPARSE(cusparseSpMV_bufferSize(
            handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha,
            matA,
            vecX,
            &beta,
            vecY,
            CUDA_R_32F,
            CUSPARSE_SPMV_ALG_DEFAULT,
            &bufferSize
        ));

        // Allocate the buffer
        cudaMalloc(&dBuffer, bufferSize);

        cusparseSpMV_preprocess(handle,
                        CUSPARSE_OPERATION_NON_TRANSPOSE,
                        &alpha,
                        matA,  // non-const descriptor supported
                        vecX,  // non-const descriptor supported
                        &beta,
                        vecY,
                        CUDA_R_32F,
                        CUSPARSE_SPMV_ALG_DEFAULT,
                        dBuffer);


        matrix_set = true;
    }

    void axpy(float* d_x, float* d_y, float alpha, float beta) {
        if (!matrix_set) {
            throw std::runtime_error("Matrix is not set. Call set_matrix() before using operator().");
        }
        // Update the input and output vector pointers
        CHECK_CUSPARSE(cusparseDnVecSetValues(vecX, (void*)d_x));
        CHECK_CUSPARSE(cusparseDnVecSetValues(vecY, (void*)d_y));

        // Perform SpMV
        CHECK_CUSPARSE(cusparseSpMV(
            handle,
            CUSPARSE_OPERATION_NON_TRANSPOSE,
            &alpha,
            matA,
            vecX,
            &beta,
            vecY,
            CUDA_R_32F,
            CUSPARSE_SPMV_ALG_DEFAULT,
            dBuffer
        ));
    }

private:
    cusparseSpMatDescr_t matA;
    cusparseDnVecDescr_t vecX, vecY;
    void* dBuffer;
    size_t bufferSize;
    cusparseHandle_t handle;
    int num_rows, num_cols, nnz;
    bool matrix_set;
};

//data structure that manages the krylov vectors and takes care of all the modulo calculation etc.
template <class TConfig>
class KrylovSubspaceBuffer
{
    public:
        typedef Vector<TConfig> VVector;

        KrylovSubspaceBuffer(): V_matrix(10200, 250), H_Matrix(251, 251), new_basis(10200){
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
        Solver<T_Config> *m_preconditioner;

        SpmvAxpy<T_Config> sp_axpy = SpmvAxpy<T_Config>();
        //DEVICE WORKSPACE
        KrylovSubspaceBuffer<T_Config> subspace;
        LeastSquaresSolver* lstsq_solver = nullptr;
        thrust::device_vector<float> e_vect;

        ValueTypeA beta;

        // VVector residual; //compute the whole residual recursively

        AMGX_STATUS checkConvergenceGMRES(bool check_V_0);

};

template<class T_Config>
class FGMRES_SolverFactory : public SolverFactory<T_Config>
{
    public:
        Solver<T_Config> *create( AMG_Config &cfg, const std::string &cfg_scope, ThreadManager *tmng ) { return new FGMRES_Solver<T_Config>( cfg, cfg_scope ); }
};

} // namespace amgx
