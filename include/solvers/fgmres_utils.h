#include <amgx_cublas.h>
#include<cusolverDn.h>

#define CUDA_CHECK(err) \
if (err != cudaSuccess) { \
std::cerr << "CUDA Error: " << cudaGetErrorString(err) \
<< " at line " << __LINE__ << std::endl; \
exit(EXIT_FAILURE); \
}

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

namespace amgx{

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
        {
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
        {   printf("-Deleting cuda matrix-");
            if (d_matrix_) {
                cudaFree(d_matrix_);
                d_matrix_ = nullptr;
            }
        }

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
        float* get_col_ptr(int col_idx) const{
            if (col_idx < 0 || col_idx >= cols_) {
                std::cerr << "Error: Column index out of bounds in getColumnPtr()." << std::endl;
                exit(EXIT_FAILURE);
            }

            return d_matrix_ + static_cast<size_t>(col_idx) * rows_;
        }

        thrust::device_vector<float> getColThrust(int col_idx) const {
            // Calculate the starting pointer of the desired column
            float* column_start_ptr = get_col_ptr(col_idx);

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
            cudaMemcpy(d_ptr, device_value, sizeof(float), cudaMemcpyDeviceToDevice);

        }

        float* get_element_ptr(int row, int col) {
            auto d_ptr = d_matrix_ + static_cast<size_t>(col) * rows_ + row;
            return d_ptr;
        }

        // Retrieves the number of rows in the matrix.
        int getRows() const { return rows_; }

        int getCols() const { return cols_; }

        // Get pointer to the device matrix
        float* getDevicePointer() const { return d_matrix_; }

        // Delete copy constructor and copy assignment operator
        CudaMatrix(const CudaMatrix&) = delete;
        CudaMatrix& operator=(const CudaMatrix&) = delete;

    private:
        int rows_;      // Number of rows
        int cols_;      // Number of columns
        float* d_matrix_; // Device pointer to the matrix
};

template <typename T>
void printvec(T* d_vec, int size) {
    T *h_vec = new T[size]; // Host vector

    // Copy the device vector to the host
    cudaMemcpy(h_vec, d_vec, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the vector on the host
    std::cout << "Vector values: ";
    // std::cout << std::fixed << std::setprecision(2);

    for (int i = 0; i < size; i++) {
        std::cout << h_vec[i] << ", ";
    }
    std::cout << std::endl;

    // Free memory
    delete[] h_vec;
}


template <typename T>
void printvec(T* d_vec, int size, const char* message ) {
    T *h_vec = new T[size]; // Host vector

    // Copy the device vector to the host
    cudaMemcpy(h_vec, d_vec, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the vector on the host
    std::cout << message << ": ";
    // std::cout << std::fixed << std::setprecision(2);

    for (int i = 0; i < size; i++) {
        std::cout << h_vec[i] << " ";
    }
    std::cout << std::endl;

    // Free memory
    delete[] h_vec;
}


inline void print_cusp_array(CudaMatrix *array) {
    // Iterate over each row
    for (size_t row = 0; row < array->getRows(); ++row) {
        // Iterate over each column in the current row
        std::cout << "[ ";
        for (size_t col = 0; col < array->getCols(); ++col) {

            auto val = array->get_element_ptr(row, col);
            float* h_val = new float;
            cudaMemcpy(h_val, val, sizeof(float), cudaMemcpyDeviceToHost);
            std::cout << *h_val << ", ";
        }
        std::cout << "], ";

        std::cout << std::endl; // Newline after each row
    }
}


// CUDA kernel to compute the reciprocal of a scalar
__global__ static void reciprocal_kernel(const float* d_input, float* d_output) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
            *d_output = 1.0f / (*d_input);
    }
}


inline void compute_L2_norm(thrust::device_vector<float>& d_x, float* d_norm){
    cublasHandle_t handle = Cublas::get_handle();
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
    int n = d_x.size();

    float* d_x_ptr = thrust::raw_pointer_cast(d_x.data());
    // d_norm = ||d_vec||
    CUBLAS_CHECK(cublasSnrm2(handle, n, d_x_ptr, 1, d_norm));
}


inline void scale_vector(thrust::device_vector<float>& d_x, float* alpha) {
    /* Scale vector by 1/alpha */
    cublasHandle_t handle = Cublas::get_handle();
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);

    float* d_inv_alpha;
    CUDA_CHECK(cudaMalloc((void**)&d_inv_alpha, sizeof(float)));

    // d_inv_norm = 1 / d_norm
    reciprocal_kernel<<<1, 1>>>(alpha, d_inv_alpha);
    // CUDA_CHECK(cudaDeviceSynchronize())

    int n = d_x.size();
    auto d_x_ptr = thrust::raw_pointer_cast(d_x.data());

    CUBLAS_CHECK(cublasSscal(handle, n, d_inv_alpha, d_x_ptr, 1));

    cudaFree(d_inv_alpha);
}


// template <typename T>
// void gram_schmidt_step(T* V, int n, int m, T* H, int ldH, T* Vm1) {
//     /*  V: pointer to V, Matrix of previous vectors. size n x (m+1), columns V(0) to V(m), stored column-major
//         H: pointer to H. Matrix Hessenberg with leading dimension ldH
//         Vm1: pointer to V(m+1), vector to be orthogonalized, size n
//     */
//     cublasHandle_t handle = Cublas::get_handle();
//
//     // Set pointer mode to device
//     cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
//
//     // Allocate device memory for neg_Him_d (scalar)
//     float* neg_Him_d;
//     cudaMalloc(&neg_Him_d, sizeof(float));
//
//     // Loop over each previous vector V(:, i)
//     for (int i = 0; i <= m; ++i) {
//         // Compute H(i, m) = <V(:, i), Vm1>
//         float* H_element = &H[i + m * ldH];  // H(i, m) in device memory
//
//         cublasSdot(handle,
//                    n,                                  // Number of elements
//                    &V[i * n],                        // V(:, i)
//                    1,                                  // Increment for V
//                    Vm1,                              // Vm1
//                    1,                                  // Increment for Vm1
//                    H_element);                         // Store result in H(i, m) (device memory)
//
//         // Compute neg_Him_d = -H(i, m)
//         compute_neg_Him<<<1,1>>>(H_element, neg_Him_d);
//
//         // Update Vm1 = Vm1 + neg_Him_d * V(:, i)
//         cublasSaxpy(handle,
//                     n,                                  // Number of elements
//                     neg_Him_d,                          // Scalar multiplier (neg_Him_d in device memory)
//                     &V[i * n],                        // V(:, i)
//                     1,                                  // Increment for V
//                     Vm1,                              // Vm1 (updated in-place)
//                     1);                                 // Increment for Vm1
//     }
// }
// template <typename T>
// void gram_schmidt_step(T* d_V, int n, int m, T* d_H, int ldH, T* Vm1) {
//     cublasHandle_t handle = Cublas::get_handle();
//     cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
//
//     float one = 1.0f;
//     float zero = 0.0f;
//     float minus_one = -1.0f;
//
//     // Compute H_col = V^T * Vm1
//     float* H_col = &d_H[m * ldH];  // H(:, m)
//
//     cublasSgemv(handle, CUBLAS_OP_T, n, m + 1, &one, d_V, n, Vm1, 1, &zero, H_col, 1);
//
//     // Update Vm1 = Vm1 - V * H_col
//     cublasSgemv(handle, CUBLAS_OP_N, n, m + 1, &minus_one, d_V, n, H_col, 1, &one, Vm1, 1);
//
//     // Reorthogonalization step
//
//     // Allocate temporary H_col_new
//     float* d_H_col_new;
//     cudaMalloc(&d_H_col_new, sizeof(float)*(m+1));
//
//     // Compute H_col_new = V^T * Vm1
//     cublasSgemv(handle, CUBLAS_OP_T, n, m + 1, &one, d_V, n, Vm1, 1, &zero, d_H_col_new, 1);
//
//     // Update Vm1 = Vm1 - V * H_col_new
//     cublasSgemv(handle, CUBLAS_OP_N, n, m + 1, &minus_one, d_V, n, d_H_col_new, 1, &one, Vm1, 1);
//
//     // Update H_col += H_col_new
//     cublasSaxpy(handle, m + 1, &one, d_H_col_new, 1, H_col, 1);
//
//     // Free temporary memory
//     cudaFree(d_H_col_new);
// }


// void gram_schmidt_step(T* d_V, int n, int m, T* d_H, int ldH, T* Vm1) {
// /*  V: pointer to V, Matrix of previous vectors. size n x (m+1), columns V(0) to V(m), stored column-major
//     H: pointer to m_H. Matrix Hessenberg with leading dimension ldH
//     Vm1: pointer to V(m+1), vector to be orthogonalized, size n*/
//     cublasHandle_t handle = Cublas::get_handle();
//     cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);
//
//     float one = 1.0f;
//     float zero = 0.0f;
//     float minus_one = -1.0f;
//
//     // Compute H(i, m) = <V(i), V(m+1)> for i = 0 to m
//     // H_col points to H(:, m)
//     float* H_col = &d_H[m * ldH];  // H(:, m)
//
//     // Compute H(:, m) = V^T * Vm1
//     cublasSgemv(handle, CUBLAS_OP_T,     // Transpose operation
//                 n,                       // Number of rows in V
//                 m + 1,                   // Number of columns in V (from V(0) to V(m))
//                 &one,                    // Scalar alpha
//                 d_V,                       // Matrix V
//                 n,                       // Leading dimension of V
//                 Vm1,                     // Vector V(m+1)
//                 1,                       // Increment for Vm1
//                 &zero,                   // Scalar beta
//                 H_col,                   // Resulting vector H(:, m)
//                 1);                      // Increment for H_col
//
//     // Update Vm1 = Vm1 - V * H(:, m)
//     cublasSgemv(handle, CUBLAS_OP_N,     // No transpose operation
//                 n,                       // Number of rows in V
//                 m + 1,                   // Number of columns in V
//                 &minus_one,              // Scalar alpha (-1.0)
//                 d_V,                       // Matrix V
//                 n,                       // Leading dimension of V
//                 H_col,                   // Vector H(:, m)
//                 1,                       // Increment for H_col
//                 &one,                    // Scalar beta
//                 Vm1,                     // Vector Vm1 (updated in-place)
//                 1);                      // Increment for Vm1
//
// }


// Kernel to compute neg_Him_d = -H_element[0]
__global__ static void compute_neg_Him(const float* H_element, float* neg_Him_d) {
    neg_Him_d[0] = -H_element[0];
}

class GramSchmidtSolver {
    public:
        // Constructor: Initializes the buffer with size m_max
        GramSchmidtSolver(int m_max, std::string& mode)
            : m_max_(m_max), d_H_col_new_(nullptr) {

            if (mode == "NORMAL")
            {
                gram_schmidt_impl_ = [this] (float* d_V, int n, int m, float* d_H, int ldH, float* Vm1) {
                    gram_schmidt_normal(d_V, n, m, d_H, ldH, Vm1);
                };
            } else if (mode == "MODIFIED")
            {

                gram_schmidt_impl_ = [this] (float* d_V, int n, int m, float* d_H, int ldH, float* Vm1) {
                    gram_schmidt_modified(d_V, n, m, d_H, ldH, Vm1);
                };
            }
            else if (mode == "REORTHOGONALIZED")
            {   // Allocate device memory for H_col_new
                CUDA_CHECK(cudaMalloc(&d_H_col_new_, sizeof(float) * m_max_));
                // Initialize to zero (optional)
                CUDA_CHECK(cudaMemset(d_H_col_new_, 0, sizeof(float) * m_max_));
                gram_schmidt_impl_ = [this] (float* d_V, int n, int m, float* d_H, int ldH, float* Vm1) {
                    gram_schmidt_reorthog(d_V, n, m, d_H, ldH, Vm1);
                };
            }
            else
            {
                std::cerr << "Invalid mode in Gram Schmidt config: " << mode << std::endl;
                exit(EXIT_FAILURE);
            }
        }

        // Destructor: Frees the allocated buffer
        ~GramSchmidtSolver() {
            printf("-Deleting Gram Schmidt Solver-");
            if (d_H_col_new_) {
                cudaFree(d_H_col_new_);
                d_H_col_new_ = nullptr;
            }
        }

        // Deleted copy constructor and assignment operator to prevent copying
        GramSchmidtSolver(const GramSchmidtSolver&) = delete;
        GramSchmidtSolver& operator=(const GramSchmidtSolver&) = delete;

        void gram_schmidt(float* d_V, int n, int m, float* d_H, int ldH, float* Vm1)
        {    /*  V: pointer to V, Matrix of previous vectors. size n x (m+1), columns V(0) to V(m), stored column-major
            H: pointer to m_H. Matrix Hessenberg with leading dimension ldH
            Vm1: pointer to V(m+1), vector to be orthogonalized, size n*/
            gram_schmidt_impl_(d_V, n, m, d_H, ldH, Vm1);
        }

    private:
        int m_max_;        // Maximum m value
        float* d_H_col_new_;   // Pre-allocated device buffer for H_col_new

        std::function<void(float* d_V, int n, int m, float* d_H, int ldH, float* Vm1)> gram_schmidt_impl_;

        // Reorthogonalized GS
        void gram_schmidt_reorthog(float* d_V, int n, int m, float* d_H, int ldH, float* Vm1) {
            assert(m < m_max_ && "m exceeds preallocated buffer size.");

            // Get cuBLAS handle
            cublasHandle_t handle = Cublas::get_handle();
            CUBLAS_CHECK(cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST));

            // Scalars
            constexpr float one = 1.0f;
            constexpr float zero = 0.0f;
            constexpr float minus_one = -1.0f;

            // Compute H_col = V^T * Vm1
            float* H_col = &d_H[m * ldH];  // H(:, m)

            CUBLAS_CHECK(
                cublasSgemv(handle,CUBLAS_OP_T,n,m + 1,&one,
                            d_V, n,Vm1, 1,&zero, H_col,1)
            );

            // Update Vm1 = Vm1 - V * H_col
            CUBLAS_CHECK(
                cublasSgemv(handle, CUBLAS_OP_N, n,m + 1,&minus_one,d_V,
                            n,H_col, 1,&one, Vm1, 1)
            );

            // Reorthogonalization step

            // Compute H_col_new = V^T * Vm1 using pre-allocated buffer
            CUBLAS_CHECK(
                cublasSgemv(handle, CUBLAS_OP_T, n,m + 1, &one,d_V, n, Vm1,
                            1, &zero,d_H_col_new_,1)
            );

            // Update Vm1 = Vm1 - V * H_col_new
            CUBLAS_CHECK(
                cublasSgemv(handle, CUBLAS_OP_N, n, m + 1, &minus_one,
                            d_V,n,d_H_col_new_,1,&one, Vm1, 1)
            );

            // Update H_col += H_col_new
            CUBLAS_CHECK(
                cublasSaxpy(handle,
                            m + 1, &one,d_H_col_new_,1, H_col,1)
            );
        }

        // Modifed GS
        void gram_schmidt_modified(float* V, int n, int m, float* H, int ldH, float* Vm1) {
        /*  V: pointer to V, Matrix of previous vectors. size n x (m+1), columns V(0) to V(m), stored column-major
            H: pointer to H. Matrix Hessenberg with leading dimension ldH
            Vm1: pointer to V(m+1), vector to be orthogonalized, size n
        */
        cublasHandle_t handle = Cublas::get_handle();

        // Set pointer mode to device
        cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);

        // Allocate device memory for neg_Him_d (scalar)
        float* neg_Him_d;
        cudaMalloc(&neg_Him_d, sizeof(float));

        // Loop over each previous vector V(:, i)
        for (int i = 0; i <= m; ++i) {
            // Compute H(i, m) = <V(:, i), Vm1>
            float* H_element = &H[i + m * ldH];  // H(i, m) in device memory

            cublasSdot(handle,
                       n,                                  // Number of elements
                       &V[i * n],                        // V(:, i)
                       1,                                  // Increment for V
                       Vm1,                              // Vm1
                       1,                                  // Increment for Vm1
                       H_element);                         // Store result in H(i, m) (device memory)

            // Compute neg_Him_d = -H(i, m)
            compute_neg_Him<<<1,1>>>(H_element, neg_Him_d);

            // Update Vm1 = Vm1 + neg_Him_d * V(:, i)
            cublasSaxpy(handle,
                        n,                                  // Number of elements
                        neg_Him_d,                          // Scalar multiplier (neg_Him_d in device memory)
                        &V[i * n],                        // V(:, i)
                        1,                                  // Increment for V
                        Vm1,                              // Vm1 (updated in-place)
                        1);                                 // Increment for Vm1
        }
}

        // Normal matrix GS
        void gram_schmidt_normal(float* d_V, int n, int m, float* d_H, int ldH, float* Vm1) {
            cublasHandle_t handle = Cublas::get_handle();
            cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);

            float one = 1.0f;
            float zero = 0.0f;
            float minus_one = -1.0f;

            // Compute H(i, m) = <V(i), V(m+1)> for i = 0 to m
            // H_col points to H(:, m)
            float* H_col = &d_H[m * ldH];  // H(:, m)

            // Compute H(:, m) = V^T * Vm1
            cublasSgemv(handle, CUBLAS_OP_T,     // Transpose operation
                        n,                       // Number of rows in V
                        m + 1,                   // Number of columns in V (from V(0) to V(m))
                        &one,                    // Scalar alpha
                        d_V,                       // Matrix V
                        n,                       // Leading dimension of V
                        Vm1,                     // Vector V(m+1)
                        1,                       // Increment for Vm1
                        &zero,                   // Scalar beta
                        H_col,                   // Resulting vector H(:, m)
                        1);                      // Increment for H_col

            // Update Vm1 = Vm1 - V * H(:, m)
            cublasSgemv(handle, CUBLAS_OP_N,     // No transpose operation
                        n,                       // Number of rows in V
                        m + 1,                   // Number of columns in V
                        &minus_one,              // Scalar alpha (-1.0)
                        d_V,                       // Matrix V
                        n,                       // Leading dimension of V
                        H_col,                   // Vector H(:, m)
                        1,                       // Increment for H_col
                        &one,                    // Scalar beta
                        Vm1,                     // Vector Vm1 (updated in-place)
                        1);                      // Increment for Vm1

        }
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

    // using Clock = std::chrono::steady_clock;
    // using TimePoint = std::chrono::time_point<Clock>;
    // using Duration = std::chrono::duration<double, std::milli>; // milliseconds
    // TimePoint start, end;
    // Duration duration;
    //
    // cudaDeviceSynchronize();
    // start = Clock::now();

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

    // cudaDeviceSynchronize();
    // end = Clock::now();
    // duration = end - start;
    // std::cout << "lstsq " << duration.count() << " ms\n";
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
    /* y = alpha * A * x + beta * y */
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
        /* y = alpha * A * x + beta * y */
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


}