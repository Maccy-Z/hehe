#include <amgx_cublas.h>

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

namespace amgx{


template <typename T>
void printvec(T* d_vec, int size) {
    T *h_vec = new T[size]; // Host vector

    // Copy the device vector to the host
    cudaMemcpy(h_vec, d_vec, size * sizeof(float), cudaMemcpyDeviceToHost);

    // Print the vector on the host
    std::cout << "Vector values:" << std::endl;
    // std::cout << std::fixed << std::setprecision(2);

    for (int i = 0; i < size; i++) {
        std::cout << h_vec[i] << " ";
    }
    std::cout << std::endl;

    // Free memory
    delete[] h_vec;
}


// CUDA kernel to compute the reciprocal of a scalar
__global__ void reciprocal_kernel(const float* d_input, float* d_output) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
            *d_output = 1.0f / (*d_input);
    }
}


inline float* compute_L2_norm(thrust::device_vector<float>& d_x){
    cublasHandle_t handle = Cublas::get_handle();
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_DEVICE);
    int n = d_x.size();

    float* d_x_ptr = thrust::raw_pointer_cast(d_x.data());

    float* d_norm;
    CUDA_CHECK(cudaMalloc((void**)&d_norm, sizeof(float)));
    // d_norm = ||d_vec||
    CUBLAS_CHECK(cublasSnrm2(handle, n, d_x_ptr, 1, d_norm));

    return d_norm;
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
}

inline void normalise_L2(thrust::device_vector<float>& d_x) {
    /* Scale vector by 1/||d_x|| */

    float* d_norm = compute_L2_norm(d_x);
    // d_x = d_x / d_norm
    scale_vector(d_x, d_norm);
    //
}


// Functor to scale elements
struct scale_by_norm {
    float norm;

    scale_by_norm(float _norm) : norm(_norm) {}

    __device__
    float operator()(const float &x) const {
        return x / norm;
    }
};


template <typename T>
void gram_schmidt_step(T* V, int n, int m, T* H, int ldH, T* Vm1) {
/*  V: pointer to V, Matrix of previous vectors. size n x (m+1), columns V(0) to V(m), stored column-major
    H: pointer to m_H. Matrix Hessenberg with leading dimension ldH
    Vm1: pointer to V(m+1), vector to be orthogonalized, size n*/
    cublasHandle_t handle = Cublas::get_handle();
    cublasSetPointerMode(handle, CUBLAS_POINTER_MODE_HOST);

    float one = 1.0f;
    float zero = 0.0f;
    float minus_one = -1.0f;

    // Compute H(i, m) = <V(i), V(m+1)> for i = 0 to m
    // H_col points to H(:, m)
    float* H_col = &H[m * ldH];  // H(:, m)

    // Compute H(:, m) = V^T * Vm1
    cublasSgemv(handle, CUBLAS_OP_T,     // Transpose operation
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
    cublasSgemv(handle, CUBLAS_OP_N,     // No transpose operation
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

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void ptr_to_vvector(float* d_x, int size, Vector<TemplateConfig<AMGX_device, t_vecPrec, t_matPrec, t_indPrec>>& vec)
{
    // These don't need to be set
    // auto rows = vec.get_num_rows();  = 0
    // auto cols = vec.get_num_cols();  = 1
    // auto lda = vec.get_lda();  = 0
    // auto dimx = vec.get_block_dimx();  = 1
    // auto dimy = vec.get_block_dimy();  = 1
    // auto bs = vec.get_block_size();  = 1
    // auto size = vec.size(); = m_dim
    //printf("rows = %d, cols = %d, lda = %d, dimx = %d, dimy = %d, bs = %d, size = %d\n", rows, cols, lda, dimx, dimy, bs, size);


    float* d_thrust_vec = (float*) vec.raw();
    cudaMemcpy(d_thrust_vec, d_x, size * sizeof(float), cudaMemcpyDeviceToDevice);

    // vec.resize(size, sizeof(float));
    // vec.assign(d_x, d_x + size * sizeof(float));

}

template <AMGX_VecPrecision t_vecPrec, AMGX_MatPrecision t_matPrec, AMGX_IndPrecision t_indPrec>
void ptr_to_vvector(float* d_x, int size, amgx::Vector<amgx::TemplateConfig<AMGX_host, t_vecPrec, t_matPrec, t_indPrec>>& vec)
{

    std::exit(123);

}

}