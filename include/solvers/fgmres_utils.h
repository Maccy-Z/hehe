#include <amgx_cublas.h>

#define CHECK_CUSPARSE(func) { \
cusparseStatus_t status = (func); \
if (status != CUSPARSE_STATUS_SUCCESS) { \
fprintf(stderr, "cuSPARSE API failed at line %d with error: %d\n", \
__LINE__, status); \
exit(EXIT_FAILURE); \
} \
}

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
// struct square {
//     __device__
//     float operator()(const float& x) const {
//         return x * x;
//     }
// };
//
//
// // Function to compute the L2 norm of a thrust::device_vector<float>
// float computeL2Norm(const thrust::device_vector<float>& d_vec) {
//     // Compute the sum of squares using thrust::transform_reduce
//     cudaDeviceSynchronize();
//     float sum_of_squares = thrust::transform_reduce(
//         d_vec.begin(), d_vec.end(), // Input range
//         square(),                   // Unary operation (square each element)
//         0.0f,                       // Initial value of the reduction
//         thrust::plus<float>()       // Binary operation (sum)
//     );
//
//     // Return the square root of the sum of squares (the L2 norm)
//     return std::sqrt(sum_of_squares);
// }

// CUDA kernel to compute the reciprocal of a scalar
__global__ void reciprocal_kernel(const float* d_input, float* d_output) {
    if (threadIdx.x == 0 && blockIdx.x == 0) {
            *d_output = 1.0f / (*d_input);
    }
}

// __global__ void reciprocal_kernel(float norm, float* inverse_norm) {
//     if (threadIdx.x == 0 && blockIdx.x == 0) {
//         inverse_norm[0] = 1.0f / norm;
//     }
// }

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
    float* d_inv_alpha;
    CUDA_CHECK(cudaMalloc((void**)&d_inv_alpha, sizeof(float)));

    // d_inv_norm = 1 / d_norm
    reciprocal_kernel<<<1, 1>>>(alpha, d_inv_alpha);
    // CUDA_CHECK(cudaDeviceSynchronize())

    int n = d_x.size();
    auto d_x_ptr = thrust::raw_pointer_cast(d_x.data());

    CUBLAS_CHECK(cublasSscal(handle, n, d_inv_alpha, d_x_ptr, 1));
}

inline void normaliseL2(thrust::device_vector<float>& d_x) {
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



template <typename T, typename T_Config>
void spmv_axpy(
    Matrix<T_Config> &A_ptr,
    T* d_x,
    float* d_y,
    float alpha, float beta
    ) {

    // Create a sparse matrix descriptor in CSR format
    int num_rows = A_ptr.get_num_rows();
    int num_cols = A_ptr.get_num_cols();
    int nnz = A_ptr.get_num_nz();
    int* row_offsets = A_ptr.row_offsets.raw();
    int* col_indices = A_ptr.col_indices.raw();
    auto values = (float*)A_ptr.values.raw();

    cusparseSpMatDescr_t matA;
    cusparseCreateCsr(&matA,
        num_rows, num_cols, nnz,
        (void*)row_offsets,   // row offsets
        (void*)col_indices,   // column indices
        (void*)values,        // non-zero values
        CUSPARSE_INDEX_32I,     // index type for row offsets
        CUSPARSE_INDEX_32I,     // index type for column indices
        CUSPARSE_INDEX_BASE_ZERO, // base index (0 or 1)
        CUDA_R_32F               // data type of values
    );

    // Create dense vector descriptors for input vector b and output vector y
    cusparseDnVecDescr_t vecX, vecY;
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecX, num_cols, (void*)d_x, CUDA_R_32F));
    CHECK_CUSPARSE(cusparseCreateDnVec(&vecY, num_rows, (void*)d_y, CUDA_R_32F));

    //todo: CUSPARSE
    auto cusparse_obj = new Cusparse();
    cusparse_obj->create_handle();
    cusparseHandle_t handle = cusparse_obj->get_handle();

    // Determine the size of the temporary buffer
    size_t bufferSize = 0;
    CHECK_CUSPARSE(cusparseSpMV_bufferSize(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        matA,
        vecX,
        &beta,
        vecY,
        CUDA_R_32F,
        CUSPARSE_MV_ALG_DEFAULT,
        &bufferSize
    ));

    void* dBuffer = NULL;
    cudaMalloc(&dBuffer, bufferSize);

    CHECK_CUSPARSE(cusparseSpMV(
        handle,
        CUSPARSE_OPERATION_NON_TRANSPOSE,
        &alpha,
        matA,
        vecX,
        &beta,
        vecY,
        CUDA_R_32F,
        CUSPARSE_MV_ALG_DEFAULT,
        dBuffer
    ));
}



}