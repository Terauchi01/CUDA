// ...existing code...
#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <vector>

extern "C" void add_launch(const float*, const float*, float*, int);

TEST(AddTest, Small){
    int n = 4;
    std::vector<float> a{1,2,3,4}, b{10,20,30,40}, c(n);
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;

    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_a, n * sizeof(float)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_b, n * sizeof(float)));
    ASSERT_EQ(cudaSuccess, cudaMalloc(&d_c, n * sizeof(float)));

    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_a, a.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    ASSERT_EQ(cudaSuccess, cudaMemcpy(d_b, b.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    add_launch(d_a, d_b, d_c, n);

    ASSERT_EQ(cudaSuccess, cudaMemcpy(c.data(), d_c, n * sizeof(float), cudaMemcpyDeviceToHost));

    EXPECT_FLOAT_EQ(c[0], 11.0f);

    cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
}

int main(int argc,char** argv){ ::testing::InitGoogleTest(&argc,argv); return RUN_ALL_TESTS(); }
// ...existing code...