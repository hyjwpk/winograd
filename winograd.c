#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const float G[4][3] = {
    {1.0, 0.0, 0.0}, {0.5, 0.5, 0.5}, {0.5, -0.5, 0.5}, {0.0, 0.0, 1.0}};
const float G_T[3][4] = {
    {1, 0.5, 0.5, 0.0}, {0.0, 0.5, -0.5, 0.0}, {0.0, 0.5, 0.5, 1.0}};
const float B[4][4] = {
    {1, 0, 0, 0}, {0, 1, -1, 1}, {-1, 1, 1, 0}, {0, 0, 0, -1}};
const float B_T[4][4] = {
    {1, 0, -1, 0}, {0, 1, 1, 0}, {0, -1, 1, 0}, {0, 1, 0, -1}};
const float A[4][2] = {{1, 0}, {1, 1}, {1, -1}, {0, -1}};
const float A_T[2][4] = {{1, 1, 1, 0}, {0, 1, -1, -1}};

void sgemm(const float *A, const float *B, float *out, const int M, const int K,
           const int N) {
    memset(out, 0, M * N * sizeof(float));
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; ++k) {
            for (int j = 0; j < N; ++j) {
                out[i * N + j] += A[i * K + k] * B[k * N + j];
            }
        }
    }
}

void sgemm_parallel(const float *A, const float *B, float *out, const int M, const int K, const int N) {
    memset(out, 0, M * N * sizeof(float));
    int s = 32;
#pragma omp parallel for collapse(2)
    for (int ih = 0; ih < M; ih += s) {
        for (int jh = 0; jh < N; jh += s) {
            for (int kh = 0; kh < K; kh += s) {
                for (int i = 0; i < s && ih + i < M; i++) {
                    for (int k = 0; k < s && kh + k < K; k++) {
                        for (int j = 0; j < s && jh + j < N; j++) {
                            out[(long)(ih + i) * N + jh + j] += A[(long)(ih + i) * K + kh + k] * B[(long)(kh + k) * N + jh + j];
                        }
                    }
                }
            }
        }
    }
}
// User API for winograd F(2,3)
// image: [batch * C * inHeight * inWidth]
// filter: [K * C * 3 * 3]
// result: [batch * K * outHeight * outWidth]
void winconv_2x3(float *__restrict__ image, const int inHeight,
                 const int inWidth, const int C, float *__restrict__ filter,
                 const int K, const int N, float *__restrict__ out,
                 float *__restrict__ U, float *__restrict__ V,
                 float *__restrict__ M) {
    // m = 2; r = 3; alpha = 4
    const int outHeight = inHeight - 2;
    const int outWidth = inWidth - 2;
    const long sizeI = inHeight * inWidth;
    const int sizeF = 3 * 3;
    const int sizeO = outHeight * outWidth;
    const long P = outHeight / 2 * outWidth / 2 * N;

    // U[:, :, k, c] = G * filters[k, c, :, :] * G.T()
    float tmp_u[12]; // 4 * 3
    float u[16];     // 4 * 4;
#pragma omp parallel for collapse(2) private(tmp_u, u)
    for (int k = 0; k < K; ++k) {
        for (int c = 0; c < C; ++c) {
            float *filters_ptr = filter + (k * C + c) * sizeF;
            sgemm(&G[0][0], filters_ptr, tmp_u, 4, 3, 3);
            sgemm(tmp_u, &G_T[0][0], u, 4, 3, 4);
            for (int xi = 0; xi < 4; ++xi) {
                for (int nu = 0; nu < 4; ++nu) {
                    U[((xi * 4 + nu) * K + k) * C + c] = u[xi * 4 + nu];
                }
            }
        }
    }

    // V[:, :, c, p] = B_T * image[c, b, :, :] * B
    float tmp_v[16];
    float d[16]; // d: [4 * 4];
    float v[16]; // v: [4 * 4];
#pragma omp parallel for collapse(2) private(tmp_v, d, v)
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int y = 0; y < outHeight / 2; ++y) {
                for (int x = 0; x < outWidth / 2; ++x) {
                    // Generate d_cb
                    for (int iy = 0; iy < 4; ++iy) {
                        for (int ix = 0; ix < 4; ++ix) {
                            d[iy * 4 + ix] = image[(n * C + c) * sizeI +
                                                   (y * 2 + iy) * inWidth + (x * 2 + ix)];
                        }
                    }
                    sgemm(&B_T[0][0], d, tmp_v, 4, 4, 4);
                    sgemm(tmp_v, &B[0][0], v, 4, 4, 4);
                    int b = ((n * outHeight / 2) + y) * outWidth / 2 + x;
                    for (int xi = 0; xi < 4; ++xi) {
                        for (int nu = 0; nu < 4; ++nu) {
                            V[((long)(xi * 4 + nu) * C + c) * P + b] = v[xi * 4 + nu];
                        }
                    }
                }
            }
        }
    }

    // M[xi, nu, :, :] = U[xi, nu, :, :] * V[xi, nu, :, :]
    for (int xi = 0; xi < 4; ++xi) {
        for (int nu = 0; nu < 4; ++nu) {
            float *M_ptr = M + (long)(xi * 4 + nu) * K * P;
            float *U_ptr = U + (long)(xi * 4 + nu) * K * C;
            float *V_ptr = V + (long)(xi * 4 + nu) * C * P;
            sgemm_parallel(U_ptr, V_ptr, M_ptr, K, C, P);
        }
    }

    // Y = A_T * m * A
#pragma omp parallel for collapse(2)
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            float *mm = (float *)malloc(outHeight / 2 * outWidth / 2 * 16 * sizeof(float));
            float *tmp_m = (float *)malloc(outHeight / 2 * outWidth / 2 * 8 * sizeof(float));
            float *temp_out = (float *)malloc(outHeight / 2 * outWidth / 2 * 4 * sizeof(float));
            for (long xi = 0; xi < 4; ++xi) {
                for (long nu = 0; nu < 4; ++nu) {
                    for (int y = 0; y < outHeight / 2; ++y) {
                        for (int x = 0; x < outWidth / 2; ++x) {
                            int b = (n * outHeight / 2 + y) * outWidth / 2 + x;
                            mm[(y * outWidth / 2 + x) * 16 + xi * 4 + nu] = M[((xi * 4 + nu) * K + k) * P + b];
                        }
                    }
                }
            }
            for (int y = 0; y < outHeight / 2; ++y) {
                for (int x = 0; x < outWidth / 2; ++x) {
                    sgemm(&A_T[0][0], &mm[(y * outWidth / 2 + x) * 16], &tmp_m[(y * outWidth / 2 + x) * 8], 2, 4, 4);
                    sgemm(&tmp_m[(y * outWidth / 2 + x) * 8], &A[0][0], &temp_out[(y * outWidth / 2 + x) * 4], 2, 4, 2);
                    for (int i = 0; i < 2; ++i) {
                        for (int j = 0; j < 2; ++j) {
                            out[((long)(n * K + k) * outHeight + y * 2 + i) * outWidth + x * 2 +
                                j] = temp_out[(y * outWidth / 2 + x) * 4 + i * 2 + j];
                        }
                    }
                }
            }
            free(mm);
            free(tmp_m);
            free(temp_out);
        }
    }
}