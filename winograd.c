#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const float G[4][3] = {{1.0, 0.0, 0.0}, {0.5, 0.5, 0.5}, {0.5, -0.5, 0.5}, {0.0, 0.0, 1.0}};

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
                            out[(ih + i) * N + jh + j] += A[(ih + i) * K + kh + k] * B[(kh + k) * N + jh + j];
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
#pragma omp parallel for collapse(2) private(tmp_u)
    for (int k = 0; k < K; ++k) {
        for (int c = 0; c < C; ++c) {
            const float *filters_ptr = filter + (k * C + c) * sizeF;
            const float *p0 = filters_ptr;
            const float *p1 = filters_ptr + 3;
            const float *p2 = filters_ptr + 6;
            for (int i = 0; i < 4; i++) {
                tmp_u[i * 3 + 0] = p0[0] * G[i][0] + p0[1] * G[i][1] + p0[2] * G[i][2];
                tmp_u[i * 3 + 1] = p1[0] * G[i][0] + p1[1] * G[i][1] + p1[2] * G[i][2];
                tmp_u[i * 3 + 2] = p2[0] * G[i][0] + p2[1] * G[i][1] + p2[2] * G[i][2];
            }
            for (int j = 0; j < 4; j++) {
                const float *tmpp = &tmp_u[j * 3];
                for (int i = 0; i < 4; i++) {
                    U[((i * 4 + j) * K + k) * C + c] = tmpp[0] * G[i][0] + tmpp[1] * G[i][1] + tmpp[2] * G[i][2];
                }
            }
        }
    }

    // V[:, :, c, p] = B_T * image[c, b, :, :] * B
    float tmp_v[16];
    float d[16]; // 4 * 4;
#pragma omp parallel for collapse(2) private(tmp_v, d)
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int y = 0; y < outHeight / 2; ++y) {
                for (int x = 0; x < outWidth / 2; ++x) {
                    const int b = ((n * outHeight / 2) + y) * outWidth / 2 + x;
                    for (int iy = 0; iy < 4; ++iy) {
                        for (int ix = 0; ix < 4; ++ix) {
                            d[iy * 4 + ix] = image[(n * C + c) * sizeI + (y * 2 + iy) * inWidth + (x * 2 + ix)];
                        }
                    }
                    for (int i = 0; i < 4; i++) {
                        tmp_v[0 * 4 + i] = d[i * 4 + 0] - d[i * 4 + 2];
                        tmp_v[1 * 4 + i] = d[i * 4 + 1] + d[i * 4 + 2];
                        tmp_v[2 * 4 + i] = -d[i * 4 + 1] + d[i * 4 + 2];
                        tmp_v[3 * 4 + i] = d[i * 4 + 1] - d[i * 4 + 3];
                    }
                    for (int i = 0; i < 4; i++) {
                        V[((0 * 4 + i) * C + c) * P + b] = tmp_v[i * 4 + 0] - tmp_v[i * 4 + 2];
                        V[((1 * 4 + i) * C + c) * P + b] = tmp_v[i * 4 + 1] + tmp_v[i * 4 + 2];
                        V[((2 * 4 + i) * C + c) * P + b] = -tmp_v[i * 4 + 1] + tmp_v[i * 4 + 2];
                        V[((3 * 4 + i) * C + c) * P + b] = tmp_v[i * 4 + 1] - tmp_v[i * 4 + 3];
                    }
                }
            }
        }
    }

    // M[xi, nu, :, :] = U[xi, nu, :, :] * V[xi, nu, :, :]
    for (int xi = 0; xi < 4; ++xi) {
        for (int nu = 0; nu < 4; ++nu) {
            float *M_ptr = M + (xi * 4 + nu) * K * P;
            float *U_ptr = U + (xi * 4 + nu) * K * C;
            float *V_ptr = V + (xi * 4 + nu) * C * P;
            sgemm_parallel(U_ptr, V_ptr, M_ptr, K, C, P);
        }
    }

    // Y = A_T * m * A
    float mm[16];   // 4 * 4
    float tmp_m[8]; // 2 * 4
#pragma omp parallel for collapse(2) private(mm, tmp_m)
    for (int n = 0; n < N; ++n) {
        for (int k = 0; k < K; ++k) {
            for (int y = 0; y < outHeight / 2; ++y) {
                for (int x = 0; x < outWidth / 2; ++x) {
                    int b = (n * outHeight / 2 + y) * outWidth / 2 + x;
                    for (int xi = 0; xi < 4; ++xi) {
                        for (int nu = 0; nu < 4; ++nu) {
                            mm[xi * 4 + nu] = M[((xi * 4 + nu) * K + k) * P + b];
                        }
                    }
                    for (int i = 0; i < 4; i++) {
                        tmp_m[0 * 4 + i] = mm[i * 4 + 0] + mm[i * 4 + 1] + mm[i * 4 + 2];
                        tmp_m[1 * 4 + i] = mm[i * 4 + 1] - mm[i * 4 + 2] - mm[i * 4 + 3];
                    }
                    for (int i = 0; i < 2; i++) {
                        out[((n * K + k) * outHeight + y * 2 + 0) * outWidth + x * 2 + i] = tmp_m[i * 4 + 0] + tmp_m[i * 4 + 1] + tmp_m[i * 4 + 2];
                        out[((n * K + k) * outHeight + y * 2 + 1) * outWidth + x * 2 + i] = tmp_m[i * 4 + 1] - tmp_m[i * 4 + 2] - tmp_m[i * 4 + 3];
                    }
                }
            }
        }
    }
}
