#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <immintrin.h>

int main() {
  const int N = 8;
  float x[N], y[N], m[N], fx[N], fy[N] , cmp[N];
  float tmpfx[N], tmpfy[N];
  for(int i=0; i<N; i++) {
    x[i] = drand48();
    y[i] = drand48();
    m[i] = drand48();
    fx[i] = fy[i] = 0;
    cmp[i]=i;
  }
  
  __m256 xvec = _mm256_load_ps(x);
  __m256 yvec = _mm256_load_ps(y);
  __m256 mvec = _mm256_load_ps(m);
  __m256 cmpvec = _mm256_load_ps(cmp);

  for(int i=0; i<N; i++) {
      __m256 xivec = _mm256_set1_ps(x[i]);
      __m256 yivec = _mm256_set1_ps(y[i]);
      
      // original code
      // float rx = x[i] - x[j];
      // float ry = y[i] - y[j];
      // float r = std::sqrt(rx * rx + ry * ry);
      // SIMD code
      __m256 rxvec = _mm256_sub_ps(xivec, xvec);
      __m256 ryvec = _mm256_sub_ps(yivec, yvec);
      __m256 rinvvec = _mm256_rsqrt_ps(_mm256_add_ps(_mm256_mul_ps(rxvec, rxvec), _mm256_mul_ps(ryvec, ryvec)));
      
      // original code
      // fx[i] -= rx * m[j] / (r * r * r);
      // fy[i] -= ry * m[j] / (r * r * r);
      // SIMD code
      __m256 rinvcubevec = _mm256_mul_ps(rinvvec, _mm256_mul_ps(rinvvec, rinvvec));
      __m256 fxivec = _mm256_mul_ps(_mm256_mul_ps(rxvec, mvec), rinvcubevec);
      __m256 fyivec = _mm256_mul_ps(_mm256_mul_ps(ryvec, mvec), rinvcubevec);

      // mask
      __m256 ivec = _mm256_set1_ps(i);
      __m256 mask = _mm256_cmp_ps(ivec, cmpvec, _CMP_NEQ_OQ);
      __m256 fxvec = _mm256_blendv_ps(_mm256_setzero_ps(), fxivec, mask);
      __m256 fyvec = _mm256_blendv_ps(_mm256_setzero_ps(), fyivec, mask);

      // reduction
      __m256 fxvecred = _mm256_permute2f128_ps(fxvec, fxvec, 1);
      fxvecred = _mm256_add_ps(fxvecred, fxvec);
      fxvecred = _mm256_hadd_ps(fxvecred, fxvecred);
      fxvecred = _mm256_hadd_ps(fxvecred, fxvecred);
      _mm256_store_ps(tmpfx, fxvecred);

      __m256 fyvecred = _mm256_permute2f128_ps(fyvec, fyvec, 1);
      fyvecred = _mm256_add_ps(fyvecred, fyvec);
      fyvecred = _mm256_hadd_ps(fyvecred, fyvecred);
      fyvecred = _mm256_hadd_ps(fyvecred, fyvecred);
      _mm256_store_ps(tmpfy, fyvecred);

      fx[i] -= tmpfx[0];
      fy[i] -= tmpfy[0];
      
      printf("%d %g %g\n",i,fx[i],fy[i]);
  }
}
