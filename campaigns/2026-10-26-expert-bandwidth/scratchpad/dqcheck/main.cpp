#include "gen.inc"
#include <cstdlib>
static uint32_t ep(uint32_t q, int p) { return p == 0 ? ext_prop<0>(q) : p == 1 ? ext_prop<1>(q) : p == 2 ? ext_prop<2>(q) : ext_prop<3>(q); }
static uint32_t em(uint32_t q, int p) { return p == 0 ? ext_master<0>(q) : p == 1 ? ext_master<1>(q) : p == 2 ? ext_master<2>(q) : ext_master<3>(q); }
int main() {
  long bad = 0;
  // (1) rotl32 is a rotate for every s used, every q bit
  const int S[4] = {4, 6, 10, 14};
  for (int si = 0; si < 4; ++si) for (int b = 0; b < 32; ++b) {
    uint32_t q = 1u << b, r = rotl32(q, S[si]);
    if (r != (1u << ((b + S[si]) & 31))) { ++bad; printf("rotl bad s=%d b=%d\n", S[si], b); }
  }
  // (2) place_prop == LUT index 2 for all pair/half/code; LUT entries are bit permutations
  uint32_t pos2[4][2][4], pos1[4][2][4], all = 0;
  for (int p = 0; p < 4; ++p) for (int h = 0; h < 2; ++h) {
    for (int c = 0; c < 16; ++c) {
      if ((int64_t)place_prop(c, p, h) != LUT[2][p][h][c]) { ++bad; printf("place_prop != LUT p%d h%d c%d\n", p, h, c); }
      if (LUT[2][p][h][c] >> 32) { ++bad; printf("LUT >32 bits\n"); }
    }
    for (int k = 0; k < 4; ++k) {
      int64_t v2 = LUT[2][p][h][1 << k], v1 = LUT[1][p][h][1 << k];
      if (__builtin_popcountll(v2) != 1 || __builtin_popcountll(v1) != 1) { ++bad; printf("not single bit\n"); }
      pos2[p][h][k] = __builtin_ctzll(v2); pos1[p][h][k] = __builtin_ctzll(v1);
      if (all & (1u << pos2[p][h][k])) { ++bad; printf("overlap\n"); }
      all |= 1u << pos2[p][h][k];
    }
    for (int c = 0; c < 16; ++c) {  // linearity: LUT[c] = OR of its bits
      int64_t v = 0; for (int k = 0; k < 4; ++k) if (c >> k & 1) v |= LUT[2][p][h][1 << k];
      if (v != LUT[2][p][h][c]) { ++bad; printf("nonlinear\n"); }
    }
  }
  if (all != 0xFFFFFFFFu) { ++bad; printf("coverage %08x\n", all); }
  // (3) byte tables: prop word -> master word (the inverse permutation through the code bits)
  static uint32_t T[4][256];
  for (int by = 0; by < 4; ++by) for (int v = 0; v < 256; ++v) {
    uint32_t m = 0;
    for (int p = 0; p < 4; ++p) for (int h = 0; h < 2; ++h) for (int k = 0; k < 4; ++k) {
      int b = pos2[p][h][k];
      if (b / 8 == by && (v >> (b & 7) & 1)) m |= 1u << pos1[p][h][k];
    }
    T[by][v] = m;
  }
  // (4) every 32-bit prop word: all four extractions equal master's on the corresponding master word
  long words = 0;
  for (uint64_t w64 = 0; w64 <= 0xFFFFFFFFull; ++w64) {
    const uint32_t w = (uint32_t)w64;
    const uint32_t m = T[0][w & 255] | T[1][(w >> 8) & 255] | T[2][(w >> 16) & 255] | T[3][w >> 24];
    const uint32_t d = (ext_prop<0>(w) ^ ext_master<0>(m)) | (ext_prop<1>(w) ^ ext_master<1>(m)) |
                       (ext_prop<2>(w) ^ ext_master<2>(m)) | (ext_prop<3>(w) ^ ext_master<3>(m));
    if (d) { if (bad < 10) printf("word %08x master %08x mismatch\n", w, m); ++bad; }
    ++words;
  }
  // (5) _ea_pack-style packing (OR of LUT[2] entries over 8 codes) vs master packing, random words
  srand(1);
  for (int it = 0; it < 1000000; ++it) {
    int code[4][2]; uint32_t q2 = 0, q1 = 0;
    for (int p = 0; p < 4; ++p) for (int h = 0; h < 2; ++h) { code[p][h] = rand() & 15; q2 |= (uint32_t)LUT[2][p][h][code[p][h]]; q1 |= (uint32_t)LUT[1][p][h][code[p][h]]; }
    for (int p = 0; p < 4; ++p) if (ep(q2, p) != em(q1, p)) { ++bad; break; }
  }
  printf("pos2 (m,e0,e1,s) per pair/half:");
  for (int p = 0; p < 4; ++p) for (int h = 0; h < 2; ++h) printf(" p%dh%d=%u,%u,%u,%u", p, h, pos2[p][h][0], pos2[p][h][1], pos2[p][h][2], pos2[p][h][3]);
  printf("\nexhaustive words %ld, mismatches/bad %ld\n", words, bad);
  return bad != 0;
}
