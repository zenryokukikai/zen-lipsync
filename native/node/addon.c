// ZEN-LipSync native frame model for Node (N-API, linux-x64), two kinds:
// - float32 (native/x86/lipf.c, the WebGPU frame model's arithmetic): AVX-512F > AVX2+FMA, picked at run time. Any
//   number of models; each run() may be called from several threads.
//   JS: a.createF32(Float32Array blob, kernel?: 'auto'|'avx512'|'avx2') -> { model, kernel };
//       a.runF32(model, feat, z, out); a.blobSizeF32() -> floats the blob must have.
// - int8 (frame.lip8): AVX-512 VNNI (native/x86/lip8.c) > AVX2 > SSE4.1 (cpu/lip8w.c through shim/). One model per
//   process (the weights are global to the library); init() again with the same bytes is allowed, other bytes refused.
//   JS: a.init(Uint8Array weights, kernel?: 'auto'|'avx512'|'avx2'|'sse41') -> kernel; a.run(feat, z, out).
// feat Float32Array [28160], z [640], out [168960]; run calls are synchronous (a few ms).
#include <node_api.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

int lip8_avx512_init(const uint8_t *w, int n); void lip8_avx512_run(const float *feat, const float *z, float *out);
int lip8_avx2_init_impl(const uint8_t *w, int n); void lip8_avx2_run_impl(const float *feat, const float *z, float *out);
int lip8_sse41_init_impl(const uint8_t *w, int n); void lip8_sse41_run_impl(const float *feat, const float *z, float *out);

typedef struct LipF LipF;
LipF *lipf_avx512_create(const float *blob, size_t n); int lipf_avx512_run(const LipF *m, const float *feat, const float *z, float *out);
void lipf_avx512_destroy(LipF *m); size_t lipf_avx512_blob_size(void);
LipF *lipf_avx2_create(const float *blob, size_t n); int lipf_avx2_run(const LipF *m, const float *feat, const float *z, float *out);
void lipf_avx2_destroy(LipF *m);

static pthread_mutex_t mu = PTHREAD_MUTEX_INITIALIZER;
static int kernel = 0;  // 0 none, 1 avx512, 2 avx2, 3 sse41
static uint64_t whash;  // FNV-1a of the weights loaded
static const char *NAMES[] = { "", "avx512", "avx2", "sse41" };

static uint64_t fnv(const uint8_t *p, size_t n) { uint64_t h = 1469598103934665603ULL; for (size_t i = 0; i < n; i++) { h ^= p[i]; h *= 1099511628211ULL; } return h; }
static int supports(int k) {
  __builtin_cpu_init();
  if (k == 1) return __builtin_cpu_supports("avx512f") && __builtin_cpu_supports("avx512bw") && __builtin_cpu_supports("avx512vl")
    && __builtin_cpu_supports("avx512vnni") && __builtin_cpu_supports("avx512vbmi");
  if (k == 2) return __builtin_cpu_supports("avx2");
  if (k == 3) return __builtin_cpu_supports("sse4.1");
  return 0;
}

#define THROW(msg) do { napi_throw_error(env, NULL, msg); return NULL; } while (0)
#define CHECK(x) do { if ((x) != napi_ok) THROW("zen-lipsync native: N-API call failed"); } while (0)

static napi_value js_init(napi_env env, napi_callback_info info) {
  size_t argc = 2; napi_value argv[2]; CHECK(napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  void *data; size_t len; napi_typedarray_type t; napi_value ab; size_t off;
  if (argc < 1 || napi_get_typedarray_info(env, argv[0], &t, &len, &data, &ab, &off) != napi_ok || t != napi_uint8_array) THROW("init(weights: Uint8Array, kernel?)");
  char want[16] = "auto"; size_t wl;
  if (argc > 1) { napi_valuetype vt; napi_typeof(env, argv[1], &vt); if (vt == napi_string) napi_get_value_string_utf8(env, argv[1], want, sizeof want, &wl); }
  int k = 0;
  if (!strcmp(want, "auto")) { for (k = 1; k <= 3 && !supports(k); k++); if (k > 3) THROW("this CPU has neither AVX-512 VNNI, AVX2 nor SSE4.1"); }
  else { for (k = 1; k <= 3 && strcmp(want, NAMES[k]); k++); if (k > 3) THROW("kernel must be auto, avx512, avx2 or sse41"); if (!supports(k)) THROW("this CPU does not support the kernel asked for"); }
  uint64_t h = fnv(data, len);
  pthread_mutex_lock(&mu);
  if (kernel) {
    int same = h == whash && k == kernel; pthread_mutex_unlock(&mu);
    if (!same) THROW("one native frame model per process: already loaded with other weights or another kernel");
  } else {
    int r = k == 1 ? lip8_avx512_init(data, (int)len) : k == 2 ? lip8_avx2_init_impl(data, (int)len) : lip8_sse41_init_impl(data, (int)len);
    if (r) { pthread_mutex_unlock(&mu); THROW("frame.lip8: invalid weights file"); }
    kernel = k; whash = h; pthread_mutex_unlock(&mu);
  }
  napi_value s; CHECK(napi_create_string_utf8(env, NAMES[k], NAPI_AUTO_LENGTH, &s)); return s;
}

static int f32(napi_env env, napi_value v, float **p, size_t n) {
  void *data; size_t len; napi_typedarray_type t; napi_value ab; size_t off;
  if (napi_get_typedarray_info(env, v, &t, &len, &data, &ab, &off) != napi_ok || t != napi_float32_array || len != n) return 0;
  *p = data; return 1;
}

static napi_value js_run(napi_env env, napi_callback_info info) {
  size_t argc = 3; napi_value argv[3]; CHECK(napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  float *feat, *z, *out;
  if (argc < 3 || !f32(env, argv[0], &feat, 32 * 20 * 44) || !f32(env, argv[1], &z, 640) || !f32(env, argv[2], &out, 3 * 160 * 352)) THROW("run(feat: Float32Array(28160), z: Float32Array(640), out: Float32Array(168960))");
  if (!kernel) THROW("init() first");
  if (kernel == 1) lip8_avx512_run(feat, z, out);  // per-thread scratch: no lock
  else { pthread_mutex_lock(&mu); if (kernel == 2) lip8_avx2_run_impl(feat, z, out); else lip8_sse41_run_impl(feat, z, out); pthread_mutex_unlock(&mu); }
  return NULL;
}

static napi_value js_supported(napi_env env, napi_callback_info info) {
  (void)info; napi_value arr; CHECK(napi_create_array(env, &arr)); uint32_t i = 0;
  for (int k = 1; k <= 3; k++) if (supports(k)) { napi_value s; CHECK(napi_create_string_utf8(env, NAMES[k], NAPI_AUTO_LENGTH, &s)); CHECK(napi_set_element(env, arr, i++, s)); }
  return arr;
}

static int supports_f32(int k) {
  __builtin_cpu_init();
  if (k == 1) return __builtin_cpu_supports("avx512f");
  if (k == 2) return __builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma");
  return 0;
}
typedef struct { LipF *m; int k; } F32Model;
static void f32_free(napi_env env, void *data, void *hint) {
  (void)env; (void)hint; F32Model *h = data;
  if (h->k == 1) lipf_avx512_destroy(h->m); else lipf_avx2_destroy(h->m);
  free(h);
}

static napi_value js_create_f32(napi_env env, napi_callback_info info) {
  size_t argc = 2; napi_value argv[2]; CHECK(napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  float *blob; void *data; size_t len; napi_typedarray_type t; napi_value ab; size_t off;
  if (argc < 1 || napi_get_typedarray_info(env, argv[0], &t, &len, &data, &ab, &off) != napi_ok || t != napi_float32_array) THROW("createF32(blob: Float32Array, kernel?)");
  blob = data;
  char want[16] = "auto"; size_t wl;
  if (argc > 1) { napi_valuetype vt; napi_typeof(env, argv[1], &vt); if (vt == napi_string) napi_get_value_string_utf8(env, argv[1], want, sizeof want, &wl); }
  int k = 0;
  if (!strcmp(want, "auto")) { for (k = 1; k <= 2 && !supports_f32(k); k++); if (k > 2) THROW("this CPU has neither AVX-512F nor AVX2 with FMA"); }
  else { k = !strcmp(want, "avx512") ? 1 : !strcmp(want, "avx2") ? 2 : 0; if (!k) THROW("float32 kernel must be auto, avx512 or avx2"); if (!supports_f32(k)) THROW("this CPU does not support the kernel asked for"); }
  F32Model *h = malloc(sizeof *h); if (!h) THROW("out of memory");
  h->k = k; h->m = k == 1 ? lipf_avx512_create(blob, len) : lipf_avx2_create(blob, len);
  if (!h->m) { free(h); THROW("frame model weights: wrong size, or out of memory"); }
  napi_value obj, ext, ks;
  CHECK(napi_create_object(env, &obj));
  if (napi_create_external(env, h, f32_free, NULL, &ext) != napi_ok) { f32_free(env, h, NULL); THROW("zen-lipsync native: N-API call failed"); }
  CHECK(napi_create_string_utf8(env, NAMES[k], NAPI_AUTO_LENGTH, &ks));
  CHECK(napi_set_named_property(env, obj, "model", ext)); CHECK(napi_set_named_property(env, obj, "kernel", ks));
  return obj;
}

static napi_value js_run_f32(napi_env env, napi_callback_info info) {
  size_t argc = 4; napi_value argv[4]; CHECK(napi_get_cb_info(env, info, &argc, argv, NULL, NULL));
  F32Model *h; float *feat, *z, *out;
  if (argc < 4 || napi_get_value_external(env, argv[0], (void **)&h) != napi_ok || !h
      || !f32(env, argv[1], &feat, 32 * 20 * 44) || !f32(env, argv[2], &z, 640) || !f32(env, argv[3], &out, 3 * 160 * 352)) THROW("runF32(model, feat: Float32Array(28160), z: Float32Array(640), out: Float32Array(168960))");
  int r = h->k == 1 ? lipf_avx512_run(h->m, feat, z, out) : lipf_avx2_run(h->m, feat, z, out);
  if (r) THROW("out of memory");
  return NULL;
}

static napi_value js_blob_size_f32(napi_env env, napi_callback_info info) {
  (void)info; napi_value v; CHECK(napi_create_double(env, (double)lipf_avx512_blob_size(), &v)); return v;
}

NAPI_MODULE_INIT() {
  napi_property_descriptor d[] = {
    { "init", NULL, js_init, NULL, NULL, NULL, napi_default, NULL },
    { "run", NULL, js_run, NULL, NULL, NULL, napi_default, NULL },
    { "supported", NULL, js_supported, NULL, NULL, NULL, napi_default, NULL },
    { "createF32", NULL, js_create_f32, NULL, NULL, NULL, napi_default, NULL },
    { "runF32", NULL, js_run_f32, NULL, NULL, NULL, napi_default, NULL },
    { "blobSizeF32", NULL, js_blob_size_f32, NULL, NULL, NULL, napi_default, NULL },
  };
  napi_define_properties(env, exports, 6, d);
  return exports;
}
