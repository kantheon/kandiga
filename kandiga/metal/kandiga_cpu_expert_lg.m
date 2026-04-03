/*
 * kandiga_cpu_expert_lg.m -- CPU expert MLP for large MoE models (122B/397B).
 *
 * Dynamic dimensions read from packed binary header at init time.
 * Uses pre-allocated scratch buffers — zero malloc in hot path.
 * GCD parallel pread, serial compute (safe alongside MLX Metal).
 *
 * Build:
 *   clang -shared -o libkandiga_cpu_expert_lg.dylib kandiga_cpu_expert_lg.m \
 *         -framework Foundation -fobjc-arc -O2 -march=native
 */

#import <Foundation/Foundation.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <fcntl.h>
#include <unistd.h>
#include <errno.h>
#include <dispatch/dispatch.h>
#include <pthread.h>

#ifdef __ARM_NEON__
#include <arm_neon.h>
#endif

#include "kandiga_cpu_expert.h"

#define MAX_EXPERTS       16
#define HEADER_SIZE       4096UL
#define GROUP_SIZE        64

typedef struct {
    int*    layer_fds;
    int     num_layers;
    char*   expert_bufs[MAX_EXPERTS];
    float*  scratch_bufs[MAX_EXPERTS];

    float*  x_f32_buf;        /* pre-allocated: hidden_size floats */
    float*  out_f32_buf;      /* pre-allocated: MAX_EXPERTS * hidden_size floats */

    int     num_experts_total;
    size_t  expert_size;
    int     hidden_size;
    int     expert_dim;
    int     bits;             /* 3 or 4 — auto-detected from expert_size */

    size_t  gate_w_off, gate_s_off, gate_b_off;
    size_t  up_w_off,   up_s_off,   up_b_off;
    size_t  down_w_off, down_s_off, down_b_off;
} Engine;

/* helpers */
static inline float bf16_to_f32(uint16_t bf) {
    uint32_t bits = (uint32_t)bf << 16; float f; memcpy(&f, &bits, 4); return f;
}
static inline uint16_t f32_to_f16(float f) {
    _Float16 h = (_Float16)f; uint16_t r; memcpy(&r, &h, sizeof(r)); return r;
}
static inline float f16_to_f32(uint16_t h) {
    _Float16 f16; memcpy(&f16, &h, sizeof(f16)); return (float)f16;
}

/* parse header */
static int parse_header(int fd, Engine* e) {
    char hdr[HEADER_SIZE];
    if (pread(fd, hdr, HEADER_SIZE, 0) != (ssize_t)HEADER_SIZE) return -1;
    if (memcmp(hdr, "BKEX", 4) != 0) return -1;

    uint32_t ne, nt; uint64_t es;
    memcpy(&ne, hdr+8, 4); memcpy(&es, hdr+12, 8); memcpy(&nt, hdr+20, 4);
    e->num_experts_total = (int)ne;
    e->expert_size = (size_t)es;

    int pos = 24;
    for (uint32_t t = 0; t < nt && t < 9; t++) {
        uint8_t nlen = (uint8_t)hdr[pos]; pos++;
        char nm[25] = {0}; memcpy(nm, hdr+pos, nlen < 24 ? nlen : 24); pos += 24;
        uint32_t off, nb, d0, d1;
        memcpy(&off, hdr+pos, 4); pos += 4;
        memcpy(&nb,  hdr+pos, 4); pos += 4;
        memcpy(&d0,  hdr+pos, 4); pos += 4;
        memcpy(&d1,  hdr+pos, 4); pos += 4;
        pos++; /* dtype */

        if      (strstr(nm,"gate_proj.weight")) { e->gate_w_off=off; e->expert_dim=(int)d0; }
        else if (strstr(nm,"gate_proj.scales")) { e->gate_s_off=off; e->hidden_size=(int)(d1*GROUP_SIZE); }
        else if (strstr(nm,"gate_proj.biases")) { e->gate_b_off=off; }
        else if (strstr(nm,"up_proj.weight"))   { e->up_w_off=off; }
        else if (strstr(nm,"up_proj.scales"))   { e->up_s_off=off; }
        else if (strstr(nm,"up_proj.biases"))   { e->up_b_off=off; }
        else if (strstr(nm,"down_proj.weight")) { e->down_w_off=off; }
        else if (strstr(nm,"down_proj.scales")) { e->down_s_off=off; }
        else if (strstr(nm,"down_proj.biases")) { e->down_b_off=off; }
    }
    /* Detect 3-bit vs 4-bit from weight size vs expected */
    /* 4-bit: gate_weight_bytes = expert_dim * hidden_size / 8 * 4 = expert_dim * hidden_size / 2 */
    /* 3-bit: gate_weight_bytes = expert_dim * hidden_size * 3 / 8 */
    size_t expected_4bit = (size_t)e->expert_dim * (size_t)e->hidden_size / 2;
    size_t actual_w_bytes = e->gate_s_off - e->gate_w_off;
    if (actual_w_bytes < expected_4bit) {
        e->bits = 3;
    } else {
        e->bits = 4;
    }

    fprintf(stderr, "[kandiga-lg] h=%d e=%d esz=%zu n=%d bits=%d\n",
            e->hidden_size, e->expert_dim, e->expert_size, e->num_experts_total, e->bits);
    return 0;
}

/* NEON 3-bit matvec — MLX packing: 3 bytes → 8 values */
static void matvec3(
    const uint8_t* w, const uint16_t* s, const uint16_t* b,
    const float* x, float* out, int od, int id, int gs
) {
    int bpr = id*3/8, ng = id/gs, ppg = gs/8; /* packs per group */
    for (int r = 0; r < od; r++) {
        const uint8_t*  wr = w + r*bpr;
        const uint16_t* sr = s + r*ng;
        const uint16_t* br = b + r*ng;
#ifdef __ARM_NEON__
        float32x4_t acc = vdupq_n_f32(0);
        for (int g = 0; g < ng; g++) {
            float sc = bf16_to_f32(sr[g]), bi = bf16_to_f32(br[g]);
            float32x4_t vs = vdupq_n_f32(sc), vb = vdupq_n_f32(bi);
            const uint8_t* wp = wr + g*ppg*3;
            int xb = g*gs;
            for (int p = 0; p < ppg; p += 2) {
                const uint8_t* a = wp + p*3;
                const uint8_t* c = a + 3;
                const float* xp = x + xb + p*8;
                /* Unpack 2×3 bytes = 16 values */
                float32x4_t n0 = {(float)(a[0]&7),(float)((a[0]>>3)&7),
                    (float)(((a[0]>>6)&3)|((a[1]&1)<<2)),(float)((a[1]>>1)&7)};
                float32x4_t n1 = {(float)((a[1]>>4)&7),
                    (float)(((a[1]>>7)&1)|((a[2]&3)<<1)),
                    (float)((a[2]>>2)&7),(float)((a[2]>>5)&7)};
                float32x4_t n2 = {(float)(c[0]&7),(float)((c[0]>>3)&7),
                    (float)(((c[0]>>6)&3)|((c[1]&1)<<2)),(float)((c[1]>>1)&7)};
                float32x4_t n3 = {(float)((c[1]>>4)&7),
                    (float)(((c[1]>>7)&1)|((c[2]&3)<<1)),
                    (float)((c[2]>>2)&7),(float)((c[2]>>5)&7)};
                acc=vfmaq_f32(acc,vfmaq_f32(vb,n0,vs),vld1q_f32(xp));
                acc=vfmaq_f32(acc,vfmaq_f32(vb,n1,vs),vld1q_f32(xp+4));
                acc=vfmaq_f32(acc,vfmaq_f32(vb,n2,vs),vld1q_f32(xp+8));
                acc=vfmaq_f32(acc,vfmaq_f32(vb,n3,vs),vld1q_f32(xp+12));
            }
        }
        float32x2_t sp=vadd_f32(vget_low_f32(acc),vget_high_f32(acc));
        out[r]=vget_lane_f32(vpadd_f32(sp,sp),0);
#else
        float a=0;
        for(int g=0;g<ng;g++){float sc=bf16_to_f32(sr[g]),bi=bf16_to_f32(br[g]);
        const uint8_t* wp=wr+g*ppg*3;
        for(int p=0;p<ppg;p++){const uint8_t* b0=wp+p*3;int xi=g*gs+p*8;
        a+=((float)(b0[0]&7)*sc+bi)*x[xi]+((float)((b0[0]>>3)&7)*sc+bi)*x[xi+1]
          +((float)(((b0[0]>>6)&3)|((b0[1]&1)<<2))*sc+bi)*x[xi+2]+((float)((b0[1]>>1)&7)*sc+bi)*x[xi+3]
          +((float)((b0[1]>>4)&7)*sc+bi)*x[xi+4]+((float)(((b0[1]>>7)&1)|((b0[2]&3)<<1))*sc+bi)*x[xi+5]
          +((float)((b0[2]>>2)&7)*sc+bi)*x[xi+6]+((float)((b0[2]>>5)&7)*sc+bi)*x[xi+7];}}
        out[r]=a;
#endif
    }
}

/* NEON matvec */
static void matvec4(
    const uint32_t* w, const uint16_t* s, const uint16_t* b,
    const float* x, float* out, int od, int id, int gs
) {
    int ppr = id/8, ng = id/gs, ppg = gs/8;
    for (int r = 0; r < od; r++) {
        const uint32_t* wr = w + r*ppr;
        const uint16_t* sr = s + r*ng;
        const uint16_t* br = b + r*ng;
#ifdef __ARM_NEON__
        float32x4_t acc = vdupq_n_f32(0);
        for (int g = 0; g < ng; g++) {
            float sc = bf16_to_f32(sr[g]), bi = bf16_to_f32(br[g]);
            float32x4_t vs = vdupq_n_f32(sc), vb = vdupq_n_f32(bi);
            for (int p = 0; p < ppg; p += 4) {
                const uint32_t* wp = wr + g*ppg + p;
                uint32_t p0=wp[0],p1=wp[1],p2=wp[2],p3=wp[3];
                int xb = g*gs + p*8;
                const float* xp = x+xb;
                #define N4(pk,sh) {(float)((pk>>sh)&0xF),(float)((pk>>(sh+4))&0xF),(float)((pk>>(sh+8))&0xF),(float)((pk>>(sh+12))&0xF)}
                float32x4_t n0=N4(p0,0),n1=N4(p0,16),n2=N4(p1,0),n3=N4(p1,16);
                float32x4_t n4=N4(p2,0),n5=N4(p2,16),n6=N4(p3,0),n7=N4(p3,16);
                #undef N4
                acc=vfmaq_f32(acc,vfmaq_f32(vb,n0,vs),vld1q_f32(xp));
                acc=vfmaq_f32(acc,vfmaq_f32(vb,n1,vs),vld1q_f32(xp+4));
                acc=vfmaq_f32(acc,vfmaq_f32(vb,n2,vs),vld1q_f32(xp+8));
                acc=vfmaq_f32(acc,vfmaq_f32(vb,n3,vs),vld1q_f32(xp+12));
                acc=vfmaq_f32(acc,vfmaq_f32(vb,n4,vs),vld1q_f32(xp+16));
                acc=vfmaq_f32(acc,vfmaq_f32(vb,n5,vs),vld1q_f32(xp+20));
                acc=vfmaq_f32(acc,vfmaq_f32(vb,n6,vs),vld1q_f32(xp+24));
                acc=vfmaq_f32(acc,vfmaq_f32(vb,n7,vs),vld1q_f32(xp+28));
            }
        }
        float32x2_t sp=vadd_f32(vget_low_f32(acc),vget_high_f32(acc));
        out[r]=vget_lane_f32(vpadd_f32(sp,sp),0);
#else
        float a=0;
        for(int g=0;g<ng;g++){float sc=bf16_to_f32(sr[g]),bi=bf16_to_f32(br[g]);
        for(int p=0;p<ppg;p++){uint32_t pk=wr[g*ppg+p];int xb=g*gs+p*8;
        for(int n=0;n<8;n++){a+=((float)((pk>>(n*4))&0xF)*sc+bi)*x[xb+n];}}}
        out[r]=a;
#endif
    }
}

/* single expert MLP — scratch pre-allocated, zero malloc */
static int expert_mlp(const Engine* e, const char* data,
                      const float* x, float* out, float* scratch) {
    int h=e->hidden_size, ed=e->expert_dim;
    float *go=scratch, *uo=scratch+ed, *act=scratch+2*ed;

    if (e->bits == 3) {
        matvec3((const uint8_t*)(data+e->gate_w_off),(const uint16_t*)(data+e->gate_s_off),
                (const uint16_t*)(data+e->gate_b_off), x, go, ed, h, GROUP_SIZE);
        matvec3((const uint8_t*)(data+e->up_w_off),(const uint16_t*)(data+e->up_s_off),
                (const uint16_t*)(data+e->up_b_off), x, uo, ed, h, GROUP_SIZE);
    } else {
        matvec4((const uint32_t*)(data+e->gate_w_off),(const uint16_t*)(data+e->gate_s_off),
                (const uint16_t*)(data+e->gate_b_off), x, go, ed, h, GROUP_SIZE);
        matvec4((const uint32_t*)(data+e->up_w_off),(const uint16_t*)(data+e->up_s_off),
                (const uint16_t*)(data+e->up_b_off), x, uo, ed, h, GROUP_SIZE);
    }

    for(int i=0;i<ed;i+=4){
        float s0=go[i]/(1+expf(-go[i])), s1=go[i+1]/(1+expf(-go[i+1]));
        float s2=go[i+2]/(1+expf(-go[i+2])), s3=go[i+3]/(1+expf(-go[i+3]));
        act[i]=s0*uo[i]; act[i+1]=s1*uo[i+1]; act[i+2]=s2*uo[i+2]; act[i+3]=s3*uo[i+3];
    }

    if (e->bits == 3) {
        matvec3((const uint8_t*)(data+e->down_w_off),(const uint16_t*)(data+e->down_s_off),
                (const uint16_t*)(data+e->down_b_off), act, out, h, ed, GROUP_SIZE);
    } else {
        matvec4((const uint32_t*)(data+e->down_w_off),(const uint16_t*)(data+e->down_s_off),
                (const uint16_t*)(data+e->down_b_off), act, out, h, ed, GROUP_SIZE);
    }
    return 0;
}

/* init */
void* bakan_cpu_expert_init(const char* dir, int nl) {
    Engine* e = (Engine*)calloc(1, sizeof(Engine));
    if(!e) return NULL;
    e->num_layers = nl;
    e->layer_fds = (int*)calloc(nl, sizeof(int));
    if(!e->layer_fds){free(e);return NULL;}
    for(int i=0;i<nl;i++) e->layer_fds[i]=-1;

    for(int i=0;i<nl;i++){
        char p[1024]; snprintf(p,sizeof(p),"%s/layer_%02d.bin",dir,i);
        int fd=open(p,O_RDONLY);
        if(fd<0){fprintf(stderr,"[kandiga-lg] open %s: %s\n",p,strerror(errno));
                  bakan_cpu_expert_destroy(e);return NULL;}
        e->layer_fds[i]=fd;
    }

    if(parse_header(e->layer_fds[0],e)!=0){bakan_cpu_expert_destroy(e);return NULL;}

    for(int i=0;i<MAX_EXPERTS;i++){
        e->expert_bufs[i]=(char*)malloc(e->expert_size);
        e->scratch_bufs[i]=(float*)calloc(3*(size_t)e->expert_dim,sizeof(float));
        if(!e->expert_bufs[i]||!e->scratch_bufs[i]){bakan_cpu_expert_destroy(e);return NULL;}
    }
    /* Pre-allocate f16 conversion buffers (zero malloc in hot path) */
    e->x_f32_buf = (float*)calloc(e->hidden_size, sizeof(float));
    e->out_f32_buf = (float*)calloc(MAX_EXPERTS * (size_t)e->hidden_size, sizeof(float));
    if (!e->x_f32_buf || !e->out_f32_buf) {
        bakan_cpu_expert_destroy(e); return NULL;
    }

    fprintf(stderr,"[kandiga-lg] Ready: %d layers, h=%d, e=%d, %zuKB/expert\n",
            nl,e->hidden_size,e->expert_dim,e->expert_size/1024);
    return e;
}

/* mlp */
/* pthread worker for parallel expert compute (no ObjC autorelease pool) */
typedef struct {
    const Engine* eng;
    const char* data;
    const float* x;
    float* out;
    float* scratch;
    int result;
} ExpertWork;

static void* _expert_thread(void* arg) {
    ExpertWork* w = (ExpertWork*)arg;
    w->result = expert_mlp(w->eng, w->data, w->x, w->out, w->scratch);
    return NULL;
}

int bakan_cpu_expert_mlp(void* ptr, int li, const float* x,
                         const int32_t* idx, int K, float* out) {
    Engine* e=(Engine*)ptr;
    if(!e||li<0||li>=e->num_layers||K<=0||K>MAX_EXPERTS) return -1;

    int fd=e->layer_fds[li]; size_t esz=e->expert_size;

    /* parallel pread (GCD is safe for pure I/O — no ObjC involved) */
    __block int ioerr=0;
    dispatch_group_t g=dispatch_group_create();
    dispatch_queue_t q=dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE,0);
    for(int k=0;k<K;k++){
        int ei=idx[k]; char* buf=e->expert_bufs[k];
        dispatch_group_async(g,q,^{
            off_t o=(off_t)HEADER_SIZE+(off_t)ei*(off_t)esz;
            if(pread(fd,buf,esz,o)!=(ssize_t)esz) ioerr=1;
        });
    }
    dispatch_group_wait(g,DISPATCH_TIME_FOREVER);
    if(ioerr) return -1;

    /* parallel compute via GCD (scratch pre-allocated, zero malloc in block) */
    int h=e->hidden_size;
    __block int cerr=0;
    dispatch_group_t cg=dispatch_group_create();
    dispatch_queue_t cq=dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE,0);
    for(int k=0;k<K;k++){
        const char* data=e->expert_bufs[k];
        float* dst=out+k*h;
        float* scr=e->scratch_bufs[k];
        const Engine* eng=e;
        dispatch_group_async(cg,cq,^{
            if(expert_mlp(eng,data,x,dst,scr)!=0) cerr=1;
        });
    }
    dispatch_group_wait(cg,DISPATCH_TIME_FOREVER);
    if(cerr) return -1;
    return 0;
}

/* f16 — uses pre-allocated buffers, zero malloc */
int bakan_cpu_expert_mlp_f16(void* ptr, int li, const void* xf16,
                              const int32_t* idx, int K, void* of16) {
    Engine* e=(Engine*)ptr; if(!e) return -1;
    int h=e->hidden_size;
    const uint16_t* xh=(const uint16_t*)xf16;
    for(int i=0;i<h;i++) e->x_f32_buf[i]=f16_to_f32(xh[i]);
    int r=bakan_cpu_expert_mlp(ptr,li,e->x_f32_buf,idx,K,e->out_f32_buf);
    if(r==0){uint16_t* oh=(uint16_t*)of16; int n=K*h; for(int i=0;i<n;i++) oh[i]=f32_to_f16(e->out_f32_buf[i]);}
    return r;
}

/* bf16 — uses pre-allocated buffers, zero malloc */
int bakan_cpu_expert_mlp_bf16(void* ptr, int li, const void* xbf,
                               const int32_t* idx, int K, void* of16) {
    Engine* e=(Engine*)ptr; if(!e) return -1;
    int h=e->hidden_size;
    const uint16_t* xr=(const uint16_t*)xbf;
    for(int i=0;i<h;i++) e->x_f32_buf[i]=bf16_to_f32(xr[i]);
    int r=bakan_cpu_expert_mlp(ptr,li,e->x_f32_buf,idx,K,e->out_f32_buf);
    if(r==0){uint16_t* oh=(uint16_t*)of16; int n=K*h; for(int i=0;i<n;i++) oh[i]=f32_to_f16(e->out_f32_buf[i]);}
    return r;
}

/* destroy */
void bakan_cpu_expert_destroy(void* ptr) {
    if(!ptr) return;
    Engine* e=(Engine*)ptr;
    if(e->layer_fds){for(int i=0;i<e->num_layers;i++) if(e->layer_fds[i]>=0) close(e->layer_fds[i]); free(e->layer_fds);}
    for(int i=0;i<MAX_EXPERTS;i++){free(e->expert_bufs[i]);free(e->scratch_bufs[i]);}
    free(e->x_f32_buf); free(e->out_f32_buf);
    free(e); fprintf(stderr,"[kandiga-lg] Destroyed\n");
}

int bakan_cpu_expert_num_layers(void* ptr) {
    return ptr ? ((Engine*)ptr)->num_layers : 0;
}

/* Batch pread for prefill */
int bakan_cpu_expert_pread_batch(void* ptr, int li,
    const int32_t* ids, int num, void* out_buf) {
    Engine* e=(Engine*)ptr;
    if(!e||li<0||li>=e->num_layers) return -1;
    int fd=e->layer_fds[li]; size_t esz=e->expert_size;
    char* out=(char*)out_buf;
    __block int err=0;
    dispatch_group_t g=dispatch_group_create();
    dispatch_queue_t q=dispatch_get_global_queue(QOS_CLASS_USER_INTERACTIVE,0);
    for(int i=0;i<num;i++){
        int eidx=ids[i]; char* dst=out+(size_t)i*esz;
        dispatch_group_async(g,q,^{
            off_t o=(off_t)HEADER_SIZE+(off_t)eidx*(off_t)esz;
            if(pread(fd,dst,esz,o)!=(ssize_t)esz) err=1;
        });
    }
    dispatch_group_wait(g,DISPATCH_TIME_FOREVER);
    return err?-1:0;
}
