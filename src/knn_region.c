/* One OpenMP team for a whole k-nearest query.
 *
 * The Rust side owns the points, the bins, and each thread's scratch.
 * This file only schedules: static shares, in thread order, so slot t is
 * the same thread for every phase. Nothing here reads an environment
 * variable. Affinity changes only when the caller asks to pin.
 */
#define _GNU_SOURCE

#include <limits.h>
#include <linux/futex.h>
#include <omp.h>
#include <pthread.h>
#include <sched.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/syscall.h>
#include <unistd.h>

typedef void (*lc_knn_phase)(void *ctx, int slot, int nthreads, int64_t begin, int64_t end);

static int g_cpu[256];
static int g_cpu_n;
static int g_pinned_once;

static unsigned long long g_prev_busy[256];
static int g_prev_id[256];
static int g_prev_n;

/* Least-loaded CPUs since the previous sample. The first sample uses the
 * counters since boot; later queries see the delta, so a warmup call
 * teaches the ranking before a timed call. */
static void least_loaded(int nt) {
    FILE *fp = fopen("/proc/stat", "r");
    int n = 0;
    int i;
    int take;
    typedef struct {
        int cpu;
        unsigned long long busy;
    } Row;
    Row rows[256];
    Row scored[256];
    int sn = 0;
    char line[384];

    g_cpu_n = 0;
    if (fp == NULL) {
        return;
    }
    while (fgets(line, (int)sizeof line, fp) != NULL && n < 256) {
        int cpu;
        unsigned long long user, nice, sys, idle, iowait, irq, soft, steal;
        if (sscanf(line, "cpu%d %llu %llu %llu %llu %llu %llu %llu %llu", &cpu, &user, &nice, &sys,
                   &idle, &iowait, &irq, &soft, &steal) == 9) {
            rows[n].cpu = cpu;
            rows[n].busy = user + nice + sys + irq + soft + steal;
            n++;
        }
    }
    fclose(fp);
    for (i = 0; i < n; i++) {
        unsigned long long delta = rows[i].busy;
        int p;
        for (p = 0; p < g_prev_n; p++) {
            if (g_prev_id[p] == rows[i].cpu) {
                if (rows[i].busy >= g_prev_busy[p]) {
                    delta = rows[i].busy - g_prev_busy[p];
                }
                break;
            }
        }
        scored[sn].cpu = rows[i].cpu;
        scored[sn].busy = delta;
        sn++;
    }
    for (i = 1; i < sn; i++) {
        Row key = scored[i];
        int j = i;
        while (j > 0 && (scored[j - 1].busy > key.busy ||
                         (scored[j - 1].busy == key.busy && scored[j - 1].cpu > key.cpu))) {
            scored[j] = scored[j - 1];
            j--;
        }
        scored[j] = key;
    }
    take = nt < sn ? nt : sn;
    /* Pin every worker or none. A short set would leave the rest wherever
     * the kernel put them and split the team across two policies. */
    if (take == nt) {
        for (i = 0; i < take; i++) {
            g_cpu[i] = scored[i].cpu;
        }
        g_cpu_n = take;
    }
    g_prev_n = n;
    for (i = 0; i < n; i++) {
        g_prev_id[i] = rows[i].cpu;
        g_prev_busy[i] = rows[i].busy;
    }
}

static __thread cpu_set_t g_orig;
static __thread int g_have_orig;

static void apply_affinity(int pin, int slot) {
    if (!g_have_orig) {
        if (sched_getaffinity(0, sizeof g_orig, &g_orig) == 0) {
            g_have_orig = 1;
        }
    }
    if (pin && slot >= 0 && slot < g_cpu_n) {
        cpu_set_t set;
        CPU_ZERO(&set);
        CPU_SET((unsigned)g_cpu[slot], &set);
        sched_setaffinity(0, sizeof set, &set);
        return;
    }
    if (g_have_orig) {
        sched_setaffinity(0, sizeof g_orig, &g_orig);
    }
}

static void share(int64_t n, int slot, int nt, int64_t *begin, int64_t *end) {
    *begin = (n * (int64_t)slot) / (int64_t)nt;
    *end = (n * ((int64_t)slot + 1)) / (int64_t)nt;
}

/* A barrier that spins only long enough to cover an idle imbalance, then
 * sleeps. An OpenMP active barrier keeps spinning on a core a burner
 * already owns, which stretches one query from a few milliseconds to
 * tens and splits the samples into two clumps. */
typedef struct {
    unsigned count;
    unsigned phase;
} LcBar;

static void futex_wait(unsigned *addr, unsigned expect) {
    syscall(SYS_futex, addr, FUTEX_WAIT, expect, NULL, NULL, 0);
}

static void futex_wake(unsigned *addr) {
    syscall(SYS_futex, addr, FUTEX_WAKE, INT_MAX, NULL, NULL, 0);
}

static void cpu_relax(void) {
#if defined(__x86_64__)
    __asm__ __volatile__("pause");
#else
    __asm__ __volatile__("yield");
#endif
}

static void passive_barrier(LcBar *bar, int nt) {
    unsigned phase;
    unsigned got;
    int spin;
    if (nt <= 1) {
        return;
    }
    phase = __atomic_load_n(&bar->phase, __ATOMIC_ACQUIRE);
    got = __atomic_add_fetch(&bar->count, 1, __ATOMIC_ACQ_REL);
    if (got == (unsigned)nt) {
        __atomic_store_n(&bar->count, 0, __ATOMIC_RELAXED);
        __atomic_store_n(&bar->phase, phase + 1, __ATOMIC_RELEASE);
        futex_wake(&bar->phase);
        return;
    }
    for (spin = 0; spin < 4000; spin++) {
        if (__atomic_load_n(&bar->phase, __ATOMIC_ACQUIRE) != phase) {
            return;
        }
        cpu_relax();
    }
    while (__atomic_load_n(&bar->phase, __ATOMIC_ACQUIRE) == phase) {
        futex_wait(&bar->phase, phase);
    }
}

int lc_omp_max_threads(void) { return omp_get_max_threads(); }

typedef struct {
    void *ctx;
    int pin;
    int restore;
    int64_t n_points;
    int64_t n_bins;
    int64_t n_blocks;
    lc_knn_phase bin;
    lc_knn_phase count;
    lc_knn_phase scan_base;
    lc_knn_phase scan;
    lc_knn_phase fill;
    lc_knn_phase select;
    int *team;
} LcJob;

/* One OpenMP region between queries. Workers sleep on `cv`. Phase edges
 * use LcBar. The region is joined from `lc_knn_shutdown`, including the
 * image destructor, so a dlclose does not leave threads in unmapped code.
 * Team size is the caller's `omp_get_max_threads()` for that query. */
#define LC_EMPTY 0
#define LC_START 1
#define LC_UP 2
#define LC_STOP 3

static struct {
    pthread_mutex_t mu;
    pthread_cond_t cv;
    pthread_t host;
    int state;
    int requested;
    int nt;
    int parked;
    int epoch;
    int done;
    int busy;
    int quit;
    LcJob job;
    LcBar bar;
} g_team = {
    .mu = PTHREAD_MUTEX_INITIALIZER,
    .cv = PTHREAD_COND_INITIALIZER,
};

static void run_phases(const LcJob *job, int nt) {
    void *ctx = job->ctx;
    int pin = job->pin;
    int restore = job->restore;
    LcBar *bar = &g_team.bar;

    if (pin || restore) {
#pragma omp single nowait
        {
            if (pin) {
                least_loaded(nt);
            }
        }
        passive_barrier(bar, nt);
        apply_affinity(pin, omp_get_thread_num());
    }

#pragma omp for schedule(static) nowait
    for (int slot = 0; slot < nt; ++slot) {
        int64_t begin, end;
        share(job->n_points, slot, nt, &begin, &end);
        job->bin(ctx, slot, nt, begin, end);
    }
    passive_barrier(bar, nt);
#pragma omp for schedule(static) nowait
    for (int slot = 0; slot < nt; ++slot) {
        int64_t begin, end;
        share(job->n_bins, slot, nt, &begin, &end);
        job->count(ctx, slot, nt, begin, end);
    }
    passive_barrier(bar, nt);
#pragma omp single nowait
    job->scan_base(ctx, 0, nt, 0, job->n_bins);
    passive_barrier(bar, nt);
#pragma omp for schedule(static) nowait
    for (int slot = 0; slot < nt; ++slot) {
        int64_t begin, end;
        share(job->n_bins, slot, nt, &begin, &end);
        job->scan(ctx, slot, nt, begin, end);
    }
    passive_barrier(bar, nt);
#pragma omp for schedule(static) nowait
    for (int slot = 0; slot < nt; ++slot) {
        int64_t begin, end;
        share(job->n_points, slot, nt, &begin, &end);
        job->fill(ctx, slot, nt, begin, end);
    }
    passive_barrier(bar, nt);
#pragma omp for schedule(static) nowait
    for (int slot = 0; slot < nt; ++slot) {
        int64_t begin, end;
        share(job->n_blocks, slot, nt, &begin, &end);
        job->select(ctx, slot, nt, begin, end);
    }
    passive_barrier(bar, nt);
#pragma omp single nowait
    {
        if (job->team != NULL) {
            *job->team = nt;
        }
    }
}

/* Every thread carries this name while it runs the region and takes its
 * own name back when it leaves. libomp keeps its workers in a pool after
 * the region ends, so the name, not the thread count, tells linkcell's
 * team apart from threads the runtime keeps. */
#define LC_TEAM_NAME "linkcell-knn"

static void *team_host(void *arg) {
    int want = (int)(intptr_t)arg;
    if (want < 1) {
        want = 1;
    }
    /* Size this region only. The caller's nthreads-var is left alone. */
#pragma omp parallel num_threads(want)
    {
        int nt = omp_get_num_threads();
        int seen = 0;
        char was[16];
        int named = pthread_getname_np(pthread_self(), was, sizeof was) == 0;
        pthread_setname_np(pthread_self(), LC_TEAM_NAME);
        pthread_mutex_lock(&g_team.mu);
        g_team.nt = nt;
        g_team.parked++;
        if (g_team.parked == nt) {
            g_team.state = LC_UP;
            pthread_cond_broadcast(&g_team.cv);
        }
        for (;;) {
            while (g_team.epoch == seen && !g_team.quit) {
                pthread_cond_wait(&g_team.cv, &g_team.mu);
            }
            if (g_team.quit) {
                pthread_mutex_unlock(&g_team.mu);
                break;
            }
            seen = g_team.epoch;
            LcJob job = g_team.job;
            pthread_mutex_unlock(&g_team.mu);
            run_phases(&job, nt);
            pthread_mutex_lock(&g_team.mu);
            g_team.done++;
            if (g_team.done == nt) {
                pthread_cond_broadcast(&g_team.cv);
            }
        }
        if (named) {
            pthread_setname_np(pthread_self(), was);
        }
    }
    return NULL;
}

static void ensure_team(int want) {
    if (want < 1) {
        want = 1;
    }
    pthread_mutex_lock(&g_team.mu);
    for (;;) {
        if (g_team.state == LC_STOP) {
            pthread_cond_wait(&g_team.cv, &g_team.mu);
            continue;
        }
        if (g_team.state == LC_UP && g_team.parked == g_team.nt && g_team.nt > 0) {
            break;
        }
        if (g_team.state == LC_EMPTY) {
            g_team.state = LC_START;
            g_team.requested = want;
            g_team.parked = 0;
            g_team.nt = 0;
            g_team.quit = 0;
            if (pthread_create(&g_team.host, NULL, team_host, (void *)(intptr_t)want) != 0) {
                g_team.state = LC_EMPTY;
                pthread_cond_broadcast(&g_team.cv);
                pthread_mutex_unlock(&g_team.mu);
                return;
            }
        }
        pthread_cond_wait(&g_team.cv, &g_team.mu);
    }
    pthread_mutex_unlock(&g_team.mu);
}

/* Join the standing region. Safe to call twice, from a query, and from
 * the image destructor (process exit, or dlclose after the workers have
 * been asked to leave). */
void lc_knn_shutdown_impl(void) {
    pthread_t host;
    int join = 0;
    pthread_mutex_lock(&g_team.mu);
    if (g_team.state == LC_EMPTY) {
        pthread_mutex_unlock(&g_team.mu);
        return;
    }
    if (g_team.state == LC_STOP) {
        while (g_team.state != LC_EMPTY) {
            pthread_cond_wait(&g_team.cv, &g_team.mu);
        }
        pthread_mutex_unlock(&g_team.mu);
        return;
    }
    while (g_team.busy || g_team.state == LC_START) {
        pthread_cond_wait(&g_team.cv, &g_team.mu);
    }
    if (g_team.state == LC_EMPTY) {
        pthread_mutex_unlock(&g_team.mu);
        return;
    }
    g_team.state = LC_STOP;
    g_team.quit = 1;
    g_team.epoch++;
    host = g_team.host;
    join = 1;
    pthread_cond_broadcast(&g_team.cv);
    pthread_mutex_unlock(&g_team.mu);
    if (join) {
        pthread_join(host, NULL);
    }
    pthread_mutex_lock(&g_team.mu);
    g_team.quit = 0;
    g_team.requested = 0;
    g_team.nt = 0;
    g_team.parked = 0;
    g_team.epoch = 0;
    g_team.done = 0;
    g_team.busy = 0;
    g_team.state = LC_EMPTY;
    g_pinned_once = 0;
    pthread_cond_broadcast(&g_team.cv);
    pthread_mutex_unlock(&g_team.mu);
}

__attribute__((destructor)) static void lc_knn_at_unload(void) { lc_knn_shutdown_impl(); }

static void run_on_caller(void *ctx, int64_t n_points, int64_t n_bins, int64_t n_blocks,
                          lc_knn_phase bin, lc_knn_phase count, lc_knn_phase scan_base,
                          lc_knn_phase scan, lc_knn_phase fill, lc_knn_phase select, int *team) {
    bin(ctx, 0, 1, 0, n_points);
    count(ctx, 0, 1, 0, n_bins);
    scan_base(ctx, 0, 1, 0, n_bins);
    scan(ctx, 0, 1, 0, n_bins);
    fill(ctx, 0, 1, 0, n_points);
    select(ctx, 0, 1, 0, n_blocks);
    if (team != NULL) {
        *team = 1;
    }
}

/* `bin`, `count`, `scan`, `fill`, and `select` run inside the standing
 * region. `scan_base` is the single that publishes each slot's prefix.
 * The caller waits on the condition variable, not on an OpenMP join.
 * `width` above zero is the team size, otherwise the caller's
 * `omp_get_max_threads()`. A team of one runs every phase on the caller
 * and starts no thread. */
void lc_knn_one_region(void *ctx, int pin, int64_t n_points, int64_t n_bins, int64_t n_blocks,
                       lc_knn_phase bin, lc_knn_phase count, lc_knn_phase scan_base,
                       lc_knn_phase scan, lc_knn_phase fill, lc_knn_phase select, int *team,
                       int width) {
    int want = width > 0 ? width : omp_get_max_threads();
    int rebuild = 0;
    if (want <= 1) {
        run_on_caller(ctx, n_points, n_bins, n_blocks, bin, count, scan_base, scan, fill, select,
                      team);
        return;
    }
    pthread_mutex_lock(&g_team.mu);
    rebuild = g_team.state == LC_UP && g_team.requested != want;
    pthread_mutex_unlock(&g_team.mu);
    if (rebuild) {
        lc_knn_shutdown_impl();
    }
    ensure_team(want);
    pthread_mutex_lock(&g_team.mu);
    rebuild = g_team.state != LC_UP;
    pthread_mutex_unlock(&g_team.mu);
    if (rebuild) {
        run_on_caller(ctx, n_points, n_bins, n_blocks, bin, count, scan_base, scan, fill, select,
                      team);
        return;
    }
    pthread_mutex_lock(&g_team.mu);
    while (g_team.busy) {
        pthread_cond_wait(&g_team.cv, &g_team.mu);
    }
    g_team.busy = 1;
    g_team.bar.count = 0;
    g_team.bar.phase = 0;
    g_team.done = 0;
    g_team.job.ctx = ctx;
    g_team.job.pin = pin;
    g_team.job.restore = g_pinned_once;
    g_team.job.n_points = n_points;
    g_team.job.n_bins = n_bins;
    g_team.job.n_blocks = n_blocks;
    g_team.job.bin = bin;
    g_team.job.count = count;
    g_team.job.scan_base = scan_base;
    g_team.job.scan = scan;
    g_team.job.fill = fill;
    g_team.job.select = select;
    g_team.job.team = team;
    g_team.epoch++;
    pthread_cond_broadcast(&g_team.cv);
    while (g_team.done < g_team.nt) {
        pthread_cond_wait(&g_team.cv, &g_team.mu);
    }
    g_team.busy = 0;
    g_pinned_once = pin;
    pthread_cond_broadcast(&g_team.cv);
    pthread_mutex_unlock(&g_team.mu);
}
