/* One linkcell-owned thread team for a whole k-nearest query.
 *
 * The Rust side owns the points, the bins, and each thread's scratch.
 * This file only schedules: static shares, in thread order, so slot t is
 * the same thread for every phase. The workers are pthreads that this
 * file creates and joins, so no OpenMP runtime holds a thread after
 * shutdown. OpenMP is only read: the team is sized as libgomp sizes a
 * parallel region the caller opens (`omp_get_max_threads()`, then
 * `omp_get_dynamic()` and `omp_get_thread_limit()`), the bind kind and
 * place list place it, and `OMP_STACKSIZE` sizes its stacks. A team of
 * one is the caller itself. Affinity changes only when the caller asks to
 * pin or the runtime binds threads.
 */
#define _GNU_SOURCE

#include <ctype.h>
#include <errno.h>
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

/* One standing team between queries. Workers sleep on `cv`. Phase edges
 * use LcBar. The team is joined from `lc_knn_shutdown`, including the
 * image destructor, so a dlclose does not leave threads in unmapped code.
 * Team size comes from `team_size`, kept for as long as the caller's
 * nthreads-var and dyn-var stay the same and the team is not joined. */
#define LC_EMPTY 0
#define LC_START 1
#define LC_UP 2
#define LC_STOP 3

static struct {
    pthread_mutex_t mu;
    pthread_cond_t cv;
    pthread_t *workers;
    int started;
    int state;
    int nt;
    int parked;
    int epoch;
    int done;
    int busy;
    int quit;
    int sized_want;
    int sized_dyn;
    int sized_nt;
    LcJob job;
    LcBar bar;
} g_team = {
    .mu = PTHREAD_MUTEX_INITIALIZER,
    .cv = PTHREAD_COND_INITIALIZER,
};

/* Slot t takes share t of every phase, as `schedule(static)` over nt
 * iterations on nt threads does. Slot 0 runs the singles. */
static void run_phases(const LcJob *job, int slot, int nt) {
    void *ctx = job->ctx;
    int pin = job->pin;
    int restore = job->restore;
    LcBar *bar = &g_team.bar;
    int64_t begin, end;

    if (pin || restore) {
        if (slot == 0 && pin) {
            least_loaded(nt);
        }
        passive_barrier(bar, nt);
        apply_affinity(pin, slot);
    }

    share(job->n_points, slot, nt, &begin, &end);
    job->bin(ctx, slot, nt, begin, end);
    passive_barrier(bar, nt);
    share(job->n_bins, slot, nt, &begin, &end);
    job->count(ctx, slot, nt, begin, end);
    passive_barrier(bar, nt);
    if (slot == 0) {
        job->scan_base(ctx, 0, nt, 0, job->n_bins);
    }
    passive_barrier(bar, nt);
    share(job->n_bins, slot, nt, &begin, &end);
    job->scan(ctx, slot, nt, begin, end);
    passive_barrier(bar, nt);
    share(job->n_points, slot, nt, &begin, &end);
    job->fill(ctx, slot, nt, begin, end);
    passive_barrier(bar, nt);
    share(job->n_blocks, slot, nt, &begin, &end);
    job->select(ctx, slot, nt, begin, end);
    passive_barrier(bar, nt);
    if (slot == 0 && job->team != NULL) {
        *job->team = nt;
    }
}

/* Every worker carries this name, so `top -H` and
 * `/proc/<pid>/task/<tid>/comm` tell linkcell's team apart from the
 * caller's threads and from any the OpenMP runtime keeps. */
#define LC_TEAM_NAME "linkcell-knn"

static void *team_worker(void *arg) {
    int slot = (int)(intptr_t)arg;
    int seen = 0;
    int nt;
    pthread_setname_np(pthread_self(), LC_TEAM_NAME);
    pthread_mutex_lock(&g_team.mu);
    nt = g_team.nt;
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
        run_phases(&job, slot, nt);
        pthread_mutex_lock(&g_team.mu);
        g_team.done++;
        if (g_team.done == nt) {
            pthread_cond_broadcast(&g_team.cv);
        }
    }
    return NULL;
}

/* The place of slot `slot` in a team of `nt` over `np` places, as libgomp
 * binds a team that a new thread starts: from place 0 of the whole list.
 * `close` and `true` put consecutive slots on consecutive places. `spread`
 * starts each slot on its own subpartition, the first `np % nt` one place
 * longer. With more slots than places both give each place a run of
 * `nt / np` consecutive slots, and the `nt % np` left over wrap round from
 * place 0. `primary` keeps every slot on place 0. */
static int place_of(int bind, int slot, int nt, int np) {
    int each, rest;
    if (bind != omp_proc_bind_true && bind != omp_proc_bind_close && bind != omp_proc_bind_spread) {
        return 0;
    }
    if (nt > np) {
        each = nt / np;
        if (slot < each * np) {
            return slot / each;
        }
        return slot - each * np;
    }
    if (bind == omp_proc_bind_spread) {
        each = np / nt;
        rest = np % nt;
        if (slot < rest) {
            return slot * (each + 1);
        }
        return rest * (each + 1) + (slot - rest) * each;
    }
    return slot;
}

static int place_mask(int place, cpu_set_t *set) {
    int n = omp_get_place_num_procs(place);
    int *ids;
    int i;
    CPU_ZERO(set);
    if (n <= 0) {
        return 0;
    }
    ids = malloc(sizeof *ids * (size_t)n);
    if (ids == NULL) {
        return 0;
    }
    omp_get_place_proc_ids(place, ids);
    for (i = 0; i < n; i++) {
        if (ids[i] >= 0 && ids[i] < CPU_SETSIZE) {
            CPU_SET((unsigned)ids[i], set);
        }
    }
    free(ids);
    return CPU_COUNT(set) > 0;
}

/* libgomp's reading of OMP_STACKSIZE: a decimal size with an optional
 * B, K, M, or G suffix (K when absent), spaces allowed around both. The
 * OpenMP API has no getter for stacksize-var, so the variable is read
 * here, once. An invalid value leaves the default, as libgomp does after
 * it has reported the value itself. */
static int parse_stacksize(const char *name, size_t *out) {
    const char *env = getenv(name);
    char *end;
    unsigned long value;
    unsigned shift = 10;
    if (env == NULL) {
        return 0;
    }
    while (isspace((unsigned char)*env)) {
        env++;
    }
    if (*env == '\0') {
        return 0;
    }
    errno = 0;
    value = strtoul(env, &end, 10);
    if (errno != 0 || end == env) {
        return 0;
    }
    while (isspace((unsigned char)*end)) {
        end++;
    }
    if (*end != '\0') {
        switch (tolower((unsigned char)*end)) {
        case 'b':
            shift = 0;
            break;
        case 'k':
            break;
        case 'm':
            shift = 20;
            break;
        case 'g':
            shift = 30;
            break;
        default:
            return 0;
        }
        end++;
        while (isspace((unsigned char)*end)) {
            end++;
        }
        if (*end != '\0') {
            return 0;
        }
    }
    if (((value << shift) >> shift) != value) {
        return 0;
    }
    *out = (size_t)(value << shift);
    return 1;
}

static pthread_once_t g_stack_once = PTHREAD_ONCE_INIT;
static size_t g_stacksize;

static void read_stacksize(void) {
    if (!parse_stacksize("OMP_STACKSIZE", &g_stacksize) &&
        !parse_stacksize("GOMP_STACKSIZE", &g_stacksize)) {
        g_stacksize = 0;
    }
}

/* A bound slot starts on its place, as an OpenMP worker does. Without a
 * bind kind, or when the place cannot be applied, it inherits the
 * caller's mask. A stack size the system refuses leaves the default. */
static int create_worker(pthread_t *th, void *arg, const cpu_set_t *place) {
    pthread_attr_t attr;
    int made;
    if (pthread_attr_init(&attr) != 0) {
        return 0;
    }
    if (g_stacksize != 0) {
        pthread_attr_setstacksize(&attr, g_stacksize);
    }
    made = (place == NULL || pthread_attr_setaffinity_np(&attr, sizeof *place, place) == 0) &&
           pthread_create(th, &attr, team_worker, arg) == 0;
    pthread_attr_destroy(&attr);
    return made;
}

static int start_worker(int slot, int nt, int bind, int np) {
    pthread_t *th = &g_team.workers[slot];
    void *arg = (void *)(intptr_t)slot;
    cpu_set_t set;
    pthread_once(&g_stack_once, read_stacksize);
    if (np > 0 && place_mask(place_of(bind, slot, nt, np), &set) && create_worker(th, arg, &set)) {
        return 1;
    }
    return create_worker(th, arg, NULL) || pthread_create(th, NULL, team_worker, arg) == 0;
}

/* Called with `mu` held and the team empty. Returns with `mu` held. A
 * short team cannot run the static shares, so a failed start joins the
 * workers it made and leaves the team empty. */
static int start_team(int want) {
    int bind = (int)omp_get_proc_bind();
    int np = bind == omp_proc_bind_false ? 0 : omp_get_num_places();
    int slot;
    g_team.workers = malloc(sizeof *g_team.workers * (size_t)want);
    if (g_team.workers == NULL) {
        return 0;
    }
    g_team.state = LC_START;
    g_team.nt = want;
    g_team.parked = 0;
    g_team.quit = 0;
    g_team.started = 0;
    for (slot = 0; slot < want; slot++) {
        if (!start_worker(slot, want, bind, np)) {
            break;
        }
        g_team.started++;
    }
    if (g_team.started == want) {
        return 1;
    }
    g_team.quit = 1;
    pthread_cond_broadcast(&g_team.cv);
    pthread_mutex_unlock(&g_team.mu);
    for (slot = 0; slot < g_team.started; slot++) {
        pthread_join(g_team.workers[slot], NULL);
    }
    pthread_mutex_lock(&g_team.mu);
    free(g_team.workers);
    g_team.workers = NULL;
    g_team.started = 0;
    g_team.quit = 0;
    g_team.nt = 0;
    g_team.parked = 0;
    g_team.state = LC_EMPTY;
    pthread_cond_broadcast(&g_team.cv);
    return 0;
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
        if (g_team.state == LC_EMPTY && !start_team(want)) {
            pthread_mutex_unlock(&g_team.mu);
            return;
        }
        pthread_cond_wait(&g_team.cv, &g_team.mu);
    }
    pthread_mutex_unlock(&g_team.mu);
}

/* Join every worker. Safe to call twice, from a query, and from the
 * image destructor (process exit, or dlclose after the workers have been
 * asked to leave). */
void lc_knn_shutdown_impl(void) {
    pthread_t *workers;
    int started;
    int slot;
    pthread_mutex_lock(&g_team.mu);
    g_team.sized_want = 0;
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
    workers = g_team.workers;
    started = g_team.started;
    pthread_cond_broadcast(&g_team.cv);
    pthread_mutex_unlock(&g_team.mu);
    for (slot = 0; slot < started; slot++) {
        pthread_join(workers[slot], NULL);
    }
    pthread_mutex_lock(&g_team.mu);
    free(workers);
    g_team.workers = NULL;
    g_team.started = 0;
    g_team.quit = 0;
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

/* libgomp's size for a parallel region the caller opens with no
 * num_threads clause. With dyn-var set the team is the processors
 * available, at most nthreads-var, less the 15-minute load average
 * rounded as libgomp rounds it, and at least one. thread-limit-var caps
 * the result. Called with `mu` held. */
static int team_size(int want, int dyn) {
    int nt = want;
    int limit;
    if (g_team.sized_want == want && g_team.sized_dyn == dyn) {
        return g_team.sized_nt;
    }
    if (dyn) {
        int procs = omp_get_num_procs();
        double load[3];
        int busy = 0;
        if (procs > want) {
            procs = want;
        }
        if (getloadavg(load, 3) == 3) {
            busy = (int)(load[2] + 0.1);
        }
        procs = busy >= procs ? 1 : procs - busy;
        if (procs < nt) {
            nt = procs;
        }
    }
    limit = omp_get_thread_limit();
    if (limit > 0 && limit < nt) {
        nt = limit;
    }
    if (nt < 1) {
        nt = 1;
    }
    g_team.sized_want = want;
    g_team.sized_dyn = dyn;
    g_team.sized_nt = nt;
    return nt;
}

/* A team of one is the caller: no worker and no handoff. For the query
 * the caller takes the CPU or the place slot 0 would have had, then gets
 * its own mask back. */
static void run_on_caller(const LcJob *job) {
    cpu_set_t mine;
    cpu_set_t set;
    int moved = 0;
    if (job->pin) {
        pthread_mutex_lock(&g_team.mu);
        while (g_team.busy) {
            pthread_cond_wait(&g_team.cv, &g_team.mu);
        }
        least_loaded(1);
        if (g_cpu_n == 1) {
            CPU_ZERO(&set);
            CPU_SET((unsigned)g_cpu[0], &set);
            moved = 1;
        }
        pthread_mutex_unlock(&g_team.mu);
    } else {
        int bind = (int)omp_get_proc_bind();
        int np = bind == omp_proc_bind_false ? 0 : omp_get_num_places();
        moved = np > 0 && place_mask(place_of(bind, 0, 1, np), &set);
    }
    moved = moved && sched_getaffinity(0, sizeof mine, &mine) == 0 &&
            sched_setaffinity(0, sizeof set, &set) == 0;
    job->bin(job->ctx, 0, 1, 0, job->n_points);
    job->count(job->ctx, 0, 1, 0, job->n_bins);
    job->scan_base(job->ctx, 0, 1, 0, job->n_bins);
    job->scan(job->ctx, 0, 1, 0, job->n_bins);
    job->fill(job->ctx, 0, 1, 0, job->n_points);
    job->select(job->ctx, 0, 1, 0, job->n_blocks);
    if (moved) {
        sched_setaffinity(0, sizeof mine, &mine);
    }
    if (job->team != NULL) {
        *job->team = 1;
    }
}

/* `bin`, `count`, `scan`, `fill`, and `select` run on the standing team.
 * `scan_base` is the single that publishes each slot's prefix. The
 * caller waits on the condition variable for every slot to finish. A
 * change of team size joins the old team first. `width` above zero
 * stands in for `omp_get_max_threads()`, as a num_threads clause does,
 * and is trimmed and capped the same way. */
void lc_knn_one_region(void *ctx, int pin, int64_t n_points, int64_t n_bins, int64_t n_blocks,
                       lc_knn_phase bin, lc_knn_phase count, lc_knn_phase scan_base,
                       lc_knn_phase scan, lc_knn_phase fill, lc_knn_phase select, int *team,
                       int width) {
    int want = width > 0 ? width : omp_get_max_threads();
    int dyn = omp_get_dynamic() != 0;
    int nt;
    int up;
    LcJob job = {
        .ctx = ctx,
        .pin = pin,
        .n_points = n_points,
        .n_bins = n_bins,
        .n_blocks = n_blocks,
        .bin = bin,
        .count = count,
        .scan_base = scan_base,
        .scan = scan,
        .fill = fill,
        .select = select,
        .team = team,
    };
    if (want < 1) {
        want = 1;
    }
    pthread_mutex_lock(&g_team.mu);
    nt = team_size(want, dyn);
    up = g_team.state == LC_UP && g_team.nt != nt;
    pthread_mutex_unlock(&g_team.mu);
    if (up) {
        lc_knn_shutdown_impl();
        pthread_mutex_lock(&g_team.mu);
        nt = team_size(want, dyn);
        pthread_mutex_unlock(&g_team.mu);
    }
    if (nt <= 1) {
        run_on_caller(&job);
        return;
    }
    ensure_team(nt);
    pthread_mutex_lock(&g_team.mu);
    while (g_team.busy) {
        pthread_cond_wait(&g_team.cv, &g_team.mu);
    }
    if (g_team.state != LC_UP) {
        pthread_mutex_unlock(&g_team.mu);
        run_on_caller(&job);
        return;
    }
    g_team.busy = 1;
    g_team.bar.count = 0;
    g_team.bar.phase = 0;
    g_team.done = 0;
    job.restore = g_pinned_once;
    g_team.job = job;
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
