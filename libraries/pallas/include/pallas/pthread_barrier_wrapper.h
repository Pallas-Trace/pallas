#ifndef PTHREAD_BARRIER_WRAPPER_H
#define PTHREAD_BARRIER_WRAPPER_H

// Wrapper of pthread barrier functionality for improved
// portability on non-linux systems -- Aaron

#include <pthread.h>

// Detect compilation on Darwin (i.e. macos) target
#if defined(__APPLE__)
    #define NEED_PTHREAD_BARRIER_FALLBACK
#endif

#ifdef NEED_PTHREAD_BARRIER_FALLBACK

#ifndef PTHREAD_BARRIER_SERIAL_THREAD
// Default value: -1
#define PTHREAD_BARRIER_SERIAL_THREAD (-1)
#endif

typedef struct {
    pthread_mutex_t mutex;
    pthread_cond_t condition_var;
    unsigned int threads_required;       // required number of threads at barrier
    unsigned int threads_remaining;      // number of threads yet to reach barrier
    unsigned int generation;             // 
} pthread_barrier_t;

static inline int pthread_barrier_init(pthread_barrier_t *barrier,
                                       void *attr, // unused pthread_barrierattr_t
                                       unsigned int count) {
    pthread_mutex_init(&barrier->mutex, NULL);
    pthread_cond_init(&barrier->condition_var, NULL);
    barrier->threads_required = count;
    barrier->threads_remaining = count;
    barrier->generation = 0;
    return 0;
}

static inline int pthread_barrier_destroy(pthread_barrier_t *barrier) {
    pthread_mutex_destroy(&barrier->mutex);
    pthread_cond_destroy(&barrier->condition_var);
    return 0;
}

static inline int pthread_barrier_wait(pthread_barrier_t *barrier) {

    pthread_mutex_lock(&barrier->mutex);

    if (--(barrier->threads_remaining) == 0) {
        barrier->generation++;
        barrier->threads_remaining = barrier->threads_required;

        pthread_cond_broadcast(&barrier->condition_var);
        pthread_mutex_unlock(&barrier->mutex);
        return PTHREAD_BARRIER_SERIAL_THREAD;
    } else {
      unsigned int gen = barrier->generation;

      while (gen == barrier->generation) {
          pthread_cond_wait(&barrier->condition_var, &barrier->mutex);
      }

      pthread_mutex_unlock(&barrier->mutex);
      return 0;
    }
}

#else // NEED_PTHREAD_BARRIER_FALLBACK

  // I.e. Linux compilation target
  // No implementation necessary, included in pthread.h

#endif // NEED_PTHREAD_BARRIER_FALLBACK

#endif // PTHREAD_BARRIER_WRAPPER_H
