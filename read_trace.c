#define _GNU_SOURCE

#include <fcntl.h>
#include <unistd.h>
#include <assert.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include "liblock.h"
#include "event.h"

struct event_data *event_data = NULL;
int nb_event_data = 0;

void load_event_data(const char* filename) {
  int fd = open(filename, O_RDONLY);
  assert(fd>0);
  struct stat s;
  fstat(fd, &s);
  size_t file_size = s.st_size;
  event_data = mmap(NULL, file_size, PROT_READ, MAP_PRIVATE, fd, 0);
  if(event_data == MAP_FAILED) {
    fprintf(stderr, "Error: cannot mmap file %s\n", filename);
    abort();
  }

  nb_event_data =file_size / sizeof(struct event_data);
  close(fd); 

  printf("There are %d data events:\n", nb_event_data);
  for(int i=0; i<nb_event_data; i++) {
    struct event_data *data = &event_data[i];
    printf("[%d]  {.ptr=%p, .tid=%lx, .func=%d, .event_type=%d}\n", i,
	   data->ptr, data->tid, data->function, data->event_type);

  }
}

struct event_data* get_data(event_data_id data_id) {
  
  if(data_id > nb_event_data) {
    fprintf(stderr, "error: looking for data %d, but there are only %d data\n", data_id, nb_event_data);
    abort();
  }

  return &event_data[data_id];
}

int main(int argc, char**argv) {
  if(argc!=3) {
    printf("usage: %s trace\n", argv[0]);
    return EXIT_FAILURE;
  }

  int fd = open(argv[1], O_RDONLY);
  assert(fd>0);
  load_event_data(argv[2]);
  struct event e;
  while(read(fd, &e, sizeof(e)) > 0) {
    struct event_data* data = get_data(e.data_id);
    if(data->event_type == function_entry)
      printf("[%lx] %lf\tenter %s(%p)\n", data->tid, e.timestamp/1e9, function_names[data->function], data->ptr);
    else
      printf("[%lx] %lf\texit %s(%p)\n", data->tid, e.timestamp/1e9, function_names[data->function], data->ptr);
  }
  close(fd);
  return EXIT_SUCCESS;
}
