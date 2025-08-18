#include <iostream>
#include <limits>
#include <string>
#include <iomanip>
#include "pallas/pallas.h" 
#include "pallas/pallas_archive.h"
#include "pallas/pallas_log.h"
#include "pallas/pallas_read.h"
#include "pallas/pallas_storage.h"
#include <time.h>
#include <fstream>
#include <float.h>
#include "pallas/pallas_dbg.h"

bool details = true;

/**
 * Adds a timestamp t to the .csv file filename
 */
void write_csv_details(const char* filename, pallas_timestamp_t t){
    std::ofstream file(std::string(filename) + ".csv", std::ios::app);
    file << t << "\n";
}



/**
 * name is a path for a .pallas trace file. It fills a csv file with all the timestamps of the trace without any header.
 */
void getTraceTimepstamps(char* name) {

  auto* copy = strdup(name);
  auto* slash = strrchr(copy, '/');
  if (slash != nullptr)
    *slash = '\0';
  fprintf(stdout, "%s\n", name);
  auto* trace = pallas_open_trace(name);

  if(details) fprintf(stdout,"nb_archives = %d\n", trace->nb_archives);


  for (uint aid = 0; aid < (uint) trace->nb_archives; aid++) {

    if(details) fprintf(stdout, "%s: archive id = %d\n", name, aid);
    auto archive = trace->archive_list[aid];

      for (uint i = 0; i < archive->nb_threads; i++) {

	    const pallas::Thread *thread = archive->getThreadAt(i);

      for (unsigned j = 0; j < thread->nb_events; j++) {
          pallas::EventSummary& e = thread->events[j];
            auto* timestamps = e.timestamps;

            for (size_t k =0; k<timestamps->size; k++){
              uint64_t t = timestamps->at(k);
              write_csv_details(copy, t);
            }
            timestamps->free_data();
            e.cleanEventSummary();
        }
      archive->freeThreadAt(i);
      }
   }

  free(copy);

trace->close();
exit(EXIT_SUCCESS);
}



/**
 * Char* c is a path for a file.
 * get_name_w_csv retruns the name of the parent forlder of the file, appended with ".csv"
 */
auto get_name_w_csv(char* c){
  auto* copy = strdup(c);
  auto* slash = strrchr(copy, '/');
  if (slash != nullptr)
    *slash = '\0';
  std::string copy_s = std::string(copy) + ".csv";
  free(copy);
  return copy_s;
}



int main(const int argc, char* argv[]) {

    if (argc < 2){
        std::cerr << "Usage: " << argv[0] << " <trace_1.pallas> ... <trace_n.pallas" << std::endl;
        return EXIT_FAILURE;
    }

    for (int i = 1; i<=argc; i++){
      auto trace_csv = get_name_w_csv(argv[i]);

      std::ofstream(trace_csv, std::ios::trunc);

      getTraceTimepstamps(argv[1]);
    }
    

    return EXIT_SUCCESS;
}