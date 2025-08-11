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
#include <sys/wait.h>
#include <malloc.h>

#define EPSILON 1e-12

bool details = true;

int thread_to_print = -1;

/**
 * Adds a timestamp t to the .csv file filename
 */
void write_csv_details(const char* filename, pallas_timestamp_t t){
    std::ofstream file(std::string(filename) + ".csv", std::ios::app);
    file << t << "\n";
}

pallas_timestamp_t get_timestamp(const pallas::EventOccurence e){ 
    return(e.timestamp);
}


/**
 * name is a path for a .pallas trace file. It fills a .csv file with all the timestamps of the trace without any header.
 */
void getTraceTimepstamps(char* name) {

  auto* copy = strdup(name);
  auto* slash = strrchr(copy, '/');
  if (slash != nullptr)
    *slash = '\0';

  auto trac = pallas_open_trace(name);
  auto trace = *trac;

  for (uint aid = 0; aid < (uint) trace.nb_archives; aid++) {
    auto archive = trace.archive_list[aid];
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
        }
      archive->freeThreadAt(i);
      }
  
      }
  free(copy);

  delete trac; 
 exit(EXIT_SUCCESS);
}


/**
 * trace1 is a path for a csv file filled with timestamps (integers) and returns the mean value of these values
 */
double mean(const char* trace){
  FILE* file = fopen(trace, "r");

  double n = 0.0;
  double y, sum_y = 0.0;

  if (!file){
    perror("[mean_fopen]");
    return EXIT_FAILURE;
  }
  while(fscanf(file, "%lf", &y) == 1){
    sum_y += y;
    n++;
  }

  fclose(file);

  double mean_y = sum_y / n ;
    if (details){
      std::cout.precision(12);
      std::cout << "------------------------------------------------------------------" << std::endl;
      std::cout << std::right << std::fixed << "Details: ";
      std::cout << " n = " << n << "," << "mean_y = " << mean_y << std::endl;
    }
    return mean_y;
}


/**
 * Test R2
 * R^2 = 1 - (SS_res / SS_tot) where:
 * SS_res = sum_1_n(y_i - x_i)^2
 * SS_tot = sum_1_n(y_i - mean(y))^2
 */
/**
 * trace1 and trace2 are paths for csv files with the timestamps get with getTraceTimestamps. 
 * Returns the statistical result of the R^2 test on these two files.
 */
double CompareTimestamps(const char* trace1, const char* trace2){
    FILE* file1 = fopen(trace1, "r");
    FILE* file2 = fopen(trace2, "r");
    if (!file1 || !file2) {
        perror("[fopen]");
        return EXIT_FAILURE;
    }

    double y, val2;
    double ss_res, ss_tot = 0.0;
    double mean_y = mean(trace1);

    while ((fscanf(file1, "%lf", &y)==1) && (fscanf(file2, "%lf", &val2)==1)) {
        ss_res += (y - val2) * (y - val2);
        ss_tot += (y - mean_y) * (y - mean_y);
    }

    fclose(file1);
    fclose(file2);

    if (ss_tot < EPSILON){
      fprintf(stderr, "\nss_tot nul\n");
      return -1.0;
    }
    if (details){
      std::cout.precision(0);
      std::cout << "------------------------------------------------------------------" << std::endl;
      std::cout << std::right << std::fixed << "Details: ";
      std::cout << " ss_res = " << ss_res << "," << " ss_tot = " << ss_tot << std::endl;
    }
    double R2 = 1.0 - ss_res / ss_tot;
    return R2;
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

    if (argc != 3){
        std::cerr << "Usage: " << argv[0] << " <trace1.pallas> <trace2.pallas> " << std::endl;
        return EXIT_FAILURE;
    }

    auto trace_csv_1 = get_name_w_csv(argv[1]);
    auto trace_csv_2 = get_name_w_csv(argv[2]);

    std::ofstream(trace_csv_1, std::ios::trunc);
    std::ofstream(trace_csv_2, std::ios::trunc);

    pid_t pid1 = fork();

    if (pid1 == 0){
      getTraceTimepstamps(argv[1]);
      exit(EXIT_SUCCESS);
    }

    int status1;
    waitpid(pid1, &status1, 0);
    if (!WIFEXITED(status1)) {
        std::cerr << "First process failed" << std::endl;
        return EXIT_FAILURE;
    }

    pid_t pid2 = fork();
    if (pid2 == 0){
      getTraceTimepstamps(argv[2]);
      fprintf(stdout, "second trace ok\n");
      exit(EXIT_SUCCESS);
    }

    int status2;
    waitpid(pid2, &status2, 0);
    if (!WIFEXITED(status2)) {
        std::cerr << "Second process failed" << std::endl;
        return EXIT_FAILURE;
    }

    double res = CompareTimestamps(trace_csv_1.c_str(), trace_csv_2.c_str());

    std::cout.precision(12);

    std::cout << "------------------------------------------------------------------" << std::endl;
    std::cout << std::right << std::setw(30) << std::fixed << "Result: R^2 = ";
    std::cout << std::right << std::setw(12) << std::fixed << res << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl;


    return EXIT_SUCCESS;
}