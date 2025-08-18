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


#define EPSILON 1e-12

bool details = true;


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
    double ss_res = 0.0, ss_tot = 0.0;
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
        std::cerr << "Usage: " << argv[0] << " <trace_1.csv> <trace_2.csv> " << std::endl;
        return EXIT_FAILURE;
    }

    auto trace_csv_1 = get_name_w_csv(argv[1]);
    auto trace_csv_2 = get_name_w_csv(argv[2]);

    double res = CompareTimestamps(argv[1], argv[2]);

    std::cout.precision(12);

    std::cout << "------------------------------------------------------------------" << std::endl;
    std::cout << std::right << std::setw(30) << std::fixed << "Result: R^2 = ";
    std::cout << std::right << std::setw(12) << std::fixed << res << std::endl;
    std::cout << "------------------------------------------------------------------" << std::endl;
    

    return EXIT_SUCCESS;
}