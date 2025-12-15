//
// Created by khatharsis on 02/12/25.
//

#include <iomanip>
#include <iostream>
#include <pallas/pallas.h>
#include <pallas/pallas_archive.h>

#include <pallas/utils/pallas_storage.h>

static int id_width = 8;
static int value_width = 14;
static int err_width = 9;

void print_header() {
    std::cout << std::setw(id_width) << std::left << "Seq_id" <<
            std::setw(value_width) << std::left << "Exact" <<
            std::setw(value_width) << std::left << "Normal" <<
            std::setw(err_width) << std::left << "Err (%)" <<
            std::setw(value_width) << std::left << "Fast" <<
            std::setw(err_width) << std::left << "Err (%)" << std::endl;
}

static inline float error(pallas_duration_t a, pallas_duration_t b) {
    if (a == 0 && b == 0) return 0;
    if (a < b) {
        return 100.0 * (0.0f + b - a) / a;
    }
    return 100.0 * (0.0f + a - b) / a;
}

int main(int argc, char** argv) {
    char* trace_name = argv[1];
    size_t nb_frames = 10;
    if (argc > 2) {
        nb_frames = atoi(argv[2]);
    }

    auto* trace = pallas_open_trace(trace_name);
    float mape_normal = 0.;
    float mape_fast = 0.;
    size_t counter = 0;
    for (auto* thread : trace->getThreadList()) {
        std::cout << std::endl << "========== Thread " << thread->id << " ==========" << std::endl;
        pallas_timestamp_t start = thread->first_timestamp;
        pallas_timestamp_t end = thread->getDuration();
        auto step = (end - start) / nb_frames;
        for (int i = 0; i < nb_frames; i++) {
            std::cout << "========== Slice " << i << " ==========" << std::endl;
            print_header();
            auto snapshotExact = thread->getSnapshotViewExact(start + i * step, start + (i + 1) * step);
            auto snapshot = thread->getSnapshotView(start + i * step, start + (i + 1) * step);
            auto snapshotFast = thread->getSnapshotViewFast(start + i * step, start + (i + 1) * step);
            for (size_t j = 1; j < thread->nb_sequences; j++) {
                auto token = pallas::Token(pallas::TypeSequence, j);
                auto* sequence = thread->getSequence(token);
                if (sequence->type == pallas::SEQUENCE_BLOCK) {
                    if (snapshot[token] + snapshotFast[token] == 0) {
                        continue;
                    }
                    float error_normal = error(snapshotExact[token], snapshot[token]);
                    mape_normal += error_normal;
                    float error_fast = error(snapshotExact[token], snapshotFast[token]);
                    mape_fast += error_fast;
                    counter ++;
                    std::cout <<
                            std::setw(id_width-1) << std::right << j << " " <<
                            std::setw(value_width) << std::left << snapshotExact[token] <<
                            std::setw(value_width) << std::left << snapshot[token] <<
                            std::setw(err_width) << std::left << std::setprecision(3) <<  error_normal <<
                            std::setw(value_width) << std::left << snapshotFast[token] <<
                            std::setw(err_width) << std::left << std::setprecision(3) << error_fast <<
                            std::endl;
                }
            }
        }
    }

    mape_normal /= counter;
    mape_fast /= counter;
    std::cout << "MAPE getSnapshotView:     " << std::setprecision(3) << mape_normal << std::endl;
    std::cout << "MAPE getSnapshotViewFast: " << std::setprecision(3) << mape_fast << std::endl;
}
