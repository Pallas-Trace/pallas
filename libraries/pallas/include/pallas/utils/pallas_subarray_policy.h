#pragma once

#include "pallas_timestamp.h"
#ifndef __cplusplus
#include <stdint.h>

#endif
#ifdef __cplusplus
#include <cstdint>
#include <set>
#include <vector>

#include "pallas_parameter_handler.h"
#include "pallas_linked_vector.h"

namespace pallas {

enum class ValueDomain : uint8_t {
    TimeStamp = 0,
    Duration = 1,
};

enum class Policy : uint8_t {
    None = 0,
    Delta = 1,
    Lossy = 2,
};

enum class LossyPolicy : uint8_t {
    Linear = 0, // Value Domain must be TimeStamp
    Normal = 1, // Value Domain must be Duration
};

enum class AddStatus : uint8_t {
    Ok = 0,
    Outlier = 1, // Will lead to a new subarray being created and some other stuff happening 
    Full = 2, // When the allocated  
};

};

constexpr long double ERR_EPSILON_NS = 1000 * 100; // 100 microsec


class Predictor {
    private:
        
    public:

    private:

    public:

};






#else 
#endif

/* -*-
   mode: c++;
   c-file-style: "k&r";
   c-basic-offset 4;
   tab-width 4 ;
   indent-tabs-mode nil
   -*- */  