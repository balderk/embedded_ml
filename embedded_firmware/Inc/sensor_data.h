//
// Created by balder on 19.02.2020.
//

#ifndef EMBEDDED_FIRMWARE_SENSOR_DATA_H
#define EMBEDDED_FIRMWARE_SENSOR_DATA_H

#include "relu_64.h"

typedef enum __sensor_data_source_t {
    SENSOR_DATA_TEST,
    SENSOR_DATA_TRAIN
} sensor_data_source_t;

void get_sensor_values(float values[AI_RELU_64_OUT_1_SIZE]);

void get_sensor_reading(float reading[AI_RELU_64_IN_1_SIZE]);

void new_sensor_reading(sensor_data_source_t source);

#endif // EMBEDDED_FIRMWARE_SENSOR_DATA_H
