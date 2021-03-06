//
// Created by balder on 19.02.2020.
//

#include "sensor_data.h"
#include "data/test_features.h"
#include "data/test_target.h"
#include "data/train_features.h"
#include "data/train_target.h"
#include <stdint.h>
#include <string.h>

static float current_reading[TEST_FEATURES_NUM_COLS];
static float current_values[TEST_TARGET_NUM_COLS];

void get_sensor_reading(float reading[TEST_FEATURES_NUM_COLS]) {
    memcpy(reading, current_reading, sizeof(current_reading));
}

void get_sensor_values(float values[TEST_TARGET_NUM_COLS]) {
    memcpy(values, current_values, sizeof(current_values));
}

static uint16_t test_i = 0, train_i = 0;

void new_sensor_reading(sensor_data_source_t source) {
    switch (source) {
        case SENSOR_DATA_TEST:
            memcpy(current_reading, test_features[test_i], sizeof(current_reading));
            memcpy(current_values, test_target[test_i], sizeof(current_values));
            test_i = (test_i + 1) % TEST_FEATURES_NUM_ROWS;
            break;
        case SENSOR_DATA_TRAIN:
            memcpy(current_reading, train_features[train_i], sizeof(current_reading));
            memcpy(current_values, train_target[train_i], sizeof(current_values));
            train_i = (train_i + 1) % TRAIN_FEATURES_NUM_ROWS;
            break;
        default:
            break;
    }
}

void get_feature_description(const char *description[TEST_FEATURES_NUM_COLS]) {
    for (int i = 0; i < TEST_FEATURES_NUM_COLS; i++) {
        description[i] = test_features_column_description[i];
    }
}

void get_target_description(const char *description[TEST_TARGET_NUM_COLS]) {
    for (int i = 0; i < TEST_TARGET_NUM_COLS; i++) {
        description[i] = test_target_column_description[i];
    }
}