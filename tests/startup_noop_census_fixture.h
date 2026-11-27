#pragma once

inline constexpr int census_test_slot = 110;
inline constexpr int census_overflow_test_slot = 111;

int record_noop_census_fixture(void* device, void* list, void* pipeline);
