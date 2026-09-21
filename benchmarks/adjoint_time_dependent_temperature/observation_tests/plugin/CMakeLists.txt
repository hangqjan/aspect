cmake_minimum_required(VERSION 3.13.4)

find_package(Aspect 2.4.0 QUIET HINTS ${Aspect_DIR} ../../../.. $ENV{ASPECT_DIR})
if (NOT Aspect_FOUND)
  message(FATAL_ERROR "Set Aspect_DIR to an ASPECT build or installation directory.")
endif ()

DEAL_II_INITIALIZE_CACHED_VARIABLES()
set(TARGET "volume_observation_test_postprocessor")
project(${TARGET})
add_library(${TARGET} SHARED volume_observation_test_postprocessor.cc)
ASPECT_SETUP_PLUGIN(${TARGET})
