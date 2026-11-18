# Rigid brick on a slope with Chrono SMC, linked against a Chrono 10 build tree.
#   core SMC (ChSystemSMC):            CHRONO_BUILD = ~/work/chrono10-cudss/build10 (default)
#   + Multicore SMC (brick_slope_mc):  CHRONO_BUILD = ~/work/chrono10-cudss/build-mc (Multicore enabled)
#
#   cmake -S test-scripts/chrono_smc -B test-scripts/chrono_smc/build-mc -DCMAKE_BUILD_TYPE=Release \
#         -DCHRONO_BUILD=$HOME/work/chrono10-cudss/build-mc -Dblaze_INCLUDE_DIR=$HOME/Packages/blaze
#   cmake --build test-scripts/chrono_smc/build-mc -j
#
# Uses the build tree's exported targets directly: its ChronoConfig.cmake insists on ChronoModels_robot,
# which these builds do not have. (~/work/chrono's source was pulled after its build, so it is not used.)
cmake_minimum_required(VERSION 3.18)
project(chrono_smc_brick CXX)

set(CHRONO_BUILD "$ENV{HOME}/work/chrono10-cudss/build10" CACHE PATH "Chrono 10 build tree")
list(APPEND CMAKE_MODULE_PATH "${CHRONO_BUILD}/cmake")
find_package(Eigen3 REQUIRED NO_MODULE)
find_package(OpenMP REQUIRED)
find_package(CUDAToolkit QUIET)  # build10's core links CUDA::cudart (cuDSS build)
include("${CHRONO_BUILD}/cmake/ChronoTargets.cmake")

add_executable(brick_slope_smc brick_slope_smc.cpp)
target_link_libraries(brick_slope_smc PRIVATE Chrono::Chrono_core)
set_target_properties(brick_slope_smc PROPERTIES BUILD_RPATH "${CHRONO_BUILD}/lib")

if(TARGET Chrono::Chrono_multicore)
  find_package(Thrust REQUIRED)
  find_package(Blaze REQUIRED)
  add_executable(brick_slope_mc brick_slope_mc.cpp)
  target_link_libraries(brick_slope_mc PRIVATE Chrono::Chrono_multicore Chrono::Chrono_core)
  set_target_properties(brick_slope_mc PROPERTIES BUILD_RPATH "${CHRONO_BUILD}/lib")
endif()
