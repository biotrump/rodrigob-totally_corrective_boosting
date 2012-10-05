# This is a CMake build file, for more information consult:
# http://en.wikipedia.org/wiki/CMake
# and
# http://www.cmake.org/Wiki/CMake
# http://www.cmake.org/cmake/help/syntax.html
# http://www.cmake.org/Wiki/CMake_Useful_Variables
# http://www.cmake.org/cmake/help/cmake-2-8-docs.html

# to compile the local code you can use: cmake ./ && make -j2

# ----------------------------------------------------------------------
# Base CMake setup
cmake_minimum_required(VERSION 2.6)
set(src_folder "../..")

# ----------------------------------------------------------------------
# Setup the project
include(FindPkgConfig)
project (TotallyCorrectiveEntropyRegularizedBoosting)

# ----------------------------------------------------------------------
# Collect source files

file(GLOB SrcCpp 
  "./*.cpp"
  "${src_folder}/*.cpp"
)


# ----------------------------------------------------------------------
# Create the executable

add_executable(test_erlpboost
  ${SrcCpp}
  ${HelpersCpp}
)

target_link_libraries(test_erlpboost
   boost_program_options-mt boost_filesystem-mt boost_system-mt boost_thread-mt protobuf
   gomp 
   
   #csparse sparse spblas mv
   #lapack blas atlas
)   


