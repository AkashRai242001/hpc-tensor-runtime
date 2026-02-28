cmake_minimum_required(VERSION 3.20)

project(hpc_tensor_runtime
        VERSION 0.1
        LANGUAGES CXX)

# Use modern C++
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# High-performance build flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")

# Create tensor library
add_library(tensor
    src/core/tensor.cpp
)

install(
  TARGETS tensor
  EXPORT hpc_tensor_runtimeTargets
  ARCHIVE DESTINATION lib
  LIBRARY DESTINATION lib
  RUNTIME DESTINATION bin
)

install(
  DIRECTORY include/
  DESTINATION include
)

# Public include directory
target_include_directories(tensor
    PUBLIC
        ${PROJECT_SOURCE_DIR}/include
)

include(FetchContent)
enable_testing()

FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/refs/tags/v1.14.0.zip
)

set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

add_executable(test_tensor
    tests/test_tensor.cpp
)

target_link_libraries(test_tensor
    PRIVATE tensor GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(test_tensor)

add_custom_target(package_all
    DEPENDS tensor
    COMMAND ${CMAKE_COMMAND} --build ${CMAKE_BINARY_DIR} --target install
    COMMENT "Building and installing the full project"
)

# ------------------------
# CPack configuration
# ------------------------

set(CPACK_PACKAGE_NAME "hpc-tensor-runtime")
set(CPACK_PACKAGE_VENDOR "Akash Rai")
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "High-performance C++ tensor runtime")
set(CPACK_PACKAGE_VERSION ${PROJECT_VERSION})

set(CPACK_GENERATOR "TGZ")

set(CPACK_PACKAGE_FILE_NAME
    "${CPACK_PACKAGE_NAME}-${CPACK_PACKAGE_VERSION}-${CMAKE_SYSTEM_NAME}"
)

include(CPack)