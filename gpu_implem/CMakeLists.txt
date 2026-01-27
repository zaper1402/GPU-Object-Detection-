cmake_minimum_required (VERSION 3.14)
project (GPGPU CUDA CXX)

include(FetchContent)
FetchContent_Declare(
  CLI11
  URL https://github.com/CLIUtils/CLI11/archive/v1.9.1.tar.gz
  )

FetchContent_Declare(
  spdlog
  URL https://github.com/gabime/spdlog/archive/v1.9.2.tar.gz
  )

FetchContent_Declare(
    stb_image
    GIT_REPOSITORY https://github.com/nothings/stb
)

FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.11.2/json.tar.xz)

FetchContent_MakeAvailable(json)
FetchContent_MakeAvailable(spdlog)
FetchContent_MakeAvailable(CLI11)
FetchContent_MakeAvailable(stb_image)

include_directories(${stb_image_SOURCE_DIR})

set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} ${PROJECT_SOURCE_DIR} )

# Find dependencies
find_package(ZLIB QUIET)
find_package(PNG QUIET)

if(NOT PNG_FOUND)
    # Try to find PNG in job-specific nvme storage or user directories
    set(SEARCH_PATHS
        "$ENV{HOME}/local"
        "${CMAKE_BINARY_DIR}/local"
    )
    
    file(GLOB NVME_JOB_DIRS "/run/nvme/job_*")
    foreach(JOB_DIR ${NVME_JOB_DIRS})
        list(APPEND SEARCH_PATHS "${JOB_DIR}/local")
    endforeach()
    
    foreach(SEARCH_PATH ${SEARCH_PATHS})
        if(EXISTS "${SEARCH_PATH}/lib/libpng.so" OR EXISTS "${SEARCH_PATH}/lib/libpng.a")
            set(CMAKE_PREFIX_PATH "${SEARCH_PATH}" ${CMAKE_PREFIX_PATH})
            find_package(PNG QUIET)
            if(PNG_FOUND)
                message(STATUS "Found PNG in: ${SEARCH_PATH}")
                break()
            endif()
        endif()
    endforeach()
endif()

if(NOT PNG_FOUND)
    # Build PNG from source if not found
    message(STATUS "PNG not found. Building from source...")
    
    # Use build directory for installation (always writable)
    set(PNG_INSTALL_DIR "${CMAKE_BINARY_DIR}/local")
    
    message(STATUS "PNG will be installed to: ${PNG_INSTALL_DIR}")
    
    FetchContent_Declare(
        libpng
        URL https://downloads.sourceforge.net/libpng/libpng-1.6.39.tar.gz
        URL_HASH SHA256=af4fb7f260f839919e5958e5ab01a275d4fe436d45442a36ee62f73e5beb75ba
    )
    
    FetchContent_GetProperties(libpng)
    if(NOT libpng_POPULATED)
        FetchContent_Populate(libpng)
        
        # Check if zlib is needed for configure
        set(ZLIB_CONFIG "")
        if(ZLIB_FOUND)
            set(ZLIB_CONFIG "--with-zlib-prefix=${ZLIB_ROOT}")
        endif()
        
        # Build libpng
        execute_process(
            COMMAND ${libpng_SOURCE_DIR}/configure --prefix=${PNG_INSTALL_DIR} ${ZLIB_CONFIG}
            WORKING_DIRECTORY ${libpng_SOURCE_DIR}
            RESULT_VARIABLE CONFIGURE_RESULT
            OUTPUT_VARIABLE CONFIGURE_OUTPUT
            ERROR_VARIABLE CONFIGURE_ERROR
        )
        
        if(CONFIGURE_RESULT EQUAL 0)
            execute_process(
                COMMAND make -j4
                WORKING_DIRECTORY ${libpng_SOURCE_DIR}
                RESULT_VARIABLE MAKE_RESULT
                OUTPUT_VARIABLE MAKE_OUTPUT
                ERROR_VARIABLE MAKE_ERROR
            )
            
            if(MAKE_RESULT EQUAL 0)
                execute_process(
                    COMMAND make install
                    WORKING_DIRECTORY ${libpng_SOURCE_DIR}
                    RESULT_VARIABLE INSTALL_RESULT
                    OUTPUT_VARIABLE INSTALL_OUTPUT
                    ERROR_VARIABLE INSTALL_ERROR
                )
                
                if(INSTALL_RESULT EQUAL 0)
                    set(CMAKE_PREFIX_PATH "${PNG_INSTALL_DIR}" ${CMAKE_PREFIX_PATH})
                    set(PNG_PNG_INCLUDE_DIR "${PNG_INSTALL_DIR}/include")
                    set(PNG_LIBRARY "${PNG_INSTALL_DIR}/lib/libpng.so")
                    
                    find_package(PNG QUIET)
                    if(PNG_FOUND)
                        message(STATUS "PNG built and installed successfully to ${PNG_INSTALL_DIR}")
                    else()
                        # Manual setup if find_package still fails
                        set(PNG_FOUND TRUE)
                        set(PNG_INCLUDE_DIRS "${PNG_INSTALL_DIR}/include")
                        set(PNG_LIBRARIES "${PNG_INSTALL_DIR}/lib/libpng.so")
                        message(STATUS "PNG manually configured at ${PNG_INSTALL_DIR}")
                    endif()
                else()
                    message(WARNING "Failed to install PNG from source: ${INSTALL_ERROR}")
                endif()
            else()
                message(WARNING "Failed to build PNG from source: ${MAKE_ERROR}")
            endif()
        else()
            message(WARNING "Failed to configure PNG from source: ${CONFIGURE_ERROR}")
        endif()
    endif()
endif()

find_package(TBB QUIET)

if(NOT TBB_FOUND)
    message(STATUS "TBB not found. Building from source...")
    
    FetchContent_Declare(
        TBB
        GIT_REPOSITORY https://github.com/oneapi-src/oneTBB.git
        GIT_TAG v2021.5.0
    )
    
    set(TBB_TEST OFF CACHE BOOL "Disable TBB tests")
    set(TBB_EXAMPLES OFF CACHE BOOL "Disable TBB examples")
    
    FetchContent_MakeAvailable(TBB)
    
    set(TBB_FOUND TRUE)
    message(STATUS "TBB built from source successfully")
endif()

find_package(benchmark QUIET)

if(NOT benchmark_FOUND)
    message(STATUS "Google Benchmark not found. Building from source...")
    
    FetchContent_Declare(
        benchmark
        GIT_REPOSITORY https://github.com/google/benchmark.git
        GIT_TAG v1.7.1
    )
    
    set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Disable benchmark tests")
    set(BENCHMARK_ENABLE_GTEST_TESTS OFF CACHE BOOL "Disable gtest tests")
    set(BENCHMARK_ENABLE_INSTALL OFF CACHE BOOL "Disable benchmark install")
    
    FetchContent_MakeAvailable(benchmark)
    
    set(benchmark_FOUND TRUE)
    message(STATUS "Google Benchmark built from source successfully")
endif()

set(CMAKE_CUDA_ARCHITECTURES "75")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -Wextra -pedantic -Wall -Wno-missing-field-initializers")
set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")

add_library(images)
target_compile_features(images PUBLIC cxx_std_14)
target_link_libraries(images PRIVATE spdlog::spdlog)
target_sources(images PRIVATE src/images.hpp src/images.cu)
set_target_properties(images PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# Executables
add_executable(gpu_implem src/main.cpp)
target_compile_features(gpu_implem PUBLIC cxx_std_17)
target_link_libraries(gpu_implem PRIVATE 
    images 
    spdlog::spdlog 
    CLI11::CLI11
    nlohmann_json::nlohmann_json
)

if(PNG_FOUND)
    if(TARGET PNG::PNG)
        target_link_libraries(gpu_implem PRIVATE PNG::PNG)
    else()
        target_include_directories(gpu_implem PRIVATE ${PNG_INCLUDE_DIRS})
        target_link_libraries(gpu_implem PRIVATE ${PNG_LIBRARIES})
    endif()
    target_compile_definitions(gpu_implem PRIVATE HAS_PNG)
    message(STATUS "PNG support enabled")
else()
    message(WARNING "Building without PNG support")
endif()

if(TBB_FOUND)
    if(TARGET TBB::tbb)
        target_link_libraries(gpu_implem PRIVATE TBB::tbb)
    else()
        target_link_libraries(gpu_implem PRIVATE tbb)
    endif()
    target_compile_definitions(gpu_implem PRIVATE HAS_TBB)
    message(STATUS "TBB support enabled")
else()
    message(WARNING "Building without TBB support")
endif()

if(benchmark_FOUND)
    if(TARGET benchmark::benchmark)
        target_link_libraries(gpu_implem PRIVATE benchmark::benchmark)
    else()
        target_link_libraries(gpu_implem PRIVATE benchmark)
    endif()
    target_compile_definitions(gpu_implem PRIVATE HAS_BENCHMARK)
    message(STATUS "Benchmark support enabled")
else()
    message(WARNING "Building without benchmark support")
endif()



