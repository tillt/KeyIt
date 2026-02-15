cmake_minimum_required(VERSION 3.20)

project(keyit
    VERSION 0.1.0
    DESCRIPTION "Musical key detection for macOS using CoreML and Accelerate"
    LANGUAGES CXX OBJCXX)

if(APPLE AND NOT DEFINED CMAKE_OSX_ARCHITECTURES)
    if(CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_CONFIGURATION_TYPES)
        set(CMAKE_OSX_ARCHITECTURES "x86_64;arm64")
    endif()
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(KEYIT_LIB_SOURCES
    src/keyit/analysis.cpp
    src/keyit/analysis_features_internal.cpp
    src/keyit/analysis_inference_internal.cpp
    src/keyit/coreml.mm
)
set(KEYIT_PUBLIC_HEADERS
    ${CMAKE_CURRENT_SOURCE_DIR}/include/keyit/keyit.h
)

add_library(keyit ${KEYIT_LIB_SOURCES})

set_target_properties(keyit PROPERTIES
    OUTPUT_NAME keyit
)

target_include_directories(keyit
    PUBLIC
        ${CMAKE_CURRENT_SOURCE_DIR}/include
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/src
)

add_executable(keyit-cli
    src/cli/main.mm
)

target_link_libraries(keyit-cli
    PRIVATE
        keyit
)

if(APPLE)
    target_compile_options(keyit PRIVATE -Wall -Wextra -Wpedantic -fobjc-arc)
    target_compile_options(keyit-cli PRIVATE -Wall -Wextra -Wpedantic -fobjc-arc)
    target_link_libraries(keyit PRIVATE
        "-framework Accelerate"
        "-framework CoreML"
        "-framework Foundation"
    )
    target_link_libraries(keyit-cli PRIVATE
        "-framework AVFoundation"
        "-framework Foundation"
    )

    add_library(keyit_framework SHARED
        ${KEYIT_LIB_SOURCES}
        ${KEYIT_PUBLIC_HEADERS}
    )

    target_include_directories(keyit_framework
        PUBLIC
            ${CMAKE_CURRENT_SOURCE_DIR}/include
        PRIVATE
            ${CMAKE_CURRENT_SOURCE_DIR}/src
    )

    set_target_properties(keyit_framework PROPERTIES
        FRAMEWORK TRUE
        FRAMEWORK_VERSION A
        OUTPUT_NAME "KeyIt"
        MACOSX_FRAMEWORK_IDENTIFIER "com.tilltoenshoff.keyit"
        PUBLIC_HEADER "${KEYIT_PUBLIC_HEADERS}"
    )

    foreach(header_file IN LISTS KEYIT_PUBLIC_HEADERS)
        set_source_files_properties("${header_file}" PROPERTIES
            MACOSX_PACKAGE_LOCATION "Headers/keyit"
        )
    endforeach()

    target_compile_options(keyit_framework PRIVATE -Wall -Wextra -Wpedantic -fobjc-arc)
    target_link_libraries(keyit_framework PRIVATE
        "-framework Accelerate"
        "-framework CoreML"
        "-framework Foundation"
    )

    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/models/keynet.mlmodelc")
        add_custom_command(TARGET keyit_framework POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${CMAKE_CURRENT_SOURCE_DIR}/models/keynet.mlmodelc"
                "$<TARGET_FILE_DIR:keyit_framework>/Resources/keynet.mlmodelc"
            COMMENT "Copying keynet.mlmodelc into KeyIt.framework resources"
        )
    elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/models/keynet.mlpackage")
        add_custom_command(TARGET keyit_framework POST_BUILD
            COMMAND ${CMAKE_COMMAND} -E copy_directory
                "${CMAKE_CURRENT_SOURCE_DIR}/models/keynet.mlpackage"
                "$<TARGET_FILE_DIR:keyit_framework>/Resources/keynet.mlpackage"
            COMMENT "Copying keynet.mlpackage into KeyIt.framework resources"
        )
    endif()
endif()

enable_testing()

add_executable(keyit_tests
    tests/keyit_tests.cpp
)

target_link_libraries(keyit_tests
    PRIVATE
        keyit
)

if(APPLE)
    target_compile_options(keyit_tests PRIVATE -Wall -Wextra -Wpedantic)
    target_link_libraries(keyit_tests PRIVATE
        "-framework Accelerate"
        "-framework CoreML"
        "-framework Foundation"
    )
endif()

add_test(NAME keyit_tests COMMAND keyit_tests)

find_package(Python3 COMPONENTS Interpreter)

set(KEYIT_TEST_AUDIO_DIR "${CMAKE_CURRENT_SOURCE_DIR}/tests/audio")
set(KEYIT_TEST_AUDIO_FILES
    "${KEYIT_TEST_AUDIO_DIR}/sine_a4_10s.wav"
    "${KEYIT_TEST_AUDIO_DIR}/c_major_triad_12s.wav"
    "${KEYIT_TEST_AUDIO_DIR}/b_major_triad_12s.wav"
    "${KEYIT_TEST_AUDIO_DIR}/e_major_triad_12s.wav"
    "${KEYIT_TEST_AUDIO_DIR}/a_minor_triad_12s.wav"
    "${KEYIT_TEST_AUDIO_DIR}/c_major_to_a_minor_20s.wav"
    "${KEYIT_TEST_AUDIO_DIR}/low_noise_8s.wav"
)

if(Python3_Interpreter_FOUND)
    add_custom_command(
        OUTPUT ${KEYIT_TEST_AUDIO_FILES}
        COMMAND ${CMAKE_COMMAND} -E make_directory "${KEYIT_TEST_AUDIO_DIR}"
        COMMAND ${Python3_EXECUTABLE} "${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_test_wavs.py"
            --out-dir "${KEYIT_TEST_AUDIO_DIR}"
        DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_test_wavs.py"
        COMMENT "Generating keyit test WAV fixtures"
    )
    add_custom_target(generate_keyit_test_audio DEPENDS ${KEYIT_TEST_AUDIO_FILES})
endif()

add_executable(keyit_pipeline_tests
    tests/keyit_pipeline_tests.cpp
)

target_link_libraries(keyit_pipeline_tests
    PRIVATE
        keyit
)

target_compile_definitions(keyit_pipeline_tests PRIVATE
    KEYIT_TEST_AUDIO_DIR="${KEYIT_TEST_AUDIO_DIR}"
    KEYIT_TEST_MODEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/models/keynet.mlmodelc"
    KEYIT_TEST_CLI_PATH="$<TARGET_FILE:keyit-cli>"
)

if(APPLE)
    target_compile_options(keyit_pipeline_tests PRIVATE -Wall -Wextra -Wpedantic)
    target_link_libraries(keyit_pipeline_tests PRIVATE
        "-framework Accelerate"
        "-framework CoreML"
        "-framework Foundation"
    )
endif()

if(TARGET generate_keyit_test_audio)
    add_dependencies(keyit_pipeline_tests generate_keyit_test_audio)
endif()

add_test(NAME keyit_pipeline_tests COMMAND keyit_pipeline_tests)

add_executable(keyit_gpu_tests
    tests/keyit_gpu_tests.cpp
)

target_link_libraries(keyit_gpu_tests
    PRIVATE
        keyit
)

target_compile_definitions(keyit_gpu_tests PRIVATE
    KEYIT_TEST_AUDIO_DIR="${KEYIT_TEST_AUDIO_DIR}"
    KEYIT_TEST_MODEL_PATH="${CMAKE_CURRENT_SOURCE_DIR}/models/keynet.mlmodelc"
    KEYIT_TEST_CLI_PATH="$<TARGET_FILE:keyit-cli>"
)

if(APPLE)
    target_compile_options(keyit_gpu_tests PRIVATE -Wall -Wextra -Wpedantic)
    target_link_libraries(keyit_gpu_tests PRIVATE
        "-framework Accelerate"
        "-framework CoreML"
        "-framework Foundation"
    )
endif()

if(TARGET generate_keyit_test_audio)
    add_dependencies(keyit_gpu_tests generate_keyit_test_audio)
endif()

add_test(NAME keyit_gpu_tests COMMAND keyit_gpu_tests)
set_tests_properties(keyit_gpu_tests PROPERTIES
    SKIP_RETURN_CODE 77
)

if(APPLE AND EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/models/keynet.mlmodelc")
    add_custom_command(TARGET keyit-cli POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_directory
            "${CMAKE_CURRENT_SOURCE_DIR}/models/keynet.mlmodelc"
            "$<TARGET_FILE_DIR:keyit-cli>/keynet.mlmodelc"
        COMMENT "Copying keynet.mlmodelc next to keyit-cli"
    )
endif()

if(APPLE)
    install(TARGETS keyit keyit-cli
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
    )
    install(TARGETS keyit_framework
        FRAMEWORK DESTINATION Library/Frameworks
    )
else()
    install(TARGETS keyit keyit-cli
        RUNTIME DESTINATION bin
        LIBRARY DESTINATION lib
        ARCHIVE DESTINATION lib
    )
endif()
