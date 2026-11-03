# SPDX-License-Identifier: GPL-3.0-only

include(FetchContent)

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    set(ONNXRUNTIME_NAME "onnxruntime-linux-x64-1.21.0")
    set(ONNXRUNTIME_HASH "SHA256=7485c7e7aac6501b27e353dcbe068e45c61ab51fbaf598d13970dfae669d20bf")
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    set(ONNXRUNTIME_NAME "onnxruntime-linux-aarch64-1.21.0")
    set(ONNXRUNTIME_HASH "SHA256=4508084bde1232ee1ab4b6fad2155be0ea2ccab1c1aae9910ddb3fb68a60805e")
else()
    message(FATAL_ERROR "Unsupported architecture: ${CMAKE_SYSTEM_PROCESSOR}")
endif()

message(STATUS "Architecture detected: ${CMAKE_SYSTEM_PROCESSOR}. Using ${ONNXRUNTIME_NAME}")

set(ONNXRUNTIME_LOCAL "${CMAKE_CURRENT_LIST_DIR}/${ONNXRUNTIME_NAME}.tgz")
set(ONNXRUNTIME_URL "https://github.com/microsoft/onnxruntime/releases/download/v1.21.0/${ONNXRUNTIME_NAME}.tgz")
set(ONNXRUNTIME_SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/onnxruntime")

if(EXISTS "${ONNXRUNTIME_LOCAL}")
    message(STATUS "ONNX Runtime: using local tarball ${ONNXRUNTIME_LOCAL}")

    string(REPLACE "SHA256=" "" ONNXRUNTIME_EXPECTED_SHA256 "${ONNXRUNTIME_HASH}")
    file(SHA256 "${ONNXRUNTIME_LOCAL}" ONNXRUNTIME_ACTUAL_SHA256)
    if(NOT ONNXRUNTIME_ACTUAL_SHA256 STREQUAL ONNXRUNTIME_EXPECTED_SHA256)
        message(FATAL_ERROR "ONNX Runtime tarball checksum mismatch: ${ONNXRUNTIME_LOCAL}")
    endif()

    if(NOT EXISTS "${ONNXRUNTIME_SOURCE_DIR}/include/onnxruntime_cxx_api.h")
        message(STATUS "ONNX Runtime: extracting local tarball")
        file(REMOVE_RECURSE "${ONNXRUNTIME_SOURCE_DIR}")
        file(MAKE_DIRECTORY "${ONNXRUNTIME_SOURCE_DIR}")
        execute_process(
            COMMAND ${CMAKE_COMMAND} -E tar xzf "${ONNXRUNTIME_LOCAL}"
            WORKING_DIRECTORY "${ONNXRUNTIME_SOURCE_DIR}"
            RESULT_VARIABLE ONNXRUNTIME_EXTRACT_RESULT)
        if(NOT ONNXRUNTIME_EXTRACT_RESULT EQUAL 0)
            message(FATAL_ERROR "Failed to extract ${ONNXRUNTIME_LOCAL}")
        endif()

        set(ONNXRUNTIME_EXTRACTED_DIR "${ONNXRUNTIME_SOURCE_DIR}/${ONNXRUNTIME_NAME}")
        if(EXISTS "${ONNXRUNTIME_EXTRACTED_DIR}")
            file(GLOB ONNXRUNTIME_EXTRACTED_CHILDREN "${ONNXRUNTIME_EXTRACTED_DIR}/*")
            foreach(ONNXRUNTIME_CHILD ${ONNXRUNTIME_EXTRACTED_CHILDREN})
                get_filename_component(ONNXRUNTIME_CHILD_NAME "${ONNXRUNTIME_CHILD}" NAME)
                file(RENAME "${ONNXRUNTIME_CHILD}" "${ONNXRUNTIME_SOURCE_DIR}/${ONNXRUNTIME_CHILD_NAME}")
            endforeach()
            file(REMOVE_RECURSE "${ONNXRUNTIME_EXTRACTED_DIR}")
        endif()
    endif()
else()
    message(STATUS "ONNX Runtime: downloading ${ONNXRUNTIME_URL}")
    FetchContent_Declare(
        onnxruntime_camera
        URL      "${ONNXRUNTIME_URL}"
        URL_HASH ${ONNXRUNTIME_HASH}
        SOURCE_DIR "${ONNXRUNTIME_SOURCE_DIR}"
    )

    FetchContent_MakeAvailable(onnxruntime_camera)
endif()

set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_SOURCE_DIR}/include")
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_SOURCE_DIR}/lib")
set(ONNXRUNTIME_LIB_FILE "${ONNXRUNTIME_LIB_DIR}/libonnxruntime.so")

set(ONNXRUNTIME_INCLUDE_DIR ${ONNXRUNTIME_INCLUDE_DIR} PARENT_SCOPE)
set(ONNXRUNTIME_LIB_DIR ${ONNXRUNTIME_LIB_DIR} PARENT_SCOPE)
set(ONNXRUNTIME_LIB_FILE ${ONNXRUNTIME_LIB_FILE} PARENT_SCOPE)
