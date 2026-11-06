# Prefer the RoboParty Debian package. Fall back to the checked-in archive.
set(_ort_prefix /opt/roboparty)
execute_process(
  COMMAND dpkg-query -W "-f=\${Status}\n\${Version}" roboparty-onnxruntime
  OUTPUT_VARIABLE _ort_package OUTPUT_STRIP_TRAILING_WHITESPACE
  ERROR_QUIET RESULT_VARIABLE _ort_query_result)
if(CMAKE_CROSSCOMPILING)
  set(_ort_query_result 1)
endif()
set(_ort_system FALSE)
if(_ort_query_result EQUAL 0 AND _ort_package MATCHES "^install ok installed\n([^\n]+)$")
  set(_ort_version "${CMAKE_MATCH_1}")
  execute_process(COMMAND dpkg --compare-versions "${_ort_version}" ge 1.21.0
    RESULT_VARIABLE _ort_version_result)
  if(_ort_version_result EQUAL 0
      AND EXISTS "${_ort_prefix}/include/onnxruntime/onnxruntime_cxx_api.h"
      AND EXISTS "${_ort_prefix}/lib/libonnxruntime.so.1")
    set(_ort_system TRUE)
  endif()
endif()

if(_ort_system)
  set(ONNXRUNTIME_INCLUDE_DIR "${_ort_prefix}/include/onnxruntime")
  set(ONNXRUNTIME_LIB_FILE "${_ort_prefix}/lib/libonnxruntime.so.1")
  set(ONNXRUNTIME_BUNDLED FALSE)
  message(STATUS "ONNX Runtime: roboparty-onnxruntime ${_ort_version}: ${ONNXRUNTIME_LIB_FILE}")
else()
  if(CMAKE_SYSTEM_PROCESSOR STREQUAL "x86_64")
    set(_ort_name onnxruntime-linux-x64-1.21.0)
    set(_ort_hash 7485c7e7aac6501b27e353dcbe068e45c61ab51fbaf598d13970dfae669d20bf)
  elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "aarch64")
    set(_ort_name onnxruntime-linux-aarch64-1.21.0)
    set(_ort_hash 4508084bde1232ee1ab4b6fad2155be0ea2ccab1c1aae9910ddb3fb68a60805e)
  else()
    message(FATAL_ERROR "Unsupported ONNX Runtime architecture: ${CMAKE_SYSTEM_PROCESSOR}")
  endif()
  set(_ort_archive "${CMAKE_CURRENT_LIST_DIR}/${_ort_name}.tgz")
  if(NOT EXISTS "${_ort_archive}")
    set(_ort_archive "${CMAKE_CURRENT_BINARY_DIR}/${_ort_name}.tgz")
    file(DOWNLOAD "https://github.com/microsoft/onnxruntime/releases/download/v1.21.0/${_ort_name}.tgz"
      "${_ort_archive}" EXPECTED_HASH SHA256=${_ort_hash} STATUS _ort_download TLS_VERIFY ON)
    list(GET _ort_download 0 _ort_download_code)
    if(NOT _ort_download_code EQUAL 0)
      message(FATAL_ERROR "ONNX Runtime download failed: ${_ort_download}")
    endif()
  endif()
  file(SHA256 "${_ort_archive}" _ort_actual_hash)
  if(NOT _ort_actual_hash STREQUAL _ort_hash)
    message(FATAL_ERROR "ONNX Runtime tarball checksum mismatch: ${_ort_archive}")
  endif()
  set(_ort_extract_dir "${CMAKE_CURRENT_BINARY_DIR}/onnxruntime-bundled")
  file(MAKE_DIRECTORY "${_ort_extract_dir}")
  set(_ort_root "${_ort_extract_dir}/${_ort_name}")
  if(NOT EXISTS "${_ort_root}/include/onnxruntime_cxx_api.h" OR
     NOT EXISTS "${_ort_root}/lib/libonnxruntime.so.1.21.0" OR
     NOT EXISTS "${_ort_root}/lib/libonnxruntime.so.1" OR
     NOT EXISTS "${_ort_root}/lib/libonnxruntime_providers_shared.so")
    file(REMOVE_RECURSE "${_ort_root}")
    execute_process(COMMAND ${CMAKE_COMMAND} -E tar xzf "${_ort_archive}"
      WORKING_DIRECTORY "${_ort_extract_dir}" RESULT_VARIABLE _ort_extract_result)
    if(NOT _ort_extract_result EQUAL 0)
      message(FATAL_ERROR "Failed to extract ${_ort_archive}")
    endif()
  endif()
  set(ONNXRUNTIME_INCLUDE_DIR "${_ort_root}/include")
  set(ONNXRUNTIME_LIB_FILE "${_ort_root}/lib/libonnxruntime.so.1")
  set(ONNXRUNTIME_BUNDLED TRUE)
  install(FILES "${_ort_root}/lib/libonnxruntime.so.1"
                "${_ort_root}/lib/libonnxruntime.so.1.21.0"
                "${_ort_root}/lib/libonnxruntime_providers_shared.so"
    DESTINATION lib/${PROJECT_NAME}/onnxruntime)
  message(STATUS "ONNX Runtime: bundled 1.21.0: ${ONNXRUNTIME_LIB_FILE}")
endif()
message(STATUS "ONNX Runtime headers: ${ONNXRUNTIME_INCLUDE_DIR}")
