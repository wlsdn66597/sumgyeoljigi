cmake_minimum_required(VERSION 3.19)
project(yamnet_atlas_tests CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
# 기기 시스템 경로를 건드리지 않도록, 실행 파일 옆 lib/ 폴더의 공유 라이브러리를 먼저 찾는다
set(CMAKE_BUILD_RPATH "$ORIGIN/lib")

find_package(PkgConfig REQUIRED)

# 1) ECO.AI 통합 API (기기에 추론 플러그인 애드온이 있어야 동작)
pkg_check_modules(ECOAI_INFERENCE REQUIRED ecoai-inference)
add_executable(yamnet_ecoai_test yamnet_ecoai_test.cpp)
target_include_directories(yamnet_ecoai_test PRIVATE ${ECOAI_INFERENCE_INCLUDE_DIRS})
target_link_directories(yamnet_ecoai_test PRIVATE ${ECOAI_INFERENCE_LIBRARY_DIRS})
target_link_libraries(yamnet_ecoai_test PRIVATE ${ECOAI_INFERENCE_LIBRARIES})

# 2) TensorFlow Lite 직접 사용 (XNNPACK)
pkg_check_modules(TFLITE QUIET tensorflowlite)
if(NOT TFLITE_FOUND)
  pkg_check_modules(TFLITE REQUIRED tensorflow2-lite)
endif()
add_executable(yamnet_tflite_test yamnet_tflite_test.cpp)
target_include_directories(yamnet_tflite_test PRIVATE ${TFLITE_INCLUDE_DIRS})
target_link_directories(yamnet_tflite_test PRIVATE ${TFLITE_LIBRARY_DIRS})
# pkg-config 목록에는 GPU 위임(EGL·GLES 의존)까지 들어 있다. CPU(XNNPACK)만 쓰므로 필요한 것만 링크한다
target_link_libraries(yamnet_tflite_test PRIVATE tensorflowlite tensorflowlite_xnnpack_delegate)
