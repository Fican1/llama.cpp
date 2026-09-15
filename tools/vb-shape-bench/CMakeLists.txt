set(TARGET vb-shape-bench)
add_executable(${TARGET} vb-shape-bench.cpp)
target_link_libraries(${TARGET} PRIVATE ggml ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
