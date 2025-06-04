add_library(imatrix_common STATIC imatrix.cpp)
target_compile_features(imatrix_common PUBLIC cxx_std_17)
target_link_libraries(imatrix_common PRIVATE common llama)
