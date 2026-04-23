target_sources(top.lbm-exe PRIVATE main.cpp)



find_package(OpenMP REQUIRED)
target_link_libraries(top.lbm-exe PRIVATE OpenMP::OpenMP_CXX)


target_link_libraries(top.lbm-exe PRIVATE top::lbm-lib MPI::MPI_CXX)

set_target_properties(top.lbm-exe PROPERTIES BUILD_RPATH ${CMAKE_BINARY_DIR}/lib)


target_sources(top.display PRIVATE display.cpp)
target_link_libraries(top.display PRIVATE top::lbm-lib)
set_target_properties(top.display PROPERTIES BUILD_RPATH ${CMAKE_BINARY_DIR}/lib)
