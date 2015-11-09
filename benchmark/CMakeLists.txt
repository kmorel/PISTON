find_package(VTK)
include(${VTK_USE_FILE})


#cuda_add_executable(cayley_timing_GPU cayley_timing.cu)
#target_link_libraries(cayley_timing_GPU pthread)

#add_executable(cayley_timing_OMP cayley_timing.cpp)
#set_target_properties(cayley_timing_OMP PROPERTIES COMPILE_FLAGS "-fopenmp -DTHRUST_DEVICE_BACKEND=THRUST_DEVICE_BACKEND_OMP")
#target_link_libraries(cayley_timing_OMP pthread gomp)

link_directories("/projects/opt/intel/compilers/composer_xe_2015.1.133/tbb/lib/intel64/gcc4.4")

cuda_add_executable(marchingCubeGPU marching_cube.cu)
target_link_libraries(marchingCubeGPU vtkImaging pthread)

add_executable(marchingCubeOMP marching_cube.cpp)
set_target_properties(marchingCubeOMP PROPERTIES COMPILE_FLAGS "-fopenmp -DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_OMP")
target_link_libraries(marchingCubeOMP pthread gomp)

add_executable(marchingCubeCPP marching_cube.cpp)
set_target_properties(marchingCubeCPP PROPERTIES COMPILE_FLAGS "-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_CPP")

add_executable(marchingCubeTBB marching_cube.cpp)
set_target_properties(marchingCubeTBB PROPERTIES COMPILE_FLAGS "-DTHRUST_DEVICE_SYSTEM=THRUST_DEVICE_SYSTEM_TBB -I/projects/opt/intel/compilers/composer_xe_2015.1.133/tbb/include")
target_link_libraries(marchingCubeTBB tbb pthread)

#cuda_add_executable(marching_tetra marching_tetra.cu)
#target_link_libraries(marching_tetra vtkImaging pthread)

#add_executable(vtkTetra vtkTetra.cpp)
#target_link_libraries(vtkTetra vtkImaging vtkGraphics)
