# CMake generated Testfile for 
# Source directory: /home/remon-emad/HPC/project/opencv/modules/ml
# Build directory: /home/remon-emad/HPC/project/build/modules/ml
# 
# This file includes the relevant testing commands required for 
# testing this directory and lists subdirectories to be tested as well.
add_test(opencv_test_ml "/home/remon-emad/HPC/project/build/bin/opencv_test_ml" "--gtest_output=xml:opencv_test_ml.xml")
set_tests_properties(opencv_test_ml PROPERTIES  LABELS "Main;opencv_ml;Accuracy" WORKING_DIRECTORY "/home/remon-emad/HPC/project/build/test-reports/accuracy" _BACKTRACE_TRIPLES "/home/remon-emad/HPC/project/opencv/cmake/OpenCVUtils.cmake;1763;add_test;/home/remon-emad/HPC/project/opencv/cmake/OpenCVModule.cmake;1375;ocv_add_test_from_target;/home/remon-emad/HPC/project/opencv/cmake/OpenCVModule.cmake;1133;ocv_add_accuracy_tests;/home/remon-emad/HPC/project/opencv/modules/ml/CMakeLists.txt;2;ocv_define_module;/home/remon-emad/HPC/project/opencv/modules/ml/CMakeLists.txt;0;")
