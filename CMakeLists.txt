cmake_minimum_required(VERSION 3.10)
project(ModelConverter CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Assimp
find_package(assimp CONFIG REQUIRED)

# nlohmann-json
find_package(nlohmann_json CONFIG REQUIRED)

add_executable(ModelConverter main.cpp)
target_link_libraries(ModelConverter PRIVATE assimp::assimp nlohmann_json::nlohmann_json)