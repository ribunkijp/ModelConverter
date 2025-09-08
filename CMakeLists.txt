# 指定所需的 CMake 最低版本
cmake_minimum_required(VERSION 3.10)

# 设置项目名称和语言
project(ModelConverter CXX)

# 设置 C++ 标准为 C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)


# 添加我们的可执行文件目标
add_executable(ModelConverter main.cpp)

# 将 Assimp 库链接到我们的可执行文件
# 这能确保编译器找到 Assimp 的头文件，链接器找到它的函数
find_package(assimp CONFIG REQUIRED)
target_link_libraries(ModelConverter PRIVATE assimp::assimp)