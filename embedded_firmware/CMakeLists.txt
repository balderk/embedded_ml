#THIS FILE IS AUTO GENERATED FROM THE TEMPLATE! DO NOT CHANGE!
SET(CMAKE_SYSTEM_NAME Generic)
SET(CMAKE_SYSTEM_VERSION 1)
cmake_minimum_required(VERSION 3.7)

# specify cross compilers and tools
SET(CMAKE_C_COMPILER_WORKS 1)
SET(CMAKE_C_COMPILER arm-none-eabi-gcc)
SET(CMAKE_CXX_COMPILER_WORKS 1)
SET(CMAKE_CXX_COMPILER arm-none-eabi-g++)
set(CMAKE_ASM_COMPILER arm-none-eabi-gcc)
set(CMAKE_AR arm-none-eabi-ar)
set(CMAKE_OBJCOPY arm-none-eabi-objcopy)
set(CMAKE_OBJDUMP arm-none-eabi-objdump)
set(SIZE arm-none-eabi-size)

SET(LINKER_SCRIPT ${CMAKE_SOURCE_DIR}/STM32F767ZITx_FLASH.ld)

#Uncomment for hardware floating point
SET(FPU_FLAGS "-mfloat-abi=hard  -mfpu=fpv5-d16")
#add_definitions(-DARM_MATH_CM4 -DARM_MATH_MATRIX_CHECK -DARM_MATH_ROUNDING -D__FPU_PRESENT=1)

#Uncomment for software floating point
#SET(FPU_FLAGS "-mfloat-abi=soft")

SET(EXTRA_OPTS "-falign-functions=32 -falign-labels=32 -falign-loops=32 \
        ")


SET(COMMON_FLAGS
        "-mcpu=cortex-m7 ${FPU_FLAGS} -mthumb -mthumb-interwork -ffunction-sections -fdata-sections \
    -g -fno-common -fmessage-length=0 -specs=nosys.specs -specs=nano.specs\
     ${EXTRA_OPTS}")

SET(OPT_LEVEL "-O2")

SET(CMAKE_CXX_FLAGS_INIT "${COMMON_FLAGS} -std=c++11 ${OPT_LEVEL}")
SET(CMAKE_C_FLAGS_INIT "${COMMON_FLAGS} -std=gnu18 ${OPT_LEVEL}")
SET(CMAKE_EXE_LINKER_FLAGS_INIT "-Wl,-gc-sections,--print-memory-usage -T ${LINKER_SCRIPT}")

PROJECT(embedded_firmware C CXX ASM)
set(CMAKE_CXX_STANDARD 11)

#add_definitions(-DARM_MATH_CM4 -DARM_MATH_MATRIX_CHECK -DARM_MATH_ROUNDING -D__FPU_PRESENT=1)
add_definitions(-DUSE_HAL_DRIVER -DSTM32F767xx -DDEBUG)

file(GLOB_RECURSE SOURCES "startup/*.*" "Drivers/*.*" "Src/*.*" )

file(GLOB_RECURSE STATIC_LIBRARY "Middlewares/ST/AI/Lib/*.a")

include_directories(
        Inc
        Middlewares/ST/AI/Inc
        Drivers/STM32F7xx_HAL_Driver/Inc
        Drivers/STM32F7xx_HAL_Driver/Inc/Legacy
        Drivers/CMSIS/Device/ST/STM32F7xx/Include
        Drivers/CMSIS/Include
)


add_executable(${PROJECT_NAME}.elf ${SOURCES} ${LINKER_SCRIPT})

target_link_libraries(${PROJECT_NAME}.elf ${STATIC_LIBRARY})

set(CMAKE_EXE_LINKER_FLAGS
        "${CMAKE_EXE_LINKER_FLAGS} -Wl,-Map=${PROJECT_BINARY_DIR}/${PROJECT_NAME}.map")

set(HEX_FILE ${PROJECT_BINARY_DIR}/${PROJECT_NAME}.hex)
set(BIN_FILE ${PROJECT_BINARY_DIR}/${PROJECT_NAME}.bin)

add_custom_command(TARGET ${PROJECT_NAME}.elf POST_BUILD
        COMMAND ${CMAKE_OBJCOPY} -Oihex $<TARGET_FILE:${PROJECT_NAME}.elf> ${HEX_FILE}
        COMMAND ${CMAKE_OBJCOPY} -Obinary $<TARGET_FILE:${PROJECT_NAME}.elf> ${BIN_FILE}
        COMMENT "Building ${HEX_FILE}
Building ${BIN_FILE}")
