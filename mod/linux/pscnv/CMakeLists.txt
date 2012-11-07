
set(sys_src "/lib/modules/${CMAKE_SYSTEM_VERSION}/build")

# generating pscnv_kapi.h
add_custom_command(OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/pscnv_kapi.h"
                   COMMAND "${CMAKE_CURRENT_SOURCE_DIR}/kapitest.sh" "${sys_src}"
                           > "${CMAKE_CURRENT_SOURCE_DIR}/pscnv_kapi.h"
                           2> "${CMAKE_CURRENT_BINARY_DIR}/kapitest.err"
                   DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/kapitest.sh"
                   WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

# generating nv98_crypt.fuc.h
add_custom_command(OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/nv98_crypt.fuc.h"
                   COMMAND envyas -m fuc -a "${CMAKE_CURRENT_SOURCE_DIR}/nv98_crypt.fuc"
                           -o "${CMAKE_CURRENT_SOURCE_DIR}/nv98_crypt.fuc.h"
                   DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/nv98_crypt.fuc"
                   WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

# generating nvc0_grhub.fuc.h
add_custom_command(OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/nvc0_grhub.fuc.h"
                   COMMAND m4 "${CMAKE_CURRENT_SOURCE_DIR}/nvc0_grhub.fuc" | envyas -a -w -m fuc -V nva3
                           -o "${CMAKE_CURRENT_SOURCE_DIR}/nvc0_grhub.fuc.h"
                   DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/nvc0_grhub.fuc"
                   WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

# generating nvc0_grgpc.fuc.h
add_custom_command(OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/nvc0_grgpc.fuc.h"
                   COMMAND m4 "${CMAKE_CURRENT_SOURCE_DIR}/nvc0_grgpc.fuc" | envyas -a -w -m fuc -V nva3
                           -o "${CMAKE_CURRENT_SOURCE_DIR}/nvc0_grgpc.fuc.h"
                   DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/nvc0_grgpc.fuc"
                   WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

# generating nvc0_copy.fuc.h
add_custom_command(OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/nvc0_copy.fuc.h"
                   COMMAND m4 -DNVC0 "${CMAKE_CURRENT_SOURCE_DIR}/nva3_copy.fuc" | envyas -a -w -m fuc -V nva3
                           -o "${CMAKE_CURRENT_SOURCE_DIR}/nvc0_copy.fuc.h"
                   DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/nva3_copy.fuc"
                   WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")


# generating makefile for building kernel module
set(sources
    nouveau_drv
    nouveau_state
    nouveau_irq
    nouveau_bios
    nouveau_hw
    nouveau_mem
    nouveau_i2c
    nouveau_calc
    nouveau_dp
    nouveau_connector
    nouveau_display
    nouveau_fbcon
    nouveau_dma
    nouveau_pm
    nouveau_volt
    nouveau_perf
    nouveau_temp
    nv04_tv
    nv04_dfp
    nv04_dac
    nv04_timer
    nv10_gpio
    nv50_gpio
    nv50_grctx
    nv50_display
    nv50_crtc
    nv50_cursor
    nv50_calc
    nv50_dac
    nv50_sor
    nv04_pm
    nv50_pm
    nva3_pm
    pscnv_mm
    pscnv_mem
    pscnv_vm
    pscnv_gem
    pscnv_ioctl
    pscnv_ramht
    pscnv_chan
    pscnv_sysram
    nv50_vram
    nv50_vm
    nv50_chan
    nv50_fifo
    nv50_graph
    nv84_crypt
    nv98_crypt
    nvc0_vram
    nvc0_vm
    nvc0_chan
    nvc0_fifo
    nvc0_copy
    nvc0_graph
    nvc0_grctx
    nv40_counter
    gdev_interface
    )

#set(makefile "${CMAKE_CURRENT_BINARY_DIR}/build/Makefile")
#
#set(all_sources "")
#foreach(src ${sources})
#    set(all_sources "${all_sources} ${src}.o")
#endforeach()
#
#file(WRITE "${makefile}" "pscnv-y := ${all_sources}\n")
#
#foreach(src ${sources})
#    file(APPEND "${makefile}" "${src}.o: ${CMAKE_CURRENT_SOURCE_DIR}/${src}.c\n")
#endforeach()
#
#file(APPEND "${makefile}" "pscnv-$(CONFIG_DRM_NOUVEAU_DEBUG) += nouveau_debugfs.o\n")
#file(APPEND "${makefile}" "pscnv-$(CONFIG_COMPAT) += nouveau_ioc32.o\n")
#file(APPEND "${makefile}" "pscnv-$(CONFIG_DRM_NOUVEAU_BACKLIGHT) += nouveau_backlight.o\n")
#file(APPEND "${makefile}" "pscnv-$(CONFIG_ACPI) += nouveau_acpi.o\n")
#
#file(APPEND "${makefile}" "obj-m := pscnv.o\n")

set(real_sources)
foreach(src ${sources})
    list(APPEND real_sources "${CMAKE_CURRENT_SOURCE_DIR}/${src}.c")
endforeach()

add_custom_command(OUTPUT "${CMAKE_CURRENT_SOURCE_DIR}/pscnv.ko"
                   COMMAND "${CMAKE_MAKE_PROGRAM}" -C "${sys_src}" "M=${CMAKE_CURRENT_SOURCE_DIR}"
                   DEPENDS ${real_sources}
                           "${CMAKE_CURRENT_SOURCE_DIR}/pscnv_kapi.h"
                           "${CMAKE_CURRENT_SOURCE_DIR}/nv98_crypt.fuc.h"
                           "${CMAKE_CURRENT_SOURCE_DIR}/nvc0_grhub.fuc.h"
                           "${CMAKE_CURRENT_SOURCE_DIR}/nvc0_grgpc.fuc.h"
                           "${CMAKE_CURRENT_SOURCE_DIR}/nvc0_copy.fuc.h"
                   WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}")

add_custom_command(OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/pscnv.ko"
                   COMMAND "${CMAKE_COMMAND}" -E copy "${CMAKE_CURRENT_SOURCE_DIR}/pscnv.ko"
                                                      "${CMAKE_CURRENT_BINARY_DIR}/pscnv.ko"
                   DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/pscnv.ko")

add_custom_target(pscnv_ko ALL DEPENDS "${CMAKE_CURRENT_BINARY_DIR}/pscnv.ko")

