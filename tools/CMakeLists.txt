add_subdirectory(polygeist-opt)
add_subdirectory(cgeist)

if(POLYGEIST_ENABLE_POLYMER)
  add_subdirectory(polymer)
endif()

if(POLYGEIST_ENABLE_LINALG_PLUGIN)
  add_subdirectory(linalg-plugin)
endif()
