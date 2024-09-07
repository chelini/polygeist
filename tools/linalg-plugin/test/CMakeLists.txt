set(PLUGIN_TEST_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(PLUGIN_BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR})

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(STANDALONE_OPT_TEST_DEPENDS
  FileCheck count not
  linalg-plugin-opt
)

add_lit_testsuite(check-linalg-plugin-opt "Running the linalg-plugin-opt regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS ${STANDALONE_OPT_TEST_DEPENDS}
)
set_target_properties(check-linalg-plugin-opt PROPERTIES FOLDER "Tests")
add_lit_testsuites(STANDALONE_OPT ${CMAKE_CURRENT_SOURCE_DIR}
  DEPENDS ${STANDALONE_OPT_TEST_DEPENDS})
