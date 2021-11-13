cmake_minimum_required(VERSION 3.10)

list(APPEND EXAMPLES
  TestImageClassification
  TestObjectDetection
)

include(cmake_utility)

find_package(OpenCV REQUIRED)

list(APPEND PUBLIC_LIBS
  ${PROJECT_NAME}
  ${OpenCV_LIBS}
)

list(APPEND PRIVATE_LIBS
)

foreach(EXAMPLE ${EXAMPLES})
  __build_example(
    ${EXAMPLE}
    "${PUBLIC_LIBS}"
    "${PRIVATE_LIBS}"
  )
  target_include_directories(${EXAMPLE}
    PUBLIC
      ${OpenCV_INCLUDE_DIRS}
  )
endforeach(EXAMPLE)

