// RUN: cgeist %s --function=foo -S | FileCheck %s

void [[clang::syntax(linalg)]] foo(float x1[10], float A[10][10], float y1[10], float x2[10]) {
  x1(i) += A(i, j) * y1(j)
  x2(i) += A(i, j) * x1(j)
}

// CHECK: #[[MAP:.+]] = affine_map<(d0, d1) -> (d0, d1)>
// CHECK: #[[MAP1:.+]] = affine_map<(d0, d1) -> (d1)>
// CHECK: #[[MAP2:.+]] = affine_map<(d0, d1) -> (d0)>

// CHECK-LABEL: foo
// CHECK-SAME: %[[ARG0:.+]]: memref<?xf32>, %[[ARG1:.+]]: memref<?x10xf32>, %[[ARG2:.+]]: memref<?xf32>, %[[ARG3:.+]]: memref<?xf32>
// CHECK: %[[ARG0_T:.+]] = bufferization.to_tensor %[[ARG0]] : memref<?xf32>
// CHECK: %[[ARG1_T:.+]] = bufferization.to_tensor %[[ARG1]] : memref<?x10xf32>
// CHECK: %[[ARG2_T:.+]] = bufferization.to_tensor %[[ARG2]] : memref<?xf32>
// CHECK: %[[CAST_ARG0_T:.+]] = tensor.cast %[[ARG0_T]] : tensor<?xf32> to tensor<10xf32>
// CHECK: %[[CAST_ARG1_T:.+]] = tensor.cast %[[ARG1_T]] : tensor<?x10xf32> to tensor<10x10xf32>
// CHECK: %[[CAST_ARG2_T:.+]] = tensor.cast %[[ARG2_T]] : tensor<?xf32> to tensor<10xf32>
// CHECK: %[[GEN:.+]] = linalg.generic
// CHECK-SAME:  indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME:  iterator_types = ["parallel", "reduction"]
// CHECK-SAME:  ins(%[[CAST_ARG1_T]], %[[CAST_ARG2_T]] : tensor<10x10xf32>, tensor<10xf32>)
// CHECK-SAME:  outs(%[[CAST_ARG0_T]] : tensor<10xf32>)
// CHECK: ^bb0
// CHECK: arith.mulf
// CHECK-NEXT: arith.addf
// CHECK-NEXT: linalg.yield
// CHECK: %[[ARG3_T:.+]] = bufferization.to_tensor %[[ARG3]] : memref<?xf32>
// CHECK: %[[GEN_1:.+]] = linalg.generic
// CHECK-SAME:  indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME:  iterator_types = ["parallel", "reduction"]
// CHECK-SAME:  ins(%[[ARG1_T]], %[[GEN]] : tensor<?x10xf32>, tensor<10xf32>)
// CHECK-SAME:  outs(%[[ARG3_T]] : tensor<?xf32>)
// CHECK: ^bb0
// CHECK: arith.mulf
// CHECK-NEXT: arith.addf
// CHECK-NEXT: linalg.yield
// CHECK: %{{.+}} = bufferization.materialize_in_destination %[[GEN_1]] in %[[ARG3_T]] : tensor<?xf32>
