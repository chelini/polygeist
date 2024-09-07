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
// CHECK: linalg.generic
// CHECK-SAME:  indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME:  iterator_types = ["parallel", "reduction"]
// CHECK-SAME:  ins(%[[ARG1]], %[[ARG2]] : memref<?x10xf32>, memref<?xf32>)
// CHECK-SAME:  outs(%[[ARG0]] : memref<?xf32>)
// CHECK: linalg.generic
// CHECK-SAME:  indexing_maps = [#[[MAP]], #[[MAP1]], #[[MAP2]]]
// CHECK-SAME:  iterator_types = ["parallel", "reduction"]
// CHECK-SAME:  ins(%[[ARG1]], %[[ARG0]] : memref<?x10xf32>, memref<?xf32>)
// CHECK-SAME:  outs(%[[ARG3]] : memref<?xf32>)
