#map = affine_map<(d0)[s0, s1] -> ((d0 - s0) ceildiv s1)>
#map1 = affine_map<(d0)[s0, s1] -> (d0 * s0 + s1)>
#map2 = affine_map<(d0) -> (d0 * 256)>
module attributes {gpu.container_module} {
  func.func @test(%arg0: memref<512x512xf16>, %arg1: memref<512x512xf16>, %arg2: memref<512x512xf32>) {
    %0 = ub.poison : f32
    %1 = ub.poison : f16
    %cst = arith.constant dense<0.000000e+00> : vector<512x512xf32>
    %c32 = arith.constant 32 : index
    %c512 = arith.constant 512 : index
    %c0 = arith.constant 0 : index
    %alloc = memref.alloc() {alignment = 64 : i64} : memref<512x512xf32>
    vector.transfer_write %cst, %alloc[%c0, %c0] {in_bounds = [true, true]} : vector<512x512xf32>, memref<512x512xf32>
    %c0_0 = arith.constant 0 : index
    %c0_1 = arith.constant 0 : index
    %c2 = arith.constant 2 : index
    %c2_2 = arith.constant 2 : index
    %c1 = arith.constant 1 : index
    %c1_3 = arith.constant 1 : index
    %c1_4 = arith.constant 1 : index
    %2 = affine.apply #map(%c2)[%c0_0, %c1]
    %3 = affine.apply #map(%c2_2)[%c0_1, %c1_3]
    %c1024 = arith.constant 1024 : index
    %c1_5 = arith.constant 1 : index
    %c1_6 = arith.constant 1 : index
    gpu.launch_func  @matmul_kernel::@matmul_kernel blocks in (%2, %3, %c1_4) threads in (%c1024, %c1_5, %c1_6)  args(%alloc : memref<512x512xf32>, %arg0 : memref<512x512xf16>, %arg1 : memref<512x512xf16>)
    memref.copy %alloc, %arg2 : memref<512x512xf32> to memref<512x512xf32>
    return
  }
  gpu.module @matmul_kernel [#xevm.target<O = 3>] {
    gpu.func @matmul_kernel(%arg0: memref<512x512xf32>, %arg1: memref<512x512xf16>, %arg2: memref<512x512xf16>) kernel attributes {known_block_size = array<i32: 1024, 1, 1>} {
      %c4 = arith.constant 4 : index
      %c32 = arith.constant 32 : index
      %c512 = arith.constant 512 : index
      %c0 = arith.constant 0 : index
      %c1 = arith.constant 1 : index
      %block_id_x = gpu.block_id x
      %block_id_y = gpu.block_id y
      %0 = affine.apply #map1(%block_id_x)[%c1, %c0]
      %1 = affine.apply #map1(%block_id_y)[%c1, %c0]
      %2 = affine.apply #map2(%0)
      %3 = affine.apply #map2(%1)
      %subview = memref.subview %arg0[%2, %3] [256, 256] [1, 1] : memref<512x512xf32> to memref<256x256xf32, strided<[512, 1], offset: ?>>
      %4 = scf.for %arg3 = %c0 to %c512 step %c32 iter_args(%arg4 = %subview) -> (memref<256x256xf32, strided<[512, 1], offset: ?>>) {
        %5 = affine.apply #map2(%0)
        %6 = xegpu.create_nd_tdesc %arg1 : memref<512x512xf16> -> !xegpu.tensor_desc<256x32xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
        %7 = xegpu.load_nd %6[%5, %arg3] <{layout = #xegpu.layout<sg_layout = [8, 1], sg_data = [32, 32], inst_data = [8, 16]>}> : !xegpu.tensor_desc<256x32xf16, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<256x32xf16>
        %8 = affine.apply #map2(%1)
        %9 = xegpu.create_nd_tdesc %arg2 : memref<512x512xf16> -> !xegpu.tensor_desc<32x256xf16, #xegpu.block_tdesc_attr<boundary_check = false>>
        %10 = xegpu.load_nd %9[%arg3, %8] <{layout = #xegpu.layout<sg_layout = [1, 8], sg_data = [32, 32], inst_data = [16, 16]>}> : !xegpu.tensor_desc<32x256xf16, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<32x256xf16>
        %base_buffer, %offset, %sizes:2, %strides:2 = memref.extract_strided_metadata %arg4 : memref<256x256xf32, strided<[512, 1], offset: ?>> -> memref<f32>, index, index, index, index, index
        %intptr = memref.extract_aligned_pointer_as_index %base_buffer : memref<f32> -> index
        %11 = arith.muli %offset, %c4 : index
        %12 = arith.addi %intptr, %11 : index
        %13 = arith.index_cast %12 : index to i64
        %14 = xegpu.create_nd_tdesc %13, shape : [256, 256], strides : [512, 1] : i64 -> !xegpu.tensor_desc<256x256xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
        %15 = xegpu.load_nd %14[0, 0]  : !xegpu.tensor_desc<256x256xf32, #xegpu.block_tdesc_attr<boundary_check = false>> -> vector<256x256xf32>
        %16 = xegpu.dpas %7, %10, %15 {layout_a = #xegpu.layout<sg_layout = [8, 1], sg_data = [32, 32], inst_data = [8, 16]>, layout_b = #xegpu.layout<sg_layout = [1, 8], sg_data = [32, 32], inst_data = [16, 16]>, layout_cd = #xegpu.layout<sg_layout = [8, 8], sg_data = [32, 32], inst_data = [8, 16]>} : vector<256x32xf16>, vector<32x256xf16>, vector<256x256xf32> -> vector<256x256xf32>
        %base_buffer_1, %offset_2, %sizes_3:2, %strides_4:2 = memref.extract_strided_metadata %arg4 : memref<256x256xf32, strided<[512, 1], offset: ?>> -> memref<f32>, index, index, index, index, index
        %intptr_5 = memref.extract_aligned_pointer_as_index %base_buffer_1 : memref<f32> -> index
        %17 = arith.muli %offset_2, %c4 : index
        %18 = arith.addi %intptr_5, %17 : index
        %19 = arith.index_cast %18 : index to i64
        %20 = xegpu.create_nd_tdesc %19, shape : [256, 256], strides : [512, 1] : i64 -> !xegpu.tensor_desc<256x256xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
        xegpu.store_nd %16, %20[0, 0] <{layout = #xegpu.layout<sg_layout = [8, 8], sg_data = [32, 32], inst_data = [8, 16]>}> : vector<256x256xf32>, !xegpu.tensor_desc<256x256xf32, #xegpu.block_tdesc_attr<boundary_check = false>>
        scf.yield %arg4 : memref<256x256xf32, strided<[512, 1], offset: ?>>
      }
      %subview_0 = memref.subview %arg0[%2, %3] [256, 256] [1, 1] : memref<512x512xf32> to memref<256x256xf32, strided<[512, 1], offset: ?>>
      memref.copy %4, %subview_0 : memref<256x256xf32, strided<[512, 1], offset: ?>> to memref<256x256xf32, strided<[512, 1], offset: ?>>
      gpu.return
    }
  }

  func.func @main() attributes {llvm.emit_c_interface} {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c512 = arith.constant 512 : index
    %c0_f32 = arith.constant 0.0 : f32
    %c2_f32 = arith.constant 2.0 : f32
    %c3_f32 = arith.constant 3.0 : f32
    %A = memref.alloc() : memref<512x512xf16>
    %B = memref.alloc() : memref<512x512xf16>
    %C = memref.alloc() : memref<512x512xf32>
    %C_ref = memref.alloc() : memref<512x512xf32>
    %A_cast = memref.cast %A : memref<512x512xf16> to memref<*xf16>
    call @fillResource1DF16(%A_cast, %c2_f32) : (memref<*xf16>, f32) -> ()
    %B_cast = memref.cast %B : memref<512x512xf16> to memref<*xf16>
    call @fillResource1DF16(%B_cast, %c3_f32) : (memref<*xf16>, f32) -> ()
    scf.for %i = %c0 to %c512 step %c1 {
      scf.for %j = %c0 to %c512 step %c1 {
        memref.store %c0_f32, %C[%i, %j] : memref<512x512xf32>
        memref.store %c0_f32, %C_ref[%i, %j] : memref<512x512xf32>
      }
    }
    call @test(%A, %B, %C) : (memref<512x512xf16>, memref<512x512xf16>, memref<512x512xf32>) -> ()
    %C_ref_cast = memref.cast %C_ref : memref<512x512xf32> to memref<*xf32>
    call @gemmF16F16F32(%A_cast, %B_cast, %C_ref_cast) : (memref<*xf16>, memref<*xf16>, memref<*xf32>) -> ()
    // CHECK: [ALLCLOSE: TRUE]
    %C_cast = memref.cast %C : memref<512x512xf32> to memref<*xf32>
    %eq = arith.constant false
    call @printAllcloseF32(%C_cast, %C_ref_cast, %eq) : (memref<*xf32>, memref<*xf32>, i1) -> ()
    return
  }
  func.func private @fillResource1DF16(memref<*xf16>, f32) attributes {llvm.emit_c_interface}
  func.func private @gemmF16F16F32(memref<*xf16>, memref<*xf16>, memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>, i1) attributes {llvm.emit_c_interface}
}
