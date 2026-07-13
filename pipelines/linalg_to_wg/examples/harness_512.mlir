module attributes {gpu.container_module} {
  // KERNEL
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
    %C_cast = memref.cast %C : memref<512x512xf32> to memref<*xf32>
    %eq = arith.constant false
    call @printAllcloseF32(%C_cast, %C_ref_cast, %eq) : (memref<*xf32>, memref<*xf32>, i1) -> ()
    return
  }
  func.func private @fillResource1DF16(memref<*xf16>, f32) attributes {llvm.emit_c_interface}
  func.func private @gemmF16F16F32(memref<*xf16>, memref<*xf16>, memref<*xf32>) attributes {llvm.emit_c_interface}
  func.func private @printAllcloseF32(memref<*xf32>, memref<*xf32>, i1) attributes {llvm.emit_c_interface}
}
