;; WASM SIMD kernels for the CPU backend (see simd.ts). Rebuild simd-bytes.ts after editing:
;; `node packages/kernels/scripts/build-simd.mjs` (needs wabt's wat2wasm on the PATH or through npx).
;;
;; Every pointer is a byte offset into the imported memory. The matvec functions compute, for n < N,
;;   y[n] = Σ_k x[k] · W[n, k]
;; with W in one of the weight formats:
;;   f32: rows of K floats (gemv_f32, x as floats; K a multiple of 4).
;;   q8:  32-value blocks of int8 codes, a scale per block (s), optionally a min per block (mn):
;;        W = code · s + min. Symmetric when mn is 0.
;;   q4:  32-value blocks of four u32 words (element 8j + e at nibble e of word j), codes 0–15:
;;        W = code · s + min (affine) or (code − 8) · s (symmetric, mn = 0).
;; For q8 and q4, x comes as int16 per block (xq) with a scale per block (xs), and the products are
;; integer dot products (i32x4.dot_i16x8_s); xsum holds each block's exact float Σx for the min term.
;; K is a multiple of 32.
(module
  (import "env" "memory" (memory 1 65536))

  (func $hsum (param $v v128) (result f32)
    (f32.add
      (f32.add (f32x4.extract_lane 0 (local.get $v)) (f32x4.extract_lane 1 (local.get $v)))
      (f32.add (f32x4.extract_lane 2 (local.get $v)) (f32x4.extract_lane 3 (local.get $v)))))

  (func (export "gemv_f32") (param $x i32) (param $w i32) (param $K i32) (param $N i32) (param $y i32)
    (local $n i32) (local $k i32) (local $xp i32) (local $wp i32)
    (local $a0 v128) (local $a1 v128) (local $a2 v128) (local $a3 v128) (local $end16 i32) (local $end i32)
    (local.set $wp (local.get $w))
    (local.set $end16 (i32.and (local.get $K) (i32.const -16)))
    (block $done
      (loop $rows
        (br_if $done (i32.ge_u (local.get $n) (local.get $N)))
        (local.set $a0 (v128.const i32x4 0 0 0 0))
        (local.set $a1 (v128.const i32x4 0 0 0 0))
        (local.set $a2 (v128.const i32x4 0 0 0 0))
        (local.set $a3 (v128.const i32x4 0 0 0 0))
        (local.set $xp (local.get $x))
        (local.set $k (i32.const 0))
        (block $d16
          (loop $l16
            (br_if $d16 (i32.ge_u (local.get $k) (local.get $end16)))
            (local.set $a0 (f32x4.add (local.get $a0) (f32x4.mul (v128.load (local.get $xp)) (v128.load (local.get $wp)))))
            (local.set $a1 (f32x4.add (local.get $a1) (f32x4.mul (v128.load offset=16 (local.get $xp)) (v128.load offset=16 (local.get $wp)))))
            (local.set $a2 (f32x4.add (local.get $a2) (f32x4.mul (v128.load offset=32 (local.get $xp)) (v128.load offset=32 (local.get $wp)))))
            (local.set $a3 (f32x4.add (local.get $a3) (f32x4.mul (v128.load offset=48 (local.get $xp)) (v128.load offset=48 (local.get $wp)))))
            (local.set $xp (i32.add (local.get $xp) (i32.const 64)))
            (local.set $wp (i32.add (local.get $wp) (i32.const 64)))
            (local.set $k (i32.add (local.get $k) (i32.const 16)))
            (br $l16)))
        (block $d4
          (loop $l4
            (br_if $d4 (i32.ge_u (local.get $k) (local.get $K)))
            (local.set $a0 (f32x4.add (local.get $a0) (f32x4.mul (v128.load (local.get $xp)) (v128.load (local.get $wp)))))
            (local.set $xp (i32.add (local.get $xp) (i32.const 16)))
            (local.set $wp (i32.add (local.get $wp) (i32.const 16)))
            (local.set $k (i32.add (local.get $k) (i32.const 4)))
            (br $l4)))
        (f32.store
          (i32.add (local.get $y) (i32.shl (local.get $n) (i32.const 2)))
          (call $hsum (f32x4.add (f32x4.add (local.get $a0) (local.get $a1)) (f32x4.add (local.get $a2) (local.get $a3)))))
        (local.set $n (i32.add (local.get $n) (i32.const 1)))
        (br $rows))))

  ;; ---- Integer dot products: x as int16 per 32-value block (xq, with a scale per block in xs) ----
  ;; q4: xq holds each block's 32 values as 16 for the low nibbles then 16 for the high ones, in the
  ;; order the code bytes hold them: lane 4j + i of the low half is x[8j + 2i], of the high half
  ;; x[8j + 2i + 1]. Symmetric codes are shifted to -8…7 before the dot, so only affine blocks need
  ;; the min · Σx term (xsum, the exact float sums).
  (func (export "gemv_q4i") (param $xq i32) (param $xs i32) (param $q i32) (param $s i32) (param $mn i32) (param $xsum i32)
    (param $K i32) (param $N i32) (param $y i32)
    (local $n i32) (local $b i32) (local $KB i32) (local $xp i32) (local $v v128) (local $lo v128) (local $hi v128)
    (local $acc v128) (local $tot v128) (local $extra f32) (local $off v128)
    (local.set $KB (i32.shr_u (local.get $K) (i32.const 5)))
    (local.set $off (select (v128.const i8x16 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0) (v128.const i8x16 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8) (local.get $mn)))
    (block $done
      (loop $rows
        (br_if $done (i32.ge_u (local.get $n) (local.get $N)))
        (local.set $tot (v128.const i32x4 0 0 0 0))
        (local.set $extra (f32.const 0))
        (local.set $xp (local.get $xq))
        (local.set $b (i32.const 0))
        (block $db
          (loop $lb
            (br_if $db (i32.ge_u (local.get $b) (local.get $KB)))
            (local.set $v (v128.load (local.get $q)))
            (local.set $lo (i8x16.sub (v128.and (local.get $v) (v128.const i8x16 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15)) (local.get $off)))
            (local.set $hi (i8x16.sub (i8x16.shr_u (local.get $v) (i32.const 4)) (local.get $off)))
            (local.set $acc (i32x4.add
              (i32x4.add
                (i32x4.dot_i16x8_s (i16x8.extend_low_i8x16_s (local.get $lo)) (v128.load (local.get $xp)))
                (i32x4.dot_i16x8_s (i16x8.extend_high_i8x16_s (local.get $lo)) (v128.load offset=16 (local.get $xp))))
              (i32x4.add
                (i32x4.dot_i16x8_s (i16x8.extend_low_i8x16_s (local.get $hi)) (v128.load offset=32 (local.get $xp)))
                (i32x4.dot_i16x8_s (i16x8.extend_high_i8x16_s (local.get $hi)) (v128.load offset=48 (local.get $xp))))))
            (local.set $tot (f32x4.add (local.get $tot) (f32x4.mul (f32x4.convert_i32x4_s (local.get $acc))
              (f32x4.splat (f32.mul (f32.load (local.get $s)) (f32.load (i32.add (local.get $xs) (i32.shl (local.get $b) (i32.const 2)))))))))
            (if (local.get $mn)
              (then
                (local.set $extra (f32.add (local.get $extra)
                  (f32.mul (f32.load (local.get $mn)) (f32.load (i32.add (local.get $xsum) (i32.shl (local.get $b) (i32.const 2)))))))
                (local.set $mn (i32.add (local.get $mn) (i32.const 4)))))
            (local.set $q (i32.add (local.get $q) (i32.const 16)))
            (local.set $s (i32.add (local.get $s) (i32.const 4)))
            (local.set $xp (i32.add (local.get $xp) (i32.const 64)))
            (local.set $b (i32.add (local.get $b) (i32.const 1)))
            (br $lb)))
        (f32.store (i32.add (local.get $y) (i32.shl (local.get $n) (i32.const 2)))
          (f32.add (call $hsum (local.get $tot)) (local.get $extra)))
        (local.set $n (i32.add (local.get $n) (i32.const 1)))
        (br $rows))))

  ;; q8: xq holds each block's 32 values in order.
  (func (export "gemv_q8i") (param $xq i32) (param $xs i32) (param $q i32) (param $s i32) (param $mn i32) (param $xsum i32)
    (param $K i32) (param $N i32) (param $y i32)
    (local $n i32) (local $b i32) (local $KB i32) (local $xp i32) (local $v0 v128) (local $v1 v128)
    (local $acc v128) (local $tot v128) (local $extra f32)
    (local.set $KB (i32.shr_u (local.get $K) (i32.const 5)))
    (block $done
      (loop $rows
        (br_if $done (i32.ge_u (local.get $n) (local.get $N)))
        (local.set $tot (v128.const i32x4 0 0 0 0))
        (local.set $extra (f32.const 0))
        (local.set $xp (local.get $xq))
        (local.set $b (i32.const 0))
        (block $db
          (loop $lb
            (br_if $db (i32.ge_u (local.get $b) (local.get $KB)))
            (local.set $v0 (v128.load (local.get $q)))
            (local.set $v1 (v128.load offset=16 (local.get $q)))
            (local.set $acc (i32x4.add
              (i32x4.add
                (i32x4.dot_i16x8_s (i16x8.extend_low_i8x16_s (local.get $v0)) (v128.load (local.get $xp)))
                (i32x4.dot_i16x8_s (i16x8.extend_high_i8x16_s (local.get $v0)) (v128.load offset=16 (local.get $xp))))
              (i32x4.add
                (i32x4.dot_i16x8_s (i16x8.extend_low_i8x16_s (local.get $v1)) (v128.load offset=32 (local.get $xp)))
                (i32x4.dot_i16x8_s (i16x8.extend_high_i8x16_s (local.get $v1)) (v128.load offset=48 (local.get $xp))))))
            (local.set $tot (f32x4.add (local.get $tot) (f32x4.mul (f32x4.convert_i32x4_s (local.get $acc))
              (f32x4.splat (f32.mul (f32.load (local.get $s)) (f32.load (i32.add (local.get $xs) (i32.shl (local.get $b) (i32.const 2)))))))))
            (if (local.get $mn)
              (then
                (local.set $extra (f32.add (local.get $extra)
                  (f32.mul (f32.load (local.get $mn)) (f32.load (i32.add (local.get $xsum) (i32.shl (local.get $b) (i32.const 2)))))))
                (local.set $mn (i32.add (local.get $mn) (i32.const 4)))))
            (local.set $q (i32.add (local.get $q) (i32.const 32)))
            (local.set $s (i32.add (local.get $s) (i32.const 4)))
            (local.set $xp (i32.add (local.get $xp) (i32.const 64)))
            (local.set $b (i32.add (local.get $b) (i32.const 1)))
            (br $lb)))
        (f32.store (i32.add (local.get $y) (i32.shl (local.get $n) (i32.const 2)))
          (f32.add (call $hsum (local.get $tot)) (local.get $extra)))
        (local.set $n (i32.add (local.get $n) (i32.const 1)))
        (br $rows))))

  ;; gemv_q4i for four x rows at once (xq rows K int16 apart, xs and xsum rows KB apart, outputs ys floats
  ;; apart): each block's codes are widened once for the four.
  (func (export "gemm4_q4i") (param $xq i32) (param $xs i32) (param $q i32) (param $s i32) (param $mn i32) (param $xsum i32)
    (param $K i32) (param $N i32) (param $y i32) (param $ys i32)
    (local $n i32) (local $b i32) (local $KB i32) (local $xp i32) (local $xsp i32) (local $sxp i32) (local $v v128) (local $v1 v128) (local $off v128)
    (local $e0 v128) (local $e1 v128) (local $e2 v128) (local $e3 v128) (local $sv f32) (local $mnv f32) (local $rs i32) (local $ss i32)
    (local $tot0 v128) (local $ex0 f32) (local $tot1 v128) (local $ex1 f32) (local $tot2 v128) (local $ex2 f32) (local $tot3 v128) (local $ex3 f32)
    (local.set $KB (i32.shr_u (local.get $K) (i32.const 5)))
    (local.set $rs (i32.shl (local.get $K) (i32.const 1)))
    (local.set $ss (i32.shl (local.get $KB) (i32.const 2)))
    (local.set $off (select (v128.const i8x16 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0) (v128.const i8x16 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8 8) (local.get $mn)))
    (block $done
      (loop $rows
        (br_if $done (i32.ge_u (local.get $n) (local.get $N)))
        (local.set $tot0 (v128.const i32x4 0 0 0 0)) (local.set $ex0 (f32.const 0))
        (local.set $tot1 (v128.const i32x4 0 0 0 0)) (local.set $ex1 (f32.const 0))
        (local.set $tot2 (v128.const i32x4 0 0 0 0)) (local.set $ex2 (f32.const 0))
        (local.set $tot3 (v128.const i32x4 0 0 0 0)) (local.set $ex3 (f32.const 0))
        (local.set $xp (local.get $xq))
        (local.set $xsp (local.get $xs))
        (local.set $sxp (local.get $xsum))
        (local.set $b (i32.const 0))
        (block $db
          (loop $lb
            (br_if $db (i32.ge_u (local.get $b) (local.get $KB)))
            (local.set $v (v128.load (local.get $q)))
            (local.set $v1 (i8x16.sub (i8x16.shr_u (local.get $v) (i32.const 4)) (local.get $off)))
            (local.set $v (i8x16.sub (v128.and (local.get $v) (v128.const i8x16 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15 15)) (local.get $off)))
            (local.set $e0 (i16x8.extend_low_i8x16_s (local.get $v)))
            (local.set $e1 (i16x8.extend_high_i8x16_s (local.get $v)))
            (local.set $e2 (i16x8.extend_low_i8x16_s (local.get $v1)))
            (local.set $e3 (i16x8.extend_high_i8x16_s (local.get $v1)))
            (local.set $sv (f32.load (local.get $s)))
            (local.set $tot0 (f32x4.add (local.get $tot0) (f32x4.mul (f32x4.convert_i32x4_s
              (i32x4.add
                (i32x4.add (i32x4.dot_i16x8_s (local.get $e0) (v128.load (local.get $xp))) (i32x4.dot_i16x8_s (local.get $e1) (v128.load offset=16 (local.get $xp))))
                (i32x4.add (i32x4.dot_i16x8_s (local.get $e2) (v128.load offset=32 (local.get $xp))) (i32x4.dot_i16x8_s (local.get $e3) (v128.load offset=48 (local.get $xp))))))
              (f32x4.splat (f32.mul (local.get $sv) (f32.load (local.get $xsp)))))))
            (local.set $tot1 (f32x4.add (local.get $tot1) (f32x4.mul (f32x4.convert_i32x4_s
              (i32x4.add
                (i32x4.add (i32x4.dot_i16x8_s (local.get $e0) (v128.load (i32.add (local.get $xp) (i32.mul (local.get $rs) (i32.const 1))))) (i32x4.dot_i16x8_s (local.get $e1) (v128.load offset=16 (i32.add (local.get $xp) (i32.mul (local.get $rs) (i32.const 1))))))
                (i32x4.add (i32x4.dot_i16x8_s (local.get $e2) (v128.load offset=32 (i32.add (local.get $xp) (i32.mul (local.get $rs) (i32.const 1))))) (i32x4.dot_i16x8_s (local.get $e3) (v128.load offset=48 (i32.add (local.get $xp) (i32.mul (local.get $rs) (i32.const 1))))))))
              (f32x4.splat (f32.mul (local.get $sv) (f32.load (i32.add (local.get $xsp) (i32.mul (local.get $ss) (i32.const 1)))))))))
            (local.set $tot2 (f32x4.add (local.get $tot2) (f32x4.mul (f32x4.convert_i32x4_s
              (i32x4.add
                (i32x4.add (i32x4.dot_i16x8_s (local.get $e0) (v128.load (i32.add (local.get $xp) (i32.mul (local.get $rs) (i32.const 2))))) (i32x4.dot_i16x8_s (local.get $e1) (v128.load offset=16 (i32.add (local.get $xp) (i32.mul (local.get $rs) (i32.const 2))))))
                (i32x4.add (i32x4.dot_i16x8_s (local.get $e2) (v128.load offset=32 (i32.add (local.get $xp) (i32.mul (local.get $rs) (i32.const 2))))) (i32x4.dot_i16x8_s (local.get $e3) (v128.load offset=48 (i32.add (local.get $xp) (i32.mul (local.get $rs) (i32.const 2))))))))
              (f32x4.splat (f32.mul (local.get $sv) (f32.load (i32.add (local.get $xsp) (i32.mul (local.get $ss) (i32.const 2)))))))))
            (local.set $tot3 (f32x4.add (local.get $tot3) (f32x4.mul (f32x4.convert_i32x4_s
              (i32x4.add
                (i32x4.add (i32x4.dot_i16x8_s (local.get $e0) (v128.load (i32.add (local.get $xp) (i32.mul (local.get $rs) (i32.const 3))))) (i32x4.dot_i16x8_s (local.get $e1) (v128.load offset=16 (i32.add (local.get $xp) (i32.mul (local.get $rs) (i32.const 3))))))
                (i32x4.add (i32x4.dot_i16x8_s (local.get $e2) (v128.load offset=32 (i32.add (local.get $xp) (i32.mul (local.get $rs) (i32.const 3))))) (i32x4.dot_i16x8_s (local.get $e3) (v128.load offset=48 (i32.add (local.get $xp) (i32.mul (local.get $rs) (i32.const 3))))))))
              (f32x4.splat (f32.mul (local.get $sv) (f32.load (i32.add (local.get $xsp) (i32.mul (local.get $ss) (i32.const 3)))))))))
            (if (local.get $mn)
              (then
                (local.set $mnv (f32.load (local.get $mn)))
                (local.set $ex0 (f32.add (local.get $ex0) (f32.mul (local.get $mnv) (f32.load (local.get $sxp)))))
                (local.set $ex1 (f32.add (local.get $ex1) (f32.mul (local.get $mnv) (f32.load (i32.add (local.get $sxp) (i32.mul (local.get $ss) (i32.const 1)))))))
                (local.set $ex2 (f32.add (local.get $ex2) (f32.mul (local.get $mnv) (f32.load (i32.add (local.get $sxp) (i32.mul (local.get $ss) (i32.const 2)))))))
                (local.set $ex3 (f32.add (local.get $ex3) (f32.mul (local.get $mnv) (f32.load (i32.add (local.get $sxp) (i32.mul (local.get $ss) (i32.const 3)))))))
                (local.set $mn (i32.add (local.get $mn) (i32.const 4)))))
            (local.set $q (i32.add (local.get $q) (i32.const 16)))
            (local.set $s (i32.add (local.get $s) (i32.const 4)))
            (local.set $xp (i32.add (local.get $xp) (i32.const 64)))
            (local.set $xsp (i32.add (local.get $xsp) (i32.const 4)))
            (local.set $sxp (i32.add (local.get $sxp) (i32.const 4)))
            (local.set $b (i32.add (local.get $b) (i32.const 1)))
            (br $lb)))
        (f32.store (i32.add (local.get $y) (i32.shl (i32.add (local.get $n) (i32.mul (local.get $ys) (i32.const 0))) (i32.const 2))) (f32.add (call $hsum (local.get $tot0)) (local.get $ex0)))
        (f32.store (i32.add (local.get $y) (i32.shl (i32.add (local.get $n) (i32.mul (local.get $ys) (i32.const 1))) (i32.const 2))) (f32.add (call $hsum (local.get $tot1)) (local.get $ex1)))
        (f32.store (i32.add (local.get $y) (i32.shl (i32.add (local.get $n) (i32.mul (local.get $ys) (i32.const 2))) (i32.const 2))) (f32.add (call $hsum (local.get $tot2)) (local.get $ex2)))
        (f32.store (i32.add (local.get $y) (i32.shl (i32.add (local.get $n) (i32.mul (local.get $ys) (i32.const 3))) (i32.const 2))) (f32.add (call $hsum (local.get $tot3)) (local.get $ex3)))
        (local.set $n (i32.add (local.get $n) (i32.const 1)))
        (br $rows))))

  ;; gemv_q8i for four x rows at once (xq rows K int16 apart, xs and xsum rows KB apart, outputs ys floats
  ;; apart): each block's codes are widened once for the four.
  (func (export "gemm4_q8i") (param $xq i32) (param $xs i32) (param $q i32) (param $s i32) (param $mn i32) (param $xsum i32)
    (param $K i32) (param $N i32) (param $y i32) (param $ys i32)
    (local $n i32) (local $b i32) (local $KB i32) (local $xp i32) (local $xsp i32) (local $sxp i32) (local $v v128) (local $v1 v128) (local $off v128)
    (local $e0 v128) (local $e1 v128) (local $e2 v128) (local $e3 v128) (local $sv f32) (local $mnv f32) (local $rs i32) (local $ss i32)
    (local $tot0 v128) (local $ex0 f32) (local $tot1 v128) (local $ex1 f32) (local $tot2 v128) (local $ex2 f32) (local $tot3 v128) (local $ex3 f32)
    (local.set $KB (i32.shr_u (local.get $K) (i32.const 5)))
    (local.set $rs (i32.shl (local.get $K) (i32.const 1)))
    (local.set $ss (i32.shl (local.get $KB) (i32.const 2)))
    (block $done
      (loop $rows
        (br_if $done (i32.ge_u (local.get $n) (local.get $N)))
        (local.set $tot0 (v128.const i32x4 0 0 0 0)) (local.set $ex0 (f32.const 0))
        (local.set $tot1 (v128.const i32x4 0 0 0 0)) (local.set $ex1 (f32.const 0))
        (local.set $tot2 (v128.const i32x4 0 0 0 0)) (local.set $ex2 (f32.const 0))
        (local.set $tot3 (v128.const i32x4 0 0 0 0)) (local.set $ex3 (f32.const 0))
        (local.set $xp (local.get $xq))
        (local.set $xsp (local.get $xs))
        (local.set $sxp (local.get $xsum))
        (local.set $b (i32.const 0))
        (block $db
          (loop $lb
            (br_if $db (i32.ge_u (local.get $b) (local.get $KB)))
            (local.set $v (v128.load (local.get $q)))
            (local.set $v1 (v128.load offset=16 (local.get $q)))
            (local.set $e0 (i16x8.extend_low_i8x16_s (local.get $v)))
            (local.set $e1 (i16x8.extend_high_i8x16_s (local.get $v)))
            (local.set $e2 (i16x8.extend_low_i8x16_s (local.get $v1)))
            (local.set $e3 (i16x8.extend_high_i8x16_s (local.get $v1)))
            (local.set $sv (f32.load (local.get $s)))
            (local.set $tot0 (f32x4.add (local.get $tot0) (f32x4.mul (f32x4.convert_i32x4_s
              (i32x4.add
                (i32x4.add (i32x4.dot_i16x8_s (local.get $e0) (v128.load (local.get $xp))) (i32x4.dot_i16x8_s (local.get $e1) (v128.load offset=16 (local.get $xp))))
                (i32x4.add (i32x4.dot_i16x8_s (local.get $e2) (v128.load offset=32 (local.get $xp))) (i32x4.dot_i16x8_s (local.get $e3) (v128.load offset=48 (local.get $xp))))))
              (f32x4.splat (f32.mul (local.get $sv) (f32.load (local.get $xsp)))))))
            (local.set $tot1 (f32x4.add (local.get $tot1) (f32x4.mul (f32x4.convert_i32x4_s
              (i32x4.add
                (i32x4.add (i32x4.dot_i16x8_s (local.get $e0) (v128.load (i32.add (local.get $xp) (i32.mul (local.get $rs) (i32.const 1))))) (i32x4.dot_i16x8_s (local.get $e1) (v128.load offset=16 (i32.add (local.get $xp) (i32.mul (local.get $rs) (i32.const 1))))))
                (i32x4.add (i32x4.dot_i16x8_s (local.get $e2) (v128.load offset=32 (i32.add (local.get $xp) (i32.mul (local.get $rs) (i32.const 1))))) (i32x4.dot_i16x8_s (local.get $e3) (v128.load offset=48 (i32.add (local.get $xp) (i32.mul (local.get $rs) (i32.const 1))))))))
              (f32x4.splat (f32.mul (local.get $sv) (f32.load (i32.add (local.get $xsp) (i32.mul (local.get $ss) (i32.const 1)))))))))
            (local.set $tot2 (f32x4.add (local.get $tot2) (f32x4.mul (f32x4.convert_i32x4_s
              (i32x4.add
                (i32x4.add (i32x4.dot_i16x8_s (local.get $e0) (v128.load (i32.add (local.get $xp) (i32.mul (local.get $rs) (i32.const 2))))) (i32x4.dot_i16x8_s (local.get $e1) (v128.load offset=16 (i32.add (local.get $xp) (i32.mul (local.get $rs) (i32.const 2))))))
                (i32x4.add (i32x4.dot_i16x8_s (local.get $e2) (v128.load offset=32 (i32.add (local.get $xp) (i32.mul (local.get $rs) (i32.const 2))))) (i32x4.dot_i16x8_s (local.get $e3) (v128.load offset=48 (i32.add (local.get $xp) (i32.mul (local.get $rs) (i32.const 2))))))))
              (f32x4.splat (f32.mul (local.get $sv) (f32.load (i32.add (local.get $xsp) (i32.mul (local.get $ss) (i32.const 2)))))))))
            (local.set $tot3 (f32x4.add (local.get $tot3) (f32x4.mul (f32x4.convert_i32x4_s
              (i32x4.add
                (i32x4.add (i32x4.dot_i16x8_s (local.get $e0) (v128.load (i32.add (local.get $xp) (i32.mul (local.get $rs) (i32.const 3))))) (i32x4.dot_i16x8_s (local.get $e1) (v128.load offset=16 (i32.add (local.get $xp) (i32.mul (local.get $rs) (i32.const 3))))))
                (i32x4.add (i32x4.dot_i16x8_s (local.get $e2) (v128.load offset=32 (i32.add (local.get $xp) (i32.mul (local.get $rs) (i32.const 3))))) (i32x4.dot_i16x8_s (local.get $e3) (v128.load offset=48 (i32.add (local.get $xp) (i32.mul (local.get $rs) (i32.const 3))))))))
              (f32x4.splat (f32.mul (local.get $sv) (f32.load (i32.add (local.get $xsp) (i32.mul (local.get $ss) (i32.const 3)))))))))
            (if (local.get $mn)
              (then
                (local.set $mnv (f32.load (local.get $mn)))
                (local.set $ex0 (f32.add (local.get $ex0) (f32.mul (local.get $mnv) (f32.load (local.get $sxp)))))
                (local.set $ex1 (f32.add (local.get $ex1) (f32.mul (local.get $mnv) (f32.load (i32.add (local.get $sxp) (i32.mul (local.get $ss) (i32.const 1)))))))
                (local.set $ex2 (f32.add (local.get $ex2) (f32.mul (local.get $mnv) (f32.load (i32.add (local.get $sxp) (i32.mul (local.get $ss) (i32.const 2)))))))
                (local.set $ex3 (f32.add (local.get $ex3) (f32.mul (local.get $mnv) (f32.load (i32.add (local.get $sxp) (i32.mul (local.get $ss) (i32.const 3)))))))
                (local.set $mn (i32.add (local.get $mn) (i32.const 4)))))
            (local.set $q (i32.add (local.get $q) (i32.const 32)))
            (local.set $s (i32.add (local.get $s) (i32.const 4)))
            (local.set $xp (i32.add (local.get $xp) (i32.const 64)))
            (local.set $xsp (i32.add (local.get $xsp) (i32.const 4)))
            (local.set $sxp (i32.add (local.get $sxp) (i32.const 4)))
            (local.set $b (i32.add (local.get $b) (i32.const 1)))
            (br $lb)))
        (f32.store (i32.add (local.get $y) (i32.shl (i32.add (local.get $n) (i32.mul (local.get $ys) (i32.const 0))) (i32.const 2))) (f32.add (call $hsum (local.get $tot0)) (local.get $ex0)))
        (f32.store (i32.add (local.get $y) (i32.shl (i32.add (local.get $n) (i32.mul (local.get $ys) (i32.const 1))) (i32.const 2))) (f32.add (call $hsum (local.get $tot1)) (local.get $ex1)))
        (f32.store (i32.add (local.get $y) (i32.shl (i32.add (local.get $n) (i32.mul (local.get $ys) (i32.const 2))) (i32.const 2))) (f32.add (call $hsum (local.get $tot2)) (local.get $ex2)))
        (f32.store (i32.add (local.get $y) (i32.shl (i32.add (local.get $n) (i32.mul (local.get $ys) (i32.const 3))) (i32.const 2))) (f32.add (call $hsum (local.get $tot3)) (local.get $ex3)))
        (local.set $n (i32.add (local.get $n) (i32.const 1)))
        (br $rows))))

  ;; ---- Attention (one query row against a run of cached keys) ----
  ;; Key j is the row at k + (j mod cap) · stride bytes (a ring when cap < the keys seen). D is a
  ;; multiple of 4 (f32 caches) or 8 (f16 caches, two halves per u32 word). Queries come scaled.

  ;; Four f16 values (zero-extended into i32 lanes) as f32: the magnitude bits moved into an f32's
  ;; exponent/mantissa and rescaled by 2^112, which is exact for normals, subnormals and zero.
  (func $h2f (param $h v128) (result v128)
    (v128.or
      (f32x4.mul
        (i32x4.shl (v128.and (local.get $h) (v128.const i32x4 0x7fff 0x7fff 0x7fff 0x7fff)) (i32.const 13))
        (v128.const f32x4 0x1p112 0x1p112 0x1p112 0x1p112))
      (i32x4.shl (v128.and (local.get $h) (v128.const i32x4 0x8000 0x8000 0x8000 0x8000)) (i32.const 16))))

  ;; out[j] = q · K_j for j in [first, n); returns the largest.
  ;; Key / value row j of the cache as floats: the row itself (f32 caches) or widened into tmp (f16).
  (func $kvrow (param $base i32) (param $j i32) (param $cap i32) (param $stride i32) (param $D i32) (param $f16 i32) (param $tmp i32) (result i32)
    (local $row i32) (local $i i32) (local $hv v128)
    (local.set $row (i32.add (local.get $base) (i32.mul (i32.rem_u (local.get $j) (local.get $cap)) (local.get $stride))))
    (if (i32.eqz (local.get $f16)) (then (return (local.get $row))))
    (block $d (loop $l
      (br_if $d (i32.ge_u (local.get $i) (local.get $D)))
      (local.set $hv (v128.load (i32.add (local.get $row) (i32.shl (local.get $i) (i32.const 1)))))
      (v128.store (i32.add (local.get $tmp) (i32.shl (local.get $i) (i32.const 2))) (call $h2f (i32x4.extend_low_i16x8_u (local.get $hv))))
      (v128.store offset=16 (i32.add (local.get $tmp) (i32.shl (local.get $i) (i32.const 2))) (call $h2f (i32x4.extend_high_i16x8_u (local.get $hv))))
      (local.set $i (i32.add (local.get $i) (i32.const 8)))
      (br $l)))
    (local.get $tmp))

  ;; For the G query heads sharing a KV head (queries D floats apart at q): out[g][j] = q_g · K_j for
  ;; j in [first, n), score rows os floats apart, and each head's largest score at mx[g]. Each key row
  ;; is read (and widened, for f16) once for the G heads.
  (func (export "att_scores") (param $q i32) (param $k i32) (param $stride i32) (param $cap i32)
    (param $first i32) (param $n i32) (param $D i32) (param $out i32) (param $G i32) (param $os i32) (param $mx i32)
    (param $f16 i32) (param $tmp i32)
    (local $j i32) (local $g i32) (local $i i32) (local $row i32) (local $qg i32) (local $acc v128) (local $d f32) (local $m i32)
    (block $z (loop $zl
      (br_if $z (i32.ge_u (local.get $g) (local.get $G)))
      (f32.store (i32.add (local.get $mx) (i32.shl (local.get $g) (i32.const 2))) (f32.const -inf))
      (local.set $g (i32.add (local.get $g) (i32.const 1)))
      (br $zl)))
    (local.set $j (local.get $first))
    (block $done
      (loop $keys
        (br_if $done (i32.ge_u (local.get $j) (local.get $n)))
        (local.set $row (call $kvrow (local.get $k) (local.get $j) (local.get $cap) (local.get $stride) (local.get $D) (local.get $f16) (local.get $tmp)))
        (local.set $g (i32.const 0))
        (block $gd
          (loop $gl
            (br_if $gd (i32.ge_u (local.get $g) (local.get $G)))
            (local.set $qg (i32.add (local.get $q) (i32.shl (i32.mul (local.get $g) (local.get $D)) (i32.const 2))))
            (local.set $acc (v128.const i32x4 0 0 0 0))
            (local.set $i (i32.const 0))
            (block $rd
              (loop $r
                (br_if $rd (i32.ge_u (local.get $i) (local.get $D)))
                (local.set $acc (f32x4.add (local.get $acc)
                  (f32x4.mul (v128.load (i32.add (local.get $qg) (i32.shl (local.get $i) (i32.const 2))))
                             (v128.load (i32.add (local.get $row) (i32.shl (local.get $i) (i32.const 2)))))))
                (local.set $i (i32.add (local.get $i) (i32.const 4)))
                (br $r)))
            (local.set $d (call $hsum (local.get $acc)))
            (f32.store (i32.add (local.get $out) (i32.shl (i32.add (i32.mul (local.get $g) (local.get $os)) (local.get $j)) (i32.const 2))) (local.get $d))
            (local.set $m (i32.add (local.get $mx) (i32.shl (local.get $g) (i32.const 2))))
            (f32.store (local.get $m) (f32.max (f32.load (local.get $m)) (local.get $d)))
            (local.set $g (i32.add (local.get $g) (i32.const 1)))
            (br $gl)))
        (local.set $j (i32.add (local.get $j) (i32.const 1)))
        (br $keys))))

  ;; acc[g] = Σ_j p[g][j] · V_j for j in [first, n), for the same G heads (weight rows ps floats apart,
  ;; accumulators D floats apart, overwritten).
  (func (export "att_wsum") (param $p i32) (param $v i32) (param $stride i32) (param $cap i32)
    (param $first i32) (param $n i32) (param $D i32) (param $acc i32) (param $G i32) (param $ps i32)
    (param $f16 i32) (param $tmp i32)
    (local $j i32) (local $g i32) (local $i i32) (local $row i32) (local $pj v128) (local $a i32) (local $ag i32)
    (block $z (loop $zl
      (br_if $z (i32.ge_u (local.get $i) (i32.mul (local.get $G) (local.get $D))))
      (v128.store (i32.add (local.get $acc) (i32.shl (local.get $i) (i32.const 2))) (v128.const i32x4 0 0 0 0))
      (local.set $i (i32.add (local.get $i) (i32.const 4)))
      (br $zl)))
    (local.set $j (local.get $first))
    (block $done
      (loop $keys
        (br_if $done (i32.ge_u (local.get $j) (local.get $n)))
        (local.set $row (call $kvrow (local.get $v) (local.get $j) (local.get $cap) (local.get $stride) (local.get $D) (local.get $f16) (local.get $tmp)))
        (local.set $g (i32.const 0))
        (block $gd
          (loop $gl
            (br_if $gd (i32.ge_u (local.get $g) (local.get $G)))
            (local.set $pj (f32x4.splat (f32.load (i32.add (local.get $p) (i32.shl (i32.add (i32.mul (local.get $g) (local.get $ps)) (local.get $j)) (i32.const 2))))))
            (local.set $ag (i32.add (local.get $acc) (i32.shl (i32.mul (local.get $g) (local.get $D)) (i32.const 2))))
            (local.set $i (i32.const 0))
            (block $rd
              (loop $r
                (br_if $rd (i32.ge_u (local.get $i) (local.get $D)))
                (local.set $a (i32.add (local.get $ag) (i32.shl (local.get $i) (i32.const 2))))
                (v128.store (local.get $a) (f32x4.add (v128.load (local.get $a))
                  (f32x4.mul (local.get $pj) (v128.load (i32.add (local.get $row) (i32.shl (local.get $i) (i32.const 2)))))))
                (local.set $i (i32.add (local.get $i) (i32.const 4)))
                (br $r)))
            (local.set $g (i32.add (local.get $g) (i32.const 1)))
            (br $gl)))
        (local.set $j (i32.add (local.get $j) (i32.const 1)))
        (br $keys))))

  ;; ---- x rows for the integer dot products ----
  ;; rows × K floats at x → int16 per 32-value block (xq; q4's lane order when q4 is set), the
  ;; block's scale amax / 32767 (xs) and its float sum (xsum). With nw set, each row is first
  ;; RMS-normalised, x · 1/√(mean(x²) + eps) · nw[k] (the mean in f64, as the TypeScript does).
  (func (export "quant_x") (param $x i32) (param $K i32) (param $rows i32) (param $nw i32) (param $eps f32)
    (param $xq i32) (param $xs i32) (param $xsum i32) (param $q4 i32)
    (local $r i32) (local $k i32) (local $np i32) (local $ss f64) (local $t f64) (local $inv v128)
    (local $v0 v128) (local $v1 v128) (local $v2 v128) (local $v3 v128)
    (local $v4 v128) (local $v5 v128) (local $v6 v128) (local $v7 v128)
    (local $m v128) (local $a f32) (local $sc f32) (local $is v128)
    (local $n0 v128) (local $n1 v128) (local $n2 v128) (local $n3 v128)
    (block $done
      (loop $rows
        (br_if $done (i32.ge_u (local.get $r) (local.get $rows)))
        (if (local.get $nw)
          (then
            (local.set $ss (f64.const 0))
            (local.set $k (i32.const 0))
            (block $ds
              (loop $ls
                (br_if $ds (i32.ge_u (local.get $k) (local.get $K)))
                (local.set $t (f64.promote_f32 (f32.load (i32.add (local.get $x) (i32.shl (local.get $k) (i32.const 2))))))
                (local.set $ss (f64.add (local.get $ss) (f64.mul (local.get $t) (local.get $t))))
                (local.set $k (i32.add (local.get $k) (i32.const 1)))
                (br $ls)))
            (local.set $inv (f32x4.splat (f32.demote_f64 (f64.div (f64.const 1)
              (f64.sqrt (f64.add
                (f64.promote_f32 (f32.demote_f64 (f64.div (local.get $ss) (f64.convert_i32_u (local.get $K)))))
                (f64.promote_f32 (local.get $eps))))))))))
        (local.set $np (local.get $nw))
        (local.set $k (i32.const 0))
        (block $db
          (loop $lb
            (br_if $db (i32.ge_u (local.get $k) (local.get $K)))
            (local.set $v0 (v128.load (local.get $x)))
            (local.set $v1 (v128.load offset=16 (local.get $x)))
            (local.set $v2 (v128.load offset=32 (local.get $x)))
            (local.set $v3 (v128.load offset=48 (local.get $x)))
            (local.set $v4 (v128.load offset=64 (local.get $x)))
            (local.set $v5 (v128.load offset=80 (local.get $x)))
            (local.set $v6 (v128.load offset=96 (local.get $x)))
            (local.set $v7 (v128.load offset=112 (local.get $x)))
            (if (local.get $nw)
              (then
                (local.set $v0 (f32x4.mul (f32x4.mul (local.get $v0) (local.get $inv)) (v128.load (local.get $np))))
                (local.set $v1 (f32x4.mul (f32x4.mul (local.get $v1) (local.get $inv)) (v128.load offset=16 (local.get $np))))
                (local.set $v2 (f32x4.mul (f32x4.mul (local.get $v2) (local.get $inv)) (v128.load offset=32 (local.get $np))))
                (local.set $v3 (f32x4.mul (f32x4.mul (local.get $v3) (local.get $inv)) (v128.load offset=48 (local.get $np))))
                (local.set $v4 (f32x4.mul (f32x4.mul (local.get $v4) (local.get $inv)) (v128.load offset=64 (local.get $np))))
                (local.set $v5 (f32x4.mul (f32x4.mul (local.get $v5) (local.get $inv)) (v128.load offset=80 (local.get $np))))
                (local.set $v6 (f32x4.mul (f32x4.mul (local.get $v6) (local.get $inv)) (v128.load offset=96 (local.get $np))))
                (local.set $v7 (f32x4.mul (f32x4.mul (local.get $v7) (local.get $inv)) (v128.load offset=112 (local.get $np))))
                (local.set $np (i32.add (local.get $np) (i32.const 128)))))
            ;; Largest magnitude and sum of the block.
            (local.set $m (f32x4.max
              (f32x4.max (f32x4.max (f32x4.abs (local.get $v0)) (f32x4.abs (local.get $v1))) (f32x4.max (f32x4.abs (local.get $v2)) (f32x4.abs (local.get $v3))))
              (f32x4.max (f32x4.max (f32x4.abs (local.get $v4)) (f32x4.abs (local.get $v5))) (f32x4.max (f32x4.abs (local.get $v6)) (f32x4.abs (local.get $v7))))))
            (local.set $a (f32.max
              (f32.max (f32x4.extract_lane 0 (local.get $m)) (f32x4.extract_lane 1 (local.get $m)))
              (f32.max (f32x4.extract_lane 2 (local.get $m)) (f32x4.extract_lane 3 (local.get $m)))))
            (f32.store (local.get $xsum) (call $hsum (f32x4.add
              (f32x4.add (f32x4.add (local.get $v0) (local.get $v1)) (f32x4.add (local.get $v2) (local.get $v3)))
              (f32x4.add (f32x4.add (local.get $v4) (local.get $v5)) (f32x4.add (local.get $v6) (local.get $v7))))))
            (local.set $sc (select (f32.div (local.get $a) (f32.const 32767)) (f32.const 1) (f32.gt (local.get $a) (f32.const 0))))
            (f32.store (local.get $xs) (local.get $sc))
            (local.set $is (f32x4.splat (f32.div (f32.const 1) (local.get $sc))))
            ;; Rounded to the nearest integer, narrowed to int16 (saturating), kept within ±32767.
            (local.set $n0 (i16x8.max_s (v128.const i16x8 -32767 -32767 -32767 -32767 -32767 -32767 -32767 -32767) (i16x8.narrow_i32x4_s
              (i32x4.trunc_sat_f32x4_s (f32x4.nearest (f32x4.mul (local.get $v0) (local.get $is))))
              (i32x4.trunc_sat_f32x4_s (f32x4.nearest (f32x4.mul (local.get $v1) (local.get $is)))))))
            (local.set $n1 (i16x8.max_s (v128.const i16x8 -32767 -32767 -32767 -32767 -32767 -32767 -32767 -32767) (i16x8.narrow_i32x4_s
              (i32x4.trunc_sat_f32x4_s (f32x4.nearest (f32x4.mul (local.get $v2) (local.get $is))))
              (i32x4.trunc_sat_f32x4_s (f32x4.nearest (f32x4.mul (local.get $v3) (local.get $is)))))))
            (local.set $n2 (i16x8.max_s (v128.const i16x8 -32767 -32767 -32767 -32767 -32767 -32767 -32767 -32767) (i16x8.narrow_i32x4_s
              (i32x4.trunc_sat_f32x4_s (f32x4.nearest (f32x4.mul (local.get $v4) (local.get $is))))
              (i32x4.trunc_sat_f32x4_s (f32x4.nearest (f32x4.mul (local.get $v5) (local.get $is)))))))
            (local.set $n3 (i16x8.max_s (v128.const i16x8 -32767 -32767 -32767 -32767 -32767 -32767 -32767 -32767) (i16x8.narrow_i32x4_s
              (i32x4.trunc_sat_f32x4_s (f32x4.nearest (f32x4.mul (local.get $v6) (local.get $is))))
              (i32x4.trunc_sat_f32x4_s (f32x4.nearest (f32x4.mul (local.get $v7) (local.get $is)))))))
            (if (local.get $q4)
              (then
                ;; Even elements to lanes 0–15, odd ones to 16–31.
                (v128.store (local.get $xq) (i8x16.shuffle 0 1 4 5 8 9 12 13 16 17 20 21 24 25 28 29 (local.get $n0) (local.get $n1)))
                (v128.store offset=16 (local.get $xq) (i8x16.shuffle 0 1 4 5 8 9 12 13 16 17 20 21 24 25 28 29 (local.get $n2) (local.get $n3)))
                (v128.store offset=32 (local.get $xq) (i8x16.shuffle 2 3 6 7 10 11 14 15 18 19 22 23 26 27 30 31 (local.get $n0) (local.get $n1)))
                (v128.store offset=48 (local.get $xq) (i8x16.shuffle 2 3 6 7 10 11 14 15 18 19 22 23 26 27 30 31 (local.get $n2) (local.get $n3))))
              (else
                (v128.store (local.get $xq) (local.get $n0))
                (v128.store offset=16 (local.get $xq) (local.get $n1))
                (v128.store offset=32 (local.get $xq) (local.get $n2))
                (v128.store offset=48 (local.get $xq) (local.get $n3))))
            (local.set $x (i32.add (local.get $x) (i32.const 128)))
            (local.set $xq (i32.add (local.get $xq) (i32.const 64)))
            (local.set $xs (i32.add (local.get $xs) (i32.const 4)))
            (local.set $xsum (i32.add (local.get $xsum) (i32.const 4)))
            (local.set $k (i32.add (local.get $k) (i32.const 32)))
            (br $lb)))
        (local.set $r (i32.add (local.get $r) (i32.const 1)))
        (br $rows))))
)
