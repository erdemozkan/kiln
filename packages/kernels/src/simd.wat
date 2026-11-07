;; WASM SIMD matrix-vector kernels for the CPU backend (see simd.ts). Rebuild simd-bytes.ts after
;; editing: `node packages/kernels/scripts/build-simd.mjs` (needs wabt's wat2wasm on the PATH or
;; through npx).
;;
;; Every pointer is a byte offset into the imported memory. Each function computes, for n < N,
;;   y[n] = Σ_k x[k] · W[n, k]
;; for one x row, with W in one of the weight formats:
;;   f32: rows of K floats.
;;   q8:  32-value blocks of int8 codes, a scale per block (s), optionally a min per block (mn):
;;        W = code · s + min. Symmetric when mn is 0.
;;   q4:  32-value blocks of four u32 words (element 8j + e at nibble e of word j), codes 0–15:
;;        W = code · s + min (affine) or (code − 8) · s (symmetric, mn = 0). x must be given
;;        permuted per block (x'[32b + 4e + j] = x[32b + 8j + e]) so a nibble lane meets
;;        contiguous x.
;; xsum holds Σ x over each block (q8 affine and q4 only). K is a multiple of 32 for q8 and q4 and
;; of 4 for f32.
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

  (func (export "gemv_q8") (param $x i32) (param $q i32) (param $s i32) (param $mn i32) (param $xsum i32)
    (param $K i32) (param $N i32) (param $y i32)
    (local $n i32) (local $b i32) (local $KB i32) (local $xp i32) (local $v v128) (local $lo v128) (local $hi v128)
    (local $blk v128) (local $tot v128) (local $extra f32)
    (local.set $KB (i32.shr_u (local.get $K) (i32.const 5)))
    (block $done
      (loop $rows
        (br_if $done (i32.ge_u (local.get $n) (local.get $N)))
        (local.set $tot (v128.const i32x4 0 0 0 0))
        (local.set $extra (f32.const 0))
        (local.set $xp (local.get $x))
        (local.set $b (i32.const 0))
        (block $db
          (loop $lb
            (br_if $db (i32.ge_u (local.get $b) (local.get $KB)))
            ;; codes 0–15 of the block
            (local.set $v (v128.load (local.get $q)))
            (local.set $lo (i16x8.extend_low_i8x16_s (local.get $v)))
            (local.set $hi (i16x8.extend_high_i8x16_s (local.get $v)))
            (local.set $blk (f32x4.mul (f32x4.convert_i32x4_s (i32x4.extend_low_i16x8_s (local.get $lo))) (v128.load (local.get $xp))))
            (local.set $blk (f32x4.add (local.get $blk) (f32x4.mul (f32x4.convert_i32x4_s (i32x4.extend_high_i16x8_s (local.get $lo))) (v128.load offset=16 (local.get $xp)))))
            (local.set $blk (f32x4.add (local.get $blk) (f32x4.mul (f32x4.convert_i32x4_s (i32x4.extend_low_i16x8_s (local.get $hi))) (v128.load offset=32 (local.get $xp)))))
            (local.set $blk (f32x4.add (local.get $blk) (f32x4.mul (f32x4.convert_i32x4_s (i32x4.extend_high_i16x8_s (local.get $hi))) (v128.load offset=48 (local.get $xp)))))
            ;; codes 16–31
            (local.set $v (v128.load offset=16 (local.get $q)))
            (local.set $lo (i16x8.extend_low_i8x16_s (local.get $v)))
            (local.set $hi (i16x8.extend_high_i8x16_s (local.get $v)))
            (local.set $blk (f32x4.add (local.get $blk) (f32x4.mul (f32x4.convert_i32x4_s (i32x4.extend_low_i16x8_s (local.get $lo))) (v128.load offset=64 (local.get $xp)))))
            (local.set $blk (f32x4.add (local.get $blk) (f32x4.mul (f32x4.convert_i32x4_s (i32x4.extend_high_i16x8_s (local.get $lo))) (v128.load offset=80 (local.get $xp)))))
            (local.set $blk (f32x4.add (local.get $blk) (f32x4.mul (f32x4.convert_i32x4_s (i32x4.extend_low_i16x8_s (local.get $hi))) (v128.load offset=96 (local.get $xp)))))
            (local.set $blk (f32x4.add (local.get $blk) (f32x4.mul (f32x4.convert_i32x4_s (i32x4.extend_high_i16x8_s (local.get $hi))) (v128.load offset=112 (local.get $xp)))))
            (local.set $tot (f32x4.add (local.get $tot) (f32x4.mul (local.get $blk) (v128.load32_splat (local.get $s)))))
            (if (local.get $mn)
              (then
                (local.set $extra (f32.add (local.get $extra)
                  (f32.mul (f32.load (local.get $mn)) (f32.load (i32.add (local.get $xsum) (i32.shl (local.get $b) (i32.const 2)))))))
                (local.set $mn (i32.add (local.get $mn) (i32.const 4)))))
            (local.set $q (i32.add (local.get $q) (i32.const 32)))
            (local.set $s (i32.add (local.get $s) (i32.const 4)))
            (local.set $xp (i32.add (local.get $xp) (i32.const 128)))
            (local.set $b (i32.add (local.get $b) (i32.const 1)))
            (br $lb)))
        (f32.store
          (i32.add (local.get $y) (i32.shl (local.get $n) (i32.const 2)))
          (f32.add (call $hsum (local.get $tot)) (local.get $extra)))
        (local.set $n (i32.add (local.get $n) (i32.const 1)))
        (br $rows))))

  (func (export "gemv_q4") (param $x i32) (param $q i32) (param $s i32) (param $mn i32) (param $xsum i32)
    (param $K i32) (param $N i32) (param $y i32)
    (local $n i32) (local $b i32) (local $KB i32) (local $xp i32) (local $v v128) (local $blk v128)
    (local $tot v128) (local $extra f32) (local $sx f32)
    (local.set $KB (i32.shr_u (local.get $K) (i32.const 5)))
    (block $done
      (loop $rows
        (br_if $done (i32.ge_u (local.get $n) (local.get $N)))
        (local.set $tot (v128.const i32x4 0 0 0 0))
        (local.set $extra (f32.const 0))
        (local.set $xp (local.get $x))
        (local.set $b (i32.const 0))
        (block $db
          (loop $lb
            (br_if $db (i32.ge_u (local.get $b) (local.get $KB)))
            (local.set $v (v128.load (local.get $q)))
            (local.set $blk (f32x4.mul (f32x4.convert_i32x4_s (v128.and (local.get $v) (v128.const i32x4 15 15 15 15))) (v128.load (local.get $xp))))
            (local.set $blk (f32x4.add (local.get $blk) (f32x4.mul (f32x4.convert_i32x4_s (v128.and (i32x4.shr_u (local.get $v) (i32.const 4)) (v128.const i32x4 15 15 15 15))) (v128.load offset=16 (local.get $xp)))))
            (local.set $blk (f32x4.add (local.get $blk) (f32x4.mul (f32x4.convert_i32x4_s (v128.and (i32x4.shr_u (local.get $v) (i32.const 8)) (v128.const i32x4 15 15 15 15))) (v128.load offset=32 (local.get $xp)))))
            (local.set $blk (f32x4.add (local.get $blk) (f32x4.mul (f32x4.convert_i32x4_s (v128.and (i32x4.shr_u (local.get $v) (i32.const 12)) (v128.const i32x4 15 15 15 15))) (v128.load offset=48 (local.get $xp)))))
            (local.set $blk (f32x4.add (local.get $blk) (f32x4.mul (f32x4.convert_i32x4_s (v128.and (i32x4.shr_u (local.get $v) (i32.const 16)) (v128.const i32x4 15 15 15 15))) (v128.load offset=64 (local.get $xp)))))
            (local.set $blk (f32x4.add (local.get $blk) (f32x4.mul (f32x4.convert_i32x4_s (v128.and (i32x4.shr_u (local.get $v) (i32.const 20)) (v128.const i32x4 15 15 15 15))) (v128.load offset=80 (local.get $xp)))))
            (local.set $blk (f32x4.add (local.get $blk) (f32x4.mul (f32x4.convert_i32x4_s (v128.and (i32x4.shr_u (local.get $v) (i32.const 24)) (v128.const i32x4 15 15 15 15))) (v128.load offset=96 (local.get $xp)))))
            (local.set $blk (f32x4.add (local.get $blk) (f32x4.mul (f32x4.convert_i32x4_s (i32x4.shr_u (local.get $v) (i32.const 28))) (v128.load offset=112 (local.get $xp)))))
            (local.set $tot (f32x4.add (local.get $tot) (f32x4.mul (local.get $blk) (v128.load32_splat (local.get $s)))))
            (local.set $sx (f32.load (i32.add (local.get $xsum) (i32.shl (local.get $b) (i32.const 2)))))
            (if (local.get $mn)
              (then
                (local.set $extra (f32.add (local.get $extra) (f32.mul (f32.load (local.get $mn)) (local.get $sx))))
                (local.set $mn (i32.add (local.get $mn) (i32.const 4))))
              (else
                (local.set $extra (f32.sub (local.get $extra) (f32.mul (f32.mul (f32.load (local.get $s)) (f32.const 8)) (local.get $sx))))))
            (local.set $q (i32.add (local.get $q) (i32.const 16)))
            (local.set $s (i32.add (local.get $s) (i32.const 4)))
            (local.set $xp (i32.add (local.get $xp) (i32.const 128)))
            (local.set $b (i32.add (local.get $b) (i32.const 1)))
            (br $lb)))
        (f32.store
          (i32.add (local.get $y) (i32.shl (local.get $n) (i32.const 2)))
          (f32.add (call $hsum (local.get $tot)) (local.get $extra)))
        (local.set $n (i32.add (local.get $n) (i32.const 1)))
        (br $rows))))

  ;; Four x rows at once (rows K floats apart, their block sums KB apart, outputs ys floats apart): each
  ;; weight block is unpacked once for all four. Per row, the same operations in the same order as
  ;; gemv_q4, so the results are identical.
  (func (export "gemm4_q4") (param $x i32) (param $q i32) (param $s i32) (param $mn i32) (param $xsum i32)
    (param $K i32) (param $N i32) (param $y i32) (param $ys i32)
    (local $n i32) (local $b i32) (local $KB i32) (local $xp i32) (local $xs i32) (local $v v128) (local $c v128) (local $sc v128)
    (local $rs i32) (local $ss i32) (local $mnv f32) (local $sv f32)
    (local $blk0 v128) (local $tot0 v128) (local $ex0 f32) (local $blk1 v128) (local $tot1 v128) (local $ex1 f32) (local $blk2 v128) (local $tot2 v128) (local $ex2 f32) (local $blk3 v128) (local $tot3 v128) (local $ex3 f32)
    (local.set $KB (i32.shr_u (local.get $K) (i32.const 5)))
    (local.set $rs (i32.shl (local.get $K) (i32.const 2)))
    (local.set $ss (i32.shl (local.get $KB) (i32.const 2)))
    (block $done
      (loop $rows
        (br_if $done (i32.ge_u (local.get $n) (local.get $N)))
        (local.set $tot0 (v128.const i32x4 0 0 0 0)) (local.set $ex0 (f32.const 0))
        (local.set $tot1 (v128.const i32x4 0 0 0 0)) (local.set $ex1 (f32.const 0))
        (local.set $tot2 (v128.const i32x4 0 0 0 0)) (local.set $ex2 (f32.const 0))
        (local.set $tot3 (v128.const i32x4 0 0 0 0)) (local.set $ex3 (f32.const 0))
        (local.set $xp (local.get $x))
        (local.set $xs (local.get $xsum))
        (local.set $b (i32.const 0))
        (block $db
          (loop $lb
            (br_if $db (i32.ge_u (local.get $b) (local.get $KB)))
            (local.set $v (v128.load (local.get $q)))
            (local.set $c (f32x4.convert_i32x4_s (v128.and (local.get $v) (v128.const i32x4 15 15 15 15))))
            (local.set $blk0 (f32x4.mul (local.get $c) (v128.load offset=0 (local.get $xp))))
            (local.set $blk1 (f32x4.mul (local.get $c) (v128.load offset=0 (i32.add (local.get $xp) (i32.mul (local.get $rs) (i32.const 1))))))
            (local.set $blk2 (f32x4.mul (local.get $c) (v128.load offset=0 (i32.add (local.get $xp) (i32.mul (local.get $rs) (i32.const 2))))))
            (local.set $blk3 (f32x4.mul (local.get $c) (v128.load offset=0 (i32.add (local.get $xp) (i32.mul (local.get $rs) (i32.const 3))))))
            (local.set $c (f32x4.convert_i32x4_s (v128.and (i32x4.shr_u (local.get $v) (i32.const 4)) (v128.const i32x4 15 15 15 15))))
            (local.set $blk0 (f32x4.add (local.get $blk0) (f32x4.mul (local.get $c) (v128.load offset=16 (local.get $xp)))))
            (local.set $blk1 (f32x4.add (local.get $blk1) (f32x4.mul (local.get $c) (v128.load offset=16 (i32.add (local.get $xp) (i32.mul (local.get $rs) (i32.const 1)))))))
            (local.set $blk2 (f32x4.add (local.get $blk2) (f32x4.mul (local.get $c) (v128.load offset=16 (i32.add (local.get $xp) (i32.mul (local.get $rs) (i32.const 2)))))))
            (local.set $blk3 (f32x4.add (local.get $blk3) (f32x4.mul (local.get $c) (v128.load offset=16 (i32.add (local.get $xp) (i32.mul (local.get $rs) (i32.const 3)))))))
            (local.set $c (f32x4.convert_i32x4_s (v128.and (i32x4.shr_u (local.get $v) (i32.const 8)) (v128.const i32x4 15 15 15 15))))
            (local.set $blk0 (f32x4.add (local.get $blk0) (f32x4.mul (local.get $c) (v128.load offset=32 (local.get $xp)))))
            (local.set $blk1 (f32x4.add (local.get $blk1) (f32x4.mul (local.get $c) (v128.load offset=32 (i32.add (local.get $xp) (i32.mul (local.get $rs) (i32.const 1)))))))
            (local.set $blk2 (f32x4.add (local.get $blk2) (f32x4.mul (local.get $c) (v128.load offset=32 (i32.add (local.get $xp) (i32.mul (local.get $rs) (i32.const 2)))))))
            (local.set $blk3 (f32x4.add (local.get $blk3) (f32x4.mul (local.get $c) (v128.load offset=32 (i32.add (local.get $xp) (i32.mul (local.get $rs) (i32.const 3)))))))
            (local.set $c (f32x4.convert_i32x4_s (v128.and (i32x4.shr_u (local.get $v) (i32.const 12)) (v128.const i32x4 15 15 15 15))))
            (local.set $blk0 (f32x4.add (local.get $blk0) (f32x4.mul (local.get $c) (v128.load offset=48 (local.get $xp)))))
            (local.set $blk1 (f32x4.add (local.get $blk1) (f32x4.mul (local.get $c) (v128.load offset=48 (i32.add (local.get $xp) (i32.mul (local.get $rs) (i32.const 1)))))))
            (local.set $blk2 (f32x4.add (local.get $blk2) (f32x4.mul (local.get $c) (v128.load offset=48 (i32.add (local.get $xp) (i32.mul (local.get $rs) (i32.const 2)))))))
            (local.set $blk3 (f32x4.add (local.get $blk3) (f32x4.mul (local.get $c) (v128.load offset=48 (i32.add (local.get $xp) (i32.mul (local.get $rs) (i32.const 3)))))))
            (local.set $c (f32x4.convert_i32x4_s (v128.and (i32x4.shr_u (local.get $v) (i32.const 16)) (v128.const i32x4 15 15 15 15))))
            (local.set $blk0 (f32x4.add (local.get $blk0) (f32x4.mul (local.get $c) (v128.load offset=64 (local.get $xp)))))
            (local.set $blk1 (f32x4.add (local.get $blk1) (f32x4.mul (local.get $c) (v128.load offset=64 (i32.add (local.get $xp) (i32.mul (local.get $rs) (i32.const 1)))))))
            (local.set $blk2 (f32x4.add (local.get $blk2) (f32x4.mul (local.get $c) (v128.load offset=64 (i32.add (local.get $xp) (i32.mul (local.get $rs) (i32.const 2)))))))
            (local.set $blk3 (f32x4.add (local.get $blk3) (f32x4.mul (local.get $c) (v128.load offset=64 (i32.add (local.get $xp) (i32.mul (local.get $rs) (i32.const 3)))))))
            (local.set $c (f32x4.convert_i32x4_s (v128.and (i32x4.shr_u (local.get $v) (i32.const 20)) (v128.const i32x4 15 15 15 15))))
            (local.set $blk0 (f32x4.add (local.get $blk0) (f32x4.mul (local.get $c) (v128.load offset=80 (local.get $xp)))))
            (local.set $blk1 (f32x4.add (local.get $blk1) (f32x4.mul (local.get $c) (v128.load offset=80 (i32.add (local.get $xp) (i32.mul (local.get $rs) (i32.const 1)))))))
            (local.set $blk2 (f32x4.add (local.get $blk2) (f32x4.mul (local.get $c) (v128.load offset=80 (i32.add (local.get $xp) (i32.mul (local.get $rs) (i32.const 2)))))))
            (local.set $blk3 (f32x4.add (local.get $blk3) (f32x4.mul (local.get $c) (v128.load offset=80 (i32.add (local.get $xp) (i32.mul (local.get $rs) (i32.const 3)))))))
            (local.set $c (f32x4.convert_i32x4_s (v128.and (i32x4.shr_u (local.get $v) (i32.const 24)) (v128.const i32x4 15 15 15 15))))
            (local.set $blk0 (f32x4.add (local.get $blk0) (f32x4.mul (local.get $c) (v128.load offset=96 (local.get $xp)))))
            (local.set $blk1 (f32x4.add (local.get $blk1) (f32x4.mul (local.get $c) (v128.load offset=96 (i32.add (local.get $xp) (i32.mul (local.get $rs) (i32.const 1)))))))
            (local.set $blk2 (f32x4.add (local.get $blk2) (f32x4.mul (local.get $c) (v128.load offset=96 (i32.add (local.get $xp) (i32.mul (local.get $rs) (i32.const 2)))))))
            (local.set $blk3 (f32x4.add (local.get $blk3) (f32x4.mul (local.get $c) (v128.load offset=96 (i32.add (local.get $xp) (i32.mul (local.get $rs) (i32.const 3)))))))
            (local.set $c (f32x4.convert_i32x4_s (i32x4.shr_u (local.get $v) (i32.const 28))))
            (local.set $blk0 (f32x4.add (local.get $blk0) (f32x4.mul (local.get $c) (v128.load offset=112 (local.get $xp)))))
            (local.set $blk1 (f32x4.add (local.get $blk1) (f32x4.mul (local.get $c) (v128.load offset=112 (i32.add (local.get $xp) (i32.mul (local.get $rs) (i32.const 1)))))))
            (local.set $blk2 (f32x4.add (local.get $blk2) (f32x4.mul (local.get $c) (v128.load offset=112 (i32.add (local.get $xp) (i32.mul (local.get $rs) (i32.const 2)))))))
            (local.set $blk3 (f32x4.add (local.get $blk3) (f32x4.mul (local.get $c) (v128.load offset=112 (i32.add (local.get $xp) (i32.mul (local.get $rs) (i32.const 3)))))))
            (local.set $sc (v128.load32_splat (local.get $s)))
            (local.set $sv (f32.load (local.get $s)))
            (local.set $tot0 (f32x4.add (local.get $tot0) (f32x4.mul (local.get $blk0) (local.get $sc))))
            (local.set $tot1 (f32x4.add (local.get $tot1) (f32x4.mul (local.get $blk1) (local.get $sc))))
            (local.set $tot2 (f32x4.add (local.get $tot2) (f32x4.mul (local.get $blk2) (local.get $sc))))
            (local.set $tot3 (f32x4.add (local.get $tot3) (f32x4.mul (local.get $blk3) (local.get $sc))))
            (if (local.get $mn)
              (then
                (local.set $mnv (f32.load (local.get $mn)))
                (local.set $ex0 (f32.add (local.get $ex0) (f32.mul (local.get $mnv) (f32.load (local.get $xs)))))
                (local.set $ex1 (f32.add (local.get $ex1) (f32.mul (local.get $mnv) (f32.load (i32.add (local.get $xs) (i32.mul (local.get $ss) (i32.const 1)))))))
                (local.set $ex2 (f32.add (local.get $ex2) (f32.mul (local.get $mnv) (f32.load (i32.add (local.get $xs) (i32.mul (local.get $ss) (i32.const 2)))))))
                (local.set $ex3 (f32.add (local.get $ex3) (f32.mul (local.get $mnv) (f32.load (i32.add (local.get $xs) (i32.mul (local.get $ss) (i32.const 3)))))))
                (local.set $mn (i32.add (local.get $mn) (i32.const 4))))
              (else
                (local.set $ex0 (f32.sub (local.get $ex0) (f32.mul (f32.mul (local.get $sv) (f32.const 8)) (f32.load (local.get $xs)))))
                (local.set $ex1 (f32.sub (local.get $ex1) (f32.mul (f32.mul (local.get $sv) (f32.const 8)) (f32.load (i32.add (local.get $xs) (i32.mul (local.get $ss) (i32.const 1)))))))
                (local.set $ex2 (f32.sub (local.get $ex2) (f32.mul (f32.mul (local.get $sv) (f32.const 8)) (f32.load (i32.add (local.get $xs) (i32.mul (local.get $ss) (i32.const 2)))))))
                (local.set $ex3 (f32.sub (local.get $ex3) (f32.mul (f32.mul (local.get $sv) (f32.const 8)) (f32.load (i32.add (local.get $xs) (i32.mul (local.get $ss) (i32.const 3)))))))
))
            (local.set $q (i32.add (local.get $q) (i32.const 16)))
            (local.set $s (i32.add (local.get $s) (i32.const 4)))
            (local.set $xp (i32.add (local.get $xp) (i32.const 128)))
            (local.set $xs (i32.add (local.get $xs) (i32.const 4)))
            (local.set $b (i32.add (local.get $b) (i32.const 1)))
            (br $lb)))
        (f32.store (i32.add (local.get $y) (i32.shl (i32.add (local.get $n) (i32.mul (local.get $ys) (i32.const 0))) (i32.const 2))) (f32.add (call $hsum (local.get $tot0)) (local.get $ex0)))
        (f32.store (i32.add (local.get $y) (i32.shl (i32.add (local.get $n) (i32.mul (local.get $ys) (i32.const 1))) (i32.const 2))) (f32.add (call $hsum (local.get $tot1)) (local.get $ex1)))
        (f32.store (i32.add (local.get $y) (i32.shl (i32.add (local.get $n) (i32.mul (local.get $ys) (i32.const 2))) (i32.const 2))) (f32.add (call $hsum (local.get $tot2)) (local.get $ex2)))
        (f32.store (i32.add (local.get $y) (i32.shl (i32.add (local.get $n) (i32.mul (local.get $ys) (i32.const 3))) (i32.const 2))) (f32.add (call $hsum (local.get $tot3)) (local.get $ex3)))
        (local.set $n (i32.add (local.get $n) (i32.const 1)))
        (br $rows))))

  ;; gemv_q8 for four x rows at once, laid out as for gemm4_q4; identical results per row.
  (func (export "gemm4_q8") (param $x i32) (param $q i32) (param $s i32) (param $mn i32) (param $xsum i32)
    (param $K i32) (param $N i32) (param $y i32) (param $ys i32)
    (local $n i32) (local $b i32) (local $KB i32) (local $xp i32) (local $xs i32) (local $v v128) (local $lo v128) (local $hi v128) (local $c v128) (local $sc v128)
    (local $rs i32) (local $ss i32) (local $mnv f32)
    (local $blk0 v128) (local $tot0 v128) (local $ex0 f32) (local $blk1 v128) (local $tot1 v128) (local $ex1 f32) (local $blk2 v128) (local $tot2 v128) (local $ex2 f32) (local $blk3 v128) (local $tot3 v128) (local $ex3 f32)
    (local.set $KB (i32.shr_u (local.get $K) (i32.const 5)))
    (local.set $rs (i32.shl (local.get $K) (i32.const 2)))
    (local.set $ss (i32.shl (local.get $KB) (i32.const 2)))
    (block $done
      (loop $rows
        (br_if $done (i32.ge_u (local.get $n) (local.get $N)))
        (local.set $tot0 (v128.const i32x4 0 0 0 0)) (local.set $ex0 (f32.const 0))
        (local.set $tot1 (v128.const i32x4 0 0 0 0)) (local.set $ex1 (f32.const 0))
        (local.set $tot2 (v128.const i32x4 0 0 0 0)) (local.set $ex2 (f32.const 0))
        (local.set $tot3 (v128.const i32x4 0 0 0 0)) (local.set $ex3 (f32.const 0))
        (local.set $xp (local.get $x))
        (local.set $xs (local.get $xsum))
        (local.set $b (i32.const 0))
        (block $db
          (loop $lb
            (br_if $db (i32.ge_u (local.get $b) (local.get $KB)))
            (local.set $v (v128.load offset=0 (local.get $q)))
            (local.set $lo (i16x8.extend_low_i8x16_s (local.get $v)))
            (local.set $hi (i16x8.extend_high_i8x16_s (local.get $v)))
            (local.set $c (f32x4.convert_i32x4_s (i32x4.extend_low_i16x8_s (local.get $lo))))
            (local.set $blk0 (f32x4.mul (local.get $c) (v128.load offset=0 (local.get $xp))))
            (local.set $blk1 (f32x4.mul (local.get $c) (v128.load offset=0 (i32.add (local.get $xp) (i32.mul (local.get $rs) (i32.const 1))))))
            (local.set $blk2 (f32x4.mul (local.get $c) (v128.load offset=0 (i32.add (local.get $xp) (i32.mul (local.get $rs) (i32.const 2))))))
            (local.set $blk3 (f32x4.mul (local.get $c) (v128.load offset=0 (i32.add (local.get $xp) (i32.mul (local.get $rs) (i32.const 3))))))
            (local.set $c (f32x4.convert_i32x4_s (i32x4.extend_high_i16x8_s (local.get $lo))))
            (local.set $blk0 (f32x4.add (local.get $blk0) (f32x4.mul (local.get $c) (v128.load offset=16 (local.get $xp)))))
            (local.set $blk1 (f32x4.add (local.get $blk1) (f32x4.mul (local.get $c) (v128.load offset=16 (i32.add (local.get $xp) (i32.mul (local.get $rs) (i32.const 1)))))))
            (local.set $blk2 (f32x4.add (local.get $blk2) (f32x4.mul (local.get $c) (v128.load offset=16 (i32.add (local.get $xp) (i32.mul (local.get $rs) (i32.const 2)))))))
            (local.set $blk3 (f32x4.add (local.get $blk3) (f32x4.mul (local.get $c) (v128.load offset=16 (i32.add (local.get $xp) (i32.mul (local.get $rs) (i32.const 3)))))))
            (local.set $c (f32x4.convert_i32x4_s (i32x4.extend_low_i16x8_s (local.get $hi))))
            (local.set $blk0 (f32x4.add (local.get $blk0) (f32x4.mul (local.get $c) (v128.load offset=32 (local.get $xp)))))
            (local.set $blk1 (f32x4.add (local.get $blk1) (f32x4.mul (local.get $c) (v128.load offset=32 (i32.add (local.get $xp) (i32.mul (local.get $rs) (i32.const 1)))))))
            (local.set $blk2 (f32x4.add (local.get $blk2) (f32x4.mul (local.get $c) (v128.load offset=32 (i32.add (local.get $xp) (i32.mul (local.get $rs) (i32.const 2)))))))
            (local.set $blk3 (f32x4.add (local.get $blk3) (f32x4.mul (local.get $c) (v128.load offset=32 (i32.add (local.get $xp) (i32.mul (local.get $rs) (i32.const 3)))))))
            (local.set $c (f32x4.convert_i32x4_s (i32x4.extend_high_i16x8_s (local.get $hi))))
            (local.set $blk0 (f32x4.add (local.get $blk0) (f32x4.mul (local.get $c) (v128.load offset=48 (local.get $xp)))))
            (local.set $blk1 (f32x4.add (local.get $blk1) (f32x4.mul (local.get $c) (v128.load offset=48 (i32.add (local.get $xp) (i32.mul (local.get $rs) (i32.const 1)))))))
            (local.set $blk2 (f32x4.add (local.get $blk2) (f32x4.mul (local.get $c) (v128.load offset=48 (i32.add (local.get $xp) (i32.mul (local.get $rs) (i32.const 2)))))))
            (local.set $blk3 (f32x4.add (local.get $blk3) (f32x4.mul (local.get $c) (v128.load offset=48 (i32.add (local.get $xp) (i32.mul (local.get $rs) (i32.const 3)))))))
            (local.set $v (v128.load offset=16 (local.get $q)))
            (local.set $lo (i16x8.extend_low_i8x16_s (local.get $v)))
            (local.set $hi (i16x8.extend_high_i8x16_s (local.get $v)))
            (local.set $c (f32x4.convert_i32x4_s (i32x4.extend_low_i16x8_s (local.get $lo))))
            (local.set $blk0 (f32x4.add (local.get $blk0) (f32x4.mul (local.get $c) (v128.load offset=64 (local.get $xp)))))
            (local.set $blk1 (f32x4.add (local.get $blk1) (f32x4.mul (local.get $c) (v128.load offset=64 (i32.add (local.get $xp) (i32.mul (local.get $rs) (i32.const 1)))))))
            (local.set $blk2 (f32x4.add (local.get $blk2) (f32x4.mul (local.get $c) (v128.load offset=64 (i32.add (local.get $xp) (i32.mul (local.get $rs) (i32.const 2)))))))
            (local.set $blk3 (f32x4.add (local.get $blk3) (f32x4.mul (local.get $c) (v128.load offset=64 (i32.add (local.get $xp) (i32.mul (local.get $rs) (i32.const 3)))))))
            (local.set $c (f32x4.convert_i32x4_s (i32x4.extend_high_i16x8_s (local.get $lo))))
            (local.set $blk0 (f32x4.add (local.get $blk0) (f32x4.mul (local.get $c) (v128.load offset=80 (local.get $xp)))))
            (local.set $blk1 (f32x4.add (local.get $blk1) (f32x4.mul (local.get $c) (v128.load offset=80 (i32.add (local.get $xp) (i32.mul (local.get $rs) (i32.const 1)))))))
            (local.set $blk2 (f32x4.add (local.get $blk2) (f32x4.mul (local.get $c) (v128.load offset=80 (i32.add (local.get $xp) (i32.mul (local.get $rs) (i32.const 2)))))))
            (local.set $blk3 (f32x4.add (local.get $blk3) (f32x4.mul (local.get $c) (v128.load offset=80 (i32.add (local.get $xp) (i32.mul (local.get $rs) (i32.const 3)))))))
            (local.set $c (f32x4.convert_i32x4_s (i32x4.extend_low_i16x8_s (local.get $hi))))
            (local.set $blk0 (f32x4.add (local.get $blk0) (f32x4.mul (local.get $c) (v128.load offset=96 (local.get $xp)))))
            (local.set $blk1 (f32x4.add (local.get $blk1) (f32x4.mul (local.get $c) (v128.load offset=96 (i32.add (local.get $xp) (i32.mul (local.get $rs) (i32.const 1)))))))
            (local.set $blk2 (f32x4.add (local.get $blk2) (f32x4.mul (local.get $c) (v128.load offset=96 (i32.add (local.get $xp) (i32.mul (local.get $rs) (i32.const 2)))))))
            (local.set $blk3 (f32x4.add (local.get $blk3) (f32x4.mul (local.get $c) (v128.load offset=96 (i32.add (local.get $xp) (i32.mul (local.get $rs) (i32.const 3)))))))
            (local.set $c (f32x4.convert_i32x4_s (i32x4.extend_high_i16x8_s (local.get $hi))))
            (local.set $blk0 (f32x4.add (local.get $blk0) (f32x4.mul (local.get $c) (v128.load offset=112 (local.get $xp)))))
            (local.set $blk1 (f32x4.add (local.get $blk1) (f32x4.mul (local.get $c) (v128.load offset=112 (i32.add (local.get $xp) (i32.mul (local.get $rs) (i32.const 1)))))))
            (local.set $blk2 (f32x4.add (local.get $blk2) (f32x4.mul (local.get $c) (v128.load offset=112 (i32.add (local.get $xp) (i32.mul (local.get $rs) (i32.const 2)))))))
            (local.set $blk3 (f32x4.add (local.get $blk3) (f32x4.mul (local.get $c) (v128.load offset=112 (i32.add (local.get $xp) (i32.mul (local.get $rs) (i32.const 3)))))))
            (local.set $sc (v128.load32_splat (local.get $s)))
            (local.set $tot0 (f32x4.add (local.get $tot0) (f32x4.mul (local.get $blk0) (local.get $sc))))
            (local.set $tot1 (f32x4.add (local.get $tot1) (f32x4.mul (local.get $blk1) (local.get $sc))))
            (local.set $tot2 (f32x4.add (local.get $tot2) (f32x4.mul (local.get $blk2) (local.get $sc))))
            (local.set $tot3 (f32x4.add (local.get $tot3) (f32x4.mul (local.get $blk3) (local.get $sc))))
            (if (local.get $mn)
              (then
                (local.set $mnv (f32.load (local.get $mn)))
                (local.set $ex0 (f32.add (local.get $ex0) (f32.mul (local.get $mnv) (f32.load (local.get $xs)))))
                (local.set $ex1 (f32.add (local.get $ex1) (f32.mul (local.get $mnv) (f32.load (i32.add (local.get $xs) (i32.mul (local.get $ss) (i32.const 1)))))))
                (local.set $ex2 (f32.add (local.get $ex2) (f32.mul (local.get $mnv) (f32.load (i32.add (local.get $xs) (i32.mul (local.get $ss) (i32.const 2)))))))
                (local.set $ex3 (f32.add (local.get $ex3) (f32.mul (local.get $mnv) (f32.load (i32.add (local.get $xs) (i32.mul (local.get $ss) (i32.const 3)))))))
                (local.set $mn (i32.add (local.get $mn) (i32.const 4)))))
            (local.set $q (i32.add (local.get $q) (i32.const 32)))
            (local.set $s (i32.add (local.get $s) (i32.const 4)))
            (local.set $xp (i32.add (local.get $xp) (i32.const 128)))
            (local.set $xs (i32.add (local.get $xs) (i32.const 4)))
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
  ;; multiple of 4 (f32 caches) or 8 (f16 caches, two halves per u32 word). q is the query, scaled.

  ;; Four f16 values (zero-extended into i32 lanes) as f32: the magnitude bits moved into an f32's
  ;; exponent/mantissa and rescaled by 2^112, which is exact for normals, subnormals and zero.
  (func $h2f (param $h v128) (result v128)
    (v128.or
      (f32x4.mul
        (i32x4.shl (v128.and (local.get $h) (v128.const i32x4 0x7fff 0x7fff 0x7fff 0x7fff)) (i32.const 13))
        (v128.const f32x4 0x1p112 0x1p112 0x1p112 0x1p112))
      (i32x4.shl (v128.and (local.get $h) (v128.const i32x4 0x8000 0x8000 0x8000 0x8000)) (i32.const 16))))

  ;; out[j] = q · K_j for j in [first, n); returns the largest.
  (func (export "att_scores_f32") (param $q i32) (param $k i32) (param $stride i32) (param $cap i32)
    (param $first i32) (param $n i32) (param $D i32) (param $out i32) (result f32)
    (local $j i32) (local $i i32) (local $row i32) (local $acc v128) (local $d f32) (local $mx f32)
    (local.set $mx (f32.const -inf))
    (local.set $j (local.get $first))
    (block $done
      (loop $keys
        (br_if $done (i32.ge_u (local.get $j) (local.get $n)))
        (local.set $row (i32.add (local.get $k) (i32.mul (i32.rem_u (local.get $j) (local.get $cap)) (local.get $stride))))
        (local.set $acc (v128.const i32x4 0 0 0 0))
        (local.set $i (i32.const 0))
        (block $rd
          (loop $r
            (br_if $rd (i32.ge_u (local.get $i) (local.get $D)))
            (local.set $acc (f32x4.add (local.get $acc)
              (f32x4.mul (v128.load (i32.add (local.get $q) (i32.shl (local.get $i) (i32.const 2))))
                         (v128.load (i32.add (local.get $row) (i32.shl (local.get $i) (i32.const 2)))))))
            (local.set $i (i32.add (local.get $i) (i32.const 4)))
            (br $r)))
        (local.set $d (call $hsum (local.get $acc)))
        (f32.store (i32.add (local.get $out) (i32.shl (local.get $j) (i32.const 2))) (local.get $d))
        (local.set $mx (f32.max (local.get $mx) (local.get $d)))
        (local.set $j (i32.add (local.get $j) (i32.const 1)))
        (br $keys)))
    (local.get $mx))

  (func (export "att_scores_f16") (param $q i32) (param $k i32) (param $stride i32) (param $cap i32)
    (param $first i32) (param $n i32) (param $D i32) (param $out i32) (result f32)
    (local $j i32) (local $i i32) (local $row i32) (local $acc v128) (local $hv v128) (local $qp i32) (local $d f32) (local $mx f32)
    (local.set $mx (f32.const -inf))
    (local.set $j (local.get $first))
    (block $done
      (loop $keys
        (br_if $done (i32.ge_u (local.get $j) (local.get $n)))
        (local.set $row (i32.add (local.get $k) (i32.mul (i32.rem_u (local.get $j) (local.get $cap)) (local.get $stride))))
        (local.set $acc (v128.const i32x4 0 0 0 0))
        (local.set $i (i32.const 0))
        (block $rd
          (loop $r
            (br_if $rd (i32.ge_u (local.get $i) (local.get $D)))
            (local.set $hv (v128.load (i32.add (local.get $row) (i32.shl (local.get $i) (i32.const 1)))))
            (local.set $qp (i32.add (local.get $q) (i32.shl (local.get $i) (i32.const 2))))
            (local.set $acc (f32x4.add (local.get $acc)
              (f32x4.add
                (f32x4.mul (v128.load (local.get $qp)) (call $h2f (i32x4.extend_low_i16x8_u (local.get $hv))))
                (f32x4.mul (v128.load offset=16 (local.get $qp)) (call $h2f (i32x4.extend_high_i16x8_u (local.get $hv)))))))
            (local.set $i (i32.add (local.get $i) (i32.const 8)))
            (br $r)))
        (local.set $d (call $hsum (local.get $acc)))
        (f32.store (i32.add (local.get $out) (i32.shl (local.get $j) (i32.const 2))) (local.get $d))
        (local.set $mx (f32.max (local.get $mx) (local.get $d)))
        (local.set $j (i32.add (local.get $j) (i32.const 1)))
        (br $keys)))
    (local.get $mx))

  ;; acc = Σ_j p[j] · V_j for j in [first, n) (acc is overwritten).
  (func (export "att_wsum_f32") (param $p i32) (param $v i32) (param $stride i32) (param $cap i32)
    (param $first i32) (param $n i32) (param $D i32) (param $acc i32)
    (local $j i32) (local $i i32) (local $row i32) (local $pj v128) (local $a i32)
    (local.set $i (i32.const 0))
    (block $z (loop $zl
      (br_if $z (i32.ge_u (local.get $i) (local.get $D)))
      (v128.store (i32.add (local.get $acc) (i32.shl (local.get $i) (i32.const 2))) (v128.const i32x4 0 0 0 0))
      (local.set $i (i32.add (local.get $i) (i32.const 4)))
      (br $zl)))
    (local.set $j (local.get $first))
    (block $done
      (loop $keys
        (br_if $done (i32.ge_u (local.get $j) (local.get $n)))
        (local.set $row (i32.add (local.get $v) (i32.mul (i32.rem_u (local.get $j) (local.get $cap)) (local.get $stride))))
        (local.set $pj (f32x4.splat (f32.load (i32.add (local.get $p) (i32.shl (local.get $j) (i32.const 2))))))
        (local.set $i (i32.const 0))
        (block $rd
          (loop $r
            (br_if $rd (i32.ge_u (local.get $i) (local.get $D)))
            (local.set $a (i32.add (local.get $acc) (i32.shl (local.get $i) (i32.const 2))))
            (v128.store (local.get $a) (f32x4.add (v128.load (local.get $a))
              (f32x4.mul (local.get $pj) (v128.load (i32.add (local.get $row) (i32.shl (local.get $i) (i32.const 2)))))))
            (local.set $i (i32.add (local.get $i) (i32.const 4)))
            (br $r)))
        (local.set $j (i32.add (local.get $j) (i32.const 1)))
        (br $keys))))

  (func (export "att_wsum_f16") (param $p i32) (param $v i32) (param $stride i32) (param $cap i32)
    (param $first i32) (param $n i32) (param $D i32) (param $acc i32)
    (local $j i32) (local $i i32) (local $row i32) (local $pj v128) (local $a i32) (local $hv v128)
    (local.set $i (i32.const 0))
    (block $z (loop $zl
      (br_if $z (i32.ge_u (local.get $i) (local.get $D)))
      (v128.store (i32.add (local.get $acc) (i32.shl (local.get $i) (i32.const 2))) (v128.const i32x4 0 0 0 0))
      (local.set $i (i32.add (local.get $i) (i32.const 4)))
      (br $zl)))
    (local.set $j (local.get $first))
    (block $done
      (loop $keys
        (br_if $done (i32.ge_u (local.get $j) (local.get $n)))
        (local.set $row (i32.add (local.get $v) (i32.mul (i32.rem_u (local.get $j) (local.get $cap)) (local.get $stride))))
        (local.set $pj (f32x4.splat (f32.load (i32.add (local.get $p) (i32.shl (local.get $j) (i32.const 2))))))
        (local.set $i (i32.const 0))
        (block $rd
          (loop $r
            (br_if $rd (i32.ge_u (local.get $i) (local.get $D)))
            (local.set $hv (v128.load (i32.add (local.get $row) (i32.shl (local.get $i) (i32.const 1)))))
            (local.set $a (i32.add (local.get $acc) (i32.shl (local.get $i) (i32.const 2))))
            (v128.store (local.get $a) (f32x4.add (v128.load (local.get $a))
              (f32x4.mul (local.get $pj) (call $h2f (i32x4.extend_low_i16x8_u (local.get $hv))))))
            (v128.store offset=16 (local.get $a) (f32x4.add (v128.load offset=16 (local.get $a))
              (f32x4.mul (local.get $pj) (call $h2f (i32x4.extend_high_i16x8_u (local.get $hv))))))
            (local.set $i (i32.add (local.get $i) (i32.const 8)))
            (br $r)))
        (local.set $j (i32.add (local.get $j) (i32.const 1)))
        (br $keys))))
)
