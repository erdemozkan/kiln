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
)
