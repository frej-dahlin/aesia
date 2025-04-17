pub fn HalvedMeanSquareError(dim: usize) type {
    return struct {
        // Fixme: Use SIMD.
        pub fn eval(y_pred: *const [dim]f32, y_real: *const [dim]f32) f32 {
            var result: f32 = 0;
            for (y_pred, y_real) |pred, real| result += (pred - real) * (pred - real);
            return result / 2;
        }

        pub fn gradient(y_pred: *const [dim]f32, y_real: *const [dim]f32, result: *[dim]f32) void {
            for (y_pred, y_real, result) |pred, real, *r| r.* = pred - real;
        }
    };
}
