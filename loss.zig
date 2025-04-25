const std = @import("std");
const assert = std.debug.assert;

// Todo: use SIMD.
pub fn HalvedMeanSquareError(dim: usize) type {
    return struct {
        pub const Label = [dim]f32;
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

pub fn DiscreteCrossEntropy(Int: type, class_count: usize) type {
    if (class_count == 0) @compileError("DiscreteCrossEntropy: invalid class_count equal to 0");
    if (class_count > std.math.maxInt(Int)) @compileError(std.fmt.comptimePrint("DiscreteCrossEntropy: integer type {s} can not hold {d} classes", .{ @typeName(Int), class_count }));
    return struct {
        pub const Label = Int;
        pub const Prediction = [class_count]f32;

        pub fn eval(prediction: *const Prediction, label: *const Label) f32 {
            assert(label.* < class_count);
            var denom: f32 = 0;
            const max = std.mem.max(f32, prediction);
            for (prediction) |logit| denom += @exp(logit - max);
            return -@log(@exp(prediction[label.*] - max) / denom);
        }

        pub fn gradient(prediction: *const Prediction, label: *const Label, output: *[class_count]f32) void {
            assert(label.* < class_count);
            var denom: f32 = 0;
            const max = std.mem.max(f32, prediction);
            for (prediction) |logit| denom += @exp(logit - max);
            for (output, prediction, 0..) |*partial, logit, j| {
                const kronecker: f32 = if (label.* == j) 1.0 else 0.0;
                partial.* = @exp(logit - max) / denom - kronecker;
            }
        }
    };
}
