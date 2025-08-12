const std = @import("std");
const assert = std.debug.assert;

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

/// Returns 1 if the prediction was miss-classified, otherwise return 0;
pub fn MissClassificationCount(Class: type, class_count: usize) type {
    return struct {
        pub fn eval(prediction: *const [class_count]f32, label: *const Class) f32 {
            return if (std.mem.indexOfMax(f32, prediction) == label.*) 1.0 else 0.0;
        }
    };
}

/// CrossEntropy specifiction for so called 'hot-encodings', e.g. if the
/// label is encoded '3' not as [0,0,1,0], then we only need the nonzero value
/// to compute the loss.
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
            const max = std.mem.max(f32, prediction);
            var denom: f32 = 0;
            for (prediction) |logit| denom += @exp(logit - max);
            for (output, prediction, 0..) |*partial, logit, j| {
                const kronecker: f32 = if (label.* == j) 1.0 else 0.0;
                partial.* = @exp(logit - max) / denom - kronecker;
            }
        }
    };
}

/// Label smoothened cross entropy loss. It smears a probability of alpha over
/// all classes, i.e. an encoding [1, 0, 0, 0] with alpha = 0.1 => [975, 0.025, 0.025, 0.025].
pub fn SmoothenedCrossEntropy(Int: type, class_count: usize, alpha: f32) type {
    if (class_count == 0) @compileError("DiscreteCrossEntropy: invalid class_count equal to 0");
    if (class_count > std.math.maxInt(Int)) @compileError(std.fmt.comptimePrint("DiscreteCrossEntropy: integer type {s} can not hold {d} classes", .{ @typeName(Int), class_count }));
    return struct {
        pub const Label = Int;
        pub const Prediction = [class_count]f32;

        pub fn smoothLabel(hot: Label) Prediction {
            var result: Prediction = @splat(alpha / @as(f32, @floatFromInt(class_count)));
            result[hot] += 1 - alpha;
            return result;
        }

        pub fn eval(prediction: *const Prediction, label: *const Label) f32 {
            assert(label.* < class_count);
            const smooth_label = smoothLabel(label.*);
            var denom: f32 = 0;
            const max = std.mem.max(f32, prediction);
            for (prediction) |logit| denom += @exp(logit - max);
            var result: f32 = 0;
            for (0..class_count) |k| {
                result += -@log(@exp(prediction[k] - max) / denom) * smooth_label[k];
            }
            return result;
        }

        pub fn gradient(prediction: *const Prediction, label: *const Label, output: *[class_count]f32) void {
            assert(label.* < class_count);
            const smooth_label = smoothLabel(label.*);
            var denom: f32 = 0;
            const max = std.mem.max(f32, prediction);
            for (prediction) |logit| denom += @exp(logit - max);
            for (output, prediction, 0..) |*partial, logit, j| {
                partial.* = @exp(logit - max) / denom - smooth_label[j];
            }
        }
    };
}
