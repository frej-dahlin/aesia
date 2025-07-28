const std = @import("std");

const aesia = @import("../aesia.zig");

const f32x4 = @Vector(4, f32);

/// A differentiable dyadic butterfly swap is one layer of a permutation network,
/// such as a butterfly diagram. It only accepts perfect powers of two, hence the layer is
/// specified by the parameter log2_dim, there are log2_dim possible 'stages'.
/// If log2_dim == 2, then there are two choices of stages,
/// 0, and 1. A stage 0 2-DyadicCrossover looks like
///      a b   c d
///       x     x
///     (a|b) (c|d).
/// Depending on its 2 steering bits a and b will possibly swap positions, and c and d
/// will possibly swap positions. A stage 1 2-DyadicCrossover will possibly swap a and c,
/// and b and d, again depending on its 2 steering bits.
// Note: When Aesia support in-place layers we can optimize this layer considerably.
pub fn ButterflySwap(log2_dim: usize, stage: usize) type {
    // Compile time checks.
    if (log2_dim == 0) @compileError("DyadicCrossover: log2_dim must be nonzero");
    if (stage >= log2_dim) @compileError("DyadicCrossover: stage must be beetween 0 and log2_dim - 1, inclusive");
    return struct {
        const Self = @This();

        const dim = 1 << log2_dim;
        // Distance between pairs.
        const delta = 1 << stage;
        const parameter_count = dim / 2;
        const parameter_alignment = 64;

        pub const info = aesia.layer.Info{
            .dim_in = dim,
            .dim_out = dim,
            .trainable = true,
            .parameter_count = parameter_count,
            .parameter_alignment = 64,
        };

        steer: [dim / 2]f32,
        input_buffer: [dim]f32,

        pub fn init(_: *Self, parameters: *[parameter_count]f32) void {
            @memset(parameters, 0);
        }

        pub fn takeParameters(self: *Self, parameters: *const [parameter_count]f32) void {
            @setFloatMode(.optimized);
            for (parameters, &self.steer) |logit, *expit| expit.* = 1 / (1 + @exp(-logit));
        }

        pub fn giveParameters(_: *Self) void {}

        pub fn eval(self: *const Self, input: *const [dim]f32, output: *[dim]f32) void {
            @setFloatMode(.optimized);
            var steer_index: usize = 0;
            for (0..dim >> (stage + 1)) |k| {
                const from = 2 * k * delta;
                const to = (2 * k + 1) * delta;
                for (from..to) |j| {
                    const a = input[j];
                    const b = input[j + delta];
                    const c = self.steer[steer_index];
                    steer_index += 1;
                    output[j] = (1 - c) * a + c * b;
                    output[j + delta] = c * a + (1 - c) * b;
                }
            }
        }

        pub fn forwardPass(self: *Self, input: *const [dim]f32, output: *[dim]f32) void {
            @setFloatMode(.optimized);
            @memcpy(&self.input_buffer, input);
            var steer_index: usize = 0;
            for (0..dim >> (stage + 1)) |k| {
                const from = 2 * k * delta;
                const to = (2 * k + 1) * delta;
                for (from..to) |j| {
                    const a = input[j];
                    const b = input[j + delta];
                    const c = self.steer[steer_index];
                    steer_index += 1;

                    output[j] = (1 - c) * a + c * b;
                    output[j + delta] = c * a + (1 - c) * b;
                }
            }
        }

        pub fn backwardPass(
            self: *const Self,
            activation_delta: *const [dim]f32,
            cost_gradient: *[dim]f32,
            argument_delta: *[dim]f32,
        ) void {
            @setFloatMode(.optimized);
            @memset(argument_delta, 0);
            var steer_index: usize = 0;
            for (0..dim >> (stage + 1)) |k| {
                const from = 2 * k * delta;
                const to = (2 * k + 1) * delta;
                for (from..to) |j| {
                    const a = self.input_buffer[j];
                    const b = self.input_buffer[j + delta];
                    const c = self.steer[steer_index];

                    cost_gradient[steer_index] += (b - a) * c * (1 - c) *
                        (activation_delta[j] - activation_delta[j + delta]);
                    steer_index += 1;

                    argument_delta[j] +=
                        activation_delta[j] * (1 - c) +
                        activation_delta[j + delta] * c;
                    argument_delta[j + delta] +=
                        activation_delta[j] * c +
                        activation_delta[j + delta] * (1 - c);
                }
            }
        }

        pub fn backwardPassFinal(
            self: *const Self,
            activation_delta: *const [dim]f32,
            cost_gradient: *[dim]f32,
        ) void {
            @setFloatMode(.optimized);
            var steer_index: usize = 0;
            for (0..dim >> (stage + 1)) |k| {
                const from = 2 * k * delta;
                const to = (2 * k + 1) * delta;
                for (from..to) |j| {
                    const a = self.input_buffer[j];
                    const b = self.input_buffer[j + delta];
                    const c = self.steer[steer_index];
                    cost_gradient[steer_index] +=
                        activation_delta[j] * (b - a) * c * (1 - c) +
                        activation_delta[j + delta] * (a - b) * c * (1 - c);
                    steer_index += 1;
                }
            }
        }
    };
}

/// A differentiable dyadic butterfly map is one layer of a mapping network,
/// such as a butterfly diagram. It only accepts perfect powers of two, hence the layer is
/// specified by the parameter log2_dim, there are log2_dim possible 'stages'.
/// Compared to ButterflySwap, this might not swap values, but instead reproduce one of the two
/// values conditionally.
pub fn ButterflyMap(log2_dim: usize, stage: usize) type {
    // Compile time checks.
    if (log2_dim == 0) @compileError("DyadicCrossover: log2_dim must be nonzero");
    if (stage >= log2_dim) @compileError("DyadicCrossover: stage must be beetween 0 and log2_dim - 1, inclusive");
    return struct {
        const Self = @This();

        const dim = 1 << log2_dim;
        // Distance between pairs.
        const delta = 1 << stage;
        const parameter_count = dim;
        const parameter_alignment = 64;

        pub const info = aesia.layer.Info{
            .dim_in = dim,
            .dim_out = dim,
            .trainable = true,
            .parameter_count = parameter_count,
            .parameter_alignment = 64,
        };

        steer: [dim]f32,
        input_buffer: [dim]f32,

        pub fn init(_: *Self, parameters: *[parameter_count]f32) void {
            @memset(parameters, -10);
        }

        pub fn takeParameters(self: *Self, parameters: *const [parameter_count]f32) void {
            @setFloatMode(.optimized);
            for (parameters, &self.steer) |logit, *expit| expit.* = 1 / (1 + @exp(-logit));
        }

        pub fn giveParameters(_: *Self) void {}

        pub fn eval(self: *const Self, input: *const [dim]f32, output: *[dim]f32) void {
            @setFloatMode(.optimized);
            var steer_index: usize = 0;
            for (0..dim >> (stage + 1)) |k| {
                const from = 2 * k * delta;
                const to = (2 * k + 1) * delta;
                for (from..to) |j| {
                    const a = input[j];
                    const b = input[j + delta];
                    const c = self.steer[steer_index];
                    steer_index += 1;
                    const d = self.steer[steer_index];
                    steer_index += 1;

                    output[j] = (1 - c) * a + c * b;
                    output[j + delta] = d * a + (1 - d) * b;
                }
            }
        }

        pub fn forwardPass(self: *Self, input: *const [dim]f32, output: *[dim]f32) void {
            @setFloatMode(.optimized);
            @memcpy(&self.input_buffer, input);
            self.eval(input, output);
        }

        pub fn backwardPass(
            self: *const Self,
            activation_delta: *const [dim]f32,
            cost_gradient: *[dim]f32,
            argument_delta: *[dim]f32,
        ) void {
            @setFloatMode(.optimized);
            @memset(argument_delta, 0);
            var steer_index: usize = 0;
            for (0..dim >> (stage + 1)) |k| {
                const from = 2 * k * delta;
                const to = (2 * k + 1) * delta;
                for (from..to) |j| {
                    const a = self.input_buffer[j];
                    const b = self.input_buffer[j + delta];

                    const c = self.steer[steer_index];
                    cost_gradient[steer_index] +=
                        activation_delta[j] * (b - a) * c * (1 - c);
                    steer_index += 1;

                    const d = self.steer[steer_index];
                    cost_gradient[steer_index] +=
                        activation_delta[j + delta] * (a - b) * d * (1 - d);
                    steer_index += 1;

                    const left_delta = activation_delta[j];
                    const right_delta = activation_delta[j + delta];

                    argument_delta[j] +=
                        left_delta * (1 - c) +
                        right_delta * d;
                    argument_delta[j + delta] +=
                        left_delta * c +
                        right_delta * (1 - d);
                }
            }
        }

        pub fn backwardPassFinal(
            self: *const Self,
            activation_delta: *const [dim]f32,
            cost_gradient: *[dim]f32,
        ) void {
            @setFloatMode(.optimized);
            var steer_index: usize = 0;
            for (0..dim >> (stage + 1)) |k| {
                const from = 2 * k * delta;
                const to = (2 * k + 1) * delta;
                for (from..to) |j| {
                    const a = self.input_buffer[j];
                    const b = self.input_buffer[j + delta];

                    const c = self.steer[steer_index];
                    cost_gradient[steer_index] +=
                        activation_delta[j] * (b - a) * c * (1 - c);
                    steer_index += 1;

                    const d = self.steer[steer_index];
                    cost_gradient[steer_index] +=
                        activation_delta[j + delta] * (a - b) * d * (1 - d);
                    steer_index += 1;
                }
            }
        }
    };
}

pub fn BenesMap(log2_dim: usize) []const type {
    var result: [2 * log2_dim - 1]type = undefined;

    var index: usize = 0;
    {
        var stage = log2_dim;
        while (stage > 0) {
            stage -= 1;
            result[index] = ButterflyMap(log2_dim, stage);
            index += 1;
        }
    }
    for (1..log2_dim) |stage| {
        result[index] = ButterflyMap(log2_dim, stage);
        index += 1;
    }
    return &result;
}

pub fn TruncatedBenesSwap(log2_dim: usize, depth: usize) []const type {
    var result: [2 * depth - 1]type = undefined;

    var index: usize = 0;
    {
        var stage = depth;
        while (stage > 0) {
            stage -= 1;
            result[index] = ButterflySwap(log2_dim, stage);
            index += 1;
        }
    }
    for (1..depth) |stage| {
        result[index] = ButterflySwap(log2_dim, stage);
        index += 1;
    }
    return &result;
}

pub fn TruncatedBenesMap(log2_dim: usize, depth: usize) []const type {
    var result: [2 * depth - 1]type = undefined;

    var index: usize = 0;
    {
        var stage = depth;
        while (stage > 0) {
            stage -= 1;
            result[index] = ButterflyMap(log2_dim, stage);
            index += 1;
        }
    }
    for (1..depth) |stage| {
        result[index] = ButterflyMap(log2_dim, stage);
        index += 1;
    }
    return &result;
}

pub fn TruncatedBenesGate(log2_dim: usize, depth: usize) []const type {
    var result: [2 * depth - 1]type = undefined;

    var index: usize = 0;
    {
        var stage = depth;
        while (stage > 0) {
            stage -= 1;
            result[index] = ButterflyGate(log2_dim, stage);
            index += 1;
        }
    }
    for (1..depth) |stage| {
        result[index] = ButterflyGate(log2_dim, stage);
        index += 1;
    }
    return &result;
}

pub fn ButterflyGate(log2_dim: usize, stage: usize) type {
    // Compile time checks.
    if (log2_dim == 0) @compileError("DyadicCrossover: log2_dim must be nonzero");
    if (stage >= log2_dim) @compileError("DyadicCrossover: stage must be beetween 0 and log2_dim - 1, inclusive");
    return struct {
        const Self = @This();

        const dim = 1 << log2_dim;
        // Distance between pairs.
        const delta = 1 << stage;
        const parameter_count = dim * 4;
        const parameter_alignment = 64;

        pub const info = aesia.layer.Info{
            .dim_in = dim,
            .dim_out = dim,
            .trainable = true,
            .parameter_count = parameter_count,
            .parameter_alignment = 64,
        };

        gates: [dim]f32x4,
        input_buffer: [dim]f32,

        pub fn init(_: *Self, parameters: *[dim]f32x4) void {
            for (0..dim) |i| {
                parameters[i] = if (i % 2 == 0)
                    .{ 4, 4, -4, -4 }
                else
                    .{ 4, -4, 4, -4 };
            }
            for (0..dim >> (stage + 1)) |k| {
                const from = 2 * k * delta;
                const to = (2 * k + 1) * delta;
                for (from..to) |j| {
                    const index_left = j;
                    const index_right = j + delta;

                    parameters[index_left] = .{ 4, 4, -4, -4 };
                    parameters[index_right] = .{ 4, -4, 4, -4 };
                }
            }
        }

        pub fn takeParameters(self: *Self, parameters: *const [dim]f32x4) void {
            @setFloatMode(.optimized);
            for (parameters, &self.gates) |logit, *expit|
                expit.* = @as(f32x4, @splat(1)) / (@as(f32x4, @splat(1)) + @exp(-logit));
        }

        pub fn giveParameters(_: *Self) void {}

        pub fn eval(self: *const Self, input: *const [dim]f32, output: *[dim]f32) void {
            @setFloatMode(.optimized);
            for (0..dim >> (stage + 1)) |k| {
                const from = 2 * k * delta;
                const to = (2 * k + 1) * delta;
                for (from..to) |j| {
                    const index_left = j;
                    const index_right = j + delta;

                    const a = input[index_left];
                    const b = input[index_right];
                    const gate_left = self.gates[index_left];
                    const gate_right = self.gates[index_right];
                    const vector: f32x4 = .{
                        a * b,
                        a * (1 - b),
                        (1 - a) * b,
                        (1 - a) * (1 - b),
                    };
                    output[index_left] = @reduce(.Add, vector * gate_left);
                    output[index_right] = @reduce(.Add, vector * gate_right);
                }
            }
        }

        pub fn forwardPass(self: *Self, input: *const [dim]f32, output: *[dim]f32) void {
            @setFloatMode(.optimized);
            @memcpy(&self.input_buffer, input);
            self.eval(input, output);
        }

        pub fn backwardPass(
            self: *const Self,
            activation_delta: *const [dim]f32,
            cost_gradient: *[dim]f32x4,
            argument_delta: *[dim]f32,
        ) void {
            @setFloatMode(.optimized);
            @memset(argument_delta, 0);
            for (0..dim >> (stage + 1)) |k| {
                const from = 2 * k * delta;
                const to = (2 * k + 1) * delta;
                for (from..to) |j| {
                    const index_left = j;
                    const index_right = j + delta;

                    const a = self.input_buffer[index_left];
                    const b = self.input_buffer[index_right];
                    const vector: f32x4 = .{
                        a * b,
                        a * (1 - b),
                        (1 - a) * b,
                        (1 - a) * (1 - b),
                    };
                    const gradient_a: f32x4 = .{
                        b,
                        (1 - b),
                        -b,
                        -(1 - b),
                    };
                    const gradient_b: f32x4 = .{
                        a,
                        -a,
                        (1 - a),
                        -(1 - a),
                    };
                    const delta_left = activation_delta[index_left];
                    const delta_right = activation_delta[index_right];
                    const gate_left = self.gates[index_left];
                    const gate_right = self.gates[index_right];

                    cost_gradient[index_left] += @as(f32x4, @splat(delta_left)) *
                        gate_left * (@as(f32x4, @splat(1)) - gate_left) * vector;
                    cost_gradient[index_right] += @as(f32x4, @splat(delta_right)) *
                        gate_right * (@as(f32x4, @splat(1)) - gate_right) * vector;

                    argument_delta[index_left] +=
                        delta_left * @reduce(.Add, gate_left * gradient_a) +
                        delta_right * @reduce(.Add, gate_right * gradient_a);
                    argument_delta[index_right] +=
                        delta_left * @reduce(.Add, gate_left * gradient_b) +
                        delta_right * @reduce(.Add, gate_right * gradient_b);
                }
            }
        }

        pub fn backwardPassFinal(
            self: *const Self,
            activation_delta: *const [dim]f32,
            cost_gradient: *[dim]f32x4,
        ) void {
            @setFloatMode(.optimized);
            for (0..dim >> (stage + 1)) |k| {
                const from = 2 * k * delta;
                const to = (2 * k + 1) * delta;
                for (from..to) |j| {
                    const index_left = j;
                    const index_right = j + delta;
                    const a = self.input_buffer[index_left];
                    const b = self.input_buffer[index_right];
                    const vector: f32x4 = .{
                        a * b,
                        a * (1 - b),
                        (1 - a) * b,
                        (1 - a) * (1 - b),
                    };
                    const delta_left = activation_delta[index_left];
                    const delta_right = activation_delta[index_right];
                    const gate_left = self.gates[index_left];
                    const gate_right = self.gates[index_right];

                    cost_gradient[index_left] += @as(f32x4, @splat(delta_left)) *
                        gate_left * (@as(f32x4, @splat(1)) - gate_left) * vector;
                    cost_gradient[index_right] += @as(f32x4, @splat(delta_right)) *
                        gate_right * (@as(f32x4, @splat(1)) - gate_right) * vector;
                }
            }
        }
    };
}
