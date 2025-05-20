const std = @import("std");
const assert = std.debug.assert;

const f32x16 = @Vector(16, f32);
const f32x8 = @Vector(8, f32);
const f32x4 = @Vector(4, f32);

pub const LogicOptions = struct {
    rand: *std.Random,
};

pub fn Logic(dim_in_: usize, dim_out_: usize, options: LogicOptions) type {
    return struct {
        const Self = @This();

        pub const ItemIn = f32;
        pub const ItemOut = f32;
        pub const dim_in = dim_in_;
        pub const dim_out = dim_out_;
        // There are 16 possible logic gates, each one is assigned a probability logit.
        pub const parameter_count = 16 * node_count;
        // For efficient SIMD we ensure that the parameters align to a cacheline.
        pub const parameter_alignment: usize = 64;

        const node_count = dim_out;
        const ParentIndex = std.math.IntFittingRange(0, dim_in - 1);

        gradient: [node_count]f32x16 align(64),
        diff: [node_count][2]f32 align(64),
        /// The preprocessed parameters, computed by softmax(parameters).
        sigma: ?*[node_count]f32x16 align(64),
        parents: [node_count][2]ParentIndex,

        /// Gate namespace for SIMD vector of relaxed logic gates.
        const gate = struct {
            /// Returns a vector representing all values of a soft logic gate.
            /// Only intended for reference. The current implementation
            /// inlines this construction in forwardPass.
            fn vector(a: f32, b: f32) f32x16 {
                @setFloatMode(.optimized);
                return .{
                    0, // false
                    a * b, // and
                    a - a * b, // a and not b
                    a, // passthrough a
                    b - a * b, // b and not a
                    b, // passthrough b
                    a + b - 2 * a * b, // xor
                    a + b - a * b, // xnor
                    // The following values are simply negation of the above ones in order.
                    // Many authors reverse this order, i.e. true is the last gate,
                    // however this leads to a less efficient SIMD construction of the vector.
                    1,
                    1 - a * b,
                    1 - (a - a * b),
                    1 - a,
                    1 - (b - a * b),
                    1 - b,
                    1 - (a + b - 2 * a * b),
                    1 - (a + b - a * b),
                };
            }

            /// Returns the soft gate vector differentiated by the first variable.
            /// Note that it only depends on the second variable.
            fn aDiff(b: f32) f32x16 {
                @setFloatMode(.optimized);
                return .{
                    0,
                    b,
                    1 - b,
                    1,
                    -b,
                    0,
                    1 - 2 * b,
                    0,
                    0,
                    -b,
                    -(1 - b),
                    -1,
                    b,
                    0,
                    -(1 - 2 * b),
                    0,
                };
            }

            /// Returns the soft gate vector differentiated by the second variable.
            /// Note that it only depends on the first variable.
            fn bDiff(a: f32) f32x16 {
                @setFloatMode(.optimized);
                return .{
                    0,
                    a,
                    -a,
                    0,
                    1 - a,
                    1,
                    1 - 2 * a,
                    1 - a,
                    0,
                    -a,
                    a,
                    0,
                    -(1 - a),
                    -1,
                    -(1 - 2 * a),
                    -(1 - a),
                };
            }
        };

        pub fn init(self: *Self, parameters: *[parameter_count]f32) void {
            parameters.* = @bitCast(@as(
                [node_count]f32x16,
                @splat(.{ 0, 0, 0, 1, 0, 0 } ++ .{0} ** 10), // passthrough a bias
            ));
            self.* = .{
                .sigma = null,
                .gradient = undefined,
                .diff = undefined,
                .parents = undefined,
            };
            for (0..node_count) |j| {
                self.parents[j] = .{
                    options.rand.intRangeLessThan(ParentIndex, 0, dim_in),
                    options.rand.intRangeLessThan(ParentIndex, 0, dim_in),
                };
            }
        }

        // Logits are normalized such that max(logits) == 0.
        // When the logits grow to big exp^logit will explode.
        fn maxNormalize(v: f32x16) f32x16 {
            return v - @as(f32x16, @splat(@reduce(.Max, v)));
        }

        // Experimentally base 2 softmax is quite a bit faster.
        fn softmax2(logit: f32x16) f32x16 {
            @setFloatMode(.optimized);
            const sigma = @exp2(maxNormalize(logit));
            const denom = @reduce(.Add, sigma);
            return sigma / @as(f32x16, @splat(denom));
        }

        fn softmax2Inverse(sigma: f32x16) f32x16 {
            @setFloatMode(.optimized);
            return @log2(sigma);
        }

        /// Evaluates the layer. Asserts that the layer has been given parameters via
        /// takeParameters.
        pub fn eval(
            noalias self: *Self,
            noalias input: *const [dim_in]ItemIn,
            noalias output: *[dim_out]ItemOut,
        ) void {
            @setFloatMode(.optimized);
            assert(self.sigma != null);
            for (self.sigma.?, output, self.parents) |sigma, *activation, parents| {
                const a = input[parents[0]];
                const b = input[parents[1]];
                activation.* = @reduce(.Add, sigma * gate.vector(a, b));
            }
        }

        /// Evaluates the layer and caches relevant data to compute the gradient with
        /// respect to the underlying logits. Asserts that the layer has been given parameters
        /// via takeParameters.
        pub fn forwardPass(
            noalias self: *Self,
            noalias input: *const [dim_in]ItemIn,
            noalias output: *[dim_out]ItemOut,
        ) void {
            @setFloatMode(.optimized);
            assert(self.sigma != null);
            // For some reason it is faster to to a simple loop rather than a multi item one.
            // I found needless memcpy calls with callgrind.
            for (0..node_count) |j| {
                const a = input[self.parents[j][0]];
                const b = input[self.parents[j][1]];

                // Inline construction of gateVector(a, b), diff_a(b), and Gate.diff_b(a).
                // We do it in this way since all three depend on mix_coef, and two out of three
                // depend on a_coef/b_coef.
                const a_coef: f32x8 = .{ 0, 0, 1, 1, 0, 0, 1, 1 };
                const b_coef: f32x8 = .{ 0, 0, 0, 0, 1, 1, 1, 1 };
                const mix_coef: f32x8 = .{ 0, 1, -1, 0, -1, 0, -2, -1 };

                // All three desired vectors have halfway symmetry so we only
                // explicitly construct the first half.
                const diff_a_half = a_coef + mix_coef * @as(f32x8, @splat(b));
                const diff_b_half = b_coef + mix_coef * @as(f32x8, @splat(a));
                const gate_half =
                    a_coef * @as(f32x8, @splat(a)) +
                    b_coef * @as(f32x8, @splat(b)) +
                    mix_coef * @as(f32x8, @splat(a * b));
                const gate_vector = std.simd.join(gate_half, @as(f32x8, @splat(1)) - gate_half);

                const sigma = self.sigma.?[j];
                self.diff[j] = .{
                    @reduce(.Add, sigma * std.simd.join(diff_a_half, -diff_a_half)),
                    @reduce(.Add, sigma * std.simd.join(diff_b_half, -diff_b_half)),
                };
                output[j] = @reduce(.Add, sigma * gate_vector);
                self.gradient[j] = sigma * (gate_vector - @as(f32x16, @splat(output[j])));
            }
        }

        // Fixme: Create a function "backward" that does not pass delta backwards, applicable for
        // the first layer in a network only.
        /// Accumulates the cost gradient using the given derivative of the loss function with
        /// respect to the layer's activations as well as computing the same derivative for the
        /// previous layer. Assert that the layer has been given parameters via takeParameters.
        pub fn backwardPass(
            noalias self: *Self,
            noalias input: *const [dim_out]f32,
            noalias cost_gradient: *[node_count]f32x16,
            noalias output: *[dim_in]f32,
        ) void {
            @setFloatMode(.optimized);
            @memset(output, 0);
            for (0..node_count) |j| {
                cost_gradient[j] += self.gradient[j] * @as(f32x16, @splat(input[j]));
                output[self.parents[j][0]] += self.diff[j][0] * input[j];
                output[self.parents[j][1]] += self.diff[j][1] * input[j];
            }
        }

        /// Takes the given parameters as logits and preprocesses them using softmax.
        /// This amortizes the cost of computing softmax over the number of evaluations
        /// before returning them with giveParameters.
        pub fn takeParameters(self: *Self, parameters: *[node_count]f32x16) void {
            @setFloatMode(.optimized);
            assert(self.sigma == null);
            self.sigma = parameters;
            for (self.sigma.?, parameters) |*sigma, logit| sigma.* = softmax2(logit);
        }

        /// Gives back the parameters as logits by posprocessing them using the 'inverse' of
        /// softmax.
        pub fn giveParameters(self: *Self) void {
            assert(self.sigma != null);
            for (self.sigma.?) |*sigma| sigma.* = softmax2Inverse(sigma.*);
            self.sigma = null;
        }
    };
}

/// A faster and leaner differentiable logic gate layer.
/// A binary logic gate is equivalent to a truth table of the following form:
///     a[0] a[1] | value
///     ----------+------
///      1    1   |  b[0]
///      1    0   |  b[1]
///      0    1   |  b[2]
///      0    0   |  b[3]
/// Where each b[?] can take the value 1 or 0, there are 2^4 = 16 possible
/// combinations. In classical differentiable logic gates each logic gate owns
/// a *joint* probability distribution vector of size 16. It describes the
/// probability that the differentiable gate is a given specific logic gate.
/// Packed differentiable logic gates instead take a vector of 4 *independent*
/// probabilities, each b[?] is the probability that its a 1. To evaluate a
/// binary logic gate we can use the expression:
///     ( a[0] &  a[1] & b[0]) v
///     ( a[0] & !a[1] & b[1]) v
///     (!a[0] &  a[1] & b[2]) v
///     (!a[0] & !a[1] & b[2]),
/// where &, v, and ! means logical and, or, negation, respectively. Furthermore
/// each or-clause is mutually exclusive. Thus a continuous relaxation of the
/// above expression is the sum of the relaxation of the or-clauses! This leaves
/// us with the expression in packed_gate.eval, the gradient and so forth can
/// easily be derived from there.
pub fn PackedLogic(dim_in_: usize, dim_out_: usize, options: LogicOptions) type {
    return struct {
        const Self = @This();

        pub const ItemIn = f32;
        pub const ItemOut = f32;
        pub const dim_in = dim_in_;
        pub const dim_out = dim_out_;
        // There are 16 possible logic gates, we use a novel packed representation
        // that packs these into 4 parameters.
        pub const parameter_count = 4 * node_count;
        // For efficient SIMD we ensure that the parameters align to a cacheline.
        pub const parameter_alignment: usize = 64;

        const node_count = dim_out;
        const ParentIndex = std.math.IntFittingRange(0, dim_in - 1);

        // Note: Do not try to be clever with SIMD operations, the compiler does a great
        // job with the following functions.
        const packed_gate = struct {
            // Evaluates the packed differentiable logic gate.
            pub fn eval(beta: f32x4, a: f32, b: f32) f32 {
                @setFloatMode(.optimized);
                const x = a * b * beta[0];
                const y = a * (1 - b) * beta[1];
                const z = (1 - a) * b * beta[2];
                const w = (1 - a) * (1 - b) * beta[3];
                return x + y + z + w;
            }

            // The derivative of the logic gate with respect to its first variable.
            pub fn aDiff(beta: f32x4, b: f32) f32 {
                @setFloatMode(.optimized);
                const x = b * beta[0];
                const y = (1 - b) * beta[1];
                const z = -b * beta[2];
                const w = -(1 - b) * beta[3];
                return x + y + z + w; // -
            }

            // The derivative of the logic gate with respect to its second variable.
            pub fn bDiff(beta: f32x4, a: f32) f32 {
                @setFloatMode(.optimized);
                const x = a * beta[0];
                const y = -a * beta[1];
                const z = (1 - a) * beta[2];
                const w = -(1 - a) * beta[3];
                return x + y + z + w;
            }

            // The gradient of the logic gate with respect to its underlying logits.
            pub fn gradient(beta: f32x4, a: f32, b: f32) f32x4 {
                @setFloatMode(.optimized);
                const x = a * b * beta[0];
                const y = a * (1 - b) * beta[1];
                const z = (1 - a) * b * beta[2];
                const w = (1 - a) * (1 - b) * beta[3];
                return .{
                    (1 - beta[0]) * x,
                    (1 - beta[1]) * y,
                    (1 - beta[2]) * z,
                    (1 - beta[3]) * w,
                };
            }
        };

        // The preprocessed parameters, computed by softmax(parameters).
        beta: [node_count]f32x4 align(64),
        // The arguments for each gate is cached such that backwardPass can compute
        // the gradient.
        arguments: [node_count][2]f32,
        parents: [node_count][2]ParentIndex,

        fn logistic(x: f32x4) f32x4 {
            @setFloatMode(.optimized);
            return @as(f32x4, @splat(1)) / (@as(f32x4, @splat(1)) + @exp(-x));
        }

        /// Initializes the layer's parameters and its parent connections.
        pub fn init(self: *Self, parameters: *[parameter_count]f32) void {
            parameters.* = @bitCast(@as([node_count]f32x4, @splat(.{ 4, 4, 0, 0 })));
            self.* = undefined;
            for (0..node_count) |j| {
                self.parents[j] = .{
                    options.rand.intRangeLessThan(ParentIndex, 0, dim_in),
                    options.rand.intRangeLessThan(ParentIndex, 0, dim_in),
                };
            }
        }

        /// Evaluates the layer.
        pub fn eval(
            noalias self: *Self,
            noalias input: *const [dim_in]ItemIn,
            noalias output: *[dim_out]ItemOut,
        ) void {
            @setFloatMode(.optimized);
            for (self.beta, output, 0..) |beta, *activation, j| {
                const a = input[self.parents[j][0]];
                const b = input[self.parents[j][1]];
                activation.* = packed_gate.eval(beta, a, b);
            }
        }

        /// Evaluates the layer and caches the arguments to each gate to be used in backwardPass.
        pub fn forwardPass(
            noalias self: *Self,
            noalias input: *const [dim_in]ItemIn,
            noalias output: *[dim_out]ItemOut,
        ) void {
            @setFloatMode(.optimized);
            // For some reason it is faster to to a simple loop rather than a multi item one.
            // I found needless memcpy calls with callgrind.
            // It is also faster to split the loop into two parts, my guess is that
            // the first one trashes the cache and the bottom one is vectorized by the compiler.
            for (0..node_count) |j| {
                self.arguments[j] = .{
                    input[self.parents[j][0]],
                    input[self.parents[j][1]],
                };
            }
            for (0..node_count) |j| {
                const a, const b = self.arguments[j];
                output[j] = packed_gate.eval(self.beta[j], a, b);
            }
        }

        /// Computes the cost gradient and the gradient with respect to the previous layer's
        /// activations.
        pub fn backwardPass(
            noalias self: *Self,
            noalias input: *const [dim_out]f32,
            noalias cost_gradient: *[node_count]f32x4,
            noalias output: *[dim_in]f32,
        ) void {
            @setFloatMode(.optimized);
            @memset(output, 0);
            for (0..node_count) |j| {
                const a, const b = self.arguments[j];
                const beta = self.beta[j];
                cost_gradient[j] += packed_gate.gradient(beta, a, b) *
                    @as(f32x4, @splat(input[j]));
                output[self.parents[j][0]] += packed_gate.aDiff(beta, b) * input[j];
                output[self.parents[j][1]] += packed_gate.bDiff(beta, a) * input[j];
            }
        }

        /// Copies parameters as logits and preprocesses them through the logistic function.
        pub fn takeParameters(self: *Self, parameters: *[node_count]f32x4) void {
            @setFloatMode(.optimized);
            for (0..node_count) |j| self.beta[j] = logistic(parameters[j]);
        }

        pub fn giveParameters(self: *Self) void {
            _ = self;
        }
    };
}

/// Divides the input into dim_out buckets, each output is the sequential sum of
/// dim_in / dim_out items of the input.
pub fn GroupSum(dim_in_: usize, dim_out_: usize) type {
    return struct {
        const Self = @This();

        pub const ItemIn = f32;
        pub const ItemOut = f32;
        pub const dim_in = dim_in_;
        pub const dim_out = dim_out_;

        const quot = dim_in / dim_out;
        const scale: f32 = 1.0 / (@as(comptime_float, @floatFromInt(dim_out)));

        pub fn eval(input: *const [dim_in]ItemIn, output: *[dim_out]ItemOut) void {
            @memset(output, 0);
            for (output, 0..) |*coord, k| {
                const from = k * quot;
                const to = from + quot;
                for (input[from..to]) |softbit| coord.* += softbit;
                coord.* *= scale;
            }
        }

        pub fn forwardPass(input: *const [dim_in]ItemIn, output: *[dim_out]ItemOut) void {
            return eval(input, output);
        }

        pub fn backwardPass(input: *const [dim_out]f32, output: *[dim_in]f32) void {
            for (input, 0..) |child, k| {
                const from = k * quot;
                const to = from + quot;
                for (output[from..to]) |*parent| parent.* = child * scale;
            }
        }
    };
}

pub fn ConvolutionLogic(height: usize, width: usize) type {
    return struct {
        const Self = @This();

        pub const ItemIn = f32;
        pub const ItemOut = f32;
        pub const dim_in = height * width;
        pub const dim_out = height * width;
        pub const parameter_count = 1 << 9;
        const ParameterVector = @Vector(parameter_count, f32);
        pub const parameter_alignment = @alignOf(ParameterVector);

        var input_buffer: [height + 2][width + 2]f32 = undefined;
        var delta_buffer: [height + 2][width + 2]f32 = undefined;

        kernel: MultiGate(9),
        receptive_fields: [height][width][9]f32,

        const Point = @Vector(2, usize);
        const receptive_offsets = [9]Point{
            .{ 0, 0 },
            .{ 0, 1 },
            .{ 0, 2 },
            .{ 1, 0 },
            .{ 1, 1 },
            .{ 1, 2 },
            .{ 2, 0 },
            .{ 2, 1 },
            .{ 2, 2 },
        };

        fn inBounds(point: Point) bool {
            return 0 < point[0] and point[0] < height and
                0 < point[1] and point[1] < width;
        }

        pub fn eval(
            self: *Self,
            input: *const [height][width]f32,
            output: *[height][width]f32,
        ) void {
            input_buffer = @splat(@splat(0));
            for (input_buffer[1 .. input_buffer.len - 1], input) |*buffer_row, row| {
                @memcpy(buffer_row[1 .. buffer_row.len - 1], &row);
            }
            for (0..height) |row| {
                for (0..width) |col| {
                    const receptions = &self.receptive_fields[row][col];
                    const center = Point{ row, col };
                    inline for (receptions, receptive_offsets) |*reception, offset| {
                        const point = center + offset;
                        reception.* = input_buffer[point[0]][point[1]];
                    }
                }
            }
            for (0..height) |row| {
                for (0..width) |col| {
                    output[row][col] = self.kernel.eval(&self.receptive_fields[row][col]);
                }
            }
        }

        pub fn forwardPass(
            self: *Self,
            input: *const [height][width]f32,
            output: *[height][width]f32,
        ) void {
            return self.eval(input, output);
        }

        pub fn backwardPass(
            self: *Self,
            activation_delta: *const [height][width]f32,
            cost_gradient: *ParameterVector,
            argument_delta: *[height][width]f32,
        ) void {
            delta_buffer = @splat(@splat(0));
            for (0..height) |row| {
                for (0..width) |col| {
                    if (activation_delta[row][col] == 0) continue;
                    const field = &self.receptive_fields[row][col];
                    cost_gradient.* += self.kernel.gradient(field) *
                        @as(ParameterVector, @splat(activation_delta[row][col]));
                    const argument_gradient = self.kernel.diff(field);
                    const center = Point{ row, col };
                    for (receptive_offsets, 0..) |offset, k| {
                        const point = center + offset;
                        delta_buffer[point[0]][point[1]] +=
                            argument_gradient[k] * activation_delta[row][col];
                    }
                }
            }
            for (delta_buffer[1 .. delta_buffer.len - 1], argument_delta) |*buffer_row, *delta| {
                @memcpy(delta, buffer_row[1 .. buffer_row.len - 1]);
            }
        }

        pub fn init(_: *Self, parameters: *ParameterVector) void {
            for (0..parameter_count / 2) |j| parameters[j] = 1;
            for (parameter_count / 2..parameter_count) |j| parameters[j] = 0;
        }

        pub fn takeParameters(self: *Self, parameters: *ParameterVector) void {
            self.kernel.beta = @as(ParameterVector, @splat(1)) /
                (@as(ParameterVector, @splat(1)) + @exp(-parameters.*));
        }

        pub fn giveParameters(_: *Self) void {}
    };
}

fn MultiGate(arity: usize) type {
    return struct {
        const Self = @This();

        const beta_len = 1 << arity;
        const Beta = @Vector(beta_len, f32);
        const ArgumentVector = @Vector(arity, f32);

        beta: Beta,

        // Evaluates the packed differentiable logic gate.
        pub fn eval(gate: *const Self, input: *const [arity]f32) f32 {
            @setFloatMode(.optimized);
            var result: f32 = 0;
            @setEvalBranchQuota(4 * beta_len * arity * arity);
            inline for (0..beta_len) |i| {
                var product: f32 = gate.beta[i];
                inline for (input, 0..) |x, bit| {
                    product *= if (comptime (1 << bit) & i != 0) x else (1 - x);
                }
                result += product;
            }
            return result;
        }

        pub fn diff(gate: *const Self, input: *const [arity]f32) ArgumentVector {
            @setFloatMode(.optimized);
            var result: ArgumentVector = @splat(0);
            @setEvalBranchQuota(4 * beta_len * arity * arity);
            inline for (0..beta_len) |i| {
                var product: f32 = gate.beta[i];
                inline for (input, 0..) |x, bit| {
                    product *= if (comptime (1 << bit) & i != 0) x else (1 - x);
                }
                inline for (0..arity) |j| {
                    const x = input[j];
                    result[j] += product /
                        if (comptime (1 << j) & i != 0) (x + 1e-09) else -(1 - x - 1e-09);
                }
            }
            return result;
        }

        pub fn gradient(gate: *const Self, input: *const [arity]f32) Beta {
            @setFloatMode(.optimized);
            var result: Beta = undefined;
            @setEvalBranchQuota(4 * beta_len * arity * arity);
            inline for (0..beta_len) |j| {
                var product: f32 = (1 - gate.beta[j]) * gate.beta[j];
                inline for (input, 0..) |x, bit| {
                    product *= if (comptime (1 << bit) & j != 0) x else (1 - x);
                }
                result[j] = product;
            }
            return result;
        }
    };
}

pub const MultiLogicOptions = struct {
    arity: usize,
    rand: *std.Random,
};

pub fn MultiLogicGate(dim_in_: usize, dim_out_: usize, options: MultiLogicOptions) type {
    return struct {
        const Self = @This();

        pub const ItemIn = f32;
        pub const ItemOut = f32;
        pub const dim_in = dim_in_;
        pub const dim_out = dim_out_;

        pub const parameter_count = parameter_vector_len * node_count;
        // For efficient SIMD we ensure that the parameters align to a cacheline.
        pub const parameter_alignment: usize = @max(64, @alignOf(ParameterVector));

        const node_count = dim_out;
        const ParentIndex = std.math.IntFittingRange(0, dim_in - 1);
        const arity = options.arity;
        const parameter_vector_len = 1 << arity;
        const ParameterVector = @Vector(parameter_vector_len, f32);
        const ArgumentVector = @Vector(arity, f32);

        const Gate = struct {
            sigma: ParameterVector,
            // Evaluates the packed differentiable logic gate.
            pub fn eval(gate: *const Gate, input: *const [arity]f32) f32 {
                @setFloatMode(.optimized);
                var result: f32 = 0;
                @setEvalBranchQuota(4 * parameter_vector_len * arity * arity);
                inline for (0..parameter_vector_len) |i| {
                    var product: f32 = gate.sigma[i];
                    inline for (input, 0..) |x, bit| {
                        product *= if (comptime (1 << bit) & i != 0) x else (1 - x);
                    }
                    result += product;
                }
                return result;
            }

            // Optimization: Use greycode.
            pub fn diff(gate: *const Gate, input: *const [arity]f32) ArgumentVector {
                @setFloatMode(.optimized);
                var result: ArgumentVector = @splat(0);
                @setEvalBranchQuota(4 * parameter_vector_len * arity * arity);
                inline for (0..parameter_vector_len) |i| {
                    var product: f32 = gate.sigma[i];
                    inline for (input, 0..) |x, bit| {
                        product *= if (comptime (1 << bit) & i != 0) x + 1e-09 else (1 - x + 1e-09);
                    }
                    inline for (0..arity) |j| {
                        const x = input[j];
                        result[j] += product /
                            if (comptime (1 << j) & i != 0) (x + 1e-09) else -(1 - x + 1e-09);
                    }
                }
                return result;
            }

            pub fn gradient(gate: *const Gate, input: *const [arity]f32) ParameterVector {
                @setFloatMode(.optimized);
                var result: ParameterVector = undefined;
                @setEvalBranchQuota(4 * parameter_vector_len * arity * arity);
                inline for (0..parameter_vector_len) |j| {
                    var product: f32 = (1 - gate.sigma[j]) * gate.sigma[j];
                    inline for (input, 0..) |x, bit| {
                        product *= if (comptime (1 << bit) & j != 0) x else (1 - x);
                    }
                    result[j] = product;
                }
                return result;
            }
        };

        const ArgumentIndex = std.math.IntFittingRange(0, parameter_vector_len - 1);

        gates: [node_count]Gate align(parameter_alignment),
        inputs: [node_count][arity]f32,
        parents: [node_count][arity]std.math.IntFittingRange(0, dim_in - 1),
        max_index: [node_count]ArgumentIndex,

        fn logistic(x: ParameterVector) ParameterVector {
            @setFloatMode(.optimized);
            return @as(ParameterVector, @splat(1)) / (@as(ParameterVector, @splat(1)) + @exp(-x));
        }

        pub fn init(self: *Self, parameters: *[parameter_count]f32) void {
            parameters.* = @bitCast(@as(
                [node_count]ParameterVector,
                @splat(.{1} ** (parameter_vector_len / 2) ++ .{0} ** (parameter_vector_len / 2)),
            ));
            self.* = undefined;
            for (0..node_count) |j| {
                inline for (&self.parents[j]) |*parent| {
                    parent.* = options.rand.intRangeLessThan(ParentIndex, 0, dim_in);
                }
            }
        }

        pub fn eval(
            noalias self: *Self,
            noalias input: *const [dim_in]ItemIn,
            noalias output: *[dim_out]ItemOut,
        ) void {
            @setFloatMode(.optimized);
            for (0..node_count) |j| {
                inline for (self.parents[j], &self.inputs[j]) |parent, *gate_input| {
                    gate_input.* = input[parent];
                }
            }
            for (0..node_count) |j| {
                output[j] = self.gates[j].eval(&self.inputs[j]);
            }
        }

        pub fn forwardPass(
            noalias self: *Self,
            noalias input: *const [dim_in]ItemIn,
            noalias output: *[dim_out]ItemOut,
        ) void {
            @setFloatMode(.optimized);
            // For some reason it is faster to to a simple loop rather than a multi item one.
            // I found needless memcpy calls with callgrind.
            // It is also faster to split the loop into two parts, my guess is that
            // the first one trashes the cache and the bottom one is vectorized by the compiler.
            for (0..node_count) |j| {
                inline for (self.parents[j], &self.inputs[j]) |parent, *gate_input| {
                    gate_input.* = input[parent];
                }
            }
            for (0..node_count) |j| {
                output[j] = self.gates[j].eval(&self.inputs[j]);
            }
        }

        /// Fixme: Create a function "backwardPassFirst" that does not pass delta backwards,
        ///  applicable for the first layer in a network only.
        pub fn backwardPass(
            noalias self: *Self,
            noalias input: *const [dim_out]f32,
            noalias cost_gradient: *[node_count]ParameterVector,
            noalias output: *[dim_in]f32,
        ) void {
            @setFloatMode(.optimized);
            @memset(output, 0);
            for (0..node_count) |j| {
                const gate = &self.gates[j];
                cost_gradient[j] += gate.gradient(&self.inputs[j]) *
                    @as(ParameterVector, @splat(input[j]));
                const argument_gradient = gate.diff(&self.inputs[j]);
                inline for (self.parents[j], 0..) |parent, k| {
                    output[parent] += argument_gradient[k] * input[j];
                }
            }
        }

        pub fn takeParameters(self: *Self, parameters: *[node_count]ParameterVector) void {
            @setFloatMode(.optimized);
            for (0..node_count) |j| self.gates[j].sigma = logistic(parameters[j]);
        }

        pub fn giveParameters(self: *Self) void {
            _ = self;
        }
    };
}

pub fn MaxPool(height: usize, width: usize) type {
    return struct {
        const Self = @This();

        comptime {
            assert(height % 2 == 0);
            assert(width % 2 == 0);
        }

        pub const ItemIn = f32;
        pub const ItemOut = f32;
        pub const dim_in = height * width;
        pub const dim_out = height * width / 4;

        activation: [height / 2][width / 2]f32,
        max_index: [height / 2][width / 2]usize,

        pub fn init(_: *Self) void {}

        pub fn eval(
            _: *Self,
            input: *const [height][width]f32,
            output: *[height / 2][width / 2]f32,
        ) void {
            for (0..height / 2) |row| {
                for (0..width / 2) |col| {
                    output[row][col] = std.mem.max(f32, &.{
                        input[2 * row][2 * col],
                        input[2 * row][2 * col + 1],
                        input[2 * row + 1][2 * col],
                        input[2 * row][2 * col + 1],
                    });
                }
            }
        }

        pub fn forwardPass(
            self: *Self,
            input: *const [height][width]f32,
            output: *[height / 2][width / 2]f32,
        ) void {
            for (0..height / 2) |row| {
                for (0..width / 2) |col| {
                    self.activation[row][col] = std.mem.max(f32, &.{
                        input[2 * row][2 * col],
                        input[2 * row][2 * col + 1],
                        input[2 * row + 1][2 * col],
                        input[2 * row][2 * col + 1],
                    });
                    output[row][col] = self.activation[row][col];
                    self.max_index[row][col] = std.mem.indexOfMax(f32, &.{
                        input[2 * row][2 * col],
                        input[2 * row][2 * col + 1],
                        input[2 * row + 1][2 * col],
                        input[2 * row][2 * col + 1],
                    });
                }
            }
        }

        pub fn backwardPass(
            self: *Self,
            activation_delta: *const [height / 2][width / 2]f32,
            argument_delta: *[height][width]f32,
        ) void {
            argument_delta.* = @splat(@splat(0));
            for (0..height / 2) |row| {
                for (0..width / 2) |col| {
                    switch (self.max_index[row][col]) {
                        0 => argument_delta[2 * row][2 * col] = activation_delta[row][col],
                        1 => argument_delta[2 * row][2 * col + 1] = activation_delta[row][col],
                        2 => argument_delta[2 * row + 1][2 * col] = activation_delta[row][col],
                        3 => argument_delta[2 * row][2 * col + 1] = activation_delta[row][col],
                        else => unreachable,
                    }
                }
            }
        }
    };
}

pub fn MultiLogicMax(dim_in_: usize, dim_out_: usize, options: MultiLogicOptions) type {
    return struct {
        const Self = @This();

        pub const ItemIn = f32;
        pub const ItemOut = f32;
        pub const dim_in = dim_in_;
        pub const dim_out = dim_out_;

        const node_count = dim_out;
        const arity = options.arity;
        const vector_len = 1 << arity;
        const Vector = @Vector(vector_len, f32);
        const ParameterIndex = std.math.IntFittingRange(0, vector_len - 1);
        const ParentIndex = std.math.IntFittingRange(0, dim_in - 1);

        pub const parameter_count = vector_len * node_count;
        // For efficient SIMD we ensure that the parameters align to a cacheline.
        pub const parameter_alignment: usize = @max(64, @alignOf(Vector));

        const EvalResult = struct {
            value: f32,
            index: ParameterIndex,
        };
        const Gate = struct {
            sigma: Vector,
            // Evaluates the packed differentiable logic gate.
            pub fn eval(gate: *const Gate, input: *const [arity]f32) EvalResult {
                @setFloatMode(.optimized);
                var value: f32 = 0;
                var max: f32 = 0;
                var index: ParameterIndex = 0;
                @setEvalBranchQuota(4 * vector_len * arity * arity);
                inline for (0..vector_len) |i| {
                    var product: f32 = gate.sigma[i];
                    inline for (input, 0..) |x, bit| {
                        product *= if (comptime (1 << bit) & i != 0) x else (1 - x);
                    }
                    value += product;
                    max = @max(max, product);
                    if (max == product) index = @truncate(i);
                }
                return .{
                    .value = value,
                    .index = index,
                };
            }

            const ArgumentVector = @Vector(arity, f32);
            pub fn diff(
                gate: *const Gate,
                input: *const [arity]f32,
                index: ParameterIndex,
            ) ArgumentVector {
                @setFloatMode(.optimized);
                var result: ArgumentVector = @splat(0);
                @setEvalBranchQuota(4 * vector_len * arity * arity);
                inline for (0..arity) |j| {
                    var product: f32 = gate.sigma[index];
                    inline for (input, 0..) |x, bit| {
                        if (comptime bit != j) {
                            product *= if ((1 << bit) & index != 0) x else (1 - x);
                        } else {
                            product *= if ((1 << bit) & index != 0) 1 else -1;
                        }
                    }
                    result[j] += product;
                }
                return result;
            }

            pub fn gradient(
                gate: *const Gate,
                input: *const [arity]f32,
                index: ParameterIndex,
            ) f32 {
                @setFloatMode(.optimized);
                @setEvalBranchQuota(4 * vector_len * arity * arity);
                var product: f32 = (1 - gate.sigma[index]) * gate.sigma[index];
                inline for (input, 0..) |x, bit| {
                    product *= if ((1 << bit) & index != 0) x else (1 - x);
                }
                return product;
            }
        };

        const ArgumentIndex = std.math.IntFittingRange(0, vector_len - 1);

        gates: [node_count]Gate align(parameter_alignment),
        inputs: [node_count][arity]f32,
        parents: [node_count][arity]std.math.IntFittingRange(0, dim_in - 1),
        max_index: [node_count]ArgumentIndex,

        fn logistic(x: Vector) Vector {
            @setFloatMode(.optimized);
            return @as(Vector, @splat(1)) / (@as(Vector, @splat(1)) + @exp(-x));
        }

        pub fn init(self: *Self, parameters: *[parameter_count]f32) void {
            parameters.* = @bitCast(@as(
                [node_count]Vector,
                @splat(.{1} ** (vector_len / 2) ++ .{0} ** (vector_len / 2)),
            ));
            self.* = undefined;
            for (0..node_count) |j| {
                inline for (&self.parents[j]) |*parent| {
                    parent.* = options.rand.intRangeLessThan(ParentIndex, 0, dim_in);
                }
            }
        }

        pub fn eval(
            noalias self: *Self,
            noalias input: *const [dim_in]ItemIn,
            noalias output: *[dim_out]ItemOut,
        ) void {
            @setFloatMode(.optimized);
            for (0..node_count) |j| {
                inline for (self.parents[j], &self.inputs[j]) |parent, *gate_input| {
                    gate_input.* = input[parent];
                }
            }
            for (0..node_count) |j| {
                const result = self.gates[j].eval(&self.inputs[j]);
                output[j] = result.value;
            }
        }

        pub fn forwardPass(
            noalias self: *Self,
            noalias input: *const [dim_in]ItemIn,
            noalias output: *[dim_out]ItemOut,
        ) void {
            @setFloatMode(.optimized);
            // For some reason it is faster to to a simple loop rather than a multi item one.
            // I found needless memcpy calls with callgrind.
            // It is also faster to split the loop into two parts, my guess is that
            // the first one trashes the cache and the bottom one is vectorized by the compiler.
            for (0..node_count) |j| {
                inline for (self.parents[j], &self.inputs[j]) |parent, *gate_input| {
                    gate_input.* = input[parent];
                }
            }
            for (0..node_count) |j| {
                const result = self.gates[j].eval(&self.inputs[j]);
                output[j] = result.value;
                self.max_index[j] = result.index;
            }
        }

        /// Fixme: Create a function "backwardPassFirst" that does not pass delta backwards,
        ///  applicable for the first layer in a network only.
        pub fn backwardPass(
            noalias self: *Self,
            noalias input: *const [dim_out]f32,
            noalias cost_gradient: *[node_count]Vector,
            noalias output: *[dim_in]f32,
        ) void {
            @setFloatMode(.optimized);
            @memset(output, 0);
            for (0..node_count) |j| {
                const gate = &self.gates[j];
                cost_gradient[j][self.max_index[j]] += gate.gradient(
                    &self.inputs[j],
                    self.max_index[j],
                ) * input[j];
                const argument_gradient = gate.diff(&self.inputs[j], self.max_index[j]);
                inline for (self.parents[j], 0..) |parent, k| {
                    output[parent] += argument_gradient[k] * input[j];
                }
            }
        }

        pub fn takeParameters(self: *Self, parameters: *[node_count]Vector) void {
            @setFloatMode(.optimized);
            for (0..node_count) |j| self.gates[j].sigma = logistic(parameters[j]);
        }

        pub fn giveParameters(self: *Self) void {
            _ = self;
        }
    };
}
