const std = @import("std");
const assert = std.debug.assert;

const aesia = @import("../aesia.zig");

const f32x16 = @Vector(16, f32);
const f32x8 = @Vector(8, f32);
const f32x4 = @Vector(4, f32);

pub const LogicOptions = struct {
    rand: *std.Random,
};

pub fn Logic(dim_in_: usize, dim_out_: usize, options: LogicOptions) type {
    return struct {
        const Self = @This();

        pub const info = aesia.layer.Info{
            .dim_in = dim_in_,
            .dim_out = dim_out_,
            .trainable = true,
            .parameter_count = 16 * dim_out_,
            // For efficient SIMD we ensure that the parameters align to a cacheline.
            .parameter_alignment = 64,
        };
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

        pub fn validationEval(
            noalias self: *Self,
            noalias input: *const [dim_in]ItemIn,
            noalias output: *[dim_out]ItemOut,
        ) void {
            @setFloatMode(.optimized);
            assert(self.sigma != null);
            for (self.sigma.?, output, self.parents) |sigma, *activation, parents| {
                const a = input[parents[0]];
                const b = input[parents[1]];
                activation.* = @reduce(.Add, @round(sigma) * gate.vector(a, b));
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

        pub fn backwardPassFinal(
            noalias self: *Self,
            noalias input: *const [dim_out]f32,
            noalias cost_gradient: *[node_count]f32x16,
        ) void {
            @setFloatMode(.optimized);
            for (0..node_count) |j| {
                cost_gradient[j] += self.gradient[j] * @as(f32x16, @splat(input[j]));
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
pub fn PackedLogic(dim_in: usize, dim_out: usize, options: LogicOptions) type {
    return struct {
        const Self = @This();

        // There are 16 possible logic gates, we use a novel packed representation
        // that packs these into 4 parameters.
        const parameter_count = 4 * node_count;

        pub const info = aesia.layer.Info{
            .dim_in = dim_in,
            .dim_out = dim_out,
            .trainable = true,
            // There are 16 possible logic gates, we use a novel packed representation
            // that packs these into 4 parameters.
            .parameter_count = 4 * node_count,
            // For efficient SIMD we ensure that the parameters align to a cacheline.
            .parameter_alignment = 64,
        };

        const node_count = dim_out;
        const ParentIndex = std.math.IntFittingRange(0, dim_in);

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
                return b * (beta[0] - beta[2]) + (1 - b) * (beta[1] - beta[3]);
            }

            // The derivative of the logic gate with respect to its second variable.
            pub fn bDiff(beta: f32x4, a: f32) f32 {
                @setFloatMode(.optimized);
                return a * (beta[0] - beta[1]) + (1 - a) * (beta[2] - beta[3]);
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
            parameters.* = @bitCast(@as([node_count]f32x4, @splat(.{ 4, 4, -4, -4 })));
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
            noalias input: *const [dim_in]f32,
            noalias output: *[dim_out]f32,
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
            noalias input: *const [dim_in]f32,
            noalias output: *[dim_out]f32,
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

        /// Computes the cost gradient and the gradient with respect to the previous layer's
        /// activations.
        pub fn backwardPassFinal(
            noalias self: *Self,
            noalias input: *const [dim_out]f32,
            noalias cost_gradient: *[node_count]f32x4,
        ) void {
            @setFloatMode(.optimized);
            for (0..node_count) |j| {
                const a, const b = self.arguments[j];
                const beta = self.beta[j];
                cost_gradient[j] += packed_gate.gradient(beta, a, b) *
                    @as(f32x4, @splat(input[j]));
            }
        }

        /// Copies parameters as logits and preprocesses them through the logistic function.
        pub fn takeParameters(self: *Self, parameters: *[node_count]f32x4) void {
            @setFloatMode(.optimized);
            for (0..node_count) |j| self.beta[j] = logistic(parameters[j]);
        }

        pub fn giveParameters(_: *Self) void {}
    };
}

pub fn LogicSequential(gate_count: usize) type {
    return struct {
        const Self = @This();

        const parameter_count = gate_count * 4;
        const parameter_alignment = 64;

        pub const info = aesia.layer.Info{
            .dim_in = gate_count * 2,
            .dim_out = gate_count,
            .trainable = true,
            .parameter_count = parameter_count,
            .parameter_alignment = parameter_alignment,
            // Fixme: .in_place = true,
        };

        luts: [gate_count]f32x4,
        input_buffer: [gate_count][2]f32,

        pub fn eval(
            self: *const Self,
            input: *const [gate_count][2]f32,
            output: *[gate_count]f32,
        ) void {
            @setFloatMode(.optimized);
            for (self.luts, output, 0..) |lut, *activation, i| {
                const a = input[i][0];
                const b = input[i][1];
                activation.* = @reduce(.Add, lut * f32x4{
                    a * b,
                    a * (1 - b),
                    (1 - a) * b,
                    (1 - a) * (1 - b),
                });
            }
        }

        pub fn validationEval(
            self: *const Self,
            input: *const [gate_count][2]f32,
            output: *[gate_count]f32,
        ) void {
            @setFloatMode(.optimized);
            for (self.luts, output, 0..) |lut, *activation, i| {
                const a = input[i][0];
                const b = input[i][1];
                activation.* = @reduce(.Add, @round(lut) * f32x4{
                    a * b,
                    a * (1 - b),
                    (1 - a) * b,
                    (1 - a) * (1 - b),
                });
            }
        }

        pub fn init(_: *Self, parameters: *[gate_count]f32x4) void {
            for (0..gate_count) |i| {
                parameters[i] = .{
                    4,
                    4,
                    -4,
                    -4,
                };
            }
        }

        pub fn takeParameters(self: *Self, parameters: *[gate_count]f32x4) void {
            @setFloatMode(.optimized);
            for (parameters, &self.luts) |logit, *expit|
                expit.* = @as(f32x4, @splat(1)) / (@as(f32x4, @splat(1)) + @exp(-logit));
        }

        pub fn giveParameters(_: *Self) void {}

        pub fn forwardPass(
            self: *Self,
            input: *const [gate_count][2]f32,
            output: *[gate_count]f32,
        ) void {
            self.input_buffer = input.*;
            self.eval(input, output);
        }

        pub fn backwardPass(
            self: *const Self,
            activation_delta: *const [gate_count]f32,
            cost_gradient: *[gate_count]f32x4,
            argument_delta: *[gate_count][2]f32,
        ) void {
            @setFloatMode(.optimized);
            argument_delta.* = @splat(@splat(0));
            for (0..gate_count) |i| {
                const a = self.input_buffer[i][0];
                const b = self.input_buffer[i][1];
                const lut = self.luts[i];
                cost_gradient[i] += @as(f32x4, @splat(activation_delta[i])) *
                    lut * (@as(f32x4, @splat(1)) - lut) * f32x4{
                    a * b,
                    a * (1 - b),
                    (1 - a) * b,
                    (1 - a) * (1 - b),
                };
                argument_delta[i][0] = activation_delta[i] * @reduce(
                    .Add,
                    lut * f32x4{ b, 1 - b, -b, -(1 - b) },
                );
                argument_delta[i][1] = activation_delta[i] * @reduce(
                    .Add,
                    lut * f32x4{ a, -a, 1 - a, -(1 - a) },
                );
            }
        }

        pub fn backwardPassFinal(
            self: *const Self,
            activation_delta: *const [gate_count]f32,
            cost_gradient: *[gate_count]f32x4,
        ) void {
            @setFloatMode(.optimized);
            for (0..gate_count) |i| {
                const a = self.input_buffer[i][0];
                const b = self.input_buffer[i][1];
                const lut = self.luts[i];
                cost_gradient[i] += @as(f32x4, @splat(activation_delta[i])) *
                    lut * (@as(f32x4, @splat(1)) - lut) * f32x4{
                    a * b,
                    a * (1 - b),
                    (1 - a) * b,
                    (1 - a) * (1 - b),
                };
            }
        }
    };
}

pub const LUTConvolutionOptions = struct {
    depth: usize,
    height: usize,
    width: usize,
    lut_count: usize,
    field_size: struct { height: usize, width: usize },
    stride: struct { row: usize, col: usize },
};

pub fn LUTConvolution(options: LUTConvolutionOptions) type {
    return struct {
        const Self = @This();

        // Unpack options.
        const depth_in = options.depth;
        const height_in = options.height;
        const width_in = options.width;
        const lut_count = options.lut_count;
        const field_size = options.field_size;
        const stride = options.stride;

        const Ply = LUTConvolutionPly(.{
            .height = height_in,
            .width = width_in,
            .lut_count = lut_count,
            .field_size = .{ .height = field_size.height, .width = field_size.width },
            .stride = .{ .row = stride.row, .col = stride.col },
        });
        const depth_out = depth_in * lut_count;
        const height_out = Ply.height_out;
        const width_out = Ply.width_out;

        pub const info = aesia.layer.Info{
            .dim_in = depth_in * height_in * width_in,
            .dim_out = depth_out * height_out * width_out,
            .parameter_count = depth_in * Ply.info.parameter_count.?,
            .parameter_alignment = Ply.info.parameter_alignment.?,
            .trainable = true,
        };

        plies: [depth_in]Ply,

        pub fn eval(
            self: *Self,
            input: *const [depth_in][height_in][width_in]f32,
            output: *[depth_in][lut_count][height_out][width_out]f32,
        ) void {
            @setFloatMode(.optimized);
            for (0..depth_in) |ply| {
                self.plies[ply].eval(&input[ply], &output[ply]);
            }
        }

        pub fn forwardPass(
            self: *Self,
            input: *const [depth_in][height_in][width_in]f32,
            output: *[depth_in][lut_count][height_out][width_out]f32,
        ) void {
            @setFloatMode(.optimized);
            return self.eval(input, output);
        }

        pub fn backwardPass(
            self: *Self,
            activation_delta: *const [depth_in][lut_count][height_out][width_out]f32,
            cost_gradient: *[depth_in][Ply.lut_parameter_count][Ply.lut_count]f32,
            argument_delta: *[depth_in][height_in][width_in]f32,
        ) void {
            @setFloatMode(.optimized);
            for (0..depth_in) |ply| {
                self.plies[ply].backwardPass(
                    &activation_delta[ply],
                    &cost_gradient[ply],
                    &argument_delta[ply],
                );
            }
        }

        pub fn backwardPassFinal(
            self: *Self,
            activation_delta: *const [depth_in][lut_count][height_out][width_out]f32,
            cost_gradient: *[depth_in][Ply.lut_parameter_count][Ply.lut_count]f32,
        ) void {
            @setFloatMode(.optimized);
            for (0..depth_in) |ply| {
                self.plies[ply].backwardPassFinal(
                    &activation_delta[ply],
                    &cost_gradient[ply],
                );
            }
        }

        pub fn init(
            self: *Self,
            parameters: *[depth_in][Ply.lut_parameter_count][Ply.lut_count]f32,
        ) void {
            for (0..depth_in) |ply| self.plies[ply].init(&parameters[ply]);
        }

        pub fn takeParameters(
            self: *Self,
            parameters: *[depth_in][Ply.lut_parameter_count][Ply.lut_count]f32,
        ) void {
            for (0..depth_in) |ply| self.plies[ply].takeParameters(&parameters[ply]);
        }

        pub fn giveParameters(_: *Self) void {}
    };
}

const LUTConvolutionPlyOptions = struct {
    height: usize,
    width: usize,
    lut_count: usize,
    field_size: struct { height: usize, width: usize },
    stride: struct { row: usize, col: usize },
};

fn LUTConvolutionPly(options: LUTConvolutionPlyOptions) type {
    return struct {
        const Self = @This();

        // Unpack options.
        const stride = options.stride;
        const height_in = options.height;
        const width_in = options.width;
        const lut_count = options.lut_count;
        const field_height = options.field_size.height;
        const field_width = options.field_size.width;
        const lut_arity = field_height * field_width;
        pub const lut_parameter_count = 1 << lut_arity;
        comptime {
            assert(field_height > 0);
            assert(field_width > 0);
            assert(lut_count > 0);
            assert(height_in >= field_height);
            assert(width_in >= field_width);
            assert((height_in - field_height) % stride.row == 0);
            assert((width_in - field_width) % stride.col == 0);
        }
        const height_out = (height_in - field_height) / stride.row + 1;
        const width_out = (width_in - field_width) / stride.col + 1;

        const ArgumentIndex = std.math.IntFittingRange(0, lut_arity);
        const ExpitIndex = std.math.IntFittingRange(0, lut_parameter_count - 1);

        pub const info = aesia.layer.Info{
            .dim_in = height_in * width_in,
            .dim_out = height_out * width_out * lut_count,
            .trainable = true,
            .parameter_count = lut_count * lut_parameter_count,
            .parameter_alignment = 64,
        };

        // The lookup tables are stored in column major order, this is because we evaluate
        // all lookup tables given a certain input.
        expits: [lut_parameter_count][lut_count]f32,
        activations: [height_in][width_in][lut_arity * (lut_arity - 1) / 2 + lut_arity + 1]ExpitIndex,
        receptions: [height_out][width_out][lut_arity]f32,

        pub fn init(_: *Self, parameters: *[lut_parameter_count][lut_count]f32) void {
            for (0..lut_parameter_count / 2) |i| {
                for (0..lut_count) |k| {
                    parameters[i][k] = 1;
                }
            }
            for (lut_parameter_count / 2..lut_parameter_count) |i| {
                for (0..lut_count) |k| {
                    parameters[i][k] = 0;
                }
            }
        }

        pub fn takeParameters(self: *Self, parameters: *[lut_parameter_count][lut_count]f32) void {
            for (0..lut_parameter_count) |i| {
                for (0..lut_count) |k| {
                    self.expits[i][k] = 1 / (1 + @exp(-parameters[i][k]));
                }
            }
        }

        pub fn giveParameters(_: *Self) void {}

        const Point = @Vector(2, usize);
        const receptive_offsets = blk: {
            var offsets: [lut_arity]Point = undefined;
            var i: usize = 0;
            for (0..field_height) |row| {
                for (0..field_width) |col| {
                    offsets[i] = Point{ row, col };
                    i += 1;
                }
            }
            break :blk offsets;
        };

        fn findActivations(
            self: *Self,
            row: usize,
            col: usize,
        ) void {
            var max_index: ExpitIndex = 0;
            inline for (0..lut_arity) |j| {
                const reception = &self.receptions[row][col];
                max_index |= @as(ExpitIndex, @intFromFloat(@round(reception[j]))) << j;
            }
            var activation_index: usize = 0;
            self.activations[row][col][activation_index] = max_index;
            activation_index += 1;
            inline for (0..lut_arity) |j| {
                self.activations[row][col][activation_index] = max_index ^ (1 << j);
                activation_index += 1;
            }
            inline for (0..lut_arity) |j| {
                inline for (j + 1..lut_arity) |k| {
                    self.activations[row][col][activation_index] =
                        max_index ^ ((1 << j) | (1 << k));
                    activation_index += 1;
                }
            }
        }

        fn evalLUTs(
            self: *Self,
            row: usize,
            col: usize,
            output: *[lut_count][height_out][width_out]f32,
        ) void {
            @setEvalBranchQuota(1000 * lut_arity * lut_count * lut_count);
            inline for (self.activations[row][col]) |index| {
                // inline for (0..1 << lut_arity) |index| {
                var product: f32 = 1;
                inline for (self.receptions[row][col], 0..) |x, bit| {
                    product *= if ((index >> bit) & 1 != 0) x else 1 - x;
                }
                inline for (0..lut_count) |k| {
                    output[k][row][col] += self.expits[index][k] * product;
                }
            }
        }

        fn backwardPassFinalLUTs(
            self: *Self,
            row: usize,
            col: usize,
            activation_delta: *const [lut_count][height_out][width_out]f32,
            cost_gradient: *[lut_parameter_count][lut_count]f32,
        ) void {
            const reception = &self.receptions[row][col];
            const activation = &self.activations[row][col];
            @setEvalBranchQuota(100 * lut_count * lut_arity * lut_arity);
            inline for (activation) |index| {
                // for (0..1 << lut_arity) |index| {
                var product: f32 = 1;
                inline for (reception, 0..) |x, bit| {
                    product *= if ((index >> bit) & 1 != 0) x else 1 - x;
                }
                inline for (0..lut_count) |k| {
                    const expit = self.expits[index][k];
                    cost_gradient[index][k] += activation_delta[k][row][col] * product * expit * (1 - expit);
                }
            }
        }

        fn backwardPassLUTs(
            self: *Self,
            row: usize,
            col: usize,
            activation_delta: *const [lut_count][height_out][width_out]f32,
            cost_gradient: *[lut_parameter_count][lut_count]f32,
            argument_delta: *[lut_arity]f32,
        ) void {
            @memset(argument_delta, 0);
            const reception = &self.receptions[row][col];
            const activation = &self.activations[row][col];
            @setEvalBranchQuota(100 * lut_count * lut_arity * lut_arity);
            inline for (activation) |index| {
                // inline for (0..1 << lut_arity) |index| {
                var product: f32 = 1;
                inline for (reception, 0..) |x, bit| {
                    product *= if ((index >> bit) & 1 != 0) x else 1 - x;
                }
                inline for (0..lut_count) |k| {
                    const expit = self.expits[index][k];
                    cost_gradient[index][k] += activation_delta[k][row][col] * product * expit * (1 - expit);
                }
            }
            @setEvalBranchQuota(1000 * lut_count * lut_arity * lut_arity);
            inline for (0..lut_arity) |j| {
                inline for (activation) |index| {
                    // inline for (0..1 << lut_arity) |index| {
                    var product: f32 = 1;
                    inline for (reception, 0..) |x, bit| {
                        if (comptime bit == j) {
                            product *= if ((index >> bit) & 1 != 0) 1 else -1;
                        } else {
                            product *= if ((index >> bit) & 1 != 0) x else 1 - x;
                        }
                    }
                    inline for (0..lut_count) |k| {
                        const expit = self.expits[index][k];
                        argument_delta[j] += activation_delta[k][row][col] * product * expit;
                    }
                }
            }
        }

        pub fn eval(
            self: *Self,
            input: *const [height_in][width_in]f32,
            output: *[lut_count][height_out][width_out]f32,
        ) void {
            @setFloatMode(.optimized);
            // Collect receptions and find the activated indices.
            output.* = @splat(@splat(@splat(0)));
            for (0..height_out) |row| {
                for (0..width_out) |col| {
                    const receptions = &self.receptions[row][col];
                    const top_left = Point{ row * stride.row, col * stride.col };
                    inline for (receptions, receptive_offsets) |*reception, offset| {
                        const point = top_left + offset;
                        reception.* = input[point[0]][point[1]];
                    }
                    self.findActivations(row, col);
                }
            }
            for (0..height_out) |row| {
                for (0..width_out) |col| {
                    self.evalLUTs(row, col, output);
                }
            }
        }

        pub fn forwardPass(
            self: *Self,
            input: *const [height_in][width_in]f32,
            output: *[height_out][width_out][lut_count]f32,
        ) void {
            return self.eval(input, output);
        }

        pub fn backwardPassFinal(
            self: *Self,
            activation_delta: *const [lut_count][height_out][width_out]f32,
            cost_gradient: *[lut_parameter_count][lut_count]f32,
        ) void {
            for (0..height_out) |row| {
                for (0..width_out) |col| {
                    self.backwardPassFinalLUTs(
                        row,
                        col,
                        activation_delta,
                        cost_gradient,
                    );
                }
            }
        }

        pub fn backwardPass(
            self: *Self,
            activation_delta: *const [lut_count][height_out][width_out]f32,
            cost_gradient: *[lut_parameter_count][lut_count]f32,
            argument_delta: *[height_in][width_in]f32,
        ) void {
            var argument_delta_buffer: [lut_arity]f32 = undefined;
            for (0..height_out) |row| {
                for (0..width_out) |col| {
                    self.backwardPassLUTs(
                        row,
                        col,
                        activation_delta,
                        cost_gradient,
                        &argument_delta_buffer,
                    );
                    const top_left = Point{ row * stride.row, col * stride.col };
                    inline for (receptive_offsets, argument_delta_buffer) |offset, delta| {
                        const point = top_left + offset;
                        argument_delta[point[0]][point[1]] += delta;
                    }
                }
            }
        }
    };
}

fn MultiLogicGate(arity: usize) type {
    return struct {
        const Self = @This();

        const tolerance = 0.25;

        const beta_len = 1 << arity;
        const Beta = @Vector(beta_len, f32);
        const ArgumentVector = @Vector(arity, f32);
        const ParameterIndex = std.math.IntFittingRange(0, beta_len - 1);
        const ArgumentIndex = std.math.IntFittingRange(0, arity);

        beta: Beta,

        // Evaluates the packed differentiable logic gate.
        pub fn eval(gate: *const Self, input: *const [arity]f32) f32 {
            @setFloatMode(.optimized);
            @setEvalBranchQuota(4 * beta_len * arity * arity);
            var result: f32 = 0;
            @setEvalBranchQuota(4 * beta_len * arity * arity);
            inline for (0..beta_len) |i| {
                var product = gate.beta[i];
                inline for (input, 0..) |x, bit| {
                    product *= if (comptime (1 << bit) & i != 0) x else (1 - x);
                }
                result += product;
            }
            return result;
        }

        pub fn argumentGradient(gate: *const Self, input: *const [arity]f32) ArgumentVector {
            @setFloatMode(.optimized);
            var result: ArgumentVector = @splat(0);
            @setEvalBranchQuota(4 * beta_len * arity * arity);
            inline for (0..arity) |j| {
                inline for (0..beta_len) |i| {
                    var product = gate.beta[i];
                    inline for (input, 0..) |x, bit| {
                        if (comptime bit != j) {
                            product *= if (comptime (1 << bit) & i != 0) x else (1 - x);
                        } else {
                            product *= if (comptime (1 << bit) & i != 0) 1 else -1;
                        }
                    }
                    result[j] += product;
                }
            }
            @setFloatMode(.optimized);
            return result;
        }

        pub fn parameterGradient(
            gate: *const Self,
            input: *const [arity]f32,
        ) Beta {
            @setFloatMode(.optimized);
            @setEvalBranchQuota(4 * beta_len * arity * arity);
            @setFloatMode(.optimized);
            var result: Beta = @splat(0);
            @setEvalBranchQuota(4 * beta_len * arity * arity);
            inline for (0..beta_len) |i| {
                var product = gate.beta[i] * (1 - gate.beta[i]);
                inline for (input, 0..) |x, bit| {
                    product *= if (comptime (1 << bit) & i != 0) x else (1 - x);
                }
                result[i] = product;
            }
            @setFloatMode(.optimized);
            return result;
        }
    };
}

pub const LUTOptions = struct {
    arity: usize,
    rand: *std.Random,
};

pub fn LUT(dim_in: usize, dim_out: usize, options: LUTOptions) type {
    return struct {
        const Self = @This();

        const parameter_count = parameter_vector_len * node_count;
        // For efficient SIMD we ensure that the parameters align to a cacheline.
        const parameter_alignment: usize = @max(64, @alignOf(ParameterVector));

        pub const info = aesia.layer.Info{
            .dim_in = dim_in,
            .dim_out = dim_out,
            .trainable = true,
            .parameter_count = parameter_count,
            .parameter_alignment = parameter_alignment,
        };

        const node_count = dim_out;
        const ParentIndex = std.math.IntFittingRange(0, dim_in);
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
                inline for (0..arity) |j| {
                    inline for (0..parameter_vector_len) |i| {
                        var product: f32 = gate.sigma[i];
                        inline for (input, 0..) |x, bit| {
                            if (comptime bit != j) {
                                product *= if (comptime (1 << bit) & i != 0) x else (1 - x);
                            } else {
                                product *= if (comptime (1 << bit) & i != 0) 1 else -1;
                            }
                        }
                        result[j] += product;
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
        parents: [node_count][arity]ParentIndex,
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
            noalias input: *const [dim_in]f32,
            noalias output: *[dim_out]f32,
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
            noalias input: *const [dim_in]f32,
            noalias output: *[dim_out]f32,
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

        pub fn backwardPassFinal(
            noalias self: *Self,
            noalias input: *const [dim_out]f32,
            noalias cost_gradient: *[node_count]ParameterVector,
        ) void {
            @setFloatMode(.optimized);
            for (0..node_count) |j| {
                const gate = &self.gates[j];
                cost_gradient[j] += gate.gradient(&self.inputs[j]) *
                    @as(ParameterVector, @splat(input[j]));
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

pub fn ExclusiveOr(dim: usize) type {
    return struct {
        const Self = @This();

        const parameter_count = dim;
        const parameter_alignment = 64;

        pub const info = aesia.layer.Info{
            .dim_in = dim,
            .dim_out = dim,
            .trainable = true,
            .parameter_count = parameter_count,
            .parameter_alignment = parameter_alignment,
            .regularize = true,
        };

        beta: [dim]f32,
        input_buffer: [dim]f32,

        pub fn regularizeError(self: *Self) f32 {
            var result: f32 = 0;
            for (0..dim) |i| {
                const x = self.beta[i];
                result += x * (1 - x);
            }
            return result;
        }

        pub fn regularize(_: *Self, _: *[dim]f32) void {}

        pub fn eval(
            self: *const Self,
            input: *const [dim]f32,
            output: *[dim]f32,
        ) void {
            @setFloatMode(.optimized);
            for (0..dim) |i| {
                output[i] = input[i] * self.beta[i];
            }
        }

        pub fn init(_: *Self, parameters: *[dim]f32) void {
            for (parameters) |*parameter| parameter.* = -4;
        }

        pub fn takeParameters(self: *Self, parameters: *[dim]f32) void {
            @setFloatMode(.optimized);
            for (parameters, &self.beta) |logit, *expit| expit.* = 1 / (1 + @exp(-logit));
        }

        pub fn giveParameters(_: *Self) void {}

        pub fn forwardPass(
            self: *Self,
            input: *const [dim]f32,
            output: *[dim]f32,
        ) void {
            self.input_buffer = input.*;
            self.eval(input, output);
        }

        pub fn backwardPass(
            self: *const Self,
            activation_delta: *const [dim]f32,
            cost_gradient: *[dim]f32,
            argument_delta: *[dim]f32,
        ) void {
            @setFloatMode(.optimized);
            for (0..dim) |i| {
                const x = self.input_buffer[i];
                const beta = self.beta[i];
                cost_gradient[i] += activation_delta[i] * x * beta * (1 - beta);
                argument_delta[i] = activation_delta[i] * beta;
            }
        }

        pub fn backwardPassFinal(
            self: *const Self,
            activation_delta: *const [dim]f32,
            cost_gradient: *[dim]f32,
        ) void {
            @setFloatMode(.optimized);
            for (0..dim) |i| {
                const x = self.input_buffer[i];
                const beta = self.beta[i];
                cost_gradient[i] += activation_delta[i] * x * beta * (1 - beta);
            }
        }
    };
}
