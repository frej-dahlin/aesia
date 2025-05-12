const std = @import("std");
const assert = std.debug.assert;

const f32x16 = @Vector(16, f32);
const f32x8 = @Vector(8, f32);
const f32x4 = @Vector(4, f32);

pub const LogicOptions = struct {
    rand: *std.Random,
};

pub fn Logic(input_dim_: usize, output_dim_: usize, options: LogicOptions) type {
    return struct {
        const Self = @This();
        pub const input_dim = input_dim_;
        pub const output_dim = output_dim_;
        pub const Input = [input_dim]f32;
        pub const Output = [output_dim]f32;
        const node_count = output_dim;
        const ParentIndex = std.math.IntFittingRange(0, input_dim - 1);
        // There are 16 possible logic gates, each one is assigned a probability logit.
        pub const parameter_count = 16 * node_count;
        // For efficient SIMD we ensure that the parameters align to a cacheline.
        pub const parameter_alignment: usize = 64;

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
            parameters.* = @bitCast(@as([node_count]f32x16, @splat(.{ 0, 0, 0, 1, 0, 1 } ++ .{0} ** 10)));
            self.* = .{
                .sigma = null,
                .gradient = undefined,
                .diff = undefined,
                .parents = undefined,
            };
            for (0..node_count) |j| {
                self.parents[j] = .{
                    options.rand.intRangeLessThan(ParentIndex, 0, input_dim),
                    options.rand.intRangeLessThan(ParentIndex, 0, input_dim),
                };
            }
        }

        /// Logits are normalized such that max(logits) == 0.
        /// When the logits grow to big exp^logit will explode.
        fn maxNormalize(v: f32x16) f32x16 {
            return v - @as(f32x16, @splat(@reduce(.Max, v)));
        }

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

        pub fn eval(noalias self: *Self, noalias input: *const Input, noalias output: *Output) void {
            @setFloatMode(.optimized);
            assert(self.sigma != null);
            for (self.sigma.?, output, self.parents) |sigma, *activation, parents| {
                const a = input[parents[0]];
                const b = input[parents[1]];
                activation.* = @reduce(.Add, sigma * gate.vector(a, b));
            }
        }

        pub fn forwardPass(noalias self: *Self, noalias input: *const Input, noalias output: *Output) void {
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

        /// Fixme: Create a function "backward" that does not pass delta backwards, applicable for
        /// the first layer in a network only.
        pub fn backwardPass(noalias self: *Self, noalias input: *const [output_dim]f32, noalias cost_gradient: *[node_count]f32x16, noalias output: *[input_dim]f32) void {
            @setFloatMode(.optimized);
            @memset(output, 0);
            for (0..node_count) |j| {
                cost_gradient[j] += self.gradient[j] * @as(f32x16, @splat(input[j]));
                output[self.parents[j][0]] += self.diff[j][0] * input[j];
                output[self.parents[j][1]] += self.diff[j][1] * input[j];
            }
        }

        pub fn takeParameters(self: *Self, parameters: *[node_count]f32x16) void {
            @setFloatMode(.optimized);
            assert(self.sigma == null);
            self.sigma = parameters;
            for (self.sigma.?, parameters) |*sigma, logit| sigma.* = softmax2(logit);
        }

        pub fn giveParameters(self: *Self) void {
            assert(self.sigma != null);
            for (self.sigma.?) |*sigma| sigma.* = softmax2Inverse(sigma.*);
            self.sigma = null;
        }

        pub fn borrowParameters(self: *Self, parameters: *[node_count]f32x16) void {
            assert(self.sigma == null);
            self.sigma = parameters;
        }

        pub fn returnParameters(self: *Self) void {
            assert(self.sigma != null);
            self.sigma = null;
        }
    };
}

/// A faster and leaner differentiable logic gate.
pub fn PackedLogic(input_dim_: usize, output_dim_: usize, options: LogicOptions) type {
    return struct {
        const Self = @This();
        pub const input_dim = input_dim_;
        pub const output_dim = output_dim_;
        pub const Input = [input_dim]f32;
        pub const Output = [output_dim]f32;
        const node_count = output_dim;
        const ParentIndex = std.math.IntFittingRange(0, input_dim - 1);
        // There are 16 possible logic gates, we use a novel packed representation.
        pub const parameter_count = 4 * node_count;
        // For efficient SIMD we ensure that the parameters align to a cacheline.
        pub const parameter_alignment: usize = 64;

        const packed_gate = struct {
            // Evaluates the packed differentiable logic gate.
            pub fn eval(sigma: f32x4, a: f32, b: f32) f32 {
                @setFloatMode(.optimized);
                const x = a * b * sigma[0];
                const y = a * (1 - b) * sigma[1];
                const z = (1 - a) * b * sigma[2];
                const w = (1 - a) * (1 - b) * sigma[3];
                return x + y + z + w;
            }

            // The derivative of the logic gate with respect to its first variable.
            pub fn aDiff(sigma: f32x4, b: f32) f32 {
                @setFloatMode(.optimized);
                const x = b * sigma[0];
                const y = (1 - b) * sigma[1];
                const z = -b * sigma[2];
                const w = -(1 - b) * sigma[3];
                return x + y + z + w; // -
            }

            // The derivative of the logic gate with respect to its second variable.
            pub fn bDiff(sigma: f32x4, a: f32) f32 {
                @setFloatMode(.optimized);
                const x = a * sigma[0];
                const y = -a * sigma[1];
                const z = (1 - a) * sigma[2];
                const w = -(1 - a) * sigma[3];
                return x + y + z + w;
            }

            // The gradient of the logic gate with respect to its underlying logits.
            pub fn gradient(sigma: f32x4, a: f32, b: f32) f32x4 {
                @setFloatMode(.optimized);
                const x = a * b * sigma[0];
                const y = a * (1 - b) * sigma[1];
                const z = (1 - a) * b * sigma[2];
                const w = (1 - a) * (1 - b) * sigma[3];
                return .{
                    (1 - sigma[0]) * x,
                    (1 - sigma[1]) * y,
                    (1 - sigma[2]) * z,
                    (1 - sigma[3]) * w,
                };
            }
        };

        /// The preprocessed parameters, computed by softmax(parameters).
        sigma: [node_count]f32x4 align(64),
        inputs: [node_count][2]f32,
        parents: [node_count][2]ParentIndex,

        fn logistic(x: f32x4) f32x4 {
            @setFloatMode(.optimized);
            return @as(f32x4, @splat(1)) / (@as(f32x4, @splat(1)) + @exp(-x));
        }

        pub fn init(self: *Self, parameters: *[parameter_count]f32) void {
            parameters.* = @bitCast(@as([node_count]f32x4, @splat(.{ 4, 4, 0, 0 })));
            self.* = .{
                .sigma = undefined,
                .inputs = undefined,
                .parents = undefined,
            };
            for (0..node_count) |j| {
                self.parents[j] = .{
                    options.rand.intRangeLessThan(ParentIndex, 0, input_dim),
                    options.rand.intRangeLessThan(ParentIndex, 0, input_dim),
                };
            }
        }

        pub fn eval(
            noalias self: *Self,
            noalias input: *const Input,
            noalias output: *Output,
        ) void {
            @setFloatMode(.optimized);
            for (self.sigma, output, 0..) |sigma, *activation, j| {
                const a = input[self.parents[j][0]];
                const b = input[self.parents[j][1]];
                activation.* = packed_gate.eval(sigma, a, b);
            }
        }

        pub fn forwardPass(
            noalias self: *Self,
            noalias input: *const Input,
            noalias output: *Output,
        ) void {
            @setFloatMode(.optimized);
            // For some reason it is faster to to a simple loop rather than a multi item one.
            // I found needless memcpy calls with callgrind.
            // It is also faster to split the loop into two parts, my guess is that
            // the first one trashes the cache and the bottom one is vectorized by the compiler.
            for (0..node_count) |j| {
                self.inputs[j] = .{
                    input[self.parents[j][0]],
                    input[self.parents[j][1]],
                };
            }
            for (0..node_count) |j| {
                const a, const b = self.inputs[j];
                output[j] = packed_gate.eval(self.sigma[j], a, b);
            }
        }

        /// Fixme: Create a function "backwardPassFirst" that does not pass delta backwards,
        ///  applicable for the first layer in a network only.
        pub fn backwardPass(
            noalias self: *Self,
            noalias input: *const [output_dim]f32,
            noalias cost_gradient: *[node_count]f32x4,
            noalias output: *[input_dim]f32,
        ) void {
            @setFloatMode(.optimized);
            @memset(output, 0);
            for (0..node_count) |j| {
                const a, const b = self.inputs[j];
                const sigma = self.sigma[j];
                cost_gradient[j] += packed_gate.gradient(sigma, a, b) * @as(f32x4, @splat(input[j]));
                output[self.parents[j][0]] += packed_gate.aDiff(sigma, b) * input[j];
                output[self.parents[j][1]] += packed_gate.bDiff(sigma, a) * input[j];
            }
        }

        pub fn takeParameters(self: *Self, parameters: *[node_count]f32x4) void {
            @setFloatMode(.optimized);
            for (0..node_count) |j| self.sigma[j] = logistic(parameters[j]);
        }

        pub fn giveParameters(self: *Self) void {
            _ = self;
        }

        pub fn borrowParameters(self: *Self, parameters: *[node_count]f32x16) void {
            _ = self;
            _ = parameters;
        }

        pub fn returnParameters(self: *Self) void {
            _ = self;
        }
    };
}

/// Divides the input into output_dim buckets, each output is the sequential sum of
/// input_dim / output_dim items of the input.
pub fn GroupSum(input_dim_: usize, output_dim_: usize) type {
    return struct {
        const Self = @This();
        pub const input_dim = input_dim_;
        pub const output_dim = output_dim_;
        pub const Input = [input_dim]f32;
        pub const Output = [output_dim]f32;

        const quot = input_dim / output_dim;
        const scale: f32 = 1.0 / (@as(comptime_float, @floatFromInt(output_dim)));

        pub fn eval(input: *const Input, output: *Output) void {
            @memset(output, 0);
            for (output, 0..) |*coord, k| {
                const from = k * quot;
                const to = from + quot;
                for (input[from..to]) |softbit| coord.* += softbit;
                coord.* *= scale;
            }
        }

        pub fn forwardPass(input: *const Input, output: *Output) void {
            return eval(input, output);
        }

        pub fn backwardPass(input: *const [output_dim]f32, output: *[input_dim]f32) void {
            for (input, 0..) |child, k| {
                const from = k * quot;
                const to = from + quot;
                for (output[from..to]) |*parent| parent.* = child * scale;
            }
        }
    };
}

pub const MultiLogicOptions = struct {
    arity: usize,
    rand: *std.Random,
};

pub fn MultiLogicGate(input_dim_: usize, output_dim_: usize, options: MultiLogicOptions) type {
    return struct {
        const Self = @This();
        pub const input_dim = input_dim_;
        pub const output_dim = output_dim_;
        pub const Input = [input_dim]f32;
        pub const Output = [output_dim]f32;

        const node_count = output_dim;
        const ParentIndex = std.math.IntFittingRange(0, input_dim - 1);
        const arity = options.arity;
        const vector_len = 1 << arity;
        const Vector = @Vector(vector_len, f32);

        pub const parameter_count = vector_len * node_count;
        // For efficient SIMD we ensure that the parameters align to a cacheline.
        pub const parameter_alignment: usize = 64;

        const Gate = struct {
            sigma: Vector,
            // Evaluates the packed differentiable logic gate.
            pub fn eval(gate: *const Gate, input: *const [arity]f32) f32 {
                @setFloatMode(.optimized);
                var result: f32 = 0;
                inline for (0..vector_len) |i| {
                    var product: f32 = gate.sigma[i];
                    inline for (input, 0..) |x, bit| {
                        product *= if ((1 << bit) & i != 0) x else (1 - x);
                    }
                    result += product;
                }
                return result;
            }

            pub fn diff(gate: *const Gate, input: *const [arity]f32, j: usize) f32 {
                @setFloatMode(.optimized);
                var result: f32 = 0;
                inline for (0..vector_len) |i| {
                    var product: f32 = gate.sigma[i];
                    inline for (input, 0..) |x, bit| {
                        if (bit != j) {
                            product *= if ((1 << bit) & i != 0) x else (1 - x);
                        } else {
                            product *= if ((1 << bit) & i != 0) 1 else -1;
                        }
                    }
                    result += product;
                }
                return result;
            }

            pub fn gradient(
                gate: *const Gate,
                input: *const [arity]f32,
            ) Vector {
                @setFloatMode(.optimized);
                var result: Vector = undefined;
                inline for (0..vector_len) |j| {
                    var product: f32 = (1 - gate.sigma[j]) * gate.sigma[j];
                    inline for (input, 0..) |x, bit| {
                        product *= if ((1 << bit) & j != 0) x else (1 - x);
                    }
                    result[j] = product;
                }
                return result;
            }
        };

        gates: [node_count]Gate align(64),
        inputs: [node_count][arity]f32,
        parents: [node_count][arity]std.math.IntFittingRange(0, input_dim - 1),

        fn logistic(x: Vector) Vector {
            @setFloatMode(.optimized);
            return @as(Vector, @splat(1)) / (@as(Vector, @splat(1)) + @exp(-x));
        }

        pub fn init(self: *Self, parameters: *[parameter_count]f32) void {
            parameters.* = @bitCast(@as(
                [node_count]Vector,
                @splat(.{4} ** (vector_len / 2) ++ .{0} ** (vector_len / 2)),
            ));
            self.* = .{
                .gates = undefined,
                .inputs = undefined,
                .parents = undefined,
            };
            for (0..node_count) |j| {
                inline for (&self.parents[j]) |*parent| {
                    parent.* = options.rand.intRangeLessThan(ParentIndex, 0, input_dim);
                }
            }
        }

        pub fn eval(
            noalias self: *Self,
            noalias input: *const Input,
            noalias output: *Output,
        ) void {
            @setFloatMode(.optimized);
            var arguments: [arity]f32 = undefined;
            for (self.gates, output, 0..) |gate, *activation, j| {
                inline for (&arguments, self.parents[j]) |*argument, parent| {
                    argument.* = input[parent];
                }
                activation.* = gate.eval(&arguments);
            }
        }

        pub fn forwardPass(
            noalias self: *Self,
            noalias input: *const Input,
            noalias output: *Output,
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
            noalias input: *const [output_dim]f32,
            noalias cost_gradient: *[node_count]Vector,
            noalias output: *[input_dim]f32,
        ) void {
            @setFloatMode(.optimized);
            @memset(output, 0);
            for (0..node_count) |j| {
                const gate = &self.gates[j];
                cost_gradient[j] += gate.gradient(&self.inputs[j]) * @as(Vector, @splat(input[j]));
                inline for (self.parents[j], 0..) |parent, k| {
                    output[parent] += gate.diff(&self.inputs[j], k) * input[j];
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

        pub fn borrowParameters(self: *Self, parameters: *[node_count]f32x16) void {
            _ = self;
            _ = parameters;
        }

        pub fn returnParameters(self: *Self) void {
            _ = self;
        }
    };
}
