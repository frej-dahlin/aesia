const std = @import("std");
const assert = std.debug.assert;

const f32x16 = @Vector(16, f32);
const f32x8 = @Vector(8, f32);
const f32x4 = @Vector(4, f32);
const f32x2 = @Vector(2, f32);

/// A layer is a type interface, it needs to declare:
///     Input      : input type
///     Output     : output type
///     input_dim  : the dimension of the input
///     output_dim : the dimension of the output
/// Optionally, the layer can make use of parameters, which have to be of type f32.
/// Therefore every layer must declare
///     parameter_count : the number of f32 parameters the layer will be allocated
/// if the parameter_count > 0 then, because some layers will utilize SIMD, the layer must declare
///     parameter_alignment : the alignment of the layer's parameters
/// as well as the methods:
///     takeParameters   : take ownership of the parameters, preprocessing them, if necessary
///     giveParameters   : give back ownership of the parameters, postprocessing them, if necessary
///     borrowParameters : store the pointer to the parameters, this is called by worker threads after
///                        the main thread has called takeParameters
///     returnParameters : release the pointer to the paramters, this is called by worked threads after
///                        the main thread has called giveParameters
///     backwardPassLast : backwardPass without passing the delta backwards, see below
/// Every layer must declare the following methods:
///     eval
///     forwardPass
///     backwardPass
/// Linear Congruential Multiplier modulo 2^32. Used for fast pseudo random numbers.
/// It is likely that the ability to invert the number generator will be important to 'efficiently' update
// a logic gate network.
pub fn LCG32(multiplier: u32, increment: u32, initial_seed: u32) type {
    return struct {
        const Self = @This();
        seed: u32,

        pub const default = Self{ .seed = initial_seed };

        pub fn next(self: *Self) u32 {
            self.seed *%= multiplier;
            self.seed +%= increment;
            return self.seed;
        }

        pub fn reset(self: *Self) void {
            self.seed = initial_seed;
        }

        pub fn init(seed_: u32) Self {
            return Self{
                .seed = seed_,
            };
        }
    };
}

pub const LogicOptions = struct {
    seed: u64,
};

pub fn Logic(input_dim_: usize, output_dim_: usize, options: LogicOptions) type {
    return struct {
        const Self = @This();
        pub const input_dim = input_dim_;
        pub const output_dim = output_dim_;
        pub const Input = [input_dim]f32;
        pub const Output = [output_dim]f32;
        const node_count = output_dim;
        // There are 16 possible logic gates, we use a novel compressed differentiable representation.
        pub const parameter_count = 4 * node_count;
        // For efficient SIMD we ensure that the parameters align to a cacheline.
        pub const parameter_alignment = 64;

        const CDLG = struct {
            a: f32,
            b: f32,
            mix: f32,
            neg: f32,

            // Evaluates the compressed differentiable logic gate.
            pub fn eval(sigma: CDLG, a: f32, b: f32) f32 {
                const mix = a * b;
                const half: f32 = sigma.a * (a - mix) + sigma.b * (b - mix) + sigma.mix * mix;
                return (1 - sigma.neg) * half + sigma.neg * (1 - half);
            }

            // The derivative of eval with respect to the first variable.
            pub fn aDiff(sigma: CDLG, b: f32) f32 {
                const half = sigma.a * (1 - b) + sigma.b * (-b) + sigma.mix * b;
                return (1 - 2 * sigma.neg) * half;
            }

            // The derivative of eval with respect to the second variable.
            pub fn bDiff(sigma: CDLG, a: f32) f32 {
                const half = sigma.a * (-a) + sigma.b * (1 - a) + sigma.mix * a;
                return (1 - 2 * sigma.neg) * half;
            }

            // The gradient of eval with respect to each respective probabilities logits.
            pub fn gradient(sigma: CDLG, a: f32, b: f32) f32x4 {
                const mix = a * b;
                const value = sigma.eval(a, b);
                return .{
                    -sigma.a * (1 - sigma.a) * (a - mix),
                    -sigma.b * (1 - sigma.b) * (b - mix),
                    -sigma.mix * (1 - sigma.mix) * mix,
                    -sigma.neg * (1 - sigma.neg) * (1 - 2 * value),
                };
            }
        };

        gradient: [node_count]f32x4 align(64),
        diff: [node_count][2]f32 align(64),
        /// The preprocessed parameters, computed by softmax(parameters).
        sigma: [node_count]CDLG align(64),
        /// Generates the random inputs on-the-fly.
        lcg: LCG32(1664525, 1013904223, options.seed),

        pub const default = Self{
            .gradient = @splat(@splat(0)),
            .diff = @splat(.{ 0, 0 }),
            .lcg = .init(options.seed),
            .sigma = undefined,
        };

        pub const default_parameters: [parameter_count]f32 =
            @bitCast(@as([node_count]f32x4, @splat(.{ 0, 0, 1, 1 })));

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

        fn logistic(x: f32) f32 {
            return 1 / (1 + @exp(x));
        }

        pub fn eval(noalias self: *Self, noalias input: *const Input, noalias output: *Output) void {
            @setFloatMode(.optimized);
            self.lcg.reset();
            for (self.sigma, output) |sigma, *activation| {
                const a = input[self.lcg.next() % input_dim];
                const b = input[self.lcg.next() % input_dim];
                activation.* = sigma.eval(a, b);
            }
        }

        pub fn forwardPass(noalias self: *Self, noalias input: *const Input, noalias output: *Output) void {
            @setFloatMode(.optimized);
            self.lcg.reset();
            // For some reason it is faster to to a simple loop rather than a multi item one.
            // I found needless memcpy calls with callgrind.
            for (0..node_count) |j| {
                const a = input[self.lcg.next() % input_dim];
                const b = input[self.lcg.next() % input_dim];

                const sigma = self.sigma[j];
                output[j] = sigma.eval(a, b);
                self.diff[j] = .{
                    sigma.aDiff(b),
                    sigma.bDiff(a),
                };
                self.gradient[j] = sigma.gradient(a, b);
            }
        }

        /// Fixme: Create a function "backward" that does not pass delta backwards, applicable for
        /// the first layer in a network only.
        pub fn backwardPass(noalias self: *Self, noalias input: *const [output_dim]f32, noalias cost_gradient: *[node_count]f32x4, noalias output: *[input_dim]f32) void {
            @setFloatMode(.optimized);
            self.lcg.reset();
            @memset(output, 0);
            for (0..node_count) |j| {
                const parent_a = self.lcg.next() % input_dim;
                const parent_b = self.lcg.next() % input_dim;
                cost_gradient[j] += self.gradient[j] * @as(f32x4, @splat(input[j]));
                output[parent_a] += self.diff[j][0] * input[j];
                output[parent_b] += self.diff[j][1] * input[j];
            }
        }

        pub fn takeParameters(self: *Self, parameters: *[node_count]f32x4) void {
            @setFloatMode(.optimized);
            for (&self.sigma, parameters) |*sigma, logit| {
                sigma.a = logistic(logit[0]);
                sigma.b = logistic(logit[1]);
                sigma.mix = logistic(logit[2]);
                sigma.neg = logistic(logit[3]);
            }
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

/// Divides the input into output_dim #buckets, each output is the sequential sum of
/// input_dim / output_dim items of the input.
pub fn GroupSum(input_dim_: usize, output_dim_: usize) type {
    return struct {
        const Self = @This();
        pub const input_dim = input_dim_;
        pub const output_dim = output_dim_;
        pub const Input = [input_dim]f32;
        pub const Output = [output_dim]f32;
        pub const parameter_count = 0;

        const quot = input_dim / output_dim;
        const scale: f32 = 1.0 / (@as(comptime_float, @floatFromInt(output_dim)));

        field: usize = 1,

        pub const default = Self{
            .field = 1,
        };

        pub fn eval(_: *Self, input: *const Input, output: *Output) void {
            @memset(output, 0);
            for (output, 0..) |*coord, k| {
                const from = k * quot;
                const to = from + quot;
                for (input[from..to]) |softbit| coord.* += softbit;
                coord.* *= scale;
            }
        }

        pub fn forwardPass(self: *Self, input: *const Input, output: *Output) void {
            return eval(self, input, output);
        }

        pub fn backwardPass(_: *Self, input: *const [output_dim]f32, _: *[0]f32, output: *[input_dim]f32) void {
            for (input, 0..) |child, k| {
                const from = k * quot;
                const to = from + quot;
                for (output[from..to]) |*parent| parent.* = child * scale;
            }
        }
    };
}
