const std = @import("std");
const assert = std.debug.assert;

const f32x16 = @Vector(16, f32);
const f32x8 = @Vector(8, f32);
const f32x4 = @Vector(4, f32);
const f32x2 = @Vector(2, f32);

/// A discrete layer is a type interface, it needs to declare:
///     Input      : input type
///     Output     : output type
///     input_dim  : the dimension of the input
///     output_dim : the dimension of the output
/// Optionally, the layer can make use of parameters, which have to be of type usize.
/// Therefore every layer must declare
///     parameter_count : the number of parameters the layer will be allocated
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

pub const LogicOptions = struct {
    rand: *std.Random,
};

pub fn Logic(input_dim_: usize, output_dim_: usize, options: LogicOptions) type {
    return struct {
        const Self = @This();
        pub const input_dim = input_dim_;
        pub const output_dim = output_dim_;
        pub const Input = [input_dim]usize;
        pub const Output = [output_dim]usize;
        const node_count = output_dim;
        const ParentIndex = std.math.IntFittingRange(0, input_dim - 1);
        // There are 16 possible logic gates, each one is assigned a probability logit.
        pub const parameter_count = 16 * node_count;
        // For efficient SIMD we ensure that the parameters align to a cacheline.
        pub const parameter_alignment = 64;

        /// The preprocessed parameters
        sigma: ?*[node_count]usize align(64),

        const gate = struct {
            /// Returns the gate applied to a and b
            pub fn eval(a: usize, b: usize, w: usize) usize {
                return switch(w) {
                    0 => 0,
                    1 => a & b,
                    2 => f: {break :f a-a & b;},
                    3 => f: {break :f a;},
                    4 => f: {break :f b-a & b;},
                    5 => f: {break :f b;},
                    6 => f: {break :f a + b - 2 * a * b;},
                    7 => f: {break :f a + b - a * b;},
                    8 => f: {break :f 1 - (a + b - a * b);},
                    9 => f: {break :f 1 - (a + b - 2 * a * b);},
                    10 => f: {break :f 1 - b;},
                    11 => f: {break :f 1 - (b - a * b);},
                    12 => f: {break :f 1 - a;},
                    13 => f: {break :f 1 - (a - a * b);},
                    14 => f: {break :f 1 - a * b;},
                    15 => 1,
                    else => 0,
                };
            }
        };
        pub fn compile(self: *Self, parameters: *const [node_count][16]f32) void {
            self.* = .{
                .sigma = null,
                .parents = undefined,
            };

            for (0..node_count) |j| {
                sigma[j] = std.mem.indexOfMax(f32, &parameters[j]);
                self.parents[j] = .{
                    options.rand.intRangeLessThan(ParentIndex, 0, input_dim),
                    options.rand.intRangeLessThan(ParentIndex, 0, input_dim),
                };
            }
        }

        pub fn unpack(beta: [4]usize) usize
        {
            if(beta[3] == 0)
                return 8*beta[3]+4*beta[2]+2*beta[1]+beta[0];
            else
                return 8*beta[3]+4*beta[0]+2*beta[1]+beta[2];
        }
        pub fn compilePacked(self: *Self, parameters: *const [node_count][4]f32) void {
            self.* = .{
                .sigma = null,
                .parents = undefined,
            };

            for (0..node_count) |j| {
                var beta0 = @round(parameters[j][0]);
                var beta1 = @round(parameters[j][1]);
                var beta2 = @round(parameters[j][2]);
                var beta3 = @round(parameters[j][3]);
                self.parents[j] = .{
                    options.rand.intRangeLessThan(ParentIndex, 0, input_dim),
                    options.rand.intRangeLessThan(ParentIndex, 0, input_dim),
                };
            }
        }

        pub fn init(self: *Self, parameters: *[node_count]usize) void {
            self.* = .{
                .sigma = null,
                .parents = undefined,
            };
            self.sigma = parameters;
            for (0..node_count) |j| {
                self.parents[j] = .{
                    options.rand.intRangeLessThan(ParentIndex, 0, input_dim),
                    options.rand.intRangeLessThan(ParentIndex, 0, input_dim),
                };
            }
        }

        pub fn eval(noalias self: *Self, noalias input: *const Input, noalias output: *Output) void {
            for (self.sigma.?, self.parents, output) |sigma, parents, *activation| {
                const a = input[parents[0]];
                const b = input[parents[1]];
                activation.* = gate.eval(a, b, w);
            }
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
        pub const Input = [input_dim]usize;
        pub const Output = [output_dim]usize;
        pub const parameter_count = 0;

        const quot = input_dim / output_dim;
        const scale: f32 = 1.0 / (@as(comptime_float, @floatFromInt(output_dim)));

        field: usize = 1,

        pub fn init(self: *Self) void {

        }

        pub fn eval(_: *Self, input: *const Input, output: *Output) void {
            @memset(output, 0);
            for (output, 0..) |*coord, k| {
                const from = k * quot;
                const to = from + quot;
                for (input[from..to]) |bit| coord.* += bit;
            }
        }
    };
}
