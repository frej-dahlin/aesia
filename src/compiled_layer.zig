const std = @import("std");
const assert = std.debug.assert;

const StaticBitSet = @import("compiled_layer/bitset.zig").StaticBitSet;

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
pub const GateRepresentation = enum {
    boolarray,
    bitset,
};

pub const LogicOptions = struct { rand: *std.Random, gateRepresentation: GateRepresentation };

pub fn Logic(input_dim_: usize, output_dim_: usize, options: LogicOptions) type {
    return struct {
        const Self = @This();
        pub const input_dim = input_dim_;
        pub const output_dim = output_dim_;
        pub const Input = if (options.gateRepresentation == .bitset) StaticBitSet(input_dim) else [input_dim]bool;
        pub const BitSet = StaticBitSet(node_count);
        pub const Output = if (options.gateRepresentation == .bitset) StaticBitSet(output_dim) else [output_dim]bool;
        const node_count = output_dim;
        const ParentIndex = std.math.IntFittingRange(0, input_dim - 1);
        // There are 16 possible logic gates, each one is assigned a probability logit.
        pub const parameter_count: usize = 16 * node_count;
        // For efficient SIMD we ensure that the parameters align to a cacheline.
        pub const parameter_alignment: usize = 64;

        /// The preprocessed parameters
        sigma: [node_count]u8 align(64),
        parents: [node_count][2]ParentIndex,
        input1: if (options.gateRepresentation == .bitset) BitSet else [node_count]bool,
        input2: if (options.gateRepresentation == .bitset) BitSet else [node_count]bool,
        beta0: if (options.gateRepresentation == .bitset) BitSet else [node_count]bool,
        beta1: if (options.gateRepresentation == .bitset) BitSet else [node_count]bool,
        beta2: if (options.gateRepresentation == .bitset) BitSet else [node_count]bool,
        beta3: if (options.gateRepresentation == .bitset) BitSet else [node_count]bool,

        permtime: u64 = 0,
        evaltime: u64 = 0,
        const gate = struct {
            /// Returns the gate applied to a and b
            pub fn eval(a: bool, b: bool, w: usize) bool {
                return switch (w) {
                    0 => false,
                    1 => a and b,
                    2 => a and !b,
                    3 => a,
                    4 => b and !a,
                    5 => b,
                    6 => a and !b or b and !a,
                    7 => a or b,
                    8 => true,
                    9 => !(a and b),
                    10 => !(a and !b),
                    11 => !a,
                    12 => !(b and !a),
                    13 => !b,
                    14 => !(a and !b or b and !a),
                    15 => !(a or b),
                    else => false,
                };
            }

            pub fn evalGate(a: bool, b: bool, beta0: bool, beta1: bool, beta2: bool, beta3: bool) bool {
                //return (a and ((b and beta0) or (!b and beta1))) or (!a and ((b and beta2) or (!b and beta3)));
                return (a & b & beta0) | (a & ~b & beta1) | (~a & b & beta2) | (~a & ~b & beta3);
            }
        };

        pub fn evalGates(self: *Self, noalias output: *Output) void {
            if (options.gateRepresentation == .bitset) {
                for (0..self.input1.masks.len) |i| {
                    const a = self.input1.masks[i];
                    const b = self.input2.masks[i];
                    const beta0 = self.beta0.masks[i];
                    const beta1 = self.beta1.masks[i];
                    const beta2 = self.beta2.masks[i];
                    const beta3 = self.beta3.masks[i];

                    output.masks[i] = (a & b & beta0) | (a & ~b & beta1) | (~a & b & beta2) | (~a & ~b & beta3);
                }
            } else {
                for (0..node_count) |i| {
                    const a = self.input1[i];
                    const b = self.input2[i];
                    const beta0 = self.beta0[i];
                    const beta1 = self.beta1[i];
                    const beta2 = self.beta2[i];
                    const beta3 = self.beta3[i];

                    output[i] = (a and b and beta0) or (a and !b and beta1) or (!a and b and beta2) or (!a and !b and beta3);
                }
            }
        }

        pub fn compile(self: *Self, parameters: *const [node_count][16]f32) void {
            self.permtime = 0;
            self.evaltime = 0;
            for (0..node_count) |j| {
                self.sigma[j] = @intCast(std.mem.indexOfMax(f32, &parameters[j]));
                self.parents[j] = .{
                    options.rand.intRangeLessThan(ParentIndex, 0, input_dim),
                    options.rand.intRangeLessThan(ParentIndex, 0, input_dim),
                };
                if (options.gateRepresentation == .bitset) {
                    if ((self.sigma[j] >> 3) % 2 == 0) {
                        self.beta0.setValue(j, (self.sigma[j] >> 0) % 2 != 0);
                        self.beta1.setValue(j, (self.sigma[j] >> 1) % 2 != 0);
                        self.beta2.setValue(j, (self.sigma[j] >> 2) % 2 != 0);
                        self.beta3.setValue(j, (self.sigma[j] >> 3) % 2 != 0);
                    } else {
                        self.beta0.setValue(j, (self.sigma[j] >> 0) % 2 == 0);
                        self.beta1.setValue(j, (self.sigma[j] >> 1) % 2 == 0);
                        self.beta2.setValue(j, (self.sigma[j] >> 2) % 2 == 0);
                        self.beta3.setValue(j, (self.sigma[j] >> 3) % 2 != 0);
                    }
                } else {
                    if ((self.sigma[j] >> 3) % 2 == 0) {
                        self.beta0[j] = (self.sigma[j] >> 0) % 2 != 0;
                        self.beta1[j] = (self.sigma[j] >> 1) % 2 != 0;
                        self.beta2[j] = (self.sigma[j] >> 2) % 2 != 0;
                        self.beta3[j] = (self.sigma[j] >> 3) % 2 != 0;
                    } else {
                        self.beta0[j] = (self.sigma[j] >> 0) % 2 == 0;
                        self.beta1[j] = (self.sigma[j] >> 1) % 2 == 0;
                        self.beta2[j] = (self.sigma[j] >> 2) % 2 == 0;
                        self.beta3[j] = (self.sigma[j] >> 3) % 2 != 0;
                    }
                }
            }
        }

        pub fn unpack(beta: [4]usize) usize {
            return if (beta[3] == 0)
                8 * beta[3] + 4 * beta[2] + 2 * beta[1] + beta[0]
            else
                8 * beta[3] + 4 * beta[0] + 2 * beta[1] + beta[2];
        }

        pub fn getPermTime(self: *Self) u64 {
            return self.permtime;
        }
        pub fn getEvalTime(self: *Self) u64 {
            return self.evaltime;
        }
        // pub fn compilePacked(self: *Self, parameters: *const [node_count][4]f32) void {
        //     self.* = .{
        //         .sigma = null,
        //         .parents = undefined,
        //     };

        //     for (0..node_count) |j| {
        //         var beta0 = @round(parameters[j][0]);
        //         var beta1 = @round(parameters[j][1]);
        //         var beta2 = @round(parameters[j][2]);
        //         var beta3 = @round(parameters[j][3]);
        //         self.parents[j] = .{
        //             options.rand.intRangeLessThan(ParentIndex, 0, input_dim),
        //             options.rand.intRangeLessThan(ParentIndex, 0, input_dim),
        //         };
        //     }
        // }

        pub fn init(self: *Self, parameters: *[node_count]bool) void {
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
            var permtimer = std.time.Timer.start() catch unreachable;
            if (options.gateRepresentation == .bitset) {
                for (0..node_count) |k| {
                    const a = input.isSet(self.parents[k][0]);
                    const b = input.isSet(self.parents[k][1]);
                    self.input1.setValue(k, a);
                    self.input2.setValue(k, b);
                }
            } else {
                for (0..node_count) |k| {
                    const a = input[self.parents[k][0]];
                    const b = input[self.parents[k][1]];
                    self.input1[k] = a;
                    self.input2[k] = b;
                }
            }

            self.permtime += permtimer.read();

            var evaltimer = std.time.Timer.start() catch unreachable;
            if (options.gateRepresentation == .bitset) {
                self.evalGates(output);
            } else {
                for (0..node_count) |k| {
                    output[k] = gate.eval(self.input1[k], self.input2[k], self.sigma[k]);
                }
            }
            self.evaltime += evaltimer.read();
        }
    };
}

/// Divides the input into output_dim #buckets, each output is the sequential sum of
/// input_dim / output_dim items of the input.
pub fn GroupSum(input_dim_: usize, output_dim_: usize, options: LogicOptions) type {
    return struct {
        const Self = @This();
        pub const input_dim = input_dim_;
        pub const output_dim = output_dim_;
        pub const Input = if (options.gateRepresentation == .bitset) StaticBitSet(input_dim) else [input_dim]bool;
        pub const Output = [output_dim]usize;
        pub const parameter_count: usize = 0;
        pub const parameter_alignment: usize = 8;

        pub var permtime: u64 = 0;
        pub var evaltime: u64 = 0;

        const quot = input_dim / output_dim;
        const scale: f32 = 1.0 / (@as(comptime_float, @floatFromInt(output_dim)));

        field: usize = 1,

        pub fn eval(_: *Self, input: *const Input, output: *Output) void {
            @memset(output, 0);
            for (output, 0..) |*coord, k| {
                const from = k * quot;
                const to = from + quot;

                if (options.gateRepresentation == .bitset) {
                    for (from..to) |l| coord.* += if (input.isSet(l)) 1 else 0;
                } else {
                    for (from..to) |l| coord.* += if (input[l]) 1 else 0;
                }
            }
        }
    };
}
