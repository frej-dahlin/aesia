const std = @import("std");
const assert = std.debug.assert;

const StaticBitSet = @import("bitset.zig").StaticBitSet;

const f32x16 = @Vector(16, f32);
const f32x8 = @Vector(8, f32);
const f32x4 = @Vector(4, f32);
const f32x2 = @Vector(2, f32);

/// A compiled layer is a type interface, it needs to declare:
///     Input      : input type
///     Output     : output type
///     dim_in  : the dimension of the input
///     dim_out : the dimension of the output
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

pub fn Logic(dim_in_: usize, dim_out_: usize, options: LogicOptions) type {
    return struct {
        const Self = @This();
        pub const ItemIn = bool;
        pub const ItemOut = bool;
        pub const dim_in = dim_in_;
        pub const dim_out = dim_out_;

        pub const Input = if (options.gateRepresentation == .bitset) StaticBitSet(dim_in) else [dim_in]bool;
        pub const BitSet = StaticBitSet(node_count);
        pub const Output = if (options.gateRepresentation == .bitset) StaticBitSet(dim_out) else [dim_out]bool;
        const node_count = dim_out;
        const ParentIndex = std.math.IntFittingRange(0, dim_in - 1);
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
                    options.rand.intRangeLessThan(ParentIndex, 0, dim_in),
                    options.rand.intRangeLessThan(ParentIndex, 0, dim_in),
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

        pub fn compilePacked(self: *Self, parameters: *const [node_count][4]f32) void {
            self.permtime = 0;
            self.evaltime = 0;

            for (0..node_count) |j| {
                self.parents[j] = .{
                    options.rand.intRangeLessThan(ParentIndex, 0, dim_in),
                    options.rand.intRangeLessThan(ParentIndex, 0, dim_in),
                };
                if (options.gateRepresentation == .bitset) {
                    self.beta0.setValue(j, @round(parameters[j][0]) != 0);
                    self.beta1.setValue(j, @round(parameters[j][1]) != 0);
                    self.beta2.setValue(j, @round(parameters[j][2]) != 0);
                    self.beta3.setValue(j, @round(parameters[j][3]) != 0);
                } else {
                    self.beta0[j] = @round(parameters[j][0]) % 2 != 0;
                    self.beta1[j] = @round(parameters[j][1]) % 2 != 0;
                    self.beta2[j] = @round(parameters[j][2]) % 2 != 0;
                    self.beta3[j] = @round(parameters[j][3]) % 2 != 0;
                }
            }
        }

        pub fn getPermTime(self: *Self) u64 {
            return self.permtime;
        }
        pub fn getEvalTime(self: *Self) u64 {
            return self.evaltime;
        }

        pub fn init(self: *Self, parameters: *[node_count]bool) void {
            self.* = .{
                .sigma = null,
                .parents = undefined,
            };
            self.sigma = parameters;
            for (0..node_count) |j| {
                self.parents[j] = .{
                    options.rand.intRangeLessThan(ParentIndex, 0, dim_in),
                    options.rand.intRangeLessThan(ParentIndex, 0, dim_in),
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
            self.evalGates(output);
            // if (options.gateRepresentation == .bitset) {
            //     self.evalGates(output);
            // } else {
            //     for (0..node_count) |k| {
            //         output[k] = gate.eval(self.input1[k], self.input2[k], self.sigma[k]);
            //     }
            // }
            self.evaltime += evaltimer.read();
        }
    };
}


pub fn PackedLogic(dim_in_: usize, dim_out_: usize, options: LogicOptions) type {
    return struct {
        const Self = @This();
        pub const ItemIn = bool;
        pub const ItemOut = bool;
        pub const dim_in = dim_in_;
        pub const dim_out = dim_out_;
        pub const Input = if (options.gateRepresentation == .bitset) StaticBitSet(dim_in) else [dim_in]bool;
        pub const BitSet = StaticBitSet(node_count);
        pub const Output = if (options.gateRepresentation == .bitset) StaticBitSet(dim_out) else [dim_out]bool;
        const node_count = dim_out;
        const ParentIndex = std.math.IntFittingRange(0, dim_in - 1);
        // There are 16 possible logic gates, each one is assigned a probability logit.
        pub const parameter_count: usize = 4 * node_count;
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


        fn logistic(x: f32x4) f32x4 {
            @setFloatMode(.optimized);
            return @as(f32x4, @splat(1)) / (@as(f32x4, @splat(1)) + @exp(-x));
        }

        pub fn compile(self: *Self, parameters: *const [node_count]f32x4) void {
            self.permtime = 0;
            self.evaltime = 0;

            for (0..node_count) |j| {
                self.parents[j] = .{
                    options.rand.intRangeLessThan(ParentIndex, 0, dim_in),
                    options.rand.intRangeLessThan(ParentIndex, 0, dim_in),
                };

                const sigma : f32x4 = logistic(parameters[j]);

                if (options.gateRepresentation == .bitset) {
                    self.beta0.setValue(j, @round(sigma[0]) != 0);
                    self.beta1.setValue(j, @round(sigma[1]) != 0);
                    self.beta2.setValue(j, @round(sigma[2]) != 0);
                    self.beta3.setValue(j, @round(sigma[3]) != 0);
                } else {
                    self.beta0[j] = @round(sigma[0]) != 0;
                    self.beta1[j] = @round(sigma[1]) != 0;
                    self.beta2[j] = @round(sigma[2]) != 0;
                    self.beta3[j] = @round(sigma[3]) != 0;
                }
            }
        }
        pub fn init(self: *Self, parameters: *[node_count]bool) void {
            self.* = .{
                .sigma = null,
                .parents = undefined,
            };
            self.sigma = parameters;
            for (0..node_count) |j| {
                self.parents[j] = .{
                    options.rand.intRangeLessThan(ParentIndex, 0, dim_in),
                    options.rand.intRangeLessThan(ParentIndex, 0, dim_in),
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
            self.evalGates(output);
            // if (options.gateRepresentation == .bitset) {
            //     self.evalGates(output);
            // } else {
            //     for (0..node_count) |k| {
            //         output[k] = gate.eval(self.input1[k], self.input2[k], self.sigma[k]);
            //     }
            // }
            self.evaltime += evaltimer.read();
        }

        pub fn getPermTime(self: *Self) u64 {
            return self.permtime;
        }
        pub fn getEvalTime(self: *Self) u64 {
            return self.evaltime;
        }

    };
}
/// Divides the input into dim_out #buckets, each output is the sequential sum of
/// dim_in / dim_out items of the input.
pub fn GroupSum(dim_in_: usize, dim_out_: usize, options: LogicOptions) type {
    return struct {
        const Self = @This();

        pub const ItemIn = bool;
        pub const ItemOut = usize;
        pub const dim_in = dim_in_;
        pub const dim_out = dim_out_;
        pub const Input = if (options.gateRepresentation == .bitset) StaticBitSet(dim_in) else [dim_in]bool;
        pub const Output = [dim_out]usize;
        //pub const parameter_count: usize = 0;
        //pub const parameter_alignment: usize = 8;

        pub var permtime: u64 = 0;
        pub var evaltime: u64 = 0;

        const quot = dim_in / dim_out;
        const scale: f32 = 1.0 / (@as(comptime_float, @floatFromInt(dim_out)));

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

pub const LUTConvolutionPliesOptions = struct {
    depth: usize,
    height: usize,
    width: usize,
    lut_count: usize,
    field_size: struct { height: usize, width: usize },
    stride: struct { row: usize, col: usize },
};

pub fn LUTConvolutionPlies(options: LUTConvolutionPliesOptions) type {
    return struct {
        const Self = @This();

        // Unpack options.
        const depth_in = options.depth;
        const height_in = options.height;
        const width_in = options.width;
        const lut_count = options.lut_count;
        const field_size = options.field_size;
        const stride = options.stride;

        const Ply = LUTConvolution(.{
            .height = height_in,
            .width = width_in,
            .lut_count = lut_count,
            .field_size = .{ .height = field_size.height, .width = field_size.width },
            .stride = .{ .row = stride.row, .col = stride.col },
        });
        const depth_out = depth_in * lut_count;
        const height_out = Ply.height_out;
        const width_out = Ply.width_out;

        pub const ItemIn = bool;
        pub const ItemOut = bool;
        pub const dim_in = depth_in * height_in * width_in;
        pub const dim_out = depth_out * height_out * width_out;

        pub const Input  = [depth_in][height_in][width_in]bool;
        pub const Output = [depth_in][lut_count][height_out][width_out]bool;
        //pub const Output = [depth_out][height_out][width_out]bool;
        pub const parameter_count = depth_in * Ply.parameter_count;
        pub const parameter_alignment = Ply.parameter_alignment;

        plies: [depth_in]Ply,


        permtime: u64 = 0,
        evaltime: u64 = 0,

        pub fn eval(
            self: *Self,
            input: *const Input,
            noalias output: *Output,
        ) void {
            for (0..depth_in) |ply| {
                self.plies[ply].eval(&input[ply], &output[ply]);
            }
        }

        pub fn compile(
            self: *Self,
            parameters: *[depth_in][Ply.lut_parameter_count][Ply.lut_count]f32,
        ) void {
            for (0..depth_in) |ply| self.plies[ply].compile(&parameters[ply]);
        }

        pub fn init(
            self: *Self,
            parameters: *[depth_in][Ply.lut_parameter_count][Ply.lut_count]bool,
        ) void {
            for (0..depth_in) |ply| self.plies[ply].init(&parameters[ply]);
        }



        pub fn getPermTime(self: *Self) u64 {
            return self.permtime;
        }
        pub fn getEvalTime(self: *Self) u64 {
            return self.evaltime;
        }

    };
}

pub const LUTConvolutionOptions = struct {
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

        pub const ItemIn = bool;
        pub const ItemOut = bool;
        pub const dim_in = height_in * width_in;
        pub const dim_out = height_out * width_out * lut_count;
        pub const parameter_count = lut_count * lut_parameter_count;
        pub const parameter_alignment = 64;

        // The lookup tables are stored in column major order, this is because we evaluate
        // all lookup tables given a certain input.
        expits: [lut_parameter_count][lut_count]bool,
        activations: [height_in][width_in][lut_arity + 1]ExpitIndex,
        receptions: [height_out][width_out][lut_arity]bool,

        pub fn init(_: *Self, parameters: *[lut_parameter_count][lut_count]bool) void {
            for (0..lut_parameter_count / 2) |i| {
                for (0..lut_count) |k| {
                    parameters[i][k] = true;
                }
            }
            for (lut_parameter_count / 2..lut_parameter_count) |i| {
                for (0..lut_count) |k| {
                    parameters[i][k] = false;
                }
            }
        }
        pub fn compile(
            self: *Self,
            parameters: *[lut_parameter_count][lut_count]f32,
        ) void {
            for (0..lut_parameter_count) |i| {
                for (0..lut_count) |k| {
                    self.expits[i][k] = (@round(1 / (1 + @exp(-parameters[i][k]))) != 0);
                    //self.expits[i][k] = (@round(parameters[i][k]) != 0);
                }
            }
        }

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
                max_index |= @as(ExpitIndex, if(reception[j]) 1 else 0) << j;
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
            output: *[lut_count][height_out][width_out]bool,
        ) void {
            inline for (self.activations[row][col]) |index| {
                var product: bool = true;
                inline for (self.receptions[row][col], 0..) |x, bit| {
                    product = product and (if ((index >> bit) & 1 != 0) x else !x);
                }
                inline for (0..lut_count) |k| {
                    //output[k][row][col] = (output[k][row][col] or (self.expits[index][k] and product));
                    output[k][row][col] = (output[k][row][col] or (self.expits[index][k] and product)) and !(output[k][row][col] and (self.expits[index][k] and product));
                }
            }
        }

        pub fn eval(
            self: *Self,
            input: *const [height_in][width_in]bool,
            output: *[lut_count][height_out][width_out]bool,
        ) void {
            // Collect receptions and find the activated indices.
            output.* = @splat(@splat(@splat(false)));

            for (0..lut_count) |i| {
                for (0..height_out) |j| {
                    for (0..width_in) |k| {
                        output[i][j][k] = false;
                    }
                }
            }
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
    };
}