const std = @import("std");
const aesia = @import("../aesia.zig");

const StaticBitSet = @import("../src/bitset.zig").StaticBitSet;
pub const logic = @import("logic.zig");

pub const GateRepresentation = logic.GateRepresentation;
pub const Options = struct {gateRepresentation: GateRepresentation };

/// A dyadic butterfly swap is one layer of a permutation network,
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
pub fn ButterflySwap(log2_dim: usize, stage: usize, options: Options) type {
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

        pub const ItemIn = bool;
        pub const ItemOut = bool;
        pub const dim_in = dim;
        pub const dim_out = dim;

        pub const Input = if (options.gateRepresentation == .bitset) StaticBitSet(dim_in) else [dim_in]bool;
        pub const Output = if (options.gateRepresentation == .bitset) StaticBitSet(dim_out) else [dim_out]bool;

        steer: [dim / 2]bool,
        input_buffer: [dim]bool,

        pub fn init(_: *Self, parameters: *[parameter_count]bool) void {
            @memset(parameters, 0);
        }


        pub fn compile(self: *Self, parameters: *[parameter_count]f32) void {
            for (0..parameter_count) |j| {
                self.steer[j] = @round(1 / (1 + @exp(-parameters[j]))) != 0;
            }
        }

        pub fn eval(self: *const Self, input: *const [dim]bool, output: *[dim]bool) void {
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
                    output[j] = (!c and a) or (c and b);
                    output[j + delta] = (c and a) or (!c and b);
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
pub fn ButterflyMap(log2_dim: usize, stage: usize, options: Options) type {
    // Compile time checks.
    if (log2_dim == 0) @compileError("DyadicCrossover: log2_dim must be nonzero");
    if (stage >= log2_dim) @compileError("DyadicCrossover: stage must be beetween 0 and log2_dim - 1, inclusive");
    return struct {
        const Self = @This();

        const dim = 1 << log2_dim;
        // Distance between pairs.
        const delta = 1 << stage;
        pub const parameter_count = dim;
        pub const parameter_alignment = 64;

        pub const ItemIn = bool;
        pub const ItemOut = bool;
        pub const dim_in = dim;
        pub const dim_out = dim;


        pub const Input = if (options.gateRepresentation == .bitset) StaticBitSet(dim_in) else [dim_in]bool;
        pub const Output = if (options.gateRepresentation == .bitset) StaticBitSet(dim_out) else [dim_out]bool;

        steer: Output,

        const c_mask = blk: {
            var mask: usize = 0;
            if(delta >= 64){
                break :blk mask;
            }
            else{
                var value : bool = true;
                for (0.. (64 / delta)) |k| {
                    if(value)
                    {
                        for(0..delta) |j| {
                            mask |=  (1 << ((k * delta + j)));
                        }
                    }
                    value = !value;
                }
                break :blk mask;
            }
        };
        const d_mask = blk: {
            var mask: usize = 0;
            if(delta >= 64){
                break :blk mask;
            }
            else{
                var value : bool = false;
                for (0.. (64 / delta)) |k| {
                    if(value)
                    {
                        for(0..delta) |j| {
                            mask |=  (1 << ((k * delta + j)));
                        }
                    }
                    value = !value;
                }
                break :blk mask;
            }
        };

        pub fn init(_: *Self, parameters: *[parameter_count]bool) void {
            @memset(parameters, -10);
        }

        pub fn compile(self: *Self, parameters: *[parameter_count]f32) void {
            if (options.gateRepresentation == .bitset) {
                if(delta < 64) {
                    var steer_index: usize = 0;
                    for (0..dim >> (stage + 1)) |k| {
                        const from = 2 * k * delta;
                        const to = (2 * k + 1) * delta;
                        for (from..to) |j| {
                            self.steer.setValue(j, @round(1 / (1 + @exp(-parameters[2*steer_index]))) != 0);
                            self.steer.setValue(j+delta, @round(1 / (1 + @exp(-parameters[2*steer_index+1]))) != 0);
                            steer_index += 1;
                        }
                    }

                    for(0..self.steer.masks.len) |l|
                    {
                        self.steer.masks[l] = (self.steer.masks[l] & c_mask) | (~self.steer.masks[l] & ~c_mask);
                    }
                }
                else{
                    const mask_delta = delta / 64;
                    var steer_index: usize = 0;
                    for (0..dim >> (stage + 1)) |k| {
                        const from = 2 * k * mask_delta;
                        const to = (2 * k + 1) * mask_delta;
                        for (from..to) |j| {
                            for (0..64) |l| {
                                self.steer.setValue(64*j+l, @round(1 / (1 + @exp(-parameters[2*steer_index]))) != 0);
                                self.steer.setValue(64*j+l+delta, @round(1 / (1 + @exp(-parameters[2*steer_index+1]))) != 0);
                                steer_index += 1;
                            }
                        }
                    }
                }
            }
            else{
                for (0..parameter_count) |j| {
                    self.steer[j] = @round(1 / (1 + @exp(-parameters[j]))) != 0;
                }
            }
        }

        pub fn eval(self: *const Self, noalias input: *const Input, noalias output: *Output) void {
            if (options.gateRepresentation == .bitset) {
                if(delta >= 64) {
                    const mask_delta = delta / 64;
                    for (0..dim >> (stage + 1)) |k| {
                        const from = 2 * k * mask_delta;
                        const to = (2 * k + 1) * mask_delta;
                        for (from..to) |j| {
                            const a = input.masks[j];
                            const b = input.masks[j+mask_delta];
                            const c = self.steer.masks[j];
                            const d = self.steer.masks[j+mask_delta];

                            output.masks[j] = (~c & a) | (c & b);
                            output.masks[j+mask_delta] = (d & a) | (~d & b);
                        }
                    }
                }
                else {
                    for(0..output.masks.len) |l|
                    {
                        const a = (input.masks[l] & c_mask) | ((input.masks[l] << delta) & d_mask);
                        const b = ((input.masks[l] >> delta) & c_mask) | (input.masks[l] & d_mask);
                        const cd = self.steer.masks[l];
                        output.masks[l] = ((~cd & a) | (cd & b));
                    }
                }
            } else {
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

                        output[j] = (!c and a) or (c and b);
                        output[j + delta] = (d and a) or (!d and b);
                    }
                }
            }
        }
    };
}

pub fn ButterflyGate(log2_dim: usize, stage: usize, options: Options) type {
    // Compile time checks.
    if (log2_dim == 0) @compileError("DyadicCrossover: log2_dim must be nonzero");
    if (stage >= log2_dim) @compileError("DyadicCrossover: stage must be beetween 0 and log2_dim - 1, inclusive");
    return struct {
        const Self = @This();

        const dim = 1 << log2_dim;
        // Distance between pairs.
        const delta = 1 << stage;
        pub const parameter_count = 4 * dim;
        pub const parameter_alignment = 64;

        pub const ItemIn = bool;
        pub const ItemOut = bool;
        pub const dim_in = dim;
        pub const dim_out = dim;

        pub const Input = if (options.gateRepresentation == .bitset) StaticBitSet(dim_in) else [dim_in]bool;
        pub const Output = if (options.gateRepresentation == .bitset) StaticBitSet(dim_out) else [dim_out]bool;

        gates: if (options.gateRepresentation == .bitset) StaticBitSet(4*dim) else [4*dim]bool,

        const c_mask = blk: {
            var mask: usize = 0;
            if(delta >= 64){
                break :blk mask;
            }
            else{
                var value : bool = true;
                for (0.. (64 / delta)) |k| {
                    if(value)
                    {
                        for(0..delta) |j| {
                            mask |=  (1 << ((k * delta + j)));
                        }
                    }
                    value = !value;
                }
                break :blk mask;
            }
        };
        const d_mask = blk: {
            var mask: usize = 0;
            if(delta >= 64){
                break :blk mask;
            }
            else{
                var value : bool = false;
                for (0.. (64 / delta)) |k| {
                    if(value)
                    {
                        for(0..delta) |j| {
                            mask |=  (1 << ((k * delta + j)));
                        }
                    }
                    value = !value;
                }
                break :blk mask;
            }
        };

        pub fn init(_: *Self, parameters: *[parameter_count]bool) void {
            @memset(parameters, -10);
        }

        pub fn compile(self: *Self, parameters: *[parameter_count]f32) void {
            if (options.gateRepresentation == .bitset) {
                for (0..parameter_count / 256) |j| {
                    for (0..256) |k| {
                        self.gates.setValue(j * 256 + (k % 4) * 64 + k / 4, @round(1 / (1 + @exp(-parameters[256*j+k]))) != 0);
                    }
                }
                for (0..parameter_count / 256) |j| {
                    for (0..256) |k| {
                        self.gates.setValue(j * 256 + (k % 4) * 64 + k / 4, @round(1 / (1 + @exp(-parameters[256*j+k]))) != 0);
                    }
                }
            }
            else{
                for (0..parameter_count) |j| {
                    self.gates[j] = @round(1 / (1 + @exp(-parameters[j]))) != 0;
                }
            }
        }

        pub fn eval(self: *const Self, noalias input: *const Input, noalias output: *Output) void {
            if (options.gateRepresentation == .bitset) {
                if(delta >= 64) {
                    const mask_delta = delta / 64;
                    for (0..dim >> (stage + 1)) |k| {
                        const from = 2 * k * mask_delta;
                        const to = (2 * k + 1) * mask_delta;
                        for (from..to) |j| {
                            const index_left = j;
                            const index_right = j + mask_delta;

                            const a = input.masks[index_left];
                            const b = input.masks[index_right];
                            
                            output.masks[index_left] =  (self.gates.masks[4 * index_left] & a & b) | (self.gates.masks[4 * index_left + 1] & a & ~b) | (self.gates.masks[4 * index_left + 2] & ~a & b) | (self.gates.masks[4 * index_left + 3] & ~a & ~b);
                            output.masks[index_right] = (self.gates.masks[4 * index_right] & a & b) | (self.gates.masks[4 * index_right + 1] & a & ~b) | (self.gates.masks[4 * index_right + 2] & ~a & b) | (self.gates.masks[4 * index_right + 3] & ~a & ~b);
                        }
                    }
                }
                else {
                    for(0..output.masks.len) |l|
                    {
                        const a = (input.masks[l] & c_mask) | ((input.masks[l] << delta) & d_mask);
                        const b = ((input.masks[l] >> delta) & c_mask) | (input.masks[l] & d_mask);
                        output.masks[l] = (self.gates.masks[4 * l] & a & b) | (self.gates.masks[4 * l + 1] & a & ~b) | (self.gates.masks[4 * l + 2] & ~a & b) | (self.gates.masks[4 * l + 3] & ~a & ~b);
                    }
                }
            } else {
                for (0..dim >> (stage + 1)) |k| {
                    const from = 2 * k * delta;
                    const to = (2 * k + 1) * delta;
                    for (from..to) |j| {
                        const index_left = j;
                        const index_right = j + delta;

                        const a = input[index_left];
                        const b = input[index_right];
                        
                        output[index_left] =  (self.gates[4 * index_left] and a and b) or (self.gates[4 * index_left + 1] and a and !b) or (self.gates[4 * index_left + 2] and !a and b) or (self.gates[4 * index_left + 3] and !a and !b);
                        output[index_right] = (self.gates[4 * index_right] and a and b) or (self.gates[4 * index_right + 1] and a and !b) or (self.gates[4 * index_right + 2] and !a and b) or (self.gates[4 * index_right + 3] and !a and !b);
                    }
                }
            }
        }
    };
}