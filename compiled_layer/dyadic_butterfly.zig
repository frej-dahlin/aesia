const aesia = @import("../aesia.zig");

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

        pub const ItemIn = bool;
        pub const ItemOut = bool;
        pub const dim_in = dim;
        pub const dim_out = dim;

        pub const Input = [dim_in]bool;
        pub const Output = [dim_out]bool;

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
pub fn ButterflyMap(log2_dim: usize, stage: usize) type {
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


        pub const Input = [dim_in]bool;
        pub const Output = [dim_out]bool;

        steer: [dim]bool,
        input_buffer: [dim]bool,

        pub fn init(_: *Self, parameters: *[parameter_count]bool) void {
            @memset(parameters, -10);
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
                    const d = self.steer[steer_index];
                    steer_index += 1;

                    output[j] = (!c and a) or (c and b);
                    output[j + delta] = (d and a) or (!d and b);
                }
            }
        }
    };
}