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
