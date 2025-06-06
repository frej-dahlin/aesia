const aesia = @import("../aesia.zig");

pub fn ZeroPad(dim_in: usize, dim_out: usize) type {
    if (dim_in >= dim_out) @compileError("ZeroPad: dim_in must be less than dim_out");
    return struct {
        const Self = @This();

        pub const info = aesia.layer.Info{
            .dim_in = dim_in,
            .dim_out = dim_out,
            .trainable = false,
            .statefull = false,
        };

        pub fn init(_: *Self) void {}

        pub fn eval(input: *const [dim_in]f32, output: *[dim_out]f32) void {
            @setFloatMode(.optimized);
            @memset(output[dim_in..], 0);
            @memcpy(output[0..dim_in], input);
        }

        pub fn forwardPass(input: *const [dim_in]f32, output: *[dim_out]f32) void {
            @setFloatMode(.optimized);
            return eval(input, output);
        }

        pub fn backwardPass(
            activation_delta: *const [dim_out]f32,
            argument_delta: *[dim_in]f32,
        ) void {
            @setFloatMode(.optimized);
            @memcpy(argument_delta, activation_delta[0..dim_in]);
        }
    };
}

pub fn Repeat(dim_in: usize, dim_out: usize) type {
    return struct {
        const Self = @This();

        const copy_count = dim_out / dim_in;

        pub const info = aesia.layer.Info{
            .dim_in = dim_in,
            .dim_out = dim_out,
            .trainable = false,
            .statefull = false,
        };

        pub fn eval(input: *const [dim_in]f32, output: *[dim_out]f32) void {
            for (0..copy_count) |k| {
                const from = k * dim_in;
                const to = (k + 1) * dim_in;
                @memcpy(output[from..to], input);
            }
            @memset(output[copy_count * dim_in ..], 0);
        }

        pub fn forwardPass(input: *const [dim_in]f32, output: *[dim_out]f32) void {
            eval(input, output);
        }

        pub fn backwardPass(
            activation_delta: *const [dim_out]f32,
            argument_delta: *[dim_in]f32,
        ) void {
            @memset(argument_delta, 0);
            for (0..copy_count) |k| {
                const from = k * dim_in;
                const to = (k + 1) * dim_in;
                const slice = activation_delta[from..to];
                for (0..dim_in) |i| {
                    argument_delta[i] += slice[i];
                }
            }
        }
    };
}
