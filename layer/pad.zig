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

pub fn Pad(options: struct {
    depth_in: usize,
    height_in: usize,
    width_in: usize,
    padding: struct {
        top: usize = 0,
        bottom: usize = 0,
        left: usize = 0,
        right: usize = 0,

        pub fn height(self: @This()) usize {
            return self.top + self.bottom;
        }

        pub fn width(self: @This()) usize {
            return self.left + self.right;
        }

        pub fn uniform(dim: usize) @This() {
            return .{
                .top = dim,
                .bottom = dim,
                .left = dim,
                .right = dim,
            };
        }
    },
}) type {
    return struct {
        const Self = @This();

        const depth_in = options.depth_in;
        const height_in = options.height_in;
        const width_in = options.width_in;
        const padding = options.padding;

        const depth_out = depth_in;
        const height_out = height_in + padding.height();
        const width_out = width_in + padding.width();

        pub const info = aesia.layer.Info{
            .dim_in = depth_in * height_in * width_in,
            .dim_out = depth_out * height_out * width_out,
            .trainable = false,
            .statefull = false,
        };

        pub fn eval(
            input: *const [depth_in][height_in][width_in]f32,
            output: *[depth_out][height_out][width_out]f32,
        ) void {
            output.* = @splat(@splat(@splat(0)));
            for (0..depth_in) |channel| {
                for (0..height_in) |row_in| {
                    const from_col = padding.left;
                    const to_col = width_out - padding.right;
                    @memcpy(
                        output[channel][row_in + padding.top][from_col..to_col],
                        &input[channel][row_in],
                    );
                }
            }
        }

        pub fn forwardPass(
            input: *const [depth_in][height_in][width_in]f32,
            output: *[depth_out][height_out][width_out]f32,
        ) void {
            return eval(input, output);
        }

        pub fn backwardPass(
            activation_error: *const [depth_out][height_out][width_out]f32,
            argument_error: *[depth_in][height_in][width_in]f32,
        ) void {
            for (0..depth_in) |channel| {
                for (0..height_in) |row_in| {
                    const from_col = padding.left;
                    const to_col = width_out - padding.right;
                    @memcpy(
                        &argument_error[channel][row_in],
                        activation_error[channel][row_in + padding.top][from_col..to_col],
                    );
                }
            }
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
            const tail = output[copy_count * dim_in ..];
            @memcpy(tail, input[0..tail.len]);
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
            const tail = activation_delta[copy_count * dim_in ..];
            for (0..tail.len) |i| {
                argument_delta[i] += tail[i];
            }
        }
    };
}
