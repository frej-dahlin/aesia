const std = @import("std");
const assert = std.debug.assert;

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

pub fn Perceive(
    options: struct {
        height_in: usize,
        width_in: usize,
        field: struct { height: usize, width: usize },
        // stride: struct { row: usize, col: usize },
    },
) type {
    return struct {
        const Self = @This();

        const height_in = options.height_in;
        const width_in = options.width_in;
        const field = options.field;
        const height_co = height_in / field.height;
        const width_co = width_in / field.width;

        pub const info: aesia.layer.Info = .{
            .dim_in = height_in * width_in,
            .dim_out = height_in * width_in,
            .trainable = false,
            .statefull = false,
        };

        const Point = @Vector(2, usize);
        const offsets = blk: {
            var result: [field.height * field.width]Point = undefined;
            var i: usize = 0;
            for (0..field.height) |row| {
                for (0..field.width) |col| {
                    result[i] = Point{ row, col };
                    i += 1;
                }
            }
            break :blk result;
        };

        pub fn eval(
            input: *const [height_in][width_in]f32,
            output: *[height_co][width_co][field.height][field.width]f32,
        ) void {
            const stride: Point = .{ field.height, field.width };
            for (0..height_co) |field_row| {
                for (0..width_co) |field_col| {
                    inline for (offsets) |offset| {
                        const point: Point = stride * Point{ field_row, field_col } + offset;
                        const offset_row = offset[0];
                        const offset_col = offset[1];
                        const row = point[0];
                        const col = point[1];
                        output[field_row][field_col][offset_row][offset_col] = input[row][col];
                    }
                }
            }
        }

        pub fn forwardPass(
            input: *const [height_in][width_in]f32,
            output: *[height_co][width_co][field.height][field.width]f32,
        ) void {
            return eval(input, output);
        }
        pub fn backwardPass(
            activation_delta: *[height_co][width_co][field.height][field.width]f32,
            _: *[height_co][width_co][field.height][field.width]f32,
            argument_delta: *const [height_in][width_in]f32,
        ) void {
            const stride: Point = .{ field.height, field.width };
            for (0..height_co) |field_row| {
                for (0..width_co) |field_col| {
                    inline for (offsets) |offset| {
                        const point: Point = stride * .{ field_row, field_col } + offset;
                        const offset_row = offset[0];
                        const offset_col = offset[1];
                        const row = point[0];
                        const col = point[1];
                        activation_delta[row][col] =
                            argument_delta[field_row][field_col][offset_row][offset_col];
                    }
                }
            }
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
            input: *const [height_in][width_in][depth_in]f32,
            output: *[height_out][width_out][depth_out]f32,
        ) void {
            output.* = @splat(@splat(@splat(0)));
            for (0..height_in) |row_in| {
                const from_col = padding.left;
                const to_col = width_out - padding.right;
                @memcpy(
                    output[row_in + padding.top][from_col..to_col],
                    &input[row_in],
                );
            }
        }

        pub fn forwardPass(
            input: *const [height_in][width_in][depth_in]f32,
            output: *[height_out][width_out][depth_out]f32,
        ) void {
            return eval(input, output);
        }

        pub fn backwardPass(
            activation_error: *const [height_out][width_out][depth_out]f32,
            argument_error: *[height_in][width_in][depth_in]f32,
        ) void {
            for (0..height_in) |row_in| {
                const from_col = padding.left;
                const to_col = width_out - padding.right;
                @memcpy(
                    &argument_error[row_in],
                    activation_error[row_in + padding.top][from_col..to_col],
                );
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
            _: *const [dim_in]f32,
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

pub fn Dropout(dim: usize, probability: f32, rand: *std.Random) type {
    return struct {
        const Self = @This();

        pub const info = aesia.layer.Info{
            .dim_in = dim,
            .dim_out = dim,
            .trainable = false,
        };

        mask: [dim]bool,

        pub fn init(_: *Self) void {}

        pub fn eval(_: *Self, input: *const [dim]f32, output: *[dim]f32) void {
            for (0..dim) |i| {
                output[i] = if (rand.float(f32) < probability) input[i] else 0;
            }
        }

        pub fn validationEval(_: *Self, input: *const [dim]f32, output: *[dim]f32) void {
            @memcpy(output, input);
        }

        pub fn forwardPass(self: *Self, input: *const [dim]f32, output: *[dim]f32) void {
            for (0..dim) |i| {
                self.mask[i] = rand.float(f32) < probability;
                output[i] = if (self.mask[i]) input[i] else 0;
            }
        }

        pub fn backwardPass(
            self: *Self,
            activation_delta: *const [dim]f32,
            _: *const [dim]f32,
            argument_delta: *[dim]f32,
        ) void {
            for (0..dim) |i| {
                argument_delta[i] = if (self.mask[i]) activation_delta[i] else 0;
            }
        }
    };
}

fn interleaveWithZeros(input: u32) u64 {
    var word: u64 = input;
    word = (word ^ (word << 16)) & 0x0000ffff0000ffff;
    word = (word ^ (word << 8)) & 0x00ff00ff00ff00ff;
    word = (word ^ (word << 4)) & 0x0f0f0f0f0f0f0f0f;
    word = (word ^ (word << 2)) & 0x3333333333333333;
    word = (word ^ (word << 1)) & 0x5555555555555555;
    return word;
}

fn interleave(x: u32, y: u32) u64 {
    return (interleaveWithZeros(x) << 1) | interleaveWithZeros(y);
}

pub fn Noise(dim: usize, probability: f32, rand: *std.Random) type {
    return struct {
        const Self = @This();

        pub const info = aesia.layer.Info{
            .dim_in = dim,
            .dim_out = dim,
            .trainable = false,
        };

        mask: [dim]bool,
        values: [dim]f32,

        pub fn init(_: *Self) void {}

        pub fn eval(self: *Self, input: *const [dim]f32, output: *[dim]f32) void {
            for (0..dim) |i| {
                output[i] = input[i];
                if (rand.float(f32) > probability) {
                    self.mask[i] = true;
                    self.values[i] = if (rand.boolean()) 1.0 else 0.0;
                    output[i] = self.values[i];
                }
            }
        }

        pub fn validationEval(_: *Self, input: *const [dim]f32, output: *[dim]f32) void {
            @memcpy(output, input);
        }

        pub fn forwardPass(self: *Self, input: *const [dim]f32, output: *[dim]f32) void {
            return self.eval(input, output);
        }

        pub fn backwardPass(
            self: *Self,
            activation_delta: *const [dim]f32,
            _: *const [dim]f32,
            argument_delta: *[dim]f32,
        ) void {
            for (0..dim) |i| {
                argument_delta[i] = if (self.mask[i]) activation_delta[i] else 0;
            }
        }
    };
}

pub fn ZOrder(log2_dim: usize) type {
    return struct {
        const Self = @This();

        comptime {
            assert(log2_dim <= 32);
        }
        const dim = 1 << log2_dim;
        pub const info: aesia.layer.Info = .{
            .dim_in = dim * dim,
            .dim_out = dim * dim,
            .trainable = false,
            .statefull = false,
        };

        pub fn eval(
            input: *const [dim][dim]f32,
            output: *[dim * dim]f32,
        ) void {
            for (0..dim) |row| {
                for (0..dim) |col| {
                    const index = interleave(@truncate(row), @truncate(col));
                    output[index] = input[row][col];
                }
            }
        }

        pub fn forwardPass(
            input: *const [dim][dim]f32,
            output: *[dim * dim]f32,
        ) void {
            return eval(input, output);
        }
        pub fn backwardPass(
            activation_delta: *[dim * dim]f32,
            _: *[dim * dim]f32,
            argument_delta: *const [dim][dim]f32,
        ) void {
            for (0..dim) |row| {
                for (0..dim) |col| {
                    const index = interleave(row, col);
                    argument_delta[row][col] = activation_delta[index];
                }
            }
        }
    };
}
