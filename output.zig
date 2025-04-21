pub fn Identity(dim_in: usize) type {
    return struct {
        const parameter_count = 0;
        const Self = @This();
        pub const dim_out = dim_in;
        pub const Prediction = [dim_out]f32;

        delta: [dim_out]f32,

        pub const default = Self{
            .delta = @splat(0),
        };

        pub fn eval(_: *Self, z: *const Prediction) *const Prediction {
            return z;
        }

        pub fn forwardPass(_: *Self, z: *const Prediction) *const Prediction {
            return z;
        }

        pub fn backwardPass(self: *Self, delta_prev: *[dim_out]f32) void {
            @memcpy(delta_prev, &self.delta);
        }
    };
}

pub fn GroupSum(tau: f32, _dim_out: usize) fn (usize) type {
    if (_dim_out == 0) @compileError("GroupSum output dimension needs to be nonzero");
    return struct {
        pub fn Init(dim_in: usize) type {
            if (dim_in % _dim_out != 0) @compileError("GroupSum input dimension must be evenly divisible by output dimension");
            return struct {
                const Self = @This();
                pub const dim_out = _dim_out;
                pub const Prediction = [dim_out]f32;
                const quot = dim_in / dim_out;
                const scale: f32 = 1.0 / (tau * @as(comptime_float, @floatFromInt(dim_out)));

                pub fn eval(buffer: anytype) *const Prediction {
                	const input = buffer.front_slice(dim_in);
                	const prediction = buffer.back_slice(dim_out);
                	@memset(prediction, 0);
                	for (prediction, 0..) |*coordinate, k| {
                		const from = k * quot;
                		const to = from + quot;
                		for (input[from..to]) |softbit| coordinate.* += softbit;
                		coordinate.* *= scale;
                    }
                    buffer.swap();
                    return prediction;
                }

                pub fn forwardPass(buffer: anytype) *const Prediction {
                    return eval(buffer);
                }

                pub fn backwardPass(buffer: anytype) void {
                	const delta_in = buffer.front_slice(dim_out);
                	const delta_out = buffer.back_slice(dim_in);
                	for (delta_in, 0..) |child, k| {
                		const from = k * quot;
                		const to = from + quot;
                		for (delta_out[from..to]) |*parent| parent.* = child * scale;
                	}
                	buffer.swap();
                }
            };
        }
    }.Init;
}
