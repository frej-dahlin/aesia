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
                value: [dim_out]f32,
                delta: [dim_out]f32,

                const quot = dim_in / dim_out;
                const scale: f32 = 1.0 / (tau * @as(comptime_float, @floatFromInt(dim_out)));

                pub fn eval(self: *Self, input: *const [dim_in]f32) *const Prediction {
                    @memset(&self.value, 0);
                    for (&self.value, 0..) |*coord, k| {
                        for (k * quot..(k + 1) * quot) |i| coord.* += input[i];
                        coord.* *= scale;
                    }
                    return &self.value;
                }

                pub fn forwardPass(self: *Self, input: *const [dim_in]f32) *const Prediction {
                    return self.eval(input);
                }

                pub fn backwardPass(self: *Self, delta_prev: *[dim_in]f32) void {
                    for (&self.delta, 0..) |delta, k| {
                        for (k * quot..(k + 1) * quot) |i| delta_prev[i] = delta * scale;
                    }
                }
            };
        }
    }.Init;
}
