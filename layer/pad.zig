pub fn ZeroPad(_dim_in: usize, _dim_out: usize) type {
    if (_dim_in >= _dim_out) @compileError("ZeroPad: dim_in must be less than dim_out");
    return struct {
        const Self = @This();

        pub const ItemIn = f32;
        pub const ItemOut = f32;
        pub const dim_in = _dim_in;
        pub const dim_out = _dim_out;

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
