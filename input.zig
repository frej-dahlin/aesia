pub fn RelaxBits(bit_count: usize) type {
    return struct {
        const Self = @This();
        const dim = bit_count;
        pub const Type = [dim]u1;
        value: [dim]f32,

        // Fixme: use SIMD.
        pub fn eval(self: *Self, x: *const Type) *const [dim]f32 {
            for (&self.value, x) |*softbit, bit| softbit.* = @floatFromInt(bit);
            return &self.value;
        }
    };
}

pub fn Identity(dim_out: usize) type {
    return struct {
        const Self = @This();
        const dim = dim_out;
        pub const Type = [dim]f32;

        pub fn eval(_: *Self, x: *const Type) *const Type {
            return x;
        }
    };
}
