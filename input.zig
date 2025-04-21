pub fn RelaxBits(bit_count: usize) type {
    return struct {
        const Self = @This();
        const dim = bit_count;
        pub const Feature = [dim]u1;
        value: [dim]f32,

        // Fixme: use SIMD.
        pub fn eval(self: *Self, x: *const Feature) *const [dim]f32 {
            for (&self.value, x) |*softbit, bit| softbit.* = @floatFromInt(bit);
            return &self.value;
        }
    };
}

pub fn Identity(dim_out: usize) type {
	return struct {
		const Self = @This();
		const dim_in = dim_out;	
		pub const Feature = [dim_in]f32;
		
		pub fn eval(x: *const Feature, buffer: anytype) void {
			@memcpy(buffer.back_slice(dim_out), x);
			buffer.swap();
		}
	};
}
