pub fn Identity(dim_in: usize) type {
    return struct {
    	const parameter_count = 0;
        const Self = @This();
        pub const dim_out = dim_in;
        pub const Type = [dim_out]f32;
        
        delta: [dim_out]f32,
        
        pub const default = Self{
       		.delta = @splat(0), 
        };
 
        pub fn eval(_: *Self, z: *const Type) *const Type {
            return z;
        }
        
        pub fn forwardPass(_: *Self, z: *const Type) *const Type {
       		return z; 
        }
        
        pub fn backwardPass(self: *Self, delta_prev: *[dim_out]f32) void {
        	@memcpy(delta_prev, &self.delta);
        }
    };
}

pub fn GroupSum(dim_out: usize) type {
    if (dim_out == 0) @compileError("GroupSum output dimension needs to be nonzero");
    return struct {
        pub fn Init(dim_in: usize) type {
            if (dim_in % dim_out != 0) @compileError("GroupSum input dimension must be evenly divisible by output dimension");
            return struct {
                const Self = @This();
                
                pub const Type = [dim_out]f32;
                value: [dim_out]f32,
                delta: [dim_out]f32,

                pub fn eval(self: *Self, input: *const [dim_in]f32) *const Type {
                    const quot = dim_in / dim_out;
                    const denom: f32 = @floatFromInt(dim_in / dim_out);
                    @memset(&self.value, 0);
                    for (&self.value, 0..) |*coord, k| {
                        for (k * quot..(k + 1) * quot) |i| coord.* += input[i];
                        coord.* /= denom;
                    }
                    return &self.value;
                }
                
                pub fn forwardPass(self: *Self, input: *const [dim_in]f32) *const Type {
                	return self.eval(input);
                }
            };
        }
    }.Init;
}
