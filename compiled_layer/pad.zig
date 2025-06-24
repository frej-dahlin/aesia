const std = @import("std");

const StaticBitSet = @import("bitset.zig").StaticBitSet;
pub const logic = @import("logic.zig");

pub const GateRepresentation = logic.GateRepresentation;
pub const Options = struct {gateRepresentation: GateRepresentation };

pub fn ZeroPad(dim_in_: usize, dim_out_: usize, options: Options) type {
    if (dim_in_ >= dim_out_) @compileError("ZeroPad: dim_in must be less than dim_out");
    return struct {
        const Self = @This();

        // const parameter_count = 0;
        // const parameter_alignment = 64;

        pub const ItemIn = bool;
        pub const ItemOut = bool;
        pub const dim_in = dim_in_;
        pub const dim_out = dim_out_;


        pub const Input = if (options.gateRepresentation == .bitset) StaticBitSet(dim_in) else [dim_in]bool;
        pub const Output = if (options.gateRepresentation == .bitset) StaticBitSet(dim_out) else [dim_out]bool;

        pub fn init(_: *Self) void {}

        pub fn eval(input: *const [dim_in]bool, output: *[dim_out]bool) void {
            @setFloatMode(.optimized);
            @memset(output[dim_in..], 0);
            @memcpy(output[0..dim_in], input);
        }
    };
}

pub fn Repeat(dim_in_: usize, dim_out_: usize, options: Options) type {
    return struct {
        const Self = @This();

        // const parameter_count = 0;
        // const parameter_alignment = 64;

        pub const ItemIn = bool;
        pub const ItemOut = bool;
        pub const dim_in = dim_in_;
        pub const dim_out = dim_out_;

        pub const Input = if (options.gateRepresentation == .bitset) StaticBitSet(dim_in) else [dim_in]bool;
        pub const Output = if (options.gateRepresentation == .bitset) StaticBitSet(dim_out) else [dim_out]bool;

        const copy_count = dim_out / dim_in;

        pub fn eval(input: *const [dim_in]bool, output: *[dim_out]bool) void {
            for (0..copy_count) |k| {
                const from = k * dim_in;
                const to = (k + 1) * dim_in;
                @memcpy(output[from..to], input);
            }
            @memset(output[copy_count * dim_in ..], false);
        }
    };
}
