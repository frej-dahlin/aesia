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

        pub fn eval(input: *const Input, output: *Output) void {
            if (options.gateRepresentation == .bitset) {
                for (0..dim_in) |i|{
                    output.setValue(i, input.isSet(i));
                }
                for (dim_in..dim_out) |i|{
                    output.setValue(i, false);
                }
            } else {
                @memset(output[dim_in..], false);
                @memcpy(output[0..dim_in], input);
            }
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
        
        const indices = blk: {
            var count = 0;
            for (0..copy_count) |k| {
                var output_first = k * dim_in;
                //const output_last = (k + 1) * dim_in - 1;
                var input_first = 0;
                const input_last = dim_in - 1;

                while(input_first <= input_last){
                    const len = @min(64 - (output_first % 64), 64 - (input_first % 64));
                    
                    count += 1;
                    input_first += len;
                    output_first += len;
                }
            }
            const nmasks = count;
            var input_mask_indices: [nmasks]usize = undefined;
            var output_mask_indices: [nmasks]usize = undefined;
            var input_first_indices: [nmasks]usize = undefined;
            var output_first_indices: [nmasks]usize = undefined;
            
            count = 0;
            for (0..copy_count) |k| {
                var output_first = k * dim_in;
                //const output_last = (k + 1) * dim_in - 1;
                var input_first = 0;
                const input_last = dim_in - 1;

                while(input_first <= input_last){
                    input_mask_indices[count] = input_first / 64;
                    output_mask_indices[count] = output_first / 64;
                    input_first_indices[count] = input_first % 64;
                    output_first_indices[count] = output_first % 64;
                    const len = @min(64 - (output_first % 64), 64 - (input_first % 64));
                    
                    count += 1;
                    input_first += len;
                    output_first += len;
                }
            }

            break :blk .{nmasks, input_mask_indices, output_mask_indices, input_first_indices, output_first_indices};
        };


        pub fn eval(input: *const Input, output: *Output) void {
            if (options.gateRepresentation == .bitset) {
                // for (0..copy_count) |k| {
                //     const from = k * dim_in;
                //     const to = (k + 1) * dim_in;
                //     for (from..to, 0..to-from) |i, j| {
                //         output.setValue(i, input.isSet(j));
                //     }
                // }
                // for (copy_count * dim_in..dim_out) |i| {
                //     output.setValue(i, false);
                // }

                // for (0..output.masks.len) |i| {
                //     output.masks[i] = 0;
                // }
                // var count : usize = 0;
                // for (0..copy_count) |k| {
                //     const from = k * dim_in;
                //     const to = (k + 1) * dim_in;
                //     const from_mask_index = Input.maskIndex(from);
                //     const to_mask_index = Input.maskIndex(to);
                //     for (from_mask_index..to_mask_index+1) |i| {
                //         output.masks[i] = input.masks[i % dim_in] & masks[count];
                //         count += 1;
                //     }
                // }
                // for (0..output.masks.len) |i| {
                //     output.masks[i] = 0;
                // }
                // var count : usize = 0;
                // for (0..masks.len) |k| {
                //     const from = k * dim_in;
                //     const to = (k + 1) * dim_in;
                //     const from_mask_index = Input.maskIndex(from);
                //     const to_mask_index = Input.maskIndex(to);
                //     for (from_mask_index..to_mask_index+1) |i| {
                //         output.masks[i] = input.masks[i % dim_in] & masks[count];
                //         count += 1;
                //     }
                // }
                for (0..output.masks.len) |i| {
                    output.masks[i] = 0;
                }
                // for (0..output.masks.len) |k| {
                //     if(input_first[k] == 0){
                //         output.masks[k] = input.masks[input_mask_index[2 * k]];
                //     }
                //     output.masks[k] = (input.masks[input_mask_index[2 * k]] >> @as(u6, @truncate(input_first[2 * k])));
                //     output.masks[k] |= ((input.masks[input_mask_index[2 * k + 1]] >> @as(u6, @truncate(input_first[2 * k + 1]))) << @as(u6, @truncate(64 - input_first[2 * k])));
                // }
                for (0..indices[0]) |k| {
                    const input_mask_index = indices[1][k];
                    const output_mask_index = indices[2][k];
                    const input_first_index = indices[3][k];
                    const output_first_index = indices[4][k];

                    output.masks[output_mask_index] |= ((input.masks[input_mask_index] >> @as(u6, @truncate(input_first_index))) << @as(u6, @truncate(output_first_index)));
                }
                for (copy_count * dim_in..dim_out) |i| {
                    output.setValue(i, false);
                }
            } else {
                for (0..copy_count) |k| {
                    const from = k * dim_in;
                    const to = (k + 1) * dim_in;
                    @memcpy(output[from..to], input);
                }
                @memset(output[copy_count * dim_in ..], false);
            }
        }
    };
}
