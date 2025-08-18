/// Divides the input into dim_out #buckets, each output is the sequential sum of
/// dim_in / dim_out items of the input.

const StaticBitSet = @import("bitset.zig").StaticBitSet;

pub const GateRepresentation = @import("../compiled_layer.zig").GateRepresentation;

pub const Options = struct { gateRepresentation: GateRepresentation };

pub fn GroupSum(dim_in_: usize, dim_out_: usize, options: Options) type {
    return struct {
        const Self = @This();

        pub const ItemIn = bool;
        pub const ItemOut = usize;
        pub const dim_in = dim_in_;
        pub const dim_out = dim_out_;
        pub const Input = if (options.gateRepresentation == .bitset) StaticBitSet(dim_in) else [dim_in]bool;
        pub const Output = [dim_out]usize;
        //pub const parameter_count: usize = 0;
        //pub const parameter_alignment: usize = 8;

        pub var permtime: u64 = 0;
        pub var evaltime: u64 = 0;

        const quot = dim_in / dim_out;
        const scale: f32 = 1.0 / (@as(comptime_float, @floatFromInt(dim_out)));

        field: usize = 1,

        const mask_first_indices = blk: {
            var first_indices: [dim_out]usize = undefined;
            for (&first_indices, 0..) |*first_index, k| {
                const first = k * quot;
                const first_mask_index = first / 64;
                first_index.* = first_mask_index;
            }
            break :blk first_indices;
        };

        const mask_last_indices = blk: {
            var last_indices: [dim_out]usize = undefined;
            for (&last_indices, 0..) |*last_index, k| {
                const first = k * quot;
                const last = first + quot - 1;
                const last_mask_index = last / 64;
                last_index.* = last_mask_index;
            }
            break :blk last_indices;
        };
        const mask_next_indices = blk: {
            var last_indices: [dim_out]usize = undefined;
            for (&last_indices, 0..) |*last_index, k| {
                const first = k * quot;
                const last = first + quot - 1;
                const last_mask_index = last / 64;
                last_index.* = last_mask_index + 1;
            }
            break :blk last_indices;
        };
        const masks = blk: {
            var nmasks = 0;
            for (0..dim_out) |k| {
                const first = k * quot;
                const last = first + quot - 1;
                const first_mask_index = first / 64;
                const last_mask_index = last / 64;
                nmasks += last_mask_index-first_mask_index + 1;
            }

            var mask_array: [nmasks]usize = undefined;
            var count = 0;
            for (0..dim_out) |k| {
                const first = k * quot;
                const last = first + quot - 1;
                const first_mask_index = first / 64;
                const last_mask_index = last / 64;
                const first_index : u8 = @truncate(first % 64);// % 64;
                const last_index : u8 = @truncate(last % 64);
                for (first_mask_index..last_mask_index+1) |i| {
                    if(i==first_mask_index and i==last_mask_index){
                        if(first_index == 63){
                            mask_array[count] = 0x8000000000000000;
                        }
                        else if(last_index == 0){
                            mask_array[count] = 0x0000000000000001;
                        }
                        else{
                            mask_array[count] = ~@as(usize, ((1 << (first_index )) - 1)) & ((1 << (last_index + 1)) - 1);
                        }
                    }
                    else if(i==first_mask_index){
                        if(first_index == 0){
                            mask_array[count] = 0xFFFFFFFFFFFFFFFF;
                        }
                        else if(first_index == 63){
                            mask_array[count] = 0x8000000000000000;
                        }
                        else{
                            mask_array[count] = ~@as(usize, ((1 << (first_index)) - 1));
                        }
                    }
                    else if(i==last_mask_index){
                        if(last_index == 0){
                            mask_array[count] = 0x0000000000000001;
                        }
                        else if(last_index == 63){
                            mask_array[count] = 0xFFFFFFFFFFFFFFFF;
                        }
                        else{
                            mask_array[count] = @as(usize, ((1 << (last_index + 1)) - 1));
                        }
                    }
                    else{
                        mask_array[count] = 0xFFFFFFFFFFFFFFFF;
                    }
                    count += 1;
                }
            }
            break :blk mask_array;
        };

        pub fn eval(_: *Self, noalias input: *const Input, noalias output: *Output) void {
            @memset(output, 0);
            var count : usize = 0;
            for (output, 0..) |*coord, k| {
                const from = k * quot;
                const to = from + quot;

                if (options.gateRepresentation == .bitset) {
                    //for (from..to) |l| coord.* += if (input.isSet(l)) 1 else 0;
                    for (mask_first_indices[k]..mask_next_indices[k]) |l| {
                        coord.* += @popCount(input.masks[l] & masks[count]);
                        count += 1;
                    }
                    
                } else {
                    for (from..to) |l| coord.* += if (input[l]) 1 else 0;
                }
            }
        }
    };
}
