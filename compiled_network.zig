const std = @import("std");
const meta = std.meta;
const assert = std.debug.assert;

/// Ranges are stored in this manner to make the illegal ranges (where to < from) unrepresentable.
const Range = struct {
    from: usize,
    len: usize,

    pub fn to(range: Range) usize {
        return range.from + range.len;
    }

    pub fn slice(comptime range: Range, T: type, ptr: anytype) *[range.len]T {
        return @ptrCast(ptr[range.from..range.to()]);
    }
};

/// A read-write double buffer consists of two equally sized memory regions: a front and back.
/// The intention of this data structure is to statically allocate transient memory regions for functions
/// to be passed input in the front half and pass output to the back half.
/// One acquires a pointer to each respective regions with the front(T) and back(t) commands,
/// where T is the type to case the pointer to the memory region to. The front is read only,
/// i.e. front(T) has return type "*const T" and back(T) returns a pointer to variable memory.
/// After the back of the buffer is filled one calls flip() to switch the two memory regions. Note that
/// no memory is moved and one can safely read from the front until the next flip() call.
pub fn DoubleBuffer(size: usize, alignment: usize) type {
    if (alignment == 0) @compileError("DoubleBuffer: invalid alignment equal to 0.");
    return struct {
        const Self = @This();

        // Ensure that the back half has the same alignment, this is guaranteed if the actual size is divisible
        // by the alignment.
        const actual_size = blk: {
            if (alignment >= size) break :blk alignment;
            const padding = (size - alignment) % alignment;
            break :blk size + padding;
        };
        data: [2 * actual_size]u8 align(alignment),
        /// Encodes which half currently is the front, 0: first half, 1: second half.
        half: u1,

        pub const default = Self{ .data = @splat(0), .half = 0 };

        /// Returns the front of the double buffer, a read only address of the given type.
        /// Asserts that the given type can be represented by the double buffer.
        pub fn front(buffer: *const Self, T: type) *align(alignment) const T {
            assert(@sizeOf(T) <= size);
            assert(@alignOf(T) <= alignment);
            const offset = buffer.half * actual_size;
            return @alignCast(@ptrCast(buffer.data[offset..]));
        }

        /// Returns the back of the double buffer, a writable address of the given type.
        /// Asserts that the given type can be represented by the double buffer.
        pub fn back(buffer: *Self, T: type) *align(alignment) T {
            assert(@sizeOf(T) <= size);
            assert(@alignOf(T) <= alignment);
            const offset = (buffer.half +% 1) * actual_size;
            return @alignCast(@ptrCast(buffer.data[offset..]));
        }

        /// Flips the double buffer, switching the front and back. No memory is moved and
        /// pointers to the front are valid until the next call to flip().
        pub fn flip(buffer: *Self) void {
            buffer.half +%= 1;
        }
    };
}

/// A feedforward network of compile time known layer types.
/// A layer is a type interface, it needs to declare:
///     Input      : input type
///     Output     : output type
///     dim_in  : the dimension of the input
///     dim_out : the dimension of the output
/// Every layer must declare the following methods:
///     eval
///     forwardPass
///     backwardPass
/// Optionally, the layer can make use of parameters, which have to be of type f32.
/// To utilize parameters, the following must be declared:
///     parameter_count : the number of f32 parameters the layer will be allocated
///     parameter_alignment : the alignment of the layer's parameters
/// as well as the methods:
///     takeParameters   : take ownership of the parameters, preprocessing them, if necessary
///     giveParameters   : give back ownership of the parameters, postprocessing them, if necessary
///     borrowParameters : store the pointer to the parameters, this is called by worker threads after
///                        the main thread has called takeParameters
///     returnParameters : release the pointer to the paramters, this is called by worked threads after
///                        the main thread has called giveParameters
///     backwardPassLast : backwardPass without passing the delta backwards, see below

const Tag = enum {
    namespace,
    statefull,
    trainable,
};

fn tag(Layer: type) Tag {
    if (@sizeOf(Layer) == 0) {
        return .namespace;
    } else if (!@hasDecl(Layer, "parameter_count")) {
        return .statefull;
    } else {
        return .trainable;
    }
}
fn printCompileError(comptime fmt: []const u8, args: anytype) void {
    @compileError(std.fmt.comptimePrint(fmt, args));
}

const DeclarationConstraint = enum {
    isConst,
    isPositiveInt,
};


fn mustNotDeclare(T: type, decl_name: []const u8, message_prefix: []const u8) void {
    if (@hasDecl(T, decl_name)) printCompileError(
        "{s} {s} must not declare '{s}'",
        .{ message_prefix, @typeName(T), decl_name },
    );
}
fn mustDeclareAs(
    decl_name: []const u8,
    constraints: []const DeclarationConstraint,
    message_prefix: []const u8,
) fn (type) void {
    return struct {
        pub fn constrain(T: type) void {
            const type_name = @typeName(T);

            if (!@hasDecl(T, decl_name)) printCompileError(
                "{s} {s} must declare '{s}'",
                .{ message_prefix, type_name, decl_name },
            );

            for (constraints) |constraint| switch (constraint) {
                .isConst => {
                    const decl_ptr = &@field(T, decl_name);
                    if (!@typeInfo(@TypeOf(decl_ptr)).pointer.is_const) printCompileError(
                        "{s} {s} must declare '{s}' as const",
                        .{ message_prefix, type_name, decl_name },
                    );
                },
                .isPositiveInt => {
                    const decl_value = @field(T, decl_name);
                    const decl_type = @TypeOf(decl_value);
                    const info = @typeInfo(decl_type);
                    const condition = (decl_type == comptime_int or info == .int) and
                        decl_value > 0;
                    if (!condition) printCompileError(
                        "{s} {s} must declare '{s}' as a positive integer",
                        .{ message_prefix, type_name, decl_name },
                    );
                },
            };
        }
    }.constrain;
}


/// Compile time checks for the layer interface.
fn check(Layer: type) void {
    const typeCheck = *const fn (type) void;
    const message_prefix = "Compiled layer:";
    @setEvalBranchQuota(4000);
    inline for ([_]typeCheck{
        mustDeclareAs("ItemIn", &.{.isConst}, message_prefix),
        mustDeclareAs("ItemOut", &.{.isConst}, message_prefix),
        mustDeclareAs("dim_in", &.{ .isConst, .isPositiveInt }, message_prefix),
        mustDeclareAs("dim_out", &.{ .isConst, .isPositiveInt }, message_prefix),
        mustDeclareAs("Input", &.{ .isConst}, message_prefix),
        mustDeclareAs("Output", &.{ .isConst}, message_prefix),
    }) |constrain| constrain(Layer);

    switch (tag(Layer)) {
        .namespace => {},
        .statefull => {},
        .trainable => {
            const trainable_prefix = "Compiled trainable layer:";
            inline for ([_]typeCheck{
                mustDeclareAs(
                    "parameter_count",
                    &.{ .isConst, .isPositiveInt },
                    trainable_prefix,
                ),
                mustDeclareAs(
                    "parameter_alignment",
                    &.{ .isConst, .isPositiveInt },
                    trainable_prefix,
                ),
            }) |constrain| constrain(Layer);
        },
    }
}

pub fn Network(Layers: []const type) type {
    inline for (Layers) |Layer| check(Layer);
    inline for (Layers[0 .. Layers.len - 1], Layers[1..]) |prev, next| {
        if (prev.ItemOut != next.ItemIn) printCompileError(
            "layers {s} and {s} have mismatched input/output item types",
            .{ @typeName(prev), @typeName(next) },
        );
        if (prev.dim_out != next.dim_in) printCompileError(
            "layers {s} and {s} have mismatched input/output dimension",
            .{ @typeName(prev), @typeName(next) },
        );
    }
    return struct {
        const Self = @This();

        // Fixme: Compile time sanity check the Layers.

        // The following parameter computations are necessary if the layer uses SIMD vectors.
        // For example @Vector(16, f32) has a natural alignment of 64 *not* 4. So if one layer
        // uses 13 parameters and the next relies on SIMD computations, then the alignment is screwed up.
        // We simply pad the parameters inbetween with some extra unused parameters.
        pub const parameter_alignment = blk: {
            var max: usize = 0;
            for (Layers) |Layer| {
                if (tag(Layer) == .trainable) max = @max(max, Layer.parameter_alignment);
            }
            break :blk max;
        };
        const parameter_ranges = blk: {
            var offset: usize = 0;
            var ranges: [Layers.len]Range = undefined;
            for (&ranges, Layers) |*range, Layer| {
                // We branch here so that parameterless layers do not need to declare all parameter info.
                if (tag(Layer) != .trainable) {
                    range.* = Range{ .from = offset, .len = 0 };
                } else {
                    // In this branch we need to pad with some f32.
                    const alignment = Layer.parameter_alignment;
                    assert(alignment % @alignOf(f32) == 0);
                    // Fixme: Compute directly instead of this stupid loop.
                    while ((@alignOf(f32) * offset) % alignment != 0) {
                        offset += 1;
                    }
                    range.* = Range{ .from = offset, .len = Layer.parameter_count };
                    offset += range.len;
                }
            }
            break :blk ranges;
        };
        // Parameter count, padding included, overallocate for SIMD operations.
        pub const parameter_count = blk: {
            var len: usize = parameter_ranges[Layers.len - 1].to();
            const vector_len = std.simd.suggestVectorLength(f32) orelse 1;
            while (len % vector_len != 0) len += 1;
            break :blk len;
        };

        pub const LastLayer = Layers[Layers.len - 1];
        pub const FirstLayer = Layers[0];
        pub const dim_in = FirstLayer.dim_in;
        pub const dim_out = LastLayer.dim_out;
        pub const ItemIn = FirstLayer.ItemIn;
        pub const ItemOut = LastLayer.ItemOut;
        //pub const Input = FirstLayer.Input;
        //pub const Output = LastLayer.Output;
        pub const Input = [dim_in]ItemIn;
        pub const Output = [dim_out]ItemOut;

        const buffer_alignment = blk: {
            var max: usize = 0;
            for (Layers) |Layer| max = @max(max, @alignOf(Layer.Output));
            break :blk max;
        };
        const buffer_size = blk: {
            var max: usize = 0;
            // Fixme: @sizeOf(Layer.Input) is only required because we do not have a
            // backwardPassFinal function.
            for (Layers) |Layer| max = @max(max, @max(@sizeOf(Layer.Input), @sizeOf(Layer.Output)));
            break :blk max;
        };

        layers: std.meta.Tuple(Layers),
        /// A network is evaluated layer by layer, either forwards or backwards.
        /// In either case one only needs to store the result of the previous
        /// layer's computation and pass it to the next. A double buffer facilitates this.
        buffer: DoubleBuffer(buffer_size, buffer_alignment),

        pub const default = Self{
            .layers = blk: {
                var result: std.meta.Tuple(Layers) = undefined;
                for (Layers, &result) |Layer, *entry| {
                    if (@sizeOf(Layer) > 0) entry.* = Layer.default;
                }
                break :blk result;
            },
            .buffer = .default,
        };

        fn LayerInput(Layer: type) type {
            return [Layer.dim_in]Layer.ItemIn;
        }
        fn LayerOutput(Layer: type) type {
            return [Layer.dim_out]Layer.ItemOut;
        }

        /// Evaluates the network, layer by layer.
        pub fn eval(self: *Self, input: *const Input) *const Output {
            const buffer = &self.buffer;
            inline for (Layers, &self.layers, 0..) |Layer, *layer, l| {
                const layer_input = if (l == 0) input else buffer.front(LayerInput(Layer));
                const layer_output = buffer.back(LayerOutput(Layer));
                if (comptime tag(Layer) == .namespace) {
                    Layer.eval(@ptrCast(layer_input), @ptrCast(layer_output));
                } else {
                    layer.eval(@ptrCast(layer_input), @ptrCast(layer_output));
                }
                buffer.flip();
            }
            return buffer.front(Output);
        }

        const parameter_count_no_padding = blk: {
            var count: usize = 0;
            for (parameter_ranges) |range| count += range.len;
            break :blk count;
        };
        pub fn compileFromFile(self: *Self, path: []const u8) !void {
            const file = try std.fs.cwd().openFile(path, .{});
            defer file.close();

            var buffered = std.io.bufferedReader(file.reader());
            var reader = buffered.reader();

            const allocator = std.heap.page_allocator;
            const bytes = try reader.readAllAlloc(allocator, @sizeOf([parameter_count_no_padding]f32));
            defer allocator.free(bytes);
            const parameters = std.mem.bytesAsSlice(f32, bytes);
            inline for (&self.layers, parameter_ranges) |*layer, range| {
                if (range.len > 0) {
                    const slice = parameters[range.from..range.to()];
                    layer.compile(@alignCast(@ptrCast(slice)));
                }
            }
        }
    };
}
