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
///     input_dim  : the dimension of the input
///     output_dim : the dimension of the output
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
    stateless,
    statefull,
    trainable,
};

fn tag(comptime Layer: type) Tag {
    if (@sizeOf(Layer) == 0) {
        return .stateless;
    } else if (!@hasDecl(Layer, "parameter_count")) {
        return .statefull;
    } else {
        return .trainable;
    }
}

fn printCompileError(comptime fmt: []const u8, args: anytype) void {
    @compileError(std.fmt.comptimePrint(fmt, args));
}

const ConstrainedDeclaration = struct {
    name: []const u8,
    constraints: []fn ([]const u8, anytype) void,
};

fn isConst(prefix: []const u8, field_ptr: anytype) void {
    if (!@typeInfo(field_ptr).pointer.is_const) @compileError(prefix ++ " must be const");
}

const Decl = struct {
    name: []const u8,
    T: ?type,
};

fn requireDecl(Struct: type, decl: Decl, prefix: []const u8) void {
    if (!@hasDecl(Struct, decl.name)) printCompileError(
        "{s} '{s}' needs to declare '{s}'",
        .{ prefix, @typeName(Struct), decl.name },
    );
    if (decl.T != null and @TypeOf(@field(Struct, decl.name)) != decl.T.?) printCompileError(
        "{s} '{s}' must declare '{s}' to be of type '{s}'.",
        .{ prefix, @typeName(Struct), decl.name, @typeName(decl.T.?) },
    );
}

fn requireNotDecl(Struct: type, decl: Decl, prefix: []const u8) void {
    if (@hasDecl(Struct, decl.name)) printCompileError(
        "{s} '{s}' is not allowed to declare '{s}'",
        .{ prefix, @typeName(Struct), decl.name },
    );
}

/// Compile time checks for the layer interface.
fn check(Layer: type) void {
    const required_decls = [_]Decl{
        .{ .name = "input_dim", .T = usize },
        .{ .name = "output_dim", .T = usize },
        .{ .name = "Input", .T = null },
        .{ .name = "Output", .T = null },
    };
    const trainable_decls = [_]Decl{
        .{ .name = "parameter_count", .T = usize },
        .{ .name = "parameter_alignment", .T = usize },
    };
    inline for (required_decls) |decl| requireDecl(Layer, decl, "layer");
    switch (tag(Layer)) {
        .stateless => {
            inline for (trainable_decls) |decl| requireNotDecl(Layer, decl, "stateless layer");
        },
        .statefull => {
            inline for (trainable_decls) |decl| requireNotDecl(Layer, decl, "statefull layer");
        },
        .trainable => {
            inline for (trainable_decls) |decl| requireDecl(Layer, decl, "statefull layer");
        },
    }
}

pub fn Network(Layers: []const type) type {
    inline for (Layers[0 .. Layers.len - 1], Layers[1..]) |prev, next| {
        if (prev.Output != next.Input) @compileError("Layers " ++ @typeName(prev) ++ " and " ++
            @typeName(next) ++ " have non matching input & output types.");
    }
    inline for (Layers) |Layer| check(Layer);
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
        pub const input_dim = FirstLayer.input_dim;
        pub const output_dim = LastLayer.output_dim;
        pub const Input = FirstLayer.Input;
        pub const Output = LastLayer.Output;

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

        /// Evaluates the network, layer by layer.
        pub fn eval(self: *Self, input: *const Input) *const Output {
            const buffer = &self.buffer;
            inline for (Layers, &self.layers, 0..) |Layer, *layer, l| {
                const layer_input = if (l == 0) input else buffer.front(Layer.Input);
                const layer_output = buffer.back(Layer.Output);
                if (@sizeOf(Layer) > 0) {
                    layer.eval(layer_input, layer_output);
                } else {
                    Layer.eval(layer_input, layer_output);
                }
                buffer.flip();
            }
            return buffer.front(Output);
        }

        const parameter_count_no_padding = blk: {
            var count: usize = 0;
            inline for (parameter_ranges) |range| count += range.len;
            break :blk count;
        };
        pub fn compileFromFile(self: *Self, path: []const u8) !void {
            const file = try std.fs.openFile(path, .{});
            defer file.close();

            var buffered = std.io.bufferedReader(file.reader());
            var reader = buffered.reader();

            const allocator = std.heap.page_allocator;
            const bytes = try reader.readAllAlloc(allocator, @sizeOf([parameter_count_no_padding]f32));
            const parameters = std.mem.bytesAsSlice(f32, bytes);
            inline for (layers, parameter_ranges) |*layer, range| {
                if (range.len > 0) {
                    const slice = parameters[range.from..range.to()];
                    layer.compile(@alignCast(@ptrCast(slice)));
                }
            }
        }
    };
}
