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

/// A read-write double buffer consists of two equally sized memory regions:
/// a front and back. The intention of this data structure is to statically
/// allocate transient memory regions for functions to be passed input in the
/// front half and pass output to the back half. One acquires a pointer to each
/// respective regions with the front(T) and back(t) commands, where T is the
/// type to case the pointer to the memory region to. The front is read only,
/// i.e. front(T) has return type "*const T" and back(T) returns a pointer to
/// variable memory. After the back of the buffer is filled one calls flip()
/// to switch the two memory regions. Note that no memory is moved and one can
/// safely read from the front until the next flip() call.
pub fn DoubleBuffer(size: usize, alignment: usize) type {
    if (alignment == 0) @compileError("DoubleBuffer: invalid alignment equal to 0.");
    return struct {
        const Self = @This();

        // Ensure that the back half has the same alignment, this is guaranteed
        // if the actual size is divisible by the alignment.
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
    const message_prefix = "Aesia layer:";
    inline for ([_]typeCheck{
        mustDeclareAs("ItemIn", &.{.isConst}, message_prefix),
        mustDeclareAs("ItemOut", &.{.isConst}, message_prefix),
        mustDeclareAs("dim_in", &.{ .isConst, .isPositiveInt }, message_prefix),
        mustDeclareAs("dim_out", &.{ .isConst, .isPositiveInt }, message_prefix),
    }) |constrain| constrain(Layer);

    switch (tag(Layer)) {
        .namespace => {},
        .statefull => {},
        .trainable => {
            const trainable_prefix = "Aesia trainable layer:";
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

/// A feedforward network of compile time known layer types.
/// A layer is a type interface, it needs to declare:
///     ItemIn  : type of input array items,
///     ItemOut : type of output array items,
///     dim_out : length of input array,
///     dim_in  : length of output array.
/// Optionally a layer may declare
///     parameter_count : a positive integer declaring how many trainable parameter the layer has.
/// Layer's are separated into three distinct tags specified by the conditions:
///     trainable : declares parameter_count,
///     statefull : does not declare parameter_count and @sizeOf(Layer) > 0,
///     namespace : @sizeOf(Layer) == 0.
/// For efficient SIMD utilization trainable layers need to declare
///     parameter_alignment : alignment of its given parameter array.
/// They also need to declare the methods
///     takeParameters(*@This(), *[parameter_count]f32),
///     giveParameters(*This()),
/// which takes/gives parameters pre/postprocessing them if necessary.
/// Every layer needs to declare the following methods with signatures:
///     eval([*@This()], *const [DimIn]ItemIn, *[DimOut]ItemOut) void,
///     forwardPass([*@This()], *const [DimIn]ItemIn, *[DimOut]ItemOut) void,
///     backwardPass([*@This()], *const [DimOut]f32, *[DimIn]f32, [*[parameter_count]f32]) void,
/// where *@This() is only passed for non namespace layers and only trainable layers act on the
/// gradient of its parameters in backwardPass.
pub fn Network(Layers: []const type) type {
    inline for (Layers) |Layer| check(Layer);
    inline for (Layers[0 .. Layers.len - 1], Layers[1..]) |prev, next| {
        if (prev.ItemOut != next.ItemIn) printCompileError(
            "Aesia layers: {s} and {s} have mismatched input/output item types",
            .{ @typeName(prev), @typeName(next) },
        );
        if (prev.dim_out != next.dim_in) printCompileError(
            "Aesia layers: {s} and {s} have mismatched input/output dimension",
            .{ @typeName(prev), @typeName(next) },
        );
    }

    return struct {
        const Self = @This();

        // The following parameter computations are necessary if the layer uses
        // SIMD vectors. For example @Vector(16, f32) has a natural alignment
        // of 64 *not* 4. So if one layer uses 13 parameters and the next relies
        // on SIMD computations, then the alignment is screwed up. We simply pad
        // the parameters inbetween with some extra unused parameters.
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
                // We branch here so that parameterless layers do not need to
                // declare all parameter info.
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

        pub const LayerLast = Layers[Layers.len - 1];
        pub const LayerFirst = Layers[0];
        pub const dim_in = LayerFirst.dim_in;
        pub const dim_out = LayerLast.dim_out;
        pub const ItemIn = LayerFirst.ItemIn;
        pub const ItemOut = LayerLast.ItemOut;

        pub const Input = [dim_in]ItemIn;
        pub const Output = [dim_out]ItemOut;

        const buffer_alignment = blk: {
            var max: usize = 0;
            for (Layers) |Layer| max = @max(max, @alignOf(LayerOutput(Layer)));
            break :blk max;
        };
        const buffer_size = blk: {
            var max: usize = 0;
            // Fixme: @sizeOf(LayerInput(Layer)) is only required because we do not have a
            // backwardPassFinal function.
            for (Layers) |Layer| max = @max(max, @max(
                @sizeOf(LayerInput(Layer)),
                @sizeOf(LayerOutput(Layer)),
            ));
            break :blk max;
        };

        layers: std.meta.Tuple(Layers),
        /// A network is evaluated layer by layer, either forwards or backwards.
        /// In either case one only needs to store the result of the previous
        /// layer's computation and pass it to the next. A double buffer facilitates this.
        buffer: DoubleBuffer(buffer_size, buffer_alignment),

        fn LayerInput(Layer: type) type {
            return [Layer.dim_in]Layer.ItemIn;
        }
        fn LayerOutput(Layer: type) type {
            return [Layer.dim_out]Layer.ItemOut;
        }

        pub fn writeToFile(parameters: *[parameter_count]f32, path: []const u8) !void {
            const file = try std.fs.cwd().createFile(path, .{});
            defer file.close();

            var buffered = std.io.bufferedWriter(file.writer());
            var writer = buffered.writer();
            inline for (parameter_ranges) |range| {
                if (range.len > 0) {
                    const slice = parameters[range.from..range.to()];
                    try writer.writeAll(std.mem.sliceAsBytes(slice));
                }
            }
        }

        pub fn readFromFile(parameters: *[parameter_count]f32, path: []const u8) !void {
            const file = try std.fs.cwd().openFile(path, .{});
            defer file.close();

            var buffered = std.io.bufferedReader(file.reader());
            var reader = buffered.reader();
            inline for (parameter_ranges) |range| {
                if (range.len > 0) {
                    const slice = parameters[range.from..range.to()];
                    _ = try reader.readAll(std.mem.sliceAsBytes(slice));
                }
            }
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

        /// The network does *not* always own its parameters.
        /// The design of this mechanism has three reasons:
        /// 1. Enabling amortization of expensive operations over the number of evaluations
        ///    inbetween. For example logic layers need to compute softmax of its parameters,
        ///    when training this cost is amortized over the batch size.
        /// 2. A model can ultimately own the parameters in a big flat array. This makes
        ///    the step part of gradient descent trivial to implement.
        /// 3. Each thread can own a separate network that is used for parallel computations,
        ///    while sharing the paremeters with the other threads. The main thread owns the
        ///    parameters and should call (take|give)Parameters, the worker threads instead
        ///    call (borrow|return)Parameters, but only *after* the main thread has succesfully
        ///    taken/given them.
        /// Takes ownership of an array of parameters, preprocessing them, if necessary.
        pub fn takeParameters(self: *Self, parameters: *[parameter_count]f32) void {
            inline for (&self.layers, parameter_ranges) |*layer, range| {
                const slice = range.slice(f32, parameters);
                if (range.len > 0) layer.takeParameters(@alignCast(@ptrCast(slice)));
            }
        }

        /// Gives the parameters back, postprocessing them, if necessary.
        pub fn giveParameters(self: *Self) void {
            inline for (Layers, &self.layers) |Layer, *layer| {
                if (comptime tag(Layer) == .trainable) layer.giveParameters();
            }
        }

        pub fn init(self: *Self, parameters: *[parameter_count]f32) void {
            inline for (Layers, &self.layers, parameter_ranges) |Layer, *layer, range| {
                if (@sizeOf(Layer) == 0) continue;
                if (range.len > 0) {
                    const slice: *[range.len]f32 = @alignCast(
                        @ptrCast(parameters[range.from..range.to()]),
                    );
                    layer.init(@alignCast(@ptrCast(slice)));
                } else {
                    layer.init();
                }
            }
        }

        /// Evaluates the network and caches relevant data it needs for the backward pass.
        pub fn forwardPass(self: *Self, input: *const Input) *const Output {
            const buffer = &self.buffer;
            inline for (Layers, &self.layers, 0..) |Layer, *layer, l| {
                const layer_input = if (l == 0) input else buffer.front(LayerInput(Layer));
                const layer_output = buffer.back(LayerOutput(Layer));
                if (@sizeOf(Layer) > 0) {
                    layer.forwardPass(@ptrCast(layer_input), @ptrCast(layer_output));
                } else {
                    Layer.forwardPass(@ptrCast(layer_input), @ptrCast(layer_output));
                }
                buffer.flip();
            }
            return buffer.front(Output);
        }

        /// Accumulates the given gradient backwards, layer by layer. Every
        /// layer passes its delta, the derivative of the loss function with
        /// respect to its activations, backwards through the buffer. This is
        /// called backpropagation. The caller is responsible for filling the
        /// network's buffer with the delta for the last layer. A pointer to the
        /// correct memory region is given by lastDeltaBuffer().
        pub fn backwardPass(self: *Self, gradient: *[parameter_count]f32) void {
            const buffer = &self.buffer;
            buffer.flip();
            comptime var l: usize = Layers.len;
            inline while (l > 0) {
                l -= 1;
                const Layer = Layers[l];
                const layer = &self.layers[l];
                const input = buffer.front([Layer.dim_out]f32);
                const output = buffer.back([Layer.dim_in]f32);
                const range = parameter_ranges[l];
                if (@sizeOf(Layer) == 0) {
                    Layer.backwardPass(@ptrCast(input), @ptrCast(output));
                } else if (range.len == 0) {
                    layer.backwardPass(@ptrCast(input), @ptrCast(output));
                } else {
                    const gradient_slice = gradient[range.from..range.to()];
                    if (l == 0) {
                        layer.backwardPassFinal(
                            @ptrCast(input),
                            @alignCast(@ptrCast(gradient_slice)),
                        );
                    } else {
                        layer.backwardPass(
                            @ptrCast(input),
                            @alignCast(@ptrCast(gradient_slice)),
                            @ptrCast(output),
                        );
                    }
                }
                buffer.flip();
            }
        }

        /// Returns the correct memory region to put the delta of the last layer
        /// before calling backwardPass.
        pub fn lastDeltaBuffer(self: *Self) *[dim_out]f32 {
            return self.buffer.back([dim_out]f32);
        }
    };
}
