const std = @import("std");
const meta = std.meta;
const assert = std.debug.assert;

const aesia = @import("aesia.zig");

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

fn printCompileError(comptime fmt: []const u8, args: anytype) void {
    @compileError(std.fmt.comptimePrint(fmt, args));
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
    @setEvalBranchQuota(1000 * Layers.len);
    inline for (Layers) |Layer| aesia.layer.check(Layer);
    inline for (Layers[0 .. Layers.len - 1], Layers[1..]) |prev, next| {
        if (prev.info.ItemOut != next.info.ItemIn) printCompileError(
            "Aesia: mismatched input/output item types in layers:\n{s}\n{s}",
            .{ @typeName(prev), @typeName(next) },
        );
        if (prev.info.dim_out != next.info.dim_in) printCompileError(
            "Aesia: mismatched input/output dimension in layers:\n{s}\n{s}",
            .{ @typeName(prev), @typeName(next) },
        );
    }
    // If every layer is flagged as in-place, then the network does *not* need a double-buffer
    // to pass input/output.
    const in_place = for (Layers) |Layer| if (!Layer.info.in_place) break false else true;

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
                if (Layer.info.trainable) max = @max(max, Layer.info.parameter_alignment.?);
            }
            break :blk max;
        };
        const parameter_ranges = blk: {
            var offset: usize = 0;
            var ranges: [Layers.len]Range = undefined;
            for (&ranges, Layers) |*range, Layer| {
                if (!Layer.info.trainable) {
                    range.* = Range{ .from = offset, .len = 0 };
                } else {
                    // In this branch we need to pad with some f32.
                    const alignment = Layer.info.parameter_alignment.?;
                    assert(alignment % @alignOf(f32) == 0);
                    // Fixme: Compute directly instead of this stupid loop.
                    while ((@alignOf(f32) * offset) % alignment != 0) {
                        offset += 1;
                    }
                    range.* = Range{ .from = offset, .len = Layer.info.parameter_count.? };
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

        // Index of the first trainable layer, used to call backwardPassFinal instead
        // of backwardPass, for efficiency.
        const first_trainable_index = for (Layers, 0..) |Layer, i| {
            if (Layer.info.trainable) break i;
        } else std.math.maxInt(usize);

        pub const LayerLast = Layers[Layers.len - 1];
        pub const LayerFirst = Layers[0];
        pub const dim_in = LayerFirst.info.dim_in;
        pub const dim_out = LayerLast.info.dim_out;
        pub const ItemIn = LayerFirst.info.ItemIn;
        pub const ItemOut = LayerLast.info.ItemOut;

        pub const Input = [dim_in]ItemIn;
        pub const Output = [dim_out]ItemOut;

        const buffer_alignment = blk: {
            var max: usize = 0;
            for (Layers) |Layer| max = @max(max, @alignOf(Layer.info.Output()));
            break :blk max;
        };
        const buffer_size = blk: {
            var max: usize = 0;
            // Fixme: @sizeOf(LayerInput(Layer)) is only required because we do not have a
            // backwardPassFinal function.
            // Fixme: Ensure we can modify this now.
            for (Layers) |Layer| max = @max(max, @max(
                @sizeOf(Layer.info.Input()),
                @sizeOf(Layer.info.Output()),
            ));
            break :blk max;
        };

        layers: std.meta.Tuple(Layers),
        /// A network is evaluated layer by layer, either forwards or backwards.
        /// In either case one only needs to store the result of the previous
        /// layer's computation and pass it to the next. A double buffer facilitates this.
        buffer: if (in_place) void else DoubleBuffer(buffer_size, buffer_alignment),

        pub fn writeToFile(parameters: *[parameter_count]f32, path: []const u8) !void {
            @setEvalBranchQuota(1000 * Layers.len);
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
            @setEvalBranchQuota(1000 * Layers.len);
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
                const input_buffer = if (l == 0) input else buffer.front(Layer.info.Input());
                const output_buffer: *Layer.info.Output() = if (Layer.info.in_place)
                    @constCast(buffer.front(Layer.info.Output()))
                else
                    buffer.back(Layer.info.Output());
                if (Layer.info.statefull) {
                    layer.eval(@ptrCast(input_buffer), @ptrCast(output_buffer));
                } else {
                    Layer.eval(@ptrCast(input_buffer), @ptrCast(output_buffer));
                }
                if (!Layer.info.in_place) buffer.flip();
            }
            return buffer.front(Output);
        }

        /// Evaluates the network using the validationEval functions, layer by layer.
        pub fn validationEval(self: *Self, input: *const Input) *const Output {
            @setEvalBranchQuota(1000 * Layers.len);
            const buffer = &self.buffer;
            inline for (Layers, &self.layers, 0..) |Layer, *layer, l| {
                const input_buffer = if (l == 0) input else buffer.front(Layer.info.Input());
                const output_buffer: *Layer.info.Output() = if (Layer.info.in_place)
                    @constCast(buffer.front(Layer.info.Output()))
                else
                    buffer.back(Layer.info.Output());
                const evalFunction = if (std.meta.hasFn(Layer, "validationEval"))
                    Layer.validationEval
                else
                    Layer.eval;
                if (Layer.info.statefull) {
                    evalFunction(layer, @ptrCast(input_buffer), @ptrCast(output_buffer));
                } else {
                    evalFunction(@ptrCast(input_buffer), @ptrCast(output_buffer));
                }
                if (!Layer.info.in_place) buffer.flip();
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
                if (Layer.info.trainable) layer.giveParameters();
            }
        }

        pub fn init(self: *Self, parameters: *[parameter_count]f32) void {
            inline for (Layers, &self.layers, parameter_ranges) |Layer, *layer, range| {
                if (!Layer.info.statefull) continue;
                if (Layer.info.trainable) {
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
                const input_buffer = if (l == 0) input else buffer.front(Layer.info.Input());
                const output_buffer: *Layer.info.Output() = if (comptime Layer.info.in_place)
                    @constCast(buffer.front(Layer.info.Output()))
                else
                    buffer.back(Layer.info.Output());
                if (Layer.info.statefull) {
                    layer.forwardPass(@ptrCast(input_buffer), @ptrCast(output_buffer));
                } else {
                    Layer.forwardPass(@ptrCast(input_buffer), @ptrCast(output_buffer));
                }
                if (comptime !Layer.info.in_place) buffer.flip();
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
            // Fixme: throw compile error if first_trainable_index is intMax(usize).
            const buffer = &self.buffer;
            buffer.flip();
            comptime var l: usize = Layers.len;
            inline while (l > first_trainable_index) {
                l -= 1;
                const Layer = Layers[l];
                const layer = &self.layers[l];
                const input = buffer.front([Layer.info.dim_out]f32);
                const output: *[Layer.info.dim_in]f32 = if (Layer.info.in_place)
                    @constCast(buffer.front([Layer.info.dim_in]f32))
                else
                    buffer.back([Layer.info.dim_in]f32);
                const range = parameter_ranges[l];
                // Todo: Simplify branches, for example, the second clause in the conjunction
                // is not needed here since we changed the loop condition.
                if (!Layer.info.statefull and l > first_trainable_index) {
                    Layer.backwardPass(@ptrCast(input), @ptrCast(output));
                } else if (!Layer.info.trainable and l > first_trainable_index) {
                    layer.backwardPass(@ptrCast(input), @ptrCast(output));
                } else if (Layer.info.trainable) {
                    const gradient_slice = gradient[range.from..range.to()];
                    if (l == first_trainable_index) {
                        layer.backwardPassFinal(
                            @alignCast(@ptrCast(input)),
                            @alignCast(@ptrCast(gradient_slice)),
                        );
                    } else {
                        layer.backwardPass(
                            @alignCast(@ptrCast(input)),
                            @alignCast(@ptrCast(gradient_slice)),
                            @ptrCast(output),
                        );
                    }
                }
                if (!Layer.info.in_place) buffer.flip();
            }
        }

        /// Returns the correct memory region to put the delta of the last layer
        /// before calling backwardPass.
        pub fn lastDeltaBuffer(self: *Self) *[dim_out]f32 {
            return self.buffer.back([dim_out]f32);
        }
    };
}
