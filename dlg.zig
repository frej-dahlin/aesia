const std = @import("std");
const meta = std.meta;
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;

pub const input_layer = @import("input.zig");
pub const output_layer = @import("output.zig");
pub const loss_function = @import("loss.zig");
pub const optim = @import("optim.zig");

pub const f32x16 = @Vector(16, f32);
pub const f32x8 = @Vector(8, f32);
pub const f32x2 = @Vector(2, f32);

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
/// i.e. front(T) returns has return type "*const T" and back(T) returns a pointer to variable memory.
/// After the back of buffer is filled one calls flip() to switch the two memory regions. Note that
/// no memory is moved and one can safely read from the front until the next flip() call.
pub fn DoubleBuffer(size: usize, alignment: usize) type {
    return struct {
        const Self = @This();

        // Ensure that the back half has the same alignment, this is guaranteed if the actual size is divisible
        // by the alignment.
        comptime {
            assert(size >= alignment);
        }
        const padding = (size - alignment) % alignment;
        const actual_size = size + padding;
        data: [2 * actual_size]u8 align(alignment),
        /// Encodes which half currently is the front, 0: first half, 1: second half.
        half: u1,

        pub const default = Self{ .data = @splat(0), .half = 0 };

        pub fn front(buffer: *const Self, T: type) *align(alignment) const T {
            assert(@sizeOf(T) <= size);
            assert(@alignOf(T) <= alignment);
            const offset = buffer.half * actual_size;
            return @alignCast(@ptrCast(buffer.data[offset..]));
        }

        pub fn back(buffer: *Self, T: type) *align(alignment) T {
            assert(@sizeOf(T) <= size);
            assert(@alignOf(T) <= alignment);
            const offset = (buffer.half +% 1) * actual_size;
            return @alignCast(@ptrCast(buffer.data[offset..]));
        }

        pub fn flip(buffer: *Self) void {
            buffer.half +%= 1;
        }
    };
}

/// Logits are normalized such that max(logits) == 0.
/// When the logits grow to big exp^logit will explode.
pub fn maxNormalize(v: f32x16) f32x16 {
    return v - @as(f32x16, @splat(@reduce(.Max, v)));
}

// Note: This is the hottest part of the code, I *think* that AVX-512 should handle it with gusto,
// but testing is required.
// Future note: No it is not! Computing the probabilities is amortized over every batch.
// So for decent size batches (say 32+) it is pretty much negligble.
pub fn softmax2(logit: f32x16) f32x16 {
    @setFloatMode(.optimized);
    const sigma = @exp2(maxNormalize(logit));
    const denom = @reduce(.Add, sigma);
    return sigma / @as(f32x16, @splat(denom));
}

pub fn softmax(logit: f32x16) f32x16 {
    @setFloatMode(.optimized);
    const sigma = @exp(maxNormalize(logit));
    const denom = @reduce(.Add, sigma);
    return sigma / @as(f32x16, @splat(denom));
}

pub fn softmaxInverse(sigma: f32x16) f32x16 {
    @setFloatMode(.optimized);
    return @log(sigma);
}

pub fn softmaxInverse2(sigma: f32x16) f32x16 {
    @setFloatMode(.optimized);
    return @log2(sigma);
}

// Computes (1 + x / 256)^256.
pub fn softmax_approx(logit: f32x16) f32x16 {
    @setFloatMode(.optimized);
    const scale = @as(f32x16, @splat(@as(f32, 1.0 / 256.0)));
    var ret = @as(f32x16, @splat(1)) + maxNormalize(logit) * scale;
    inline for (0..8) |_| ret *= ret;
    return ret / @as(f32x16, @splat(@reduce(.Add, ret)));
}

const Gate = struct {
    const count = 16;
    /// Returns a vector representing all values of a soft logic gate.
    /// Only intended for reference. The current implementation
    /// inlines this construction in forwardPass.
    pub fn vector(a: f32, b: f32) f32x16 {
        @setFloatMode(.optimized);
        return .{
            0, // false
            a * b, // and
            a - a * b, // a and not b
            a, // passthrough a
            b - a * b, // b and not a
            b, // passthrough b
            a + b - 2 * a * b, // xor
            a + b - a * b, // xnor
            // The following values are simply negation of the above ones in order.
            // Many authors reverse this order, i.e. true is the last gate,
            // however this leads to a less efficient SIMD construction of the vector.
            1,
            1 - a * b,
            1 - (a - a * b),
            1 - a,
            1 - (b - a * b),
            1 - b,
            1 - (a + b - 2 * a * b),
            1 - (a + b - a * b),
        };
    }

    /// Returns the soft gate vector differentiated by the first variable.
    /// Note that it only depends on the second variable.
    pub fn diff_a(b: f32) f32x16 {
        @setFloatMode(.optimized);
        return .{
            0,
            b,
            1 - b,
            1,
            -b,
            0,
            1 - 2 * b,
            0,
            0,
            -b,
            -(1 - b),
            -1,
            b,
            0,
            -(1 - 2 * b),
            0,
        };
    }

    /// Returns the soft gate vector differentiated by the second variable.
    /// Note that it only depends on the first variable.
    pub fn diff_b(a: f32) f32x16 {
        @setFloatMode(.optimized);
        return .{
            0,
            a,
            -a,
            0,
            1 - a,
            1,
            1 - 2 * a,
            1 - a,
            0,
            -a,
            a,
            0,
            -(1 - a),
            -1,
            -(1 - 2 * a),
            -(1 - a),
        };
    }
};

pub fn Network(Layers: []const type) type {
    inline for (Layers[0 .. Layers.len - 1], Layers[1..]) |prev, next| {
        if (prev.Output != next.Input) @compileError("Layers " ++ @typeName(prev) ++ " and " ++ @typeName(next) ++ " have non matching input & output types.");
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
                if (Layer.parameter_count > 0) max = @max(max, Layer.parameter_alignment);
            }
            break :blk max;
        };
        const parameter_ranges = blk: {
            var offset: usize = 0;
            var ranges: [Layers.len]Range = undefined;
            for (&ranges, Layers) |*range, Layer| {
                // We branch here so that parameterless layers do not need to declare all parameter info.
                if (Layer.parameter_count == 0) {
                    range.* = Range{ .from = offset, .len = 0 };
                    continue;
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
        const Input = Layers[0].Input;
        pub const input_dim = Layers[0].input_dim;
        const LastLayer = Layers[Layers.len - 1];
        pub const output_dim = LastLayer.output_dim;
        const Output = LastLayer.Output;
        const buffer_alignment = blk: {
            var max: usize = 0;
            for (Layers) |Layer| max = @max(max, @alignOf(Layer.Output));
            break :blk max;
        };
        const buffer_size = blk: {
            var max: usize = 0;
            for (Layers) |Layer| max = @max(max, @sizeOf(Layer.Output));
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
                layer.eval(layer_input, layer_output);
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
                if (Layer.parameter_count > 0) layer.giveParameters();
            }
        }

        /// Borrows parameters, without preprocessing, called by worker threads after the
        /// main thread has called takeParameters.
        pub fn borrowParameters(self: *Self, parameters: *[parameter_count]f32) void {
            inline for (self.layers, parameter_ranges) |*layer, range| {
                const slice = range.slice(f32, &parameters);
                if (range.len > 0) layer.borrowParameters(@alignCast(@ptrCast(slice)));
            }
        }

        /// Returns parameters, without postprocessing, called by worker threads before the
        /// main thread calls giveParameters.
        pub fn returnParameters(self: *Self) void {
            inline for (self.layers) |*layer| {
                if (layer.parameter_count > 0) layer.returnParameters();
            }
        }

        pub fn initParameters(_: *Self, parameters: *[parameter_count]f32) void {
            inline for (Layers, parameter_ranges) |Layer, range| {
                const slice = parameters[range.from..range.to()];
                if (range.len > 0) @memcpy(slice, &Layer.default_parameters);
            }
        }

        /// Evaluates the network and caches relevant data it needs for the backward pass.
        pub fn forwardPass(self: *Self, input: *const Input) *const Output {
            const buffer = &self.buffer;
            inline for (Layers, &self.layers, 0..) |Layer, *layer, l| {
                const layer_input = if (l == 0) input else buffer.front(Layer.Input);
                const layer_output = buffer.back(Layer.Output);
                layer.forwardPass(layer_input, layer_output);
                buffer.flip();
            }
            return buffer.front(Output);
        }

        /// Accumulates the given gradient backwards, layer by layer. Every layer passes its delta,
        /// the derivative of the loss function with respect to its activations, backwards through the buffer.
        /// This is called backpropagation.
        /// The caller is responsible for filling the network's buffer with the delta for the last layer.
        /// A pointer to the correct memory region is given by lastDeltaBuffer().
        pub fn backwardPass(self: *Self, gradient: *[parameter_count]f32) void {
            const buffer = &self.buffer;
            buffer.flip();
            comptime var l: usize = Layers.len;
            inline while (l > 0) {
                l -= 1;
                const Layer = Layers[l];
                const layer = &self.layers[l];
                const input = buffer.front([Layer.output_dim]f32);
                const output = buffer.back([Layer.input_dim]f32);
                const range = parameter_ranges[l];
                const gradient_slice = gradient[range.from..range.to()];
                layer.backwardPass(input, @alignCast(@ptrCast(gradient_slice)), output);
                buffer.flip();
            }
        }

        /// Returns the correct memory region to put the delta of the last layer before calling backwardPass.
        pub fn lastDeltaBuffer(self: *Self) *[LastLayer.output_dim]f32 {
            return self.buffer.back([LastLayer.output_dim]f32);
        }
    };
}

pub fn GroupSum(input_dim_: usize, output_dim_: usize) type {
    return struct {
        const Self = @This();
        pub const input_dim = input_dim_;
        pub const output_dim = output_dim_;
        pub const Input = [input_dim]f32;
        pub const Output = [output_dim]f32;
        pub const parameter_count = 0;

        const quot = input_dim / output_dim;
        const scale: f32 = 1.0 / (@as(comptime_float, @floatFromInt(output_dim)));

        field: usize = 1,

        pub const default = Self{
            .field = 1,
        };

        pub fn eval(_: *Self, input: *const Input, output: *Output) void {
            @memset(output, 0);
            for (output, 0..) |*coord, k| {
                const from = k * quot;
                const to = from + quot;
                for (input[from..to]) |softbit| coord.* += softbit;
                coord.* *= scale;
            }
        }

        pub fn forwardPass(self: *Self, input: *const Input, output: *Output) void {
            return eval(self, input, output);
        }

        pub fn backwardPass(_: *Self, input: *const [output_dim]f32, _: *[0]f32, output: *[input_dim]f32) void {
            for (input, 0..) |child, k| {
                const from = k * quot;
                const to = from + quot;
                for (output[from..to]) |*parent| parent.* = child * scale;
            }
        }
    };
}

pub const LogicOptions = struct {
    input_dim: usize,
    output_dim: usize,
    seed: u64,
};

pub fn LCG32(multiplier: u32, increment: u32, initial_seed: u32) type {
    return struct {
        const Self = @This();
        seed: u32,

        pub const default = Self{ .seed = initial_seed };

        pub inline fn next(self: *Self) u32 {
            self.seed *%= multiplier;
            self.seed +%= increment;
            return self.seed;
        }

        pub fn reset(self: *Self) void {
            self.seed = initial_seed;
        }

        pub fn init(seed_: u32) Self {
            return Self{
                .seed = seed_,
            };
        }
    };
}

pub fn Logic(options: LogicOptions) type {
    return struct {
        const Self = @This();
        pub const input_dim = options.input_dim;
        pub const output_dim = options.output_dim;
        pub const Input = [input_dim]f32;
        pub const Output = [output_dim]f32;
        const node_count = output_dim;
        pub const parameter_count = 16 * node_count;
        pub const parameter_alignment = 64;

        gradient: [node_count]f32x16 align(64),
        diff: [node_count][2]f32 align(64),
        /// The preprocessed parameters, computed by softmax(parameters).
        sigma: ?*[node_count]f32x16,
        /// Generates the random inputs on-the-fly.
        lcg: LCG32(1664525, 1013904223, options.seed),

        pub const default = Self{
            .gradient = @splat(@splat(0)),
            .diff = @splat(.{ 0, 0 }),
            .lcg = .init(options.seed),
            .sigma = null,
        };

        pub const default_parameters: [parameter_count]f32 =
            @bitCast(@as([node_count]f32x16, @splat(.{ 0, 0, 0, 8, 0, 8 } ++ .{0} ** 10)));

        pub fn eval(noalias self: *Self, noalias input: *const Input, noalias output: *Output) void {
            @setFloatMode(.optimized);
            assert(self.sigma != null);
            self.lcg.reset();
            for (self.sigma.?, output) |sigma, *activation| {
                const a = input[self.lcg.next() % input_dim];
                const b = input[self.lcg.next() % input_dim];
                activation.* = @reduce(.Add, sigma * Gate.vector(a, b));
            }
        }

        pub fn forwardPass(noalias self: *Self, noalias input: *const Input, noalias output: *Output) void {
            @setFloatMode(.optimized);
            assert(self.sigma != null);
            self.lcg.reset();
            // For some reason it is faster to to a simple loop rather than a multi item one.
            // I detected needless memcpy calls with callgrind.
            for (0..node_count) |j| {
                const a = input[self.lcg.next() % input_dim];
                const b = input[self.lcg.next() % input_dim];

                // Inline construction of Gate.vector(a, b), Gate.diff_a(b), and Gate.diff_b(a).
                // We do it in this way since all three depend on mix_coef, and two out of three
                // depend on a_coef/b_coef.
                const a_coef: f32x8 = .{ 0, 0, 1, 1, 0, 0, 1, 1 };
                const b_coef: f32x8 = .{ 0, 0, 0, 0, 1, 1, 1, 1 };
                const mix_coef: f32x8 = .{ 0, 1, -1, 0, -1, 0, -2, -1 };

                // All three desired vectors have halfway symmetry so we only
                // explicitly construct the first half.
                const diff_a_half = a_coef + mix_coef * @as(f32x8, @splat(b));
                const diff_b_half = b_coef + mix_coef * @as(f32x8, @splat(a));
                const gate_half =
                    a_coef * @as(f32x8, @splat(a)) +
                    b_coef * @as(f32x8, @splat(b)) +
                    mix_coef * @as(f32x8, @splat(a * b));
                const gate_vector = std.simd.join(gate_half, @as(f32x8, @splat(1)) - gate_half);

                const sigma = self.sigma.?[j];
                self.diff[j] = .{
                    @reduce(.Add, sigma * std.simd.join(diff_a_half, -diff_a_half)),
                    @reduce(.Add, sigma * std.simd.join(diff_b_half, -diff_b_half)),
                };
                output[j] = @reduce(.Add, sigma * gate_vector);
                self.gradient[j] = sigma * (gate_vector - @as(f32x16, @splat(output[j])));
            }
        }

        /// Fixme: Create a function "backward" that does not pass delta backwards, applicable for
        /// the first layer in a network only.
        pub fn backwardPass(noalias self: *Self, noalias input: *const [output_dim]f32, noalias cost_gradient: *[node_count]f32x16, noalias output: *[input_dim]f32) void {
            @setFloatMode(.optimized);
            self.lcg.reset();
            @memset(output, 0);
            for (0..node_count) |j| {
                const parent_a = self.lcg.next() % input_dim;
                const parent_b = self.lcg.next() % input_dim;
                cost_gradient[j] += self.gradient[j] * @as(f32x16, @splat(input[j]));
                const out = @as(f32x2, self.diff[j]) * @as(f32x2, @splat(input[j]));
                output[parent_a] += out[0];
                output[parent_b] += out[1];
            }
        }

        pub fn takeParameters(self: *Self, parameters: *[node_count]f32x16) void {
            @setFloatMode(.optimized);
            assert(self.sigma == null);
            self.sigma = parameters;
            for (self.sigma.?) |*sigma| sigma.* = softmax(sigma.*);
        }

        pub fn giveParameters(self: *Self) void {
            assert(self.sigma != null);
            for (self.sigma.?) |*sigma| sigma.* = softmaxInverse(sigma.*);
            self.sigma = null;
        }

        pub fn borrowParameters(self: *Self, parameters: *[node_count]f32x16) void {
            assert(self.sigma == null);
            self.sigma = parameters;
        }

        pub fn returnParameters(self: *Self) void {
            assert(self.sigma != null);
            self.sigma = null;
        }

        pub fn initParameters(_: *const Self, parameters: *[node_count]f32x16) void {
            // Initialize to be biased toward the two passthrough gates.
            @memset(parameters, [_]f32{ 0, 0, 0, 8, 0, 8 } ++ .{0} ** 10);
        }
    };
}

pub const ModelOptions = struct {
    Optimizer: ?fn (usize) type,
    Loss: ?type,

    pub const default = ModelOptions{
        .Optimizer = null,
        .Loss = null,
    };
};

// zig fmt: off
pub fn Model(Layers: []const type, options: ModelOptions) type {
    return struct {
        const Self = @This();

        const NetworkType = Network(Layers);
        pub const parameter_count = NetworkType.parameter_count;
        const vector_len = std.simd.suggestVectorLength(f32) orelse 1;
        comptime {
            assert(parameter_count % vector_len == 0);
        }
        const parameter_alignment = @max(
            NetworkType.parameter_alignment,
            @alignOf(@Vector(vector_len, f32))
        );

        pub const Feature = NetworkType.Input;
        pub const input_dim = NetworkType.input_dim;
        pub const Prediction = NetworkType.Output;
        pub const output_dim = NetworkType.output_dim;
        pub const Optimizer = (if (options.Optimizer) |O| O else optim.Adam(.default))(parameter_count);
        pub const Loss = if (options.Loss) |L| L else loss_function.HalvedMeanSquareError(output_dim);
        pub const Label = Loss.Label;

        pub const Dataset = struct {
            features: []const Feature,
            labels: []const Label,

            pub const empty = Dataset{ .features = &.{}, .labels = &.{} };

            pub fn len(dataset: Dataset) usize {
                assert(dataset.features.len == dataset.labels.len);
                return dataset.features.len;
            }

            pub fn init(features: []const Feature, labels: []const Label) Dataset {
                assert(features.len == labels.len);
                return .{
                    .features = features,
                    .labels = labels,
                };
            }

            pub fn slice(dataset: Dataset, from: usize, to: usize) Dataset {
                assert(from <= to);
                return .{
                    .features = dataset.features[from..to],
                    .labels = dataset.labels[from..to],
                };
            }
        };

        network: NetworkType,
        parameters: [parameter_count]f32 align(parameter_alignment),        
        gradient: [parameter_count]f32 align(parameter_alignment),
        optimizer: Optimizer,
        locked: bool,

        pub const default = Self{
            .network = .default,
            .parameters = @splat(0),
            .gradient = @splat(0),
            .optimizer = .default,
            .locked = false,
        };

        pub fn initParameters(self: *Self) void {
            self.network.initParameters(&self.parameters);
        }

        // Not thread safe unless you know the parameter_lock is true.
        pub fn eval(model: *Self, input: *const Feature) *const Prediction {
            assert(model.locked);
            return model.network.eval(input);
        }

        pub fn forwardPass(model: *Self, feature: *const Feature) *const Prediction {
            assert(model.locked);
            return model.network.forwardPass(feature);
        }

        pub fn backwardPass(model: *Self) void {
            assert(model.locked);
            return model.network.backwardPass(&model.gradient);
        }

        pub fn lock(model: *Self) void {
            assert(!model.locked);
            model.network.takeParameters(&model.parameters);
            model.locked = true;
        }

        pub fn unlock(model: *Self) void {
            assert(model.locked);
            model.network.giveParameters();
            model.locked = false;
        }

        pub fn loss(model: *Self, feature: *const Feature, label: *const Label) f32 {
            assert(model.locked);
            return Loss.eval(model.eval(feature), label);
        }

        /// Returns the mean loss over a dataset.
        pub fn cost(model: *Self, dataset: Dataset) f32 {
            assert(dataset.len() > 0);
            assert(model.locked);
            var result: f32 = 0;
            for (dataset.features, dataset.labels) |feature, label| result += model.loss(&feature, &label);
            return result / @as(f32, @floatFromInt(dataset.len()));
        }

        pub fn differentiate(model: *Self, dataset: Dataset) void {
            @setFloatMode(.optimized);
            assert(model.locked);
            @memset(&model.gradient, 0);
            for (dataset.features, dataset.labels) |feature, label| {
                const prediction = model.forwardPass(&feature);
                Loss.gradient(prediction, &label, model.network.lastDeltaBuffer());
                model.backwardPass();
            }
        }

        /// Trains the model on a given dataset for a specified amount of epochs and batch size.
        /// Every epoch the model is 'validated' on another given dataset.
        pub fn train(model: *Self, training: Dataset, validate: Dataset, epoch_count: usize, batch_size: usize) void {
            assert(!model.locked);
            assert(batch_size > 0);
            for (0..epoch_count) |epoch| {
                var offset: usize = 0;
                while (training.len() > offset) : (offset += batch_size) {
                    model.lock();
                    const subset = training.slice(offset, @min(offset + batch_size, training.len()));
                    model.differentiate(subset);
                    model.unlock();
                    model.optimizer.step(@ptrCast(&model.parameters), @ptrCast(&model.gradient));
                }
                model.lock();
                if (validate.len() > 0) std.debug.print("epoch: {d}\tloss: {d}\n", .{ epoch, model.cost(validate) });
                model.unlock();
           }
        }
    };
}
