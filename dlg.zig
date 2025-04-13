const std = @import("std");
const meta = std.meta;
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const EnumArray = std.EnumArray;
const MultiArrayList = std.MultiArrayList;
const ArrayList = std.ArrayList;
const AutoArrayHashMap = std.AutoArrayHashMap;

pub const f32x16 = @Vector(16, f32);

const SoftGate = struct {
    /// Returns a vector representing all values of a soft logic gate.
    pub fn vector(a: f32, b: f32) f32x16 {
        @setFloatMode(.optimized);
        return .{
            0,
            a * b,
            a - a * b,
            a,
            b - a * b,
            b,
            a + b - 2 * a * b,
            a + b - a * b,
            1 - (a + b - a * b),
            1 - (a + b - 2 * a * b),
            1 - b,
            1 - (b - a * b),
            1 - a,
            1 - (a - a * b),
            1 - a * b,
            1,
        };
    }

    /// Returns the soft gate vector differentiated by the first variable.
    /// Note that it only depends on the second variable.
    pub fn del_a(b: f32) f32x16 {
        @setFloatMode(.optimized);
        return .{
            0,
            b,
            1 - b,
            1,
            -b,
            0,
            1 - 2 * b,
            1 - b,
            -(1 - b),
            -(1 - 2 * b),
            0,
            b,
            -1,
            -(1 - b),
            -b,
            0,
        };
    }

    /// Returns the soft gate vector differentiated by the second variable.
    /// Note that it only depends on the first variable.
    pub fn del_b(a: f32) f32x16 {
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
            -(1 - a),
            -(1 - 2 * a),
            -1,
            -(1 - a),
            0,
            a,
            -a,
            0,
        };
    }
};

pub fn InputRelaxBits(bit_count: usize) type {
    return struct {
        const Self = @This();
        const dim = bit_count;
        const InputType = [dim]bool;
        data: [dim]f32 = [_]f32{0} ** dim,

        // Fixme: Use SIMD.
        pub fn eval(self: *Self, x: InputType) void {
            for (&self.data, x) |*value, xj| value.* = if (xj) 1.0 else 0.0;
        }

        // Note: since the input layer owns the underlying data, it is a bug to not pass a pointer here.
        pub fn values(self: *Self) []const f32 {
            return &self.data;
        }
    };
}

pub fn InputIdentity(_dim: usize) type {
    return struct {
        const Self = @This();
        const dim = _dim;
        const InputType = [dim]f32;
        slice: []const f32 = &.{},

        pub fn eval(self: *Self, x: InputType) void {
            self.slice = &x;
        }

        pub fn values(self: Self) []const f32 {
            return self.slice;
        }
    };
}

pub fn HalvedMeanSquareError(dim: usize) type {
    return struct {
        pub fn eval(y_pred: *const [dim]f32, y_real: *const [dim]f32) f32 {
            assert(y_pred.len == y_real.len);
            var result: f32 = 0;
            inline for (y_pred, y_real) |pred, real| result += (pred - real) * (pred - real);
            return result / 2;
        }

        pub fn gradient(y_pred: *const [dim]f32, y_real: *const [dim]f32, out: *[dim]f32) void {
            assert(y_pred.len == y_real.len and y_real.len == out.len);
            for (y_pred, y_real, out) |pred, real, *o| o.* = pred - real;
        }
    };
}

// Fixme: Add FloatType as a field.
// Fixme: Make float type an optional parameter.
pub const NetworkOptions = struct {
    shape: []const usize,
    InputLayer: ?fn (usize) type = null,
    Cost: ?type = null,
};

pub fn Network(comptime options: NetworkOptions) type {
    const shape = options.shape;
    // Assert validity of the shape array.
    if (shape.len < 2) @compileError("options.shape: at least two dimensions are required, input and output");
    for (shape) |dim| if (dim == 0) @compileError("options.shape: all dimensions need to be nonzero");
    for (shape[0 .. shape.len - 1], shape[1..]) |dim, next| {
        if (2 * next < dim) @compileError("options.shape: not allowed to shrink dimensions by factor > 2");
    }
    const node_count = blk: {
        var result: usize = 0;
        inline for (shape[1..]) |dim| result += dim;
        break :blk result;
    };
    const layer_offsets = blk: {
   		var result: [shape.len - 1]usize = undefined;
   		var offset: usize = 0;
   		for (&result, shape[1..]) |*r, dim| {
   			r.* = offset;	
   			offset += dim;
   		}
   		break :blk result;
    };
    // Unpack options.
    const InputLayer = if (options.InputLayer) |IL| IL(shape[0]) else InputIdentity(shape[0]);
    const output_dim = shape[shape.len - 1];
    const Cost = if (options.Cost) |C| C else HalvedMeanSquareError(output_dim);

    return struct {
        const Self = @This();

        // Fixme: For simplicity these are currently *global* indices, but using an array of offsets or similar,
        // one can contract these into a much smaller type.
        const NodeIndex = u32;
        // Currently the graph is stored as a list of lists.
        // Consider using a compressed sparse row format instead.
        // We use a struct of arrays for cache efficiency, for example eval never touches the delta and gradient.
        // Note: Consider creating a static version of MultiArrayList 'MultiArray' for flexibility and elegance.
        const Nodes = struct {
            parents: [node_count][2]NodeIndex,

            logits: [node_count]f32x16,
            // Cached value of softmax(logit).
            sigma: [node_count]f32x16,
            // The gradient of the cost function with respect to the logit of the node.
            gradient: [node_count]f32x16,
            // The current feedforward value.
            value: [node_count]f32,
            // Used for backpropagation, defined as dC/dvalue,
            // where C is the cost function. This in turn is used to compute the gradient.
            delta: [node_count]f32,
        };
        const Node = struct {
            parents: [2]NodeIndex,

            logits: f32x16,
            // Cached probability distribution.
            sigma: f32x16,
            // The gradient of the cost function with respect to the logits of the node.
            gradient: f32x16,
            // The current feedforward value.
            value: f32 = 0,
            // Used for backpropagation, defined as dC/dvalue,
            // where C is the cost function. This in turn is used to compute the gradient.
            delta: f32,

            // Adam optimizer data.
            adam_v: f32x16,
            adam_m: f32x16,
        };

        nodes: Nodes,
        input_layer: InputLayer,

        // Note: This is the hottest part of the code, I *think* that AVX-512 should handle it with gusto,
        // but testing is required. Consider caching the result, this can be a compile time option.
        pub fn softmax(w: f32x16) f32x16 {
            @setFloatMode(.optimized);
            const sigma = @exp2(w);
            const denom = @reduce(.Add, sigma);
            return sigma / @as(f32x16, @splat(denom));
        }

        pub fn eval(net: *Self, x: InputLayer.InputType) void {
            @setFloatMode(.optimized);
            assert(x.len == InputLayer.dim);

            @setEvalBranchQuota(node_count);
            const nodes = &net.nodes;
            // Gotta go fast?!
            // Fixme: benchmark inlining.
            // Evaluate the network, layer by layer.
            net.input_layer.eval(x);
            const input_values = net.input_layer.values();
            for (nodes.value[0..shape[1]], nodes.sigma[0..shape[1]], nodes.parents[0..shape[1]], nodes.logits[0..shape[1]]) |*value, *sigma, parents, logits| {
                const a = input_values[parents[0]];
                const b = input_values[parents[1]];
                sigma.* = softmax(logits);
                value.* = @reduce(.Add, sigma.* * SoftGate.vector(a, b));
            }
            for (nodes.value[shape[1]..], nodes.sigma[shape[1]..], nodes.parents[shape[1]..], nodes.logits[shape[1]..]) |*value, *sigma, parents, logits| {
                const a = nodes.value[parents[0]];
                const b = nodes.value[parents[1]];
                sigma.* = softmax(logits);
                value.* = @reduce(.Add, sigma.* * SoftGate.vector(a, b));
            }
        }

        // Compute the delta for each node relative to a given datapoint.
        // See the definition of the Node struct.
        // Fixme: Currently assumes the network is trained with a halved mean square error.
        pub fn backprop(net: *Self, x: InputLayer.InputType, y: *const [output_dim]f32) void {
            @setFloatMode(.optimized);
            assert(x.len == InputLayer.dim);
            assert(y.len == output_dim);

            net.eval(x);
            @memset(&net.nodes.delta, 0);
            // Compute the delta for the last layer.
            Cost.gradient(net.nodes.value[node_count - output_dim ..], y, net.nodes.delta[node_count - output_dim ..]);
            // Compute the delta for the remaining layers, back to front.
            // Each child is responsible for correctly updating their parents.
            for (1..node_count + 1 - shape[1]) |i| {
                const index = node_count - i;
                const parents = net.nodes.parents[index];
                const delta = net.nodes.delta[index];
                const sigma = net.nodes.sigma[index];
                const a = net.nodes.value[parents[0]];
                const b = net.nodes.value[parents[1]];
                net.nodes.delta[parents[0]] += delta * @reduce(.Add, sigma * SoftGate.del_a(b));
                net.nodes.delta[parents[1]] += delta * @reduce(.Add, sigma * SoftGate.del_b(a));
            }
        }

        // Updates the gradient of the network for a given datapoint.
        // Fixme: Now the gradient of softmax is hardcoded.
        // Fixme: Move the inner 2 for loops to a separate function.
        pub fn update_gradient(net: *Self, x: InputLayer.InputType, y: *const [output_dim]f32) void {
            @setFloatMode(.optimized);
            assert(x.len == InputLayer.dim);
            assert(y.len == output_dim);

            net.backprop(x, y);
            // Compensating factor for using base 2 instead of e in softmax.
            const tau = @log(2.0);

            const nodes = &net.nodes;
            for (&nodes.gradient, nodes.parents, nodes.sigma, nodes.delta) |*gradient, parents, sigma, delta| {
                const a = nodes.value[parents[0]];
                const b = nodes.value[parents[1]];
                const gate = SoftGate.vector(a, b);
                const value = @reduce(.Add, sigma * gate);
                gradient.* += @as(f32x16, @splat(tau * delta)) * sigma * (gate - @as(f32x16, @splat(value)));
            }
        }

        pub fn deinit(net: Self, allocator: Allocator) void {
            allocator.free(net.nodes);
            allocator.free(net.shape);
        }
        
        pub fn cost(net: *Self, y_pred: *const [output_dim]f32) f32 {
    		return Cost.eval(net.nodes.value[node_count - output_dim..], y_pred);
        }
        
        /// Initializes a network with specified shape with random connections inbetween.
        /// Each node is guaranteed to have the same number of children modulo +-1.
        /// Note that shape[0] is the input dimension and shape[shape.len - 1] is the output dimension.
        pub fn initRandom() !Self {
            // Shape must at least specify input and output dimensions.
            assert(shape.len >= 2);
            // Assert that the network does not shrink by more than a factor of 2,
            // this forces some nodes to have no children, which causes those nodes to become useless.
            for (shape[0 .. shape.len - 1], shape[1..]) |dim, next| {
                assert(next * 2 >= dim);
                assert(dim > 0);
            }
            assert(shape[0] == InputLayer.dim);

            var prng = std.Random.DefaultPrng.init(blk: {
                var seed: u64 = undefined;
                try std.posix.getrandom(std.mem.asBytes(&seed));
                break :blk seed;
            });
            const rand = prng.random();

            var net: Self = undefined;

            // Initialize nodes.
            // Initialize logits to be biased toward the pass through gates.
            @memset(&net.nodes.logits, [_]f32{ 0, 0, 0, 8, 0, 8 } ++ .{0} ** 10);
            @memset(&net.nodes.gradient, [_]f32{0} ** 16);

            // Initialize node parents such that each node has at least one child, excluding the last layer.
            // Moreover, the following scheme guarantees that the number of children each node has is
            // uniformly distributed, i.e. every node has an equal number of children modulo +-1.
            inline for (shape[0..shape.len - 2], layer_offsets[0..layer_offsets.len - 1], layer_offsets[1..]) |dim, from, to| {
                for (net.nodes.parents[from..to], 0..) |*parents, i| {
                	parents.* = .{@truncate(i % dim), @truncate((i + 1) % dim)};
               	}
               	rand.shuffle(NodeIndex, @as(*[2 * (to - from)]NodeIndex, @ptrCast(net.nodes.parents[from..to])));
            }

            return net;
        }
    };
}
