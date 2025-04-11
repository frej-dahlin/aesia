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
		data: [dim]f32 = [_]f32{0} ** 16,
		
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

// Fixme: The structure of the network is compile-time known, in principle everything is static, utilize this!
// Fixme: Make float type an optional parameter.
pub fn Network(InputLayer: type) type {
	return struct {
		const Self = @This();

	    // Relative indices into each layer.
	    const NodeIndex = u16;
	    // Currently the graph is stored as a list of lists.
	    // Consider using a compressed sparse row format instead.
	    const Node = struct {
	        parents: [2]NodeIndex,
	        children: []NodeIndex,

	        weights: f32x16,
	        // The gradient of the cost function with respect to the weights of the node.
	        gradient: f32x16,
	        // The current feedforward value.
	        value: f32 = 0,
	        // Used for backpropagation, defined as dC/dvalue,
	        // where C is the cost function. This in turn is used to compute the gradient.
	        delta: f32,

	        // Adam optimizer data.
	        adam_v: f32x16,
	        adam_m: f32x16,

	        // Returns the derivative of eval with respect to the first parent.
	        pub fn del_a(node: Node, b: f32) f32 {
	            const sigma = softmax(node.weights);
	            return @reduce(.Add, sigma * SoftGate.del_a(b));
	        }

	        // Returns the derivative of eval with respect to the second parent.
	        pub fn del_b(node: Node, a: f32) f32 {
	            const sigma = softmax(node.weights);
	            return @reduce(.Add, sigma * SoftGate.del_b(a));
	        }

	        pub fn eval(weights: f32x16, a: f32, b: f32) f32 {
	            const sigma = softmax(weights);
	            return @reduce(.Add, sigma * SoftGate.vector(a, b));
	        }
	    };

		input_layer: InputLayer,
	    logic_layers: []MultiArrayList(Node).Slice,

	    // Note: This is the hottest part of the code, I *think* that AVX-512 should handle it with gusto,
	    // but testing is required. Consider caching the result, this can be a compile time option.
	    pub fn softmax(w: f32x16) f32x16 {
	        @setFloatMode(.optimized);
	        const sigma = @exp2(w);
	        const denom = @reduce(.Add, sigma);
	        return sigma / @as(f32x16, @splat(denom));
	    }

	    pub fn lastLayer(net: Self) MultiArrayList(Node).Slice {
	        return net.logic_layers[net.logic_layers.len - 1];
	    }

	    pub fn eval(net: *Self, x: InputLayer.InputType) void {
	        @setFloatMode(.optimized);
	        assert(x.len == InputLayer.dim);

	        // Evaluate the network, layer by layer.
	        // Note: Two for loops is faster than a single with a branch.
			net.input_layer.eval(x);
	        const first = net.logic_layers[0];
	        for (first.items(.value), first.items(.parents), first.items(.weights)) |*value, parents, weights| {
	            const a = net.input_layer.values()[parents[0]];
	            const b = net.input_layer.values()[parents[1]];
	            value.* = Node.eval(weights, a, b);
	        }
	        for (net.logic_layers[1..], net.logic_layers[0 .. net.logic_layers.len - 1]) |layer, prev| {
	            const prev_values = prev.items(.value);
	            for (layer.items(.value), layer.items(.parents), layer.items(.weights)) |*value, parents, weights| {
	                const a = prev_values[parents[0]];
	                const b = prev_values[parents[1]];
	                value.* = Node.eval(weights, a, b);
	            }
	        }
	    }

	    // Compute the delta for each node relative to a given datapoint.
	    // See the definition of the Node struct.
	    // Fixme: Currently assumes the network is trained with a halved mean square error.
	    pub fn backprop(net: *Self, x: InputLayer.InputType, y: []const f32) void {
	        @setFloatMode(.optimized);
	        const last = net.lastLayer();
	        assert(x.len == InputLayer.dim);
	        assert(y.len == last.len);

	        net.eval(x);
	        for (net.logic_layers) |layer| @memset(layer.items(.delta), 0);
	        // Compute the delta for the last layer.
	        for (last.items(.delta), last.items(.value), y) |*delta, value, y_value| {
	            // Fixme: Hardcoded gradient of the halved mean square error.
	            delta.* = value - y_value;
	        }
	        // Compute the delta of the previous layers, back to front.
	        for (1..net.logic_layers.len) |i| {
	            const layer = net.logic_layers[net.logic_layers.len - i];
	            const prev = net.logic_layers[net.logic_layers.len - i - 1];
	            const prev_deltas = prev.items(.delta);
	            const prev_values = prev.items(.value);
	            for (0..layer.len) |child_index| {
	                const child = layer.get(child_index);
	                const parents = child.parents;
	                const a = prev_values[parents[0]];
	                const b = prev_values[parents[1]];
	                prev_deltas[parents[0]] += child.delta * child.del_a(b);
	                prev_deltas[parents[1]] += child.delta * child.del_b(a);
	            }
	        }
	    }

	    // Updates the gradient of the network for a given datapoint.
	    // Fixme: Now the gradient of softmax is hardcoded.
	    // Fixme: Move the inner 2 for loops to a separate function.
	    pub fn update_gradient(net: *Self, x: InputLayer.InputType, y: []const f32) void {
	        @setFloatMode(.optimized);
	        assert(x.len == InputLayer.dim);
	        assert(y.len == net.lastLayer().len);

	        net.backprop(x, y);
	        // Compensating factor for using base 2 instead of e in softmax.
	        const tau = @log(2.0);
			
	        for (net.logic_layers, 0..) |layer, i| {
	            const prev_values = if (i == 0) net.input_layer.values() else net.logic_layers[i - 1].items(.value);
	            for (layer.items(.parents), layer.items(.weights), layer.items(.gradient), layer.items(.delta)) |parents, weights, *gradient, delta| {
	                const a = prev_values[parents[0]];
	                const b = prev_values[parents[1]];
	                const gate = SoftGate.vector(a, b);
	                const sigma = softmax(weights);
	                const value = @reduce(.Add, sigma * gate);
	                gradient.* += @as(f32x16, @splat(tau * delta)) * sigma * (gate - @as(f32x16, @splat(value)));
	            }
	        }
	    }

	    pub fn deinit(net: Self, allocator: Allocator) void {
	        for (net.nodes) |node| allocator.free(node.children);
	        allocator.free(net.nodes);
	        allocator.free(net.shape);
	    }

	    /// Initializes a network with specified shape with random connections inbetween.
	    /// Each node is guaranteed to have the same number of children modulo +-1.
	    /// Note that shape[0] is the input dimension and shape[shape.len - 1] is the output dimension.
	    pub fn initRandom(allocator: Allocator, comptime shape: []const usize) !Self {
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
	        net.input_layer = .{};
	        net.logic_layers = try allocator.alloc(MultiArrayList(Node).Slice, shape.len - 1);
	        for (net.logic_layers, shape[1..]) |*layer, dim| {
	            var tmp: MultiArrayList(Node) = .{};
	            try tmp.resize(allocator, dim);
	            layer.* = tmp.slice();
	        }

	        // Initialize nodes.
	        for (net.logic_layers) |*layer| {
	            for (layer.items(.weights)) |*weights| {
	                // Initialize weights to be biased toward the pass through gates.
	                weights.* = [_]f32{ 0, 0, 0, 8, 0, 8 } ++ .{0} ** 10;
	            }
	            @memset(layer.items(.gradient), [_]f32{0} ** 16);
	            @memset(layer.items(.children), &.{});
	            @memset(layer.items(.adam_m), [_]f32{0} ** 16);
	            @memset(layer.items(.adam_v), [_]f32{0} ** 16);
	        }

	        // Initialize node parents such that each node has at least one child, excluding the last layer.
	        // Moreover, the following scheme guarantees that the number of children each node has is
	        // uniformly distributed, i.e. every node has an equal number of children modulo +-1.
	        var stack = try ArrayList(usize).initCapacity(allocator, std.mem.max(usize, shape));
	        defer stack.deinit();
	        for (net.logic_layers, shape[0 .. shape.len - 1]) |layer, prev_dim| {
	            for (layer.items(.parents)) |*parents| {
	            	const first = if (stack.pop()) |index| index else refill: {
			        // Fill the stack with all possible indices of the previous layer in random order.
	            		for (0..prev_dim) |i| stack.appendAssumeCapacity(i);	
	            		rand.shuffle(usize, stack.items);
	            		break :refill stack.pop().?;
	            	};
	            	const second = if (stack.pop()) |index| index else refill: {
	            		for (0..prev_dim) |i| stack.appendAssumeCapacity(i);	
	            		rand.shuffle(usize, stack.items);
	            		break :refill stack.pop().?;
	            	};
	                parents.* = .{ @truncate(first), @truncate(second) };
	            }
	        }
	        // Find the children of each node.
	        var buffer = ArrayList(NodeIndex).init(allocator);
	        defer buffer.deinit();
	        for (net.logic_layers[0 .. net.logic_layers.len - 1], net.logic_layers[1..]) |*layer, next| {
	            for (layer.items(.children), 0..) |*children, parent_index| {
	                for (next.items(.parents), 0..) |parents, child_index| {
	                    if (parents[0] == parent_index or parents[1] == parent_index) try buffer.append(@truncate(child_index));
	                }
	                children.* = try buffer.toOwnedSlice();
	            }
	        }

	        // Assert parent indices and that every node has at least one child excepting the ones in the last layer.
	        for (net.logic_layers, shape[0 .. shape.len - 1], shape[1..], 0..) |layer, dim_prev, dim_next, i| {
	            for (layer.items(.parents), layer.items(.children)) |parents, children| {
	                assert(parents[0] < dim_prev);
	                assert(parents[1] < dim_prev);
	                assert(children.len > 0 or i == net.logic_layers.len - 1);
	                for (children) |child_index| assert(child_index < dim_next);
	            }
	        }
	        // Assert child indices.
	        for (net.logic_layers[0 .. net.logic_layers.len - 1], net.logic_layers[1..]) |layer, next| {
	            for (layer.items(.children), 0..) |children, parent_index| {
	                for (children) |child_index| {
	                    const parents = next.items(.parents)[child_index];
	                    assert(parents[0] == parent_index or parents[1] == parent_index);
	                }
	            }
	        }

	        return net;
	    }
	};
}
