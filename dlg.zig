const std = @import("std");
const meta = std.meta;
const assert = std.debug.assert;
const Allocator = std.mem.Allocator;
const EnumArray = std.EnumArray;
const MultiArrayList = std.MultiArrayList;
const ArrayList = std.ArrayList;
const AutoArrayHashMap = std.AutoArrayHashMap;

const SoftGate = enum(u4) {
	@"false",
	@"and",
	a_and_not_b,
	a,
	b_and_not_a,
	b,
	xor,
	@"or",
	nor,
	xnor,
	not_b,
	a_or_not_b,
	not_a,
	b_or_not_a,
	nand,
	@"true",
	
	pub fn eval(gate: SoftGate, a: f32, b: f32) f32 {
		return switch (gate) {
			.@"false"	=> 0,
			.@"and"		=> a * b,
			.a_and_not_b	=> a - a * b,
			.a		=> a,
			.b_and_not_a	=> b - a * b,
			.b		=> b,
			.xor		=> a + b - 2 * a * b,
			.@"or"		=> a + b - a * b,
			.nor		=> 1 - (a + b - a * b),
			.xnor		=> 1 - (a + b - 2 * a * b),
			.not_b		=> 1 - b,
			.a_or_not_b	=> 1 - b + a * b,
			.not_a		=> 1 - a,
			.b_or_not_a	=> 1 - a + a * b,
			.nand		=> 1 - a * b,
			.@"true"	=> 1,
		};
	}

	// Derivative with respect to the first variable.
	pub fn del_a(gate: SoftGate, b: f32) f32 {
		return switch (gate) {
			.@"false"	=> 0,
			.@"and"		=> b,
			.a_and_not_b	=> (1 - b),
			.a		=> 1,
			.b_and_not_a	=> -b,
			.b		=> 0,
			.xor		=> 1 - 2 * b,
			.@"or"		=> 1 - b,
			.nor		=> b - 1,
			.xnor		=> -1 + 2 * b,
			.not_b		=> 0,
			.a_or_not_b	=> b,
			.not_a		=> -1,
			.b_or_not_a	=> b - 1,
			.nand		=> -b,
			.@"true"	=> 0,
		};
	}
	
	// Derivative with respect to the second variable.
	pub fn del_b(gate: SoftGate, a: f32) f32 {
		return switch (gate) {
			.@"false"	=> 0,
			.@"and"		=> a,
			.a_and_not_b	=> -a,
			.a		=> 0,
			.b_and_not_a	=> (1 - a),
			.b		=> 1,
			.xor		=> 1 - 2 * a,
			.@"or"		=> 1 - a,
			.nor		=> a - 1,
			.xnor		=> -1 + 2 * a,
			.not_b		=> -1,
			.a_or_not_b	=> -1 + a,
			.not_a		=> 0,
			.b_or_not_a	=> a,
			.nand		=> -a,
			.@"true"	=> 0,
		};
	}
};
// Array containing all possible gates in order.
const gates = blk: {
	const fields = meta.fields(SoftGate);
	var result: [fields.len]SoftGate = undefined;
	for (fields, &result) |field, *g| {
		g.* = @enumFromInt(field.value);
	}
	break :blk result;
};

pub fn softmax(dim: comptime_int, input: [dim]f32) [dim]f32 {
	var denom: f32 = 0;
	var result = [_]f32{0} ** dim;
	inline for (input) |a| denom += @exp(a);
	inline for (&result, input) |*r, a| r.* = @exp(a) / denom;
	return result;
}

pub fn kronecker_delta(T: type, i: anytype, j: anytype) T {
	return if (i == j) 1 else 0;
}

// Fixme: Use @Vector(16, f32) SIMD.
// Fixme: Make float type an optional parameter.
const Network = struct {
	// Indices into the nodes slice.
	const NodeIndex = u32;
	// Currently the graph is stored as a list of lists.
	// Consider using a compressed sparse row format instead.
	const Node = struct {
		// Note: One can make these relative to each layer; this will save memory.
		// For example if each layer at most has 60_000 nodes, then u16 suffices
		// as a relative index type.
		parents: [2]NodeIndex,
		children: []NodeIndex,
		
		weights: [gates.len]f32,
		// The gradient of the cost function with respect to the weights of the node.
		gradient: [gates.len]f32,
		// The current feedforward value.
		value: f32 = 0,
		// Used for backpropagation, defined as dC/dvalue,
		// where C is the cost function. This in turn is used to compute the gradient.
		delta: f32,
		
		pub fn eval(node: Node, a: f32, b: f32) f32 {
			var result: f32 = 0;
			const sigma = softmax(node.weights.len, node.weights);
			inline for (sigma, gates) |prob, gate| {
				result += prob * gate.eval(a, b);
			}
			return result;
		}
		
		// Returns the derivative of eval with respect to the first parent.
		pub fn del_a(node: Node, b: f32) f32 {
			var result: f32 = 0;
			const sigma = softmax(node.weights.len, node.weights);
			inline for (sigma, gates) |prob, gate| {
				result += prob * gate.del_a(b);
			}
			return result;
		}

		// Returns the derivative of eval with respect to the second parent.
		pub fn del_b(node: Node, a: f32) f32 {
			var result: f32 = 0;
			const sigma = softmax(node.weights.len, node.weights);
			inline for (sigma, gates) |prob, gate| {
				result += prob * gate.del_b(a);	
			}
			return result;
		}
	};

	// Fixme: Use a MultiArrayList instead (aka SoA), this would be much more cache-friendly.
	// The forward pass only needs the nodes parents, while the backward pass
	// does not need the parent indices.
	nodes: []Node,
	// The shape of the network, shape[0] and shape[shape.len - 1] are the
	// input and output dimensions, respectively. For l >= 1, shape[l] is
	// the dimension of layer l.
	// Note: With the above relative index changes this will need to become a slice of offsets into the separate layers.
	shape: []usize,
	
	pub fn lastLayer(network: Network) []Node {
		return network.nodes[network.nodes.len - network.shape[network.shape.len - 1]..];
	}

	pub fn eval(network: Network, x: []const f32) void {
		assert(x.len == network.shape[0]);
		
		const nodes = network.nodes;
		const shape = network.shape;
		
		// Evaluate first layer.
		for (nodes[0..shape[1]]) |*node| {
			const a = x[node.parents[0]];
			const b = x[node.parents[1]];
			node.value = node.eval(a, b);
		}
		// Evaluate the rest of the network.
		for (nodes[shape[1]..]) |*node| {
			const a = nodes[node.parents[0]].value;
			const b = nodes[node.parents[1]].value;
			node.value = node.eval(a, b);
		}
	}
	
	pub fn output(net: Network, out: []f32) void {
		assert(out.len == net.lastLayer().len);
		for (net.lastLayer(), out) |node, *y| y.* = node.value;
	}

	// Compute the delta for each node relative to a given datapoint.
	// See the definition of the Node struct.
	// Fixme: Currently assumes the network is trained with a halved mean square error.
	pub fn backprop(network: Network, x: []const f32, y: []const f32) void {
		assert(x.len == network.shape[0]);
		assert(y.len == network.shape[network.shape.len - 1]);
		
		network.eval(x);
		
		const nodes = network.nodes;
		const shape = network.shape;
		
		// Compute the delta of the last layer.
		const last_layer = network.lastLayer();
		for (last_layer, 0..) |*node, j| {
			// Fixme: Hardcoded gradient of the halved mean square error.
			node.delta = node.value - y[j];
		}
		
		// Compute the delta of the previous layers, back to front.
		var offset = nodes.len - last_layer.len;
		for (2..shape.len) |i| {
			const from = offset - shape[shape.len - i];
			const to = offset;
			for (nodes[from..to], 0..) |*node, j| {
				node.delta = 0;
				for (node.children) |child_index| {
					const child = nodes[child_index];
					const a = nodes[child.parents[0]].value;
					const b = nodes[child.parents[1]].value;
					node.delta += child.delta * if (child.parents[0] == j)
						child.del_a(b) else
						child.del_b(a);
				}
			}
			offset = from;
		}
	}
	
	// Updates the gradient of the network for a given datapoint.
	// Fixme: Now the gradient of softmax is hardcoded.
	// Fixme: Move the inner 2 for loops to a separate function.
	pub fn update_gradient(network: Network, x: []const f32, y: []const f32) void {
		assert(x.len == network.shape[0]);
		assert(y.len == network.shape[network.shape.len - 1]);
		
		network.backprop(x, y);
		const nodes = network.nodes;
		
		// Note: Can use inline for-loops to contract these into one.
		for (nodes[0..network.shape[1]]) |*node| {
			const a = x[node.parents[0]];
			const b = x[node.parents[1]];
			const sigma = softmax(node.weights.len, node.weights);
			for (&node.gradient, 0..) |*partial, k| {
				// Compute partial derivative with respect to the k:th weight.
				var coef: f32 = 0;
				for (gates, 0..) |gate, n| {
					coef += sigma[n] * (kronecker_delta(f32, n, k) - sigma[k]) * gate.eval(a, b);
				}
				partial.* += coef * node.delta;
			}
		}
		for (nodes[network.shape[1]..]) |*node| {
			const a = nodes[node.parents[0]].value;
			const b = nodes[node.parents[1]].value;
			const sigma = softmax(node.weights.len, node.weights);
			for (&node.gradient, 0..) |*partial, k| {
				var coef: f32 = 0;
				for (gates, 0..) |gate, n| {
					coef += sigma[n] * (kronecker_delta(f32, n, k) - sigma[k]) * gate.eval(a, b);
				}
				partial.* += coef * node.delta;
			}
		}
	}

	pub fn deinit(network: Network, allocator: Allocator) void {
		for (network.nodes) |node| allocator.free(node.children);
		allocator.free(network.nodes);
		allocator.free(network.shape);
	}

	/// Initializes a network with specified shape with random connections inbetween.
	/// Each node is guaranteed to have at least one child.
	/// Note that shape[0] is the input dimension and shape[shape.len - 1] is the output dimension.
	pub fn initRandom(allocator: Allocator, shape: []const usize) !Network {
		// Shape must at least specify input and output dimensions.
		assert(shape.len >= 2);
		// Assert that the network does not shrink by more than a factor of 2,
		// this forces some nodes to have no children, which causes those nodes to become useless.
		for (shape[0..shape.len - 1], shape[1..]) |dim, next| {
			assert(next * 2 >= dim);
		}

		var prng = std.Random.DefaultPrng.init(blk: {
		        var seed: u64 = undefined;
		        try std.posix.getrandom(std.mem.asBytes(&seed));
			break :blk seed;
		});
		const rand = prng.random();
		
		var network: Network = undefined;
		
		// Copy shape slice.
		network.shape = try allocator.alloc(usize, shape.len);
		@memcpy(network.shape, shape);
		errdefer allocator.free(network.shape);
		
		// Count number of nodes in the network, this excludes the input layer.
		var node_count: usize = 0;
		for (shape[1..]) |dim| node_count += dim;
		network.nodes = try allocator.alloc(Node, node_count);
		errdefer allocator.free(network.nodes);

		// Temporary slice of the layers.
		const layers: [][]Node = try allocator.alloc([]Node, shape.len - 1);
		defer allocator.free(layers);
		var offset: usize = 0;
		for (shape[1..], layers) |dim, *layer| {
			const from = offset;
			const to = offset + dim;
			layer.* = network.nodes[from..to];
			offset = to;
		}
		
		// Zero initialize.
		for (network.nodes) |*node| {
			@memset(&node.gradient, 0);
			@memset(&node.weights, 0);
			//for (&node.weights) |*w| w.* = rand.float(f32);
			node.weights[3] = 8;
			node.weights[5] = 8;
			node.children = &.{};
		}
		
		// Initialize nodes with random parents such that each node has at least one child, excluding the output layer.
		var stack = try ArrayList(usize).initCapacity(allocator, shape[0]);
		defer stack.deinit();
		for (0..shape[0]) |i| stack.appendAssumeCapacity(i);
		rand.shuffle(usize, stack.items);
		for (layers[0]) |*node| {
			node.parents[0] = if (stack.items.len > 0)
				@truncate(stack.pop().?) else
				rand.uintLessThan(NodeIndex, @truncate(shape[0]));
			node.parents[1] = if (stack.items.len > 0)
				@truncate(stack.pop().?) else
				while (true) {
					const i = rand.uintLessThan(NodeIndex, @truncate(shape[0]));
					// Fixme: Will fail if shape[0] == 1.
					if (i != node.parents[0]) break i;
				} else unreachable;
		}

		offset = 0;
		for (layers[1..], layers[0..layers.len - 1]) |layer, prev| {
			for (0..prev.len) |i| try stack.append(i);
			rand.shuffle(usize, stack.items);
			for (layer) |*node| {
				node.parents[0] = if (stack.items.len > 0)
					@truncate(offset + stack.pop().?) else
					@truncate(offset + rand.uintLessThan(NodeIndex, @truncate(prev.len)));
				node.parents[1] = if (stack.items.len > 0)
					@truncate(offset + stack.pop().?) else
					while (true) {
						const i: NodeIndex = @truncate(offset + rand.uintLessThan(NodeIndex, @truncate(prev.len)));
						// Fixme: Will fail if shape[0] == 1.
						if (i != node.parents[0]) break i;
					} else unreachable;
			}
			offset += prev.len;
		}

		// Find the children of each node.
		if (shape.len == 2) return network;
		
		var buffer = ArrayList(NodeIndex).init(allocator);
		errdefer buffer.deinit();
		errdefer for (network.nodes) |node| allocator.free(node.children);
		offset = 0;
		for (layers[0..layers.len - 1], layers[1..]) |layer, next| {
			for (layer, 0..) |*parent, j| {
				const parent_index: NodeIndex = @truncate(offset + j);
				for (next, 0..) |child, k| {
					const child_index: NodeIndex = @truncate(offset + layer.len + k);
					if (child.parents[0] == parent_index or
					    child.parents[1] == parent_index) try buffer.append(child_index);
				}
				parent.children = try buffer.toOwnedSlice();
			}
			offset += layer.len;
		}
		
		return network;
	}
};
