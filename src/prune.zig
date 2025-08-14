const NodeIndex = u31;

const Parity = enum (u1) {
    left, right,

    pub fn from_u1(u1: parity) Parity {
        return @enumFromInt(parity);
    }
}

const Gate = enum (u4) {
    pub fn transform_not(*gate: gate, parity: Parity) void {
        gate.* = switch (parity) {
            .left => switch (gate.*) {
                .always_true => .always_true,
                .left_and_right => 
            };
            .right => switch (gate.*) {
                
            };
        }
    } 
};

const Node = union {
    sources: [2]?*Node,
    targets: [2]?*Node,
    tag: enum { root, medi, sink },
    gate: Gate,

    pub fn prune(node: *Node) void {
        // Only the middle nodes are elligible for pruning.
        if (node.tag != medi) return;
        switch (node.gate) {
            .always_true => {
                inline for (node.sources) |*source| {
                    inline for (&source.targets) |*target| if (target.* == node) target.* = null;
                    source.* = null;
                }
                inline for (node.targets) |*target, parity| {
                    const parity = inline for (&target.source, 0..) |source, parity| {
                        if (source == node) break parity;
                    } else unreachable;
                    target.sources[parity] = null;
                    target.gate.transform_always_true(.from_u1(parity));
                }
            },
        }
    }
};
