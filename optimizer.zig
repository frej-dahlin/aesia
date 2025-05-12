const std = @import("std");
const assert = std.debug.assert;
const comptimePrint = std.fmt.comptimePrint;

const vector_len = std.simd.suggestVectorLength(f32) orelse 1;
const Vector = @Vector(vector_len, f32);

pub const GradientDescentOptions = struct {
    learn_rate: f32 = 0.01,
    momentum: f32 = 0,

    pub const default = @This(){};
};

fn assertParameterCount(parameter_count: usize) void {
    if (parameter_count % vector_len != 0) @compileError("optimizer.zig: parameter count is not divisible by natural vector length.");
}

pub fn GradientDescent(options: GradientDescentOptions) fn (type) type {
    if (options.momentum < 0) @compileError(comptimePrint("GradientDescent: invalid momentum {d} < 0", .{options.momentum}));
    if (options.learn_rate <= 0) @compileError(comptimePrint("GradientDescent: invalid learn_rate {d} <= 0", .{options.learn_rate}));

    return struct {
        pub fn Optimizer(parameter_count: usize) type {
            assertParameterCount(parameter_count);
            const vector_count = parameter_count / vector_len;
            return struct {
                const Self = @This();

                const momentum = options.momentum;
                const learn_rate = options.learn_rate;

                velocity: if (momentum == 0) void else [vector_count]Vector,

                pub const default = Self{
                    .gradient = @splat(@splat(0)),
                    .velocity = if (momentum == 0) {} else @splat(@splat(0)),
                };

                pub fn step(self: *Self, parameters: *[vector_count]Vector, gradient: *[vector_count]Vector) void {
                    if (momentum == 0) {
                        for (parameters, gradient) |*parameter, partial| {
                            parameter.* -= @as(Vector, @splat(learn_rate)) * partial;
                        }
                    } else {
                        for (parameters, gradient, &self.velocity) |*parameter, partial, *velocity| {
                            velocity.* = @as(Vector, @splat(momentum)) * velocity.* -
                                @as(Vector, @splat(learn_rate)) * partial;
                            parameter.* += velocity.*;
                        }
                    }
                }
            };
        }
    }.Optimizer;
}

pub const AdamOptions = struct {
    learn_rate: f32 = 0.01,
    beta: [2]f32 = .{ 0.9, 0.999 },
    epsilon: f32 = 1e-8,

    pub const default = @This(){};
};

pub fn Adam(options: AdamOptions) fn (usize) type {
    if (options.learn_rate <= 0)
        @compileError(comptimePrint("Adam: invalid learn_rate {d} <= 0", .{options.learn_rate}));
    if (options.epsilon <= 0)
        @compileError(comptimePrint("Adam: invalid epsilon {d} <= 0", .{options.epsilon}));
    return struct {
        pub fn Optimizer(parameter_count: usize) type {
            assertParameterCount(parameter_count);
            const vector_count = parameter_count / vector_len;
            return struct {
                const Self = @This();

                const learn_rate = options.learn_rate;
                const beta = options.beta;
                const epsilon = options.epsilon;

                m: [vector_count]Vector,
                v: [vector_count]Vector,

                pub const default = Self{
                    .m = @splat(@splat(0)),
                    .v = @splat(@splat(0)),
                };

                pub fn step(
                    self: *Self,
                    parameters: *[vector_count]Vector,
                    gradient: *[vector_count]Vector,
                ) void {
                    @setFloatMode(.optimized);
                    for (parameters, gradient, &self.m, &self.v) |*parameter, partial, *m, *v| {
                        m.* = @as(Vector, @splat(beta[0])) * m.* +
                            @as(Vector, @splat(1 - beta[0])) * partial;
                        v.* = @as(Vector, @splat(beta[1])) * v.* +
                            @as(Vector, @splat(1 - beta[1])) * partial * partial;
                        const m_hat = m.* / @as(Vector, @splat(1 - beta[0]));
                        const v_hat = v.* / @as(Vector, @splat(1 - beta[1]));
                        parameter.* -= @as(Vector, @splat(learn_rate)) * m_hat /
                            (@sqrt(v_hat) + @as(Vector, @splat(epsilon)));
                    }
                }
            };
        }
    }.Optimizer;
}
