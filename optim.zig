const std = @import("std");
const assert = std.debug.assert;

const f32x16 = @Vector(16, f32);

pub const GradientDescentOptions = struct {
    learn_rate: f32 = 0.01,
    momentum: f32 = 0,

    pub const default = @This(){};
};

pub fn GradientDescent(options: GradientDescentOptions) fn (type) type {
    if (options.momentum < 0) @compileError("GradientDescent: momentum must be nonzero");
    if (options.learn_rate <= 0) @compileError("GradientDescent: learn_rate must be positive");

    return struct {
        pub fn Optimizer(parameter_count: usize) type {
            return struct {
                const Self = @This();

                const momentum = options.momentum;
                const learn_rate = options.learn_rate;

                velocity: if (momentum == 0) void else [parameter_count]f32x16,

                pub const default = Self{
                    .gradient = @splat(@splat(0)),
                    .velocity = if (momentum == 0) {} else @splat(@splat(0)),
                };

                pub fn step(self: *Self, parameters: *[parameter_count]f32, gradient: *[parameter_count]f32) void {
                    if (momentum == 0) {
                        for (parameters, gradient) |*parameter, partial| {
                            parameter.* -= @as(f32x16, @splat(learn_rate)) * partial;
                        }
                    } else {
                        for (parameters, gradient, &self.velocity) |*parameter, partial, *velocity| {
                            velocity.* = @as(f32x16, @splat(momentum)) * velocity.* -
                                @as(f32x16, @splat(learn_rate)) * partial;
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
    if (options.learn_rate <= 0) @compileError("Adam: learn_rate must be positive");
    if (options.epsilon <= 0) @compileError("Adam: epsilon must be positive");

    return struct {
        pub fn Optimizer(parameter_count: usize) type {
            return struct {
                const Self = @This();

                const learn_rate = options.learn_rate;
                const beta = options.beta;
                const epsilon = options.epsilon;

                m: [parameter_count]f32,
                v: [parameter_count]f32,

                pub const default = Self{
                    .m = @splat(0),
                    .v = @splat(0),
                };

                pub fn step(self: *Self, parameters: *[parameter_count]f32, gradient: *[parameter_count]f32) void {
                    for (parameters, gradient, &self.m, &self.v) |*parameter, partial, *m, *v| {
                        m.* = beta[0] * m.* + (1 - beta[0]) * partial;
                        v.* = beta[1] * v.* + (1 - beta[1]) * partial * partial;
                        const m_hat = m.* / (1 - beta[0]);
                        const v_hat = v.* / (1 - beta[1]);
                        parameter.* -= learn_rate * m_hat / (@sqrt(v_hat) + epsilon);
                    }
                }
            };
        }
    }.Optimizer;
}
