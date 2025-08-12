const std = @import("std");
const assert = std.debug.assert;
const comptimePrint = std.fmt.comptimePrint;

const aesia = @import("aesia.zig");

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

                learn_rate: f32 = options.learn_rate,
                beta: [2]f32 = options.beta,
                epsilon: f32 = options.epsilon,

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
                    const learn_rate = self.learn_rate;
                    const beta = self.beta;
                    const epsilon = self.epsilon;
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

pub const SGDOptions = struct {
    learn_rate: f32 = 0.01,
    momentum: f32 = 0.9,
    weight_decay: f32 = 0.0002,
    schedules: []const aesia.schedule.Schedule = &.{},

    pub const default = @This(){};
};

pub fn SGD(options: SGDOptions) fn (usize) type {
    return struct {
        pub fn Optimizer(parameter_count: usize) type {
            return struct {
                const Self = @This();

                const momentum = options.momentum;
                const weight_decay = options.weight_decay;

                b: [parameter_count]f32,
                learn_rate: f32 = options.learn_rate,

                pub const default = Self{
                    .b = @splat(0),
                };

                pub fn step(
                    self: *Self,
                    info: aesia.schedule.Info,
                    parameters: *[parameter_count]f32,
                    gradient: *[parameter_count]f32,
                ) void {
                    @setFloatMode(.optimized);
                    inline for (options.schedules) |schedule| schedule(self, info);
                    const learn_rate = self.learn_rate;
                    for (parameters, gradient, &self.b) |*parameter, partial, *b| {
                        parameter.* -= learn_rate * weight_decay * parameter.*;
                        b.* = momentum * b.* + partial;
                        parameter.* -= learn_rate * b.*;
                    }
                }
            };
        }
    }.Optimizer;
}

fn softPlus(beta: f32, x: f32) f32 {
    return @log(1 + @exp(beta * x)) / beta;
}

pub const AdamWOptions = struct {
    learn_rate: f32 = 0.01,
    weight_decay: f32 = 0.0002,
    beta: [2]f32 = .{ 0.95, 0.999 },
    epsilon: f32 = 1e-6,
    schedules: []const aesia.schedule.Schedule = &.{},

    pub const default = @This(){};
};
pub fn AdamW(options: AdamWOptions) fn (usize) type {
    if (options.learn_rate <= 0)
        @compileError(comptimePrint("AdamW: invalid learn_rate {d} <= 0", .{options.learn_rate}));
    if (options.epsilon <= 0)
        @compileError(comptimePrint("AdamW: invalid epsilon {d} <= 0", .{options.epsilon}));
    return struct {
        pub fn Optimizer(parameter_count: usize) type {
            return struct {
                const Self = @This();

                learn_rate: f32 = options.learn_rate,
                weight_decay: f32 = options.weight_decay,
                beta: [2]f32 = options.beta,
                epsilon: f32 = options.epsilon,

                m: [parameter_count]f32,
                v: [parameter_count]f32,

                pub const default = Self{
                    .m = @splat(0),
                    .v = @splat(0),
                };

                pub fn step(
                    self: *Self,
                    info: aesia.schedule.Info,
                    noalias parameters: *[parameter_count]f32,
                    noalias gradient: *[parameter_count]f32,
                ) void {
                    @setFloatMode(.optimized);
                    inline for (options.schedules) |schedule| schedule(self, info);

                    const learn_rate = self.learn_rate;
                    const weight_decay = self.weight_decay;
                    const epsilon = self.epsilon;

                    const beta = self.beta;
                    const beta_t: [2]f32 = .{
                        std.math.pow(f32, beta[0], @floatFromInt(info.iteration)),
                        std.math.pow(f32, beta[1], @floatFromInt(info.iteration)),
                    };

                    for (parameters, gradient, &self.m, &self.v) |*parameter, partial, *m, *v| {
                        // Decoupled weight decay.
                        parameter.* -= learn_rate * weight_decay * parameter.*;

                        // Update first and second momentums.
                        m.* = beta[0] * m.* + (1 - beta[0]) * partial;
                        v.* = beta[1] * v.* + (1 - beta[1]) * partial * partial;

                        // Update parameters.
                        const m_hat = m.* / (1 - beta_t[0]);
                        const v_hat = v.* / (1 - beta_t[1]);
                        parameter.* -= learn_rate * m_hat / (@sqrt(v_hat) + epsilon);
                        // parameter.* -= learn_rate / softPlus(200, @sqrt(v.*)) * m.*;
                    }
                }
            };
        }
    }.Optimizer;
}

pub const LookAheadOptions = struct {
    cycle: usize,
    blend: f32,

    pub const default: LookAheadOptions = .{
        .cycle = 10,
        .blend = 0.5,
    };
};

// Wraps an optimizer in a lookahead optimizer.
pub fn LookAhead(options: LookAheadOptions, InnerOptimizer: fn (usize) type) fn (usize) type {
    return struct {
        pub fn OuterOptimizer(parameter_count: usize) type {
            return struct {
                const Self = @This();

                inner_optimizer: InnerOptimizer(parameter_count),
                // The "slow" parameters.
                phi: [parameter_count]f32,

                pub const default: Self = .{
                    .inner_optimizer = .default,
                    .phi = @splat(0),
                };

                pub fn step(
                    self: *Self,
                    info: aesia.schedule.Info,
                    parameters: *[parameter_count]f32,
                    gradient: *[parameter_count]f32,
                ) void {
                    @setFloatMode(.optimized);
                    if (info.iteration % options.cycle == 0) {
                        const blend = options.blend;
                        for (&self.phi, parameters) |*phi, *parameter| {
                            phi.* = blend * phi.* + (1 - blend) * parameter.*;
                            parameter.* = phi.*;
                        }
                    } else {
                        self.inner_optimizer.step(info, parameters, gradient);
                    }
                }
            };
        }
    }.OuterOptimizer;
}

pub const RAdamWOptions = struct {
    learn_rate: f32 = 0.01,
    weight_decay: f32 = 0.0001,
    beta: [2]f32 = .{ 0.90, 0.99 },
    epsilon: f32 = 1e-6,

    pub const default = @This(){};
};

pub fn RAdamW(options: AdamWOptions) fn (usize) type {
    if (options.learn_rate <= 0)
        @compileError(comptimePrint("AdamW: invalid learn_rate {d} <= 0", .{options.learn_rate}));
    if (options.epsilon <= 0)
        @compileError(comptimePrint("AdamW: invalid epsilon {d} <= 0", .{options.epsilon}));
    return struct {
        pub fn Optimizer(parameter_count: usize) type {
            return struct {
                const Self = @This();

                const mu = 0.5;
                const k = 6;

                learn_rate: f32 = options.learn_rate,
                weight_decay: f32 = options.weight_decay,
                beta: [2]f32 = options.beta,
                epsilon: f32 = options.epsilon,

                m: [parameter_count]f32,
                v: [parameter_count]f32,
                phi: [parameter_count]f32,

                pub const default = Self{
                    .m = @splat(0),
                    .v = @splat(0),
                    .phi = @splat(0),
                };

                pub fn step(
                    self: *Self,
                    info: aesia.schedule.Info,
                    parameters: *[parameter_count]f32,
                    gradient: *[parameter_count]f32,
                ) void {
                    @setFloatMode(.optimized);
                    const t: f32 = @floatFromInt(info.iteration);
                    const learn_rate = self.learn_rate;
                    const weight_decay = self.weight_decay;
                    const beta = self.beta;
                    const beta_t: [2]f32 = .{
                        std.math.pow(f32, beta[0], t),
                        std.math.pow(f32, beta[1], t),
                    };
                    const epsilon = self.epsilon;
                    const rho_inf = 2 / (1 - beta[1]) - 1;
                    for (gradient, parameters, &self.m, &self.v, &self.phi) |
                        partial,
                        *parameter,
                        *m,
                        *v,
                        *phi,
                    | {
                        m.* = beta[0] * m.* + (1 - beta[0]) * partial;
                        v.* = beta[1] * v.* + (1 - beta[1]) * partial * partial;
                        parameter.* -= learn_rate * weight_decay * parameter.*;
                        const m_hat = m.* / (1 - beta_t[0]);
                        const rho_t = rho_inf - 2 * t * beta_t[1] / (1 - beta_t[1]);
                        if (info.iteration % k != 0) {
                            if (rho_t > 5) {
                                const l_t = @sqrt(1 - beta_t[1]) / (epsilon + @sqrt(v.*));
                                const r_t = @sqrt(
                                    (rho_t - 4) * (rho_t - 2) * rho_inf /
                                        ((rho_inf - 4) * (rho_inf - 2) * rho_t),
                                );
                                parameter.* -= learn_rate * m_hat * r_t * l_t;
                            } else {
                                parameter.* -= learn_rate * m_hat;
                            }
                        } else {
                            phi.* = mu * phi.* + (1 - mu) * parameter.*;
                            parameter.* = phi.*;
                        }
                    }
                }
            };
        }
    }.Optimizer;
}

pub const SAMOptions = struct {
    rho: f32 = 0.05,

    pub const default = @This(){};
};

pub fn SAM(options: SAMOptions) fn (usize) type {
    return struct {
        pub fn Optimizer(parameter_count: usize) type {
            return struct {
                const Self = @This();

                const rho = options.rho;

                cache: [parameter_count]f32,

                pub fn step(
                    self: *Self,
                    parameters: *[parameter_count]f32,
                    gradient: *[parameter_count]f32,
                ) void {
                    @setFloatMode(.optimized);
                    @memcpy(&self.cache, parameters);
                    var norm: f32 = 0;
                    for (gradient) |partial| norm += partial * partial;
                    norm = @sqrt(norm);
                    for (parameters, gradient) |*parameter, partial| {
                        parameter.* += rho * partial / norm;
                    }
                }

                pub fn unstep(noalias self: *Self, noalias parameters: *[parameter_count]f32) void {
                    @memcpy(parameters, &self.cache);
                }
            };
        }
    }.Optimizer;
}
