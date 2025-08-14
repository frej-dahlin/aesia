const std = @import("std");

pub const Info = struct {
    iteration: usize,
    iteration_count: usize,
    epoch: usize,
    epoch_count: usize,
};
pub const Schedule = *const fn (anytype, Info) void;

pub const LinearWarmupOptions = struct {
    epoch_count: usize,
    learn_rate_min: f32 = 0,
    learn_rate_max: f32,
};
pub fn LinearWarmUp(comptime options: LinearWarmupOptions) Schedule {
    const learn_rate_min = options.learn_rate_min;
    const learn_rate_max = options.learn_rate_max;
    return struct {
        pub fn schedule(optimizer: anytype, info: Info) void {
            if (info.epoch > options.epoch_count) return;
            const iterations_per_epoch = @as(f32, @floatFromInt(info.iteration_count)) /
                @as(f32, @floatFromInt(info.epoch_count));
            const total_iterations = @as(f32, @floatFromInt(options.epoch_count)) *
                iterations_per_epoch;
            const progress = @as(f32, @floatFromInt(info.iteration)) / total_iterations;
            optimizer.learn_rate = learn_rate_min + progress * (learn_rate_max - learn_rate_min);
        }
    }.schedule;
}

pub const LinearWarmDownOptions = struct {
    percent_begin: f32,
    field_name: []const u8,
    max: f32,
    min: f32 = 0,
};
pub fn LinearWarmDown(comptime options: LinearWarmDownOptions) Schedule {
    const percent_begin = options.percent_begin;
    const field_name = options.field_name;
    const max = options.max;
    const min = options.min;
    return struct {
        pub fn schedule(target: anytype, info: Info) void {
            const quotient = @as(f32, @floatFromInt(info.iteration)) /
                @as(f32, @floatFromInt(info.iteration_count));
            if (quotient < percent_begin) return;
            @field(target, field_name) = min +
                (quotient - percent_begin) / (1 - percent_begin) * (max - min);
        }
    }.schedule;
}

pub const CosineAnnealOptions = struct {
    percent_begin: f32,
    learn_rate_max: f32,
    learn_rate_min: f32 = 0,
};
pub fn CosineAnneal(comptime options: CosineAnnealOptions) Schedule {
    const percent_begin = options.percent_begin;
    const learn_rate_min = options.learn_rate_min;
    const learn_rate_max = options.learn_rate_max;
    return struct {
        pub fn schedule(optimizer: anytype, info: Info) void {
            const quotient = @as(f32, @floatFromInt(info.epoch)) /
                @as(f32, @floatFromInt(info.epoch_count));
            if (quotient < percent_begin) return;
            const percent_active = 1 - percent_begin;
            optimizer.learn_rate = learn_rate_min +
                0.5 * (learn_rate_max - learn_rate_min) * (1 + @cos(
                    (quotient - percent_begin) / percent_active * std.math.pi,
                ));
        }
    }.schedule;
}
