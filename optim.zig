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
		pub fn Optimizer(ModelType: type) type {
			return struct {
				const Self = @This();
				
				const momentum = options.momentum;
				const learn_rate = options.learn_rate;
				
				velocity: if (momentum == 0) void else [ModelType.NetworkType.node_count]f32x16,
				
				pub const default = if (momentum == 0) Self{.velocity = {}} else Self{.velocity = @splat(@splat(0))};
				
				pub fn step(self: *Self, model: *ModelType) void {
					const training = model.dataset_training;
					assert(training.input.len == training.output.len);
					const net = &model.network;
					@memset(&net.gradient, @splat(0));
					for (training.input, training.output) |x, y_real| {
						model.backprop(&x, &y_real); 
					}
					if (momentum == 0) {
						for (&net.logit, net.gradient) |*logit, gradient| {
							logit.* -= @as(f32x16, @splat(learn_rate)) * gradient;
						}
					} else {
						for (&net.logit, net.gradient, &self.velocity) |*logit, gradient,  *velocity| {
							velocity.* = @as(f32x16, @splat(momentum)) * velocity.* -
								@as(f32x16, @splat(learn_rate)) * gradient;
							logit.* += velocity.*;	
						}
					}
				}
			};
		}
	}.Optimizer;
}
