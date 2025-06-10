const std = @import("std");

const logic = @import("layer/logic.zig");
const dyadic_butterfly = @import("layer/dyadic_butterfly.zig");
const pad = @import("layer/pad.zig");

pub const GroupSum = @import("layer/group_sum.zig").GroupSum;
pub const ZeroPad = @import("layer/pad.zig").ZeroPad;
pub const Repeat = pad.Repeat;

pub const ButterflySwap = dyadic_butterfly.ButterflySwap;
pub const ButterflyMap = dyadic_butterfly.ButterflyMap;
pub const BenesMap = dyadic_butterfly.BenesMap;

pub const Logic = logic.Logic;
pub const PackedLogic = logic.PackedLogic;
pub const LUTConvolution = logic.LUTConvolution;
pub const LUT = logic.LUT;
pub const LogicSequential = logic.LogicSequential;

/// Aesia layers need to declare a public constant of name "info" of the following type.
/// For more info about the layer API, see network.zig.
pub const Info = struct {
    dim_in: usize,
    dim_out: usize,
    ItemIn: type = f32,
    ItemOut: type = f32,
    in_place: bool = false,
    trainable: bool,
    statefull: bool = true,
    parameter_count: ?usize = null,
    parameter_alignment: ?usize = null,

    pub fn Input(info: Info) type {
        return [info.dim_in]info.ItemIn;
    }

    pub fn Output(info: Info) type {
        return [info.dim_out]info.ItemOut;
    }
};

/// Compile time checks on the Layer type to ensure that it is in a valid configuration and
/// satisfies the Aesia API.
pub fn check(Layer: type) void {
    const Demand = struct {
        cond: bool,
        msg: []const u8,

        pub fn inspect(demand: @This()) void {
            if (!demand.cond) printCompileError("Aesia: {s} in layer:\n{s}", .{
                demand.msg,
                @typeName(Layer),
            });
        }
    };
    // Check that the "info" is declared of the correct type and const.
    Demand.inspect(.{
        .cond = @hasDecl(Layer, "info") and @TypeOf(Layer.info) == Info,
        .msg = "must declare 'info' of type aesia.layer.Info",
    });
    Demand.inspect(.{
        .cond = @typeInfo(@TypeOf(&Layer.info)).pointer.is_const,
        .msg = "must declare 'info' to be const",
    });
    // Check that the layer's info is valid.
    const info = Layer.info;
    const demands = [_]Demand{
        .{ .cond = info.dim_in > 0, .msg = "invalid dim_in equals 0" },
        .{ .cond = info.dim_out > 0, .msg = "invalid dim_out equals 0" },
        .{
            .cond = implies(info.statefull, @sizeOf(Layer) > 0),
            .msg = "invalid size of layer type equals 0 but info.statefull is true",
        },
        .{
            .cond = implies(!info.statefull, @sizeOf(Layer) == 0),
            .msg = "invalid size of layer type > 0 but info.statefull is false",
        },
        .{
            .cond = implies(info.trainable, info.statefull),
            .msg = "invalid size of layer type equals 0 but info.trainable is true",
        },
        .{
            .cond = implies(info.trainable, info.parameter_count != null),
            .msg = "info.parameter_count is null but info.trainable is true",
        },
        .{
            .cond = implies(info.trainable, info.parameter_alignment != null),
            .msg = "info.parameter_alignment is null but info.trainable is true",
        },
        .{
            .cond = implies(!info.trainable, info.parameter_count == null),
            .msg = "info.parameter_count is not null but info.trainable is false",
        },
        .{
            .cond = implies(!info.trainable, info.parameter_alignment == null),
            .msg = "info.parameter_alignment is not null but info.trainable is false",
        },
    };
    for (demands) |demand| demand.inspect();
    // Todo: check the layers methods.
}

fn implies(p: bool, q: bool) bool {
    return !p or q;
}

fn printCompileError(comptime fmt: []const u8, args: anytype) void {
    @compileError(std.fmt.comptimePrint(fmt, args));
}
