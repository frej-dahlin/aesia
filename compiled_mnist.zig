const std = @import("std");
const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const assert = std.debug.assert;

const rep : compiled_layer.GateRepresentation = compiled_layer.GateRepresentation.boolarray;
const StaticBitSet = @import("compiled_layer/bitset.zig").StaticBitSet;

const aesia = @import("aesia.zig");

const dim = 28;
const Image = if(rep == .bitset) StaticBitSet(dim*dim) else [dim * dim]bool;
const Label = u8;


fn loadImages(allocator: Allocator, path: []const u8) ![]Image {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    var decompressor = std.compress.gzip.decompressor(file.reader());
    var buffered = std.io.bufferedReader(decompressor.reader());
    var reader = buffered.reader();

    // Assert magic value.
    assert(try reader.readByte() == 0); // Specified by IDX
    assert(try reader.readByte() == 0); // --/--
    assert(try reader.readByte() == 0x08); // 0x08 means bytes.
    assert(try reader.readByte() == 0x03); // Three dimensios, image count, rows, and columns.
    const image_count = try reader.readInt(u32, .big);
    std.debug.print("Reading {d} images...\n", .{image_count});
    const row_count = try reader.readInt(u32, .big);
    const col_count = try reader.readInt(u32, .big);
    assert(row_count == dim);
    assert(col_count == dim);
    const images = try allocator.alloc(Image, image_count);
    for (images) |*image| {
        for (0..dim*dim) |k| {
            if(rep == .bitset){
                image.setValue(k, if (try reader.readByte() > 255/2) true else false);
            }
            else{
                image[k] = if (try reader.readByte() >= 128) true else false;
            }
        }
    }
    return images;
}

fn loadLabels(allocator: Allocator, path: []const u8) ![]Label {
    const file = try std.fs.cwd().openFile(path, .{});
    defer file.close();

    var decompressor = std.compress.gzip.decompressor(file.reader());
    var buffered = std.io.bufferedReader(decompressor.reader());
    var reader = buffered.reader();

    // Assert magic value.
    if (try reader.readByte() != 0) return error.InvalidIDXMagicValue;
    if (try reader.readByte() != 0) return error.InvalidIDXMagicValue;
    if (try reader.readByte() != 0x08) return error.InvalidEntrySize;
    if (try reader.readByte() != 0x01) return error.UnexpectedDimensionCount;
    const label_count = try reader.readInt(u32, .big);
    std.debug.print("Reading {d} labels...\n", .{label_count});
    const labels = try allocator.alloc(Label, label_count);
    assert(try reader.readAll(labels) == label_count);
    return labels;
}

const compiled_layer = @import("compiled_layer/logic.zig");
const compiled_layer_pad = @import("compiled_layer/pad.zig");
const compiled_layer_dyadic = @import("compiled_layer/dyadic_butterfly.zig");
const LogicLayer = compiled_layer.Logic;
const PackedLogicLayer = compiled_layer.PackedLogic;
const LogicSequential = compiled_layer.LogicSequential;
const LUTConvolution = compiled_layer.LUTConvolutionPlies;
const GroupSum = compiled_layer.GroupSum;
const Repeat = compiled_layer_pad.Repeat;
const ButterflyMap = compiled_layer_dyadic.ButterflyMap;

var pcg = std.Random.Pcg.init(0);
var rand = pcg.random();
var pcg2 = std.Random.Pcg.init(0);
var rand2 = pcg2.random();
const width = 8000;
const Network = @import("compiled_network.zig").Network(&.{
    LogicLayer(784, width, .{ .rand = &rand, .gateRepresentation = rep}),
    LogicLayer(width, width, .{ .rand = &rand, .gateRepresentation = rep }),
    LogicLayer(width, width, .{ .rand = &rand, .gateRepresentation = rep }),
    LogicLayer(width, width, .{ .rand = &rand, .gateRepresentation = rep }),
    LogicLayer(width, width, .{ .rand = &rand, .gateRepresentation = rep }),
    LogicLayer(width, width, .{ .rand = &rand, .gateRepresentation = rep }),
    GroupSum(width, 10, .{ .rand = &rand, .gateRepresentation = rep }),
});


const width2 = 24000;
const Network2 = @import("compiled_network.zig").Network(&.{
    PackedLogicLayer(784, width2, .{ .rand = &rand, .gateRepresentation = rep  }),
    PackedLogicLayer(width2, width2, .{ .rand = &rand, .gateRepresentation = rep  }),
    GroupSum(width2, 10, .{ .rand = &rand, .gateRepresentation = rep }),
});


const Network3 = @import("compiled_network.zig").Network(&.{
    Repeat(784, 8192, .{ .gateRepresentation = rep  }),
    ButterflyMap(13, 12),
    ButterflyMap(13, 11),
    ButterflyMap(13, 10),
    ButterflyMap(13, 9),
    ButterflyMap(13, 8),
    ButterflyMap(13, 7),
    ButterflyMap(13, 6),
    ButterflyMap(13, 5),
    ButterflyMap(13, 4),
    ButterflyMap(13, 3),
    ButterflyMap(13, 2),
    ButterflyMap(13, 1),
    ButterflyMap(13, 0),
    ButterflyMap(13, 1),
    ButterflyMap(13, 2),
    ButterflyMap(13, 3),
    ButterflyMap(13, 4),
    ButterflyMap(13, 5),
    ButterflyMap(13, 6),
    ButterflyMap(13, 7),
    ButterflyMap(13, 8),
    ButterflyMap(13, 9),
    ButterflyMap(13, 10),
    ButterflyMap(13, 11),
    ButterflyMap(13, 12),
    LogicSequential(4096, .{ .gateRepresentation = rep  }),
    Repeat(4096, 8192, .{.gateRepresentation = rep  }),
    ButterflyMap(13, 12),
    ButterflyMap(13, 11),
    ButterflyMap(13, 10),
    ButterflyMap(13, 9),
    ButterflyMap(13, 8),
    ButterflyMap(13, 7),
    ButterflyMap(13, 6),
    ButterflyMap(13, 5),
    ButterflyMap(13, 4),
    ButterflyMap(13, 3),
    ButterflyMap(13, 2),
    ButterflyMap(13, 1),
    ButterflyMap(13, 0),
    LogicSequential(4096, .{ .gateRepresentation = rep  }),
    Repeat(4096, 8192, .{.gateRepresentation = rep  }),
    ButterflyMap(13, 0),
    ButterflyMap(13, 1),
    ButterflyMap(13, 2),
    ButterflyMap(13, 3),
    ButterflyMap(13, 4),
    ButterflyMap(13, 5),
    ButterflyMap(13, 6),
    ButterflyMap(13, 7),
    ButterflyMap(13, 8),
    ButterflyMap(13, 9),
    ButterflyMap(13, 10),
    ButterflyMap(13, 11),
    ButterflyMap(13, 12),
    LogicSequential(4096, .{ .gateRepresentation = rep  }),
    GroupSum(4096, 10, .{ .gateRepresentation = rep }),
});

const model_scale = 4;
const ConvolutionalNetwork = @import("compiled_network.zig").Network(&.{
    LUTConvolution(.{
        .depth = 1,
        .height = 28,
        .width = 28,
        .lut_count = model_scale,
        .field_size = .{ .height = 3, .width = 3 },
        .stride = .{ .row = 1, .col = 1 },
    }),
    LUTConvolution(.{
        .depth = model_scale,
        .height = 26,
        .width = 26,
        .lut_count = 1,
        .field_size = .{ .height = 2, .width = 2 },
        .stride = .{ .row = 2, .col = 2 },
    }),
    LUTConvolution(.{
        .depth = model_scale,
        .height = 13,
        .width = 13,
        .lut_count = 4,
        .field_size = .{ .height = 3, .width = 3 },
        .stride = .{ .row = 2, .col = 2 },
    }),
    LUTConvolution(.{
        .depth = 4 * model_scale,
        .height = 6,
        .width = 6,
        .lut_count = 4,
        .field_size = .{ .height = 2, .width = 2 },
        .stride = .{ .row = 2, .col = 2 },
    }),
    PackedLogicLayer(16 * model_scale * 3 * 3, 32_000, .{ .rand = &rand2, .gateRepresentation = rep }),
    PackedLogicLayer(32_000, 16_000, .{ .rand = &rand2, .gateRepresentation = rep }),
    PackedLogicLayer(16_000, 8_000, .{ .rand = &rand2, .gateRepresentation = rep }),
    GroupSum(8_000, 10, .{ .gateRepresentation = rep }),
});
//var network: Network = undefined;
//var network: Network2 = undefined;
var network: Network3 = undefined;
var convNetwork: ConvolutionalNetwork = undefined;

pub fn main() !void {
    //try network.compileFromFile("mnist.model");
    //try network.compileFromFile("mnist_packed.model");
    try network.compileFromFile("mnist_butterfly.model");
    const allocator = std.heap.page_allocator;
    const images_validate = try loadImages(allocator, "data/t10k-images-idx3-ubyte.gz");
    const labels_validate = try loadLabels(allocator, "data/t10k-labels-idx1-ubyte.gz");

    var timer = try std.time.Timer.start();

    var correct_count: usize = 0;
    for (images_validate, labels_validate) |image, label| {
        const prediction = network.eval(&image);
        if (std.mem.indexOfMax(usize, prediction) == label) correct_count += 1;
    }

    std.debug.print(
        "Correctly classified {d} / {d} ~ {d}%\n",
        .{
            correct_count,
            images_validate.len,
            100 * @as(f32, @floatFromInt(correct_count)) /
                @as(f32, @floatFromInt(images_validate.len)),
        },
    );
    std.debug.print("Evaluation took: {d}ms\n", .{timer.read() / std.time.ns_per_ms});

    //std.debug.print("Permutation took: {d}ms\n", .{network.layers[1].permtime / std.time.ns_per_ms});
    //std.debug.print("Gate evaluation took: {d}us\n", .{network.layers[1].evaltime / std.time.ns_per_us});


    try convNetwork.compileFromFile("lut-convolution-discrete.model");
    timer = try std.time.Timer.start();

    correct_count = 0;
    for (images_validate, labels_validate) |image, label| {
        const prediction = convNetwork.eval(&image);
        if (std.mem.indexOfMax(usize, prediction) == label) correct_count += 1;
    }

    std.debug.print(
        "Correctly classified {d} / {d} ~ {d}%\n",
        .{
            correct_count,
            images_validate.len,
            100 * @as(f32, @floatFromInt(correct_count)) /
                @as(f32, @floatFromInt(images_validate.len)),
        },
    );
    std.debug.print("Evaluation took: {d}ms\n", .{timer.read() / std.time.ns_per_ms});

    std.debug.print("Permutation took: {d}ms\n", .{convNetwork.layers[1].permtime / std.time.ns_per_ms});
    std.debug.print("Gate evaluation took: {d}us\n", .{convNetwork.layers[1].evaltime / std.time.ns_per_us});
}
