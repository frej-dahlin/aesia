const std = @import("std");

const Allocator = std.mem.Allocator;
const ArrayList = std.ArrayList;
const assert = std.debug.assert;

const aesia = @import("aesia");

const rep : compiled_layer.GateRepresentation = compiled_layer.GateRepresentation.bitset;
const StaticBitSet = aesia.compiler.bitset.StaticBitSet;

const dim = 28;
const encoding_bits = 1;
const ImageMNIST = if(rep == .bitset) StaticBitSet(encoding_bits * dim * dim) else [dim * dim]bool;
const LabelMNIST = u8;

// "Error function" used belof for the Gaussian CDF.
fn erf(x_: f64) f32 {
    // constants
    const a1 = 0.254829592;
    const a2 = -0.284496736;
    const a3 = 1.421413741;
    const a4 = -1.453152027;
    const a5 = 1.061405429;
    const p = 0.3275911;

    // Save the sign of x
    const sign = std.math.sign(x_);
    const x = @abs(x_);

    // A&S formula 7.1.26
    const t = 1.0 / (1.0 + p * x);
    const y = 1.0 - (((((a5 * t + a4) * t) + a3) * t + a2) * t + a1) * t * @exp(-x * x);

    return @floatCast(sign * y);
}

// The Gaussian cumulative distribution function.
fn zScoreToProbability(z: f32) f32 {
    return 0.5 * (1 + erf(z / @sqrt(2.0)));
}

// Used to Z-order input indices.
fn interleaveWithZeros(input: u32) u64 {
    var word: u64 = input;
    word = (word ^ (word << 16)) & 0x0000ffff0000ffff;
    word = (word ^ (word << 8)) & 0x00ff00ff00ff00ff;
    word = (word ^ (word << 4)) & 0x0f0f0f0f0f0f0f0f;
    word = (word ^ (word << 2)) & 0x3333333333333333;
    word = (word ^ (word << 1)) & 0x5555555555555555;
    return word;
}

fn interleave(x: u32, y: u32) u64 {
    return (interleaveWithZeros(x) << 1) | interleaveWithZeros(y);
}

fn loadImages(allocator: Allocator, path: []const u8) ![]ImageMNIST {
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
    assert(row_count == 28);
    assert(col_count == 28);

    // Read raw image data and normalize.
    const RawImage = [28 * 28]f32;
    const scale = 1.0 / 255.0;
    const raw_images = try allocator.alloc(RawImage, image_count);
    defer allocator.free(raw_images);
    for (raw_images) |*image| {
        for (image) |*pixel| {
            pixel.* = @as(f32, @floatFromInt(try reader.readByte())) * scale;
        }
    }

    // Standardize using the mean and deviation and thermometer encode.
    // Fashion MNIST:
    // const mean = 0.2994749;
    // const deviation = 0.3270475;
    // Regular MNIST:
    const mean = 0.13113596;
    const deviation = 0.2885157;
    const images = try allocator.alloc(ImageMNIST, image_count);
    for (images, raw_images) |*image, raw_image| {
        inline for (0..encoding_bits) |bit| {
            for (0..28) |row| {
                for (0..28) |col| {
                    const raw_pixel = raw_image[28 * row + col];
                    const z_score = (raw_pixel - mean) / deviation;
                    const probability = zScoreToProbability(z_score);
                    const percentile = @as(f32, @floatFromInt(bit + 1)) /
                        @as(f32, @floatFromInt(encoding_bits + 1));
                    if(rep == .bitset){
                        image.setValue(dim * dim * bit + dim * row + col, if (probability >= percentile) true else false);
                    }
                    else{
                        image[28 * row + col] = if (probability >= percentile) true else false;
                    }
                }
            }
        }
    }
    return images;
}

fn loadLabels(allocator: Allocator, path: []const u8) ![]LabelMNIST {
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
    const labels = try allocator.alloc(LabelMNIST, label_count);
    assert(try reader.readAll(labels) == label_count);
    return labels;
}

const compiled_network = aesia.compiler.compiled_network;
const compiled_layer = aesia.compiler.compiled_layer;
const compiled_layer_pad = aesia.compiler.pad;
const compiled_layer_dyadic = aesia.compiler.dyadic_butterfly;
const LogicLayer = compiled_layer.Logic;
const PackedLogicLayer = compiled_layer.PackedLogic;
const LogicSequential = compiled_layer.LogicSequential;
const LUTConvolution = compiled_layer.LUTConvolutionPlies;
const GroupSum = aesia.compiler.group_sum.GroupSum;
const Repeat = compiled_layer_pad.Repeat;
const ButterflyMap = compiled_layer_dyadic.ButterflyMap;
const ButterflyGate = compiled_layer_dyadic.ButterflyGate;

var pcg = std.Random.Pcg.init(0);
var rand = pcg.random();
var pcg2 = std.Random.Pcg.init(0);
var rand2 = pcg2.random();
const width = 8000;
const Network = compiled_network.Network(&.{
    LogicLayer(784, width, .{ .rand = &rand, .gateRepresentation = rep}),
    LogicLayer(width, width, .{ .rand = &rand, .gateRepresentation = rep }),
    LogicLayer(width, width, .{ .rand = &rand, .gateRepresentation = rep }),
    LogicLayer(width, width, .{ .rand = &rand, .gateRepresentation = rep }),
    LogicLayer(width, width, .{ .rand = &rand, .gateRepresentation = rep }),
    LogicLayer(width, width, .{ .rand = &rand, .gateRepresentation = rep }),
    GroupSum(width, 10, .{ .rand = &rand, .gateRepresentation = rep }),
});


const width2 = 24000;
const Network2 = compiled_network.Network(&.{
    PackedLogicLayer(784, width2, .{ .rand = &rand, .gateRepresentation = rep  }),
    PackedLogicLayer(width2, width2, .{ .rand = &rand, .gateRepresentation = rep  }),
    GroupSum(width2, 10, .{.gateRepresentation = rep }),
});


const Network3 = compiled_network.Network(&.{
    Repeat(784, 8192, .{ .gateRepresentation = rep  }),
    ButterflyMap(13, 12, .{ .gateRepresentation = rep  }),
    ButterflyMap(13, 11, .{ .gateRepresentation = rep  }),
    ButterflyMap(13, 10, .{ .gateRepresentation = rep  }),
    ButterflyMap(13, 9, .{ .gateRepresentation = rep  }),
    ButterflyMap(13, 8, .{ .gateRepresentation = rep  }),
    ButterflyMap(13, 7, .{ .gateRepresentation = rep  }),
    ButterflyMap(13, 6, .{ .gateRepresentation = rep  }),
    ButterflyMap(13, 5, .{ .gateRepresentation = rep  }),
    ButterflyMap(13, 4, .{ .gateRepresentation = rep  }),
    ButterflyMap(13, 3, .{ .gateRepresentation = rep  }),
    ButterflyMap(13, 2, .{ .gateRepresentation = rep  }),
    ButterflyMap(13, 1, .{ .gateRepresentation = rep  }),
    ButterflyMap(13, 0, .{ .gateRepresentation = rep  }),
    ButterflyMap(13, 1, .{ .gateRepresentation = rep  }),
    ButterflyMap(13, 2, .{ .gateRepresentation = rep  }),
    ButterflyMap(13, 3, .{ .gateRepresentation = rep  }),
    ButterflyMap(13, 4, .{ .gateRepresentation = rep  }),
    ButterflyMap(13, 5, .{ .gateRepresentation = rep  }),
    ButterflyMap(13, 6, .{ .gateRepresentation = rep  }),
    ButterflyMap(13, 7, .{ .gateRepresentation = rep  }),
    ButterflyMap(13, 8, .{ .gateRepresentation = rep  }),
    ButterflyMap(13, 9, .{ .gateRepresentation = rep  }),
    ButterflyMap(13, 10, .{ .gateRepresentation = rep  }),
    ButterflyMap(13, 11, .{ .gateRepresentation = rep  }),
    ButterflyMap(13, 12, .{ .gateRepresentation = rep  }),
    LogicSequential(4096, .{ .gateRepresentation = rep  }),
    Repeat(4096, 8192, .{.gateRepresentation = rep  }),
    ButterflyMap(13, 12, .{ .gateRepresentation = rep  }),
    ButterflyMap(13, 11, .{ .gateRepresentation = rep  }),
    ButterflyMap(13, 10, .{ .gateRepresentation = rep  }),
    ButterflyMap(13, 9, .{ .gateRepresentation = rep  }),
    ButterflyMap(13, 8, .{ .gateRepresentation = rep  }),
    ButterflyMap(13, 7, .{ .gateRepresentation = rep  }),
    ButterflyMap(13, 6, .{ .gateRepresentation = rep  }),
    ButterflyMap(13, 5, .{ .gateRepresentation = rep  }),
    ButterflyMap(13, 4, .{ .gateRepresentation = rep  }),
    ButterflyMap(13, 3, .{ .gateRepresentation = rep  }),
    ButterflyMap(13, 2, .{ .gateRepresentation = rep  }),
    ButterflyMap(13, 1, .{ .gateRepresentation = rep  }),
    ButterflyMap(13, 0, .{ .gateRepresentation = rep  }),
    LogicSequential(4096, .{ .gateRepresentation = rep  }),
    Repeat(4096, 8192, .{.gateRepresentation = rep  }),
    ButterflyMap(13, 0, .{ .gateRepresentation = rep  }),
    ButterflyMap(13, 1, .{ .gateRepresentation = rep  }),
    ButterflyMap(13, 2, .{ .gateRepresentation = rep  }),
    ButterflyMap(13, 3, .{ .gateRepresentation = rep  }),
    ButterflyMap(13, 4, .{ .gateRepresentation = rep  }),
    ButterflyMap(13, 5, .{ .gateRepresentation = rep  }),
    ButterflyMap(13, 6, .{ .gateRepresentation = rep  }),
    ButterflyMap(13, 7, .{ .gateRepresentation = rep  }),
    ButterflyMap(13, 8, .{ .gateRepresentation = rep  }),
    ButterflyMap(13, 9, .{ .gateRepresentation = rep  }),
    ButterflyMap(13, 10, .{ .gateRepresentation = rep  }),
    ButterflyMap(13, 11, .{ .gateRepresentation = rep  }),
    ButterflyMap(13, 12, .{ .gateRepresentation = rep  }),
    LogicSequential(4096, .{ .gateRepresentation = rep  }),
    GroupSum(4096, 10, .{ .gateRepresentation = rep }),
});

const Network4 = compiled_network.Network(&.{
    Repeat(28*28, 1 << 10, .{ .gateRepresentation = rep  }),
    ButterflyGate(10, 0, .{ .gateRepresentation = rep  }),
    ButterflyGate(10, 1, .{ .gateRepresentation = rep  }),
    ButterflyGate(10, 2, .{ .gateRepresentation = rep  }),
    ButterflyGate(10, 3, .{ .gateRepresentation = rep  }),
    ButterflyGate(10, 4, .{ .gateRepresentation = rep  }),
    ButterflyGate(10, 5, .{ .gateRepresentation = rep  }),
    ButterflyGate(10, 6, .{ .gateRepresentation = rep  }),
    ButterflyGate(10, 7, .{ .gateRepresentation = rep  }),
    ButterflyGate(10, 8, .{ .gateRepresentation = rep  }),
    ButterflyGate(10, 9, .{ .gateRepresentation = rep  }),
    GroupSum(1 << 10, 10, .{ .gateRepresentation = rep }),
});

const model_scale = 4;
// const ConvolutionalNetwork = @import("compiled_network.zig").Network(&.{
//     LUTConvolution(.{
//         .depth = 1,
//         .height = 28,
//         .width = 28,
//         .lut_count = model_scale,
//         .field_size = .{ .height = 3, .width = 3 },
//         .stride = .{ .row = 1, .col = 1 },
//     }),
//     LUTConvolution(.{
//         .depth = model_scale,
//         .height = 26,
//         .width = 26,
//         .lut_count = 1,
//         .field_size = .{ .height = 2, .width = 2 },
//         .stride = .{ .row = 2, .col = 2 },
//     }),
//     LUTConvolution(.{
//         .depth = model_scale,
//         .height = 13,
//         .width = 13,
//         .lut_count = 4,
//         .field_size = .{ .height = 3, .width = 3 },
//         .stride = .{ .row = 2, .col = 2 },
//     }),
//     LUTConvolution(.{
//         .depth = 4 * model_scale,
//         .height = 6,
//         .width = 6,
//         .lut_count = 4,
//         .field_size = .{ .height = 2, .width = 2 },
//         .stride = .{ .row = 2, .col = 2 },
//     }),
//     PackedLogicLayer(16 * model_scale * 3 * 3, 32_000, .{ .rand = &rand2, .gateRepresentation = rep }),
//     PackedLogicLayer(32_000, 16_000, .{ .rand = &rand2, .gateRepresentation = rep }),
//     PackedLogicLayer(16_000, 8_000, .{ .rand = &rand2, .gateRepresentation = rep }),
//     GroupSum(8_000, 10, .{ .gateRepresentation = rep }),
// });
//var network: Network = undefined;
//var network: Network2 = undefined;
//var network: Network3 = undefined;
var network: Network4 = undefined;
//var convNetwork: ConvolutionalNetwork = undefined;

pub fn main() !void {
    //try network.compileFromFile("mnist.model");
    //try network.compileFromFile("mnist_packed.model");
    //try network.compileFromFile("mnist_butterfly_inv.model");
    try network.compileFromFile("mnist_emil.model");
    var gpa = std.heap.GeneralPurposeAllocator(.{}){};
    const allocator = gpa.allocator();

    const images_validate = try loadImages(allocator, "data/mnist/t10k-images-idx3-ubyte.gz");
    const labels_validate = try loadLabels(allocator, "data/mnist/t10k-labels-idx1-ubyte.gz");


    var correct_count: usize = 0;
    for (images_validate, labels_validate) |image, label| {
        const prediction = network.eval(&image);
        if (std.mem.indexOfMax(usize, prediction) == label) correct_count += 1;
    }

    var timer = try std.time.Timer.start();
    const samplesize = 100;
    for(0..samplesize) |k|{
        _ = k;
        for (images_validate) |image| {
            const prediction = network.eval(&image);
            _ = prediction;
        }
    }

    std.debug.print("Evaluation took: {d}ms\n", .{timer.read() / std.time.ns_per_ms});
    std.debug.print("Evaluation per image: {d}ns\n", .{timer.read() / (images_validate.len*samplesize)});
    std.debug.print(
        "Correctly classified {d} / {d} ~ {d}%\n",
        .{
            correct_count,
            images_validate.len,
            100 * @as(f32, @floatFromInt(correct_count)) /
                @as(f32, @floatFromInt(images_validate.len)),
        },
    );

    //std.debug.print("Permutation took: {d}ms\n", .{network.layers[1].permtime / std.time.ns_per_ms});
    //std.debug.print("Gate evaluation took: {d}us\n", .{network.layers[1].evaltime / std.time.ns_per_us});


    // try convNetwork.compileFromFile("lut-convolution-discrete.model");
    // timer = try std.time.Timer.start();

    // correct_count = 0;
    // for (images_validate, labels_validate) |image, label| {
    //     const prediction = convNetwork.eval(&image);
    //     if (std.mem.indexOfMax(usize, prediction) == label) correct_count += 1;
    // }

    // std.debug.print(
    //     "Correctly classified {d} / {d} ~ {d}%\n",
    //     .{
    //         correct_count,
    //         images_validate.len,
    //         100 * @as(f32, @floatFromInt(correct_count)) /
    //             @as(f32, @floatFromInt(images_validate.len)),
    //     },
    // );
    // std.debug.print("Evaluation took: {d}ms\n", .{timer.read() / std.time.ns_per_ms});

    // std.debug.print("Permutation took: {d}ms\n", .{convNetwork.layers[1].permtime / std.time.ns_per_ms});
    // std.debug.print("Gate evaluation took: {d}us\n", .{convNetwork.layers[1].evaltime / std.time.ns_per_us});
}
