#import <CoreML/CoreML.h>
#import <Foundation/Foundation.h>

#include "keyit/keyit.h"

#include <algorithm>
#include <cmath>
#include <dlfcn.h>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

namespace keyit {
namespace {

bool fill_multiarray(MLMultiArray* array, const std::vector<float>& input_nchw) {
    if (!array || array.dataType != MLMultiArrayDataTypeFloat32) {
        return false;
    }
    if (array.count != static_cast<NSInteger>(input_nchw.size())) {
        return false;
    }

    float* ptr = static_cast<float*>(array.dataPointer);
    if (!ptr) {
        return false;
    }
    std::copy(input_nchw.begin(), input_nchw.end(), ptr);
    return true;
}

std::vector<float> flatten_feature_value(MLFeatureValue* value) {
    if (!value || value.type != MLFeatureTypeMultiArray) {
        return {};
    }
    MLMultiArray* array = value.multiArrayValue;
    if (!array || array.dataType != MLMultiArrayDataTypeFloat32) {
        return {};
    }
    const float* ptr = static_cast<const float*>(array.dataPointer);
    if (!ptr) {
        return {};
    }
    return std::vector<float>(ptr, ptr + array.count);
}

NSString* resolve_model_path(const KeyitConfig& config) {
    NSFileManager* fm = [NSFileManager defaultManager];
    if (!config.model_path.empty()) {
        NSString* candidate = [NSString stringWithUTF8String:config.model_path.c_str()];
        if ([fm fileExistsAtPath:candidate]) {
            return candidate;
        }
    }

    // Resolve model from KeyIt.framework bundle resources when linked as a framework.
    Dl_info self_info{};
    if (dladdr(reinterpret_cast<const void*>(&resolve_model_path), &self_info) != 0 && self_info.dli_fname) {
        NSString* self_path = [NSString stringWithUTF8String:self_info.dli_fname];
        NSString* framework_path = [[[self_path stringByDeletingLastPathComponent]
                                     stringByDeletingLastPathComponent]
                                    stringByDeletingLastPathComponent];
        NSBundle* framework_bundle = [NSBundle bundleWithPath:framework_path];
        if (framework_bundle) {
            NSString* bundled_modelc = [framework_bundle pathForResource:@"keynet" ofType:@"mlmodelc"];
            if (bundled_modelc && [fm fileExistsAtPath:bundled_modelc]) {
                return bundled_modelc;
            }
            NSString* bundled_mlpackage = [framework_bundle pathForResource:@"keynet" ofType:@"mlpackage"];
            if (bundled_mlpackage && [fm fileExistsAtPath:bundled_mlpackage]) {
                return bundled_mlpackage;
            }
        }
    }

    NSArray<NSString*>* defaults = @[
        @"models/keynet.mlmodelc",
        @"models/keynet.mlpackage",
        @"/opt/homebrew/share/keyit/keynet.mlmodelc",
        @"/usr/local/share/keyit/keynet.mlmodelc"
    ];

    for (NSString* path in defaults) {
        if ([fm fileExistsAtPath:path]) {
            return path;
        }
    }

    NSString* bundled = [[NSBundle mainBundle] pathForResource:@"keynet" ofType:@"mlmodelc"];
    if (bundled) {
        return bundled;
    }
    bundled = [[NSBundle mainBundle] pathForResource:@"keynet" ofType:@"mlpackage"];
    return bundled;
}

NSString* nsstring(const std::string& text) {
    return [NSString stringWithUTF8String:text.c_str()];
}

NSString* compute_unit_name(MLComputeUnits units) {
    switch (units) {
        case MLComputeUnitsCPUOnly:
            return @"CPUOnly";
        case MLComputeUnitsCPUAndGPU:
            return @"CPUAndGPU";
        case MLComputeUnitsCPUAndNeuralEngine:
            return @"CPUAndNeuralEngine";
        case MLComputeUnitsAll:
        default:
            return @"All";
    }
}

} // namespace

bool run_keynet_coreml(const std::vector<float>& input_nchw,
                       const KeyitConfig& config,
                       std::vector<float>* logits,
                       std::string* error) {
    if (!logits) {
        if (error) {
            *error = "Internal error: logits output pointer is null";
        }
        return false;
    }

    @autoreleasepool {
        NSString* model_path = resolve_model_path(config);
        if (!model_path) {
            if (error) {
                *error = "CoreML model not found. Set --model, bundle keynet.mlmodelc in KeyIt.framework, or place keynet.mlmodelc in models/.";
            }
            return false;
        }

        NSURL* model_url = [NSURL fileURLWithPath:model_path];
        NSError* ns_error = nil;
        NSString* ext = model_url.pathExtension.lowercaseString;
        if (![ext isEqualToString:@"mlmodelc"]) {
            NSURL* compiled_url = [MLModel compileModelAtURL:model_url error:&ns_error];
            if (!compiled_url || ns_error) {
                if (error) {
                    *error = [[NSString stringWithFormat:@"Failed to compile model %@: %@",
                               model_path,
                               ns_error.localizedDescription] UTF8String];
                }
                return false;
            }
            model_url = compiled_url;
        }

        if (config.cqt_bins == 0 || input_nchw.empty()) {
            if (error) {
                *error = "Invalid model input tensor size";
            }
            return false;
        }
        if (input_nchw.size() % config.cqt_bins != 0) {
            if (error) {
                *error = "Input tensor size is not divisible by cqt_bins";
            }
            return false;
        }
        const std::size_t frames = input_nchw.size() / config.cqt_bins;
        MLMultiArray* input_array = [[MLMultiArray alloc] initWithShape:@[@1, @1, @(config.cqt_bins), @(frames)]
                                                                dataType:MLMultiArrayDataTypeFloat32
                                                                   error:&ns_error];
        if (!input_array || ns_error) {
            if (error) {
                *error = [[NSString stringWithFormat:@"Failed to allocate input tensor: %@",
                           ns_error.localizedDescription] UTF8String];
            }
            return false;
        }

        if (!fill_multiarray(input_array, input_nchw)) {
            if (error) {
                *error = "Failed to fill CoreML input tensor";
            }
            return false;
        }

        MLFeatureValue* input_value = [MLFeatureValue featureValueWithMultiArray:input_array];
        NSString* input_name = nsstring(config.input_name);
        NSDictionary<NSString*, MLFeatureValue*>* dict = @{input_name: input_value};

        MLDictionaryFeatureProvider* provider =
            [[MLDictionaryFeatureProvider alloc] initWithDictionary:dict error:&ns_error];
        if (!provider || ns_error) {
            if (error) {
                *error = [[NSString stringWithFormat:@"Failed to build feature provider: %@",
                           ns_error.localizedDescription] UTF8String];
            }
            return false;
        }

        NSMutableArray<NSNumber*>* unit_candidates = [NSMutableArray array];
        if (config.coreml_cpu_only) {
            [unit_candidates addObject:@(MLComputeUnitsCPUOnly)];
        } else {
            // Prefer accelerated paths first and fall back to CPU.
            [unit_candidates addObject:@(MLComputeUnitsAll)];
            [unit_candidates addObject:@(MLComputeUnitsCPUAndGPU)];
            [unit_candidates addObject:@(MLComputeUnitsCPUAndNeuralEngine)];
            [unit_candidates addObject:@(MLComputeUnitsCPUOnly)];
        }

        NSMutableArray<NSString*>* failures = [NSMutableArray array];
        for (NSNumber* unit_number in unit_candidates) {
            MLComputeUnits units = static_cast<MLComputeUnits>(unit_number.integerValue);
            MLModelConfiguration* model_config = [[MLModelConfiguration alloc] init];
            model_config.computeUnits = units;

            ns_error = nil;
            MLModel* model = [MLModel modelWithContentsOfURL:model_url
                                               configuration:model_config
                                                       error:&ns_error];
            if (!model || ns_error) {
                NSString* load_desc = ns_error.localizedDescription ? ns_error.localizedDescription : @"unknown error";
                NSString* msg = [NSString stringWithFormat:@"%@: load failed: %@",
                                 compute_unit_name(units),
                                 load_desc];
                [failures addObject:msg];
                continue;
            }

            ns_error = nil;
            id<MLFeatureProvider> output = [model predictionFromFeatures:provider error:&ns_error];
            if (!output || ns_error) {
                NSString* pred_desc = ns_error.localizedDescription ? ns_error.localizedDescription : @"unknown error";
                NSString* msg = [NSString stringWithFormat:@"%@: predict failed: %@",
                                 compute_unit_name(units),
                                 pred_desc];
                [failures addObject:msg];
                continue;
            }

            NSString* output_name = nsstring(config.output_name);
            MLFeatureValue* logits_value = [output featureValueForName:output_name];
            if (!logits_value) {
                for (NSString* candidate in output.featureNames) {
                    MLFeatureValue* candidate_value = [output featureValueForName:candidate];
                    if (candidate_value.type == MLFeatureTypeMultiArray) {
                        logits_value = candidate_value;
                        break;
                    }
                }
            }

            std::vector<float> out = flatten_feature_value(logits_value);
            if (out.empty()) {
                NSString* msg = [NSString stringWithFormat:@"%@: output parse failed",
                                 compute_unit_name(units)];
                [failures addObject:msg];
                continue;
            }

            if (config.verbose) {
                std::cerr << "keyit: CoreML compute units selected: "
                          << [compute_unit_name(units) UTF8String] << "\n";
            }
            *logits = std::move(out);
            return true;
        }

        if (error) {
            std::ostringstream oss;
            oss << "CoreML inference failed for all compute unit modes.";
            if (failures.count > 0) {
                oss << " Attempts: ";
                bool first = true;
                for (NSString* failure in failures) {
                    if (!first) {
                        oss << " | ";
                    }
                    first = false;
                    oss << [failure UTF8String];
                }
            }
            *error = oss.str();
        }
        return false;
    }
}

} // namespace keyit
