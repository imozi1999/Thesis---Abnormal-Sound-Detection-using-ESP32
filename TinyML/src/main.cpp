    /* Copyright 2018 The TensorFlow Authors. All Rights Reserved.

	Licensed under the Apache License, Version 2.0 (the "License");
	you may not use this file except in compliance with the License.
	You may obtain a copy of the License at

		http://www.apache.org/licenses/LICENSE-2.0

	Unless required by applicable law or agreed to in writing, software
	distributed under the License is distributed on an "AS IS" BASIS,
	WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
	See the License for the specific language governing permissions and
	limitations under the License.

	Modified by shaynejohnson00@gmail.com (Thinh Hoang Quoc)
==============================================================================*/

#include <TensorFlowLite_ESP32.h>
#include <Arduino.h>

#include "main_functions.h"

#include "audio_provider.h"
#include "feature_provider.h"
#include "micro_model_settings.h"
#include "model.h"
#include "utils.h"

#include "tensorflow/lite/experimental/micro/kernels/micro_ops.h"
#include "tensorflow/lite/experimental/micro/micro_error_reporter.h"
#include "tensorflow/lite/experimental/micro/micro_interpreter.h"
#include "tensorflow/lite/experimental/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#define PIN22 22

// Settings
constexpr int NUM_AXES = 1;           // Number of axes on accelerometer
constexpr int MAX_MEASUREMENTS = 160; // Number of samples to keep in each axis
constexpr float THRESHOLD =     ;    // Any MSE over this is an anomaly
int count = 0;

// Globals, used for compatibility with Arduino-style sketches.
namespace {
	tflite::ErrorReporter* error_reporter = nullptr;
	const tflite::Model* model = nullptr;
	tflite::MicroInterpreter* interpreter = nullptr;
	TfLiteTensor* model_input = nullptr;
	FeatureProvider* feature_provider = nullptr;
	int32_t previous_time = 0;

	// Create an area of memory to use for input, output, and intermediate arrays.
	// The size of this will depend on the model you're using, and may need to be
	// determined by experimentation.
	constexpr int kTensorArenaSize = 10 * 1024;
	uint8_t tensor_arena[kTensorArenaSize];
}  // namespace

QueueHandle_t xQueueAudioWave;
#define QueueAudioWaveSize 32

// The name of this function is important for Arduino compatibility.
void setup() {
	pinMode(PIN22, OUTPUT);
	xQueueAudioWave = xQueueCreate(QueueAudioWaveSize, sizeof(int16_t));
	
	// Set up logging. Google style is to avoid globals or statics because of
	// lifetime uncertainty, but since this has a trivial destructor it's okay.
	// NOLINTNEXTLINE(runtime-global-variables)
	static tflite::MicroErrorReporter micro_error_reporter;
	error_reporter = &micro_error_reporter;

	// Map the model into a usable data structure. This doesn't involve any
	// copying or parsing, it's a very lightweight operation.
	model = tflite::GetModel(model_tflite);
	if (model->version() != TFLITE_SCHEMA_VERSION) {
		error_reporter->Report(
		"Model provided is schema version %d not equal "
		"to supported version %d.",
		model->version(), TFLITE_SCHEMA_VERSION);
		return;
	}

	// Pull in only the operation implementations we need.
	// This relies on a complete list of all the ops needed by this graph.
	// An easier approach is to just use the AllOpsResolver, but this will
	// incur some penalty in code space for op implementations that are not
	// needed by this graph.
	//
	// tflite::ops::micro::AllOpsResolver resolver;
	// NOLINTNEXTLINE(runtime-global-variables)
	static tflite::MicroMutableOpResolver micro_mutable_op_resolver;
	micro_mutable_op_resolver.AddBuiltin(
		tflite::BuiltinOperator_FULLY_CONNECTED,
		tflite::ops::micro::Register_FULLY_CONNECTED(),
		1, 9);

	// Build an interpreter to run the model with.
	static tflite::MicroInterpreter static_interpreter(
		model, micro_mutable_op_resolver, tensor_arena, kTensorArenaSize,
		error_reporter);
	interpreter = &static_interpreter;

	// Allocate memory from the tensor_arena for the model's tensors.
	TfLiteStatus allocate_status = interpreter->AllocateTensors();
	if (allocate_status != kTfLiteOk) {
		error_reporter->Report("AllocateTensors() failed");
		return;
	}

	// Get information about the memory area to use for the model's input.
	model_input = interpreter->input(0);

	// Prepare to access the audio spectrograms from a microphone or other source
	// that will provide the inputs to the neural network.
	// NOLINTNEXTLINE(runtime-global-variables)
	static FeatureProvider static_feature_provider(kFeatureElementCount, model_input->data.f);
	feature_provider = &static_feature_provider;

	previous_time = 0;

	Serial.begin(115200);

	Serial.printf("model_input->name          : %s\n", model_input->name);
	Serial.printf("model_input->type          : %d\n", model_input->type);
	Serial.printf("model_input->bytes         : %d\n", model_input->bytes);
	Serial.printf("model_input->dims->size    : %d\n", model_input->dims->size);
	Serial.printf("model_input->dims->data[0] : %d\n", model_input->dims->data[0]);
	Serial.printf("model_input->dims->data[1] : %d\n", model_input->dims->data[1]);
	Serial.printf("model_input->dims->data[2] : %d\n", model_input->dims->data[2]);

}

// The name of this function is important for Arduino compatibility.
void loop() {

	float measurements[MAX_MEASUREMENTS];
    float mad;
	float x_total = (float)0;
	TfLiteStatus invoke_status;

	int16_t wave;
	for (int i = 0; i < QueueAudioWaveSize; i++) {
		if (xQueueReceive(xQueueAudioWave, &wave, 0) == pdTRUE) {
			// Serial.printf("Wave 1:	%d\n", wave);
		}
	}  

	// Fetch the spectrogram for the current time.
	const int32_t current_time = LatestAudioTimestamp();
	int how_many_new_slices = 0;
	TfLiteStatus feature_status = feature_provider->PopulateFeatureData(
									error_reporter, previous_time, current_time, &how_many_new_slices);
	if (feature_status != kTfLiteOk) {
		error_reporter->Report("Feature generation failed");
		delay(1);
		return;
	}
	previous_time = current_time;
	// If no new audio samples have been received since last time, don't bother
	// running the network model.
	if (how_many_new_slices == 0) {
		delay(1);
		return;
	}


	if (count < MAX_MEASUREMENTS){
        measurements[count] = (float)wave;
		x_total += measurements[count];
		#if DEBUG
		// Serial.printf("Wave:		%d\n", measurements[count]);
		#endif
        count++;
    }
    else{

		// Put data into model
		model_input->data.f = measurements;
        count = 0;

		TfLiteTensor* model_output = interpreter->output(0);
        // Run inference
        invoke_status = interpreter->Invoke();
        if (invoke_status != kTfLiteOk) {
            error_reporter->Report("Invoke failed on input");
        }

		// Get the output array
		float* y_val = model_output->data.f;

		// Calculate the MSE
		float mse = calc_mse(measurements, y_val, MAX_MEASUREMENTS);
		Serial.printf("Prediction score:	%f\n", mse);

		// Any mse > THRESHOLD is abnormal
		if(mse > THRESHOLD){
			Serial.printf("ABNORMAL\n\n");
			digitalWrite(PIN22, LOW);
		}
		else{
			Serial.printf("NORMAL\n\n");
			digitalWrite(PIN22,  HIGH);
		}
    }

	delay(1);
}
