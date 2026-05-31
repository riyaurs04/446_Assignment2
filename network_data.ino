#include <TensorFlowLite.h>
#include "network_model.h"
#include "tensorflow/lite/micro/kernels/micro_ops.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#define NUMBER_OF_INPUTS 38
#define NUMBER_OF_OUTPUTS 1
#define TENSOR_ARENA_SIZE 16*1024

uint8_t tensor_arena[TENSOR_ARENA_SIZE];
tflite::ErrorReporter* error_reporter;
tflite::MicroInterpreter* interpreter;
TfLiteTensor* input;
TfLiteTensor* output;

const float X_test[10][38] = {
    -0.11024922321249885, -0.12470615670462065, 1.3931828317609012, 
    0.7511112872365361, -0.00764657389418498, -0.00483682909271775, 
    -0.08948642202040107, -0.09507567152556495, 1.2356940323701657, 
    -0.011663642603760032, -0.036651869142258646, -0.024436507262009306, 
    -0.01238515036740332, -0.02618002418454278, -0.018609896340735923, 
    -0.04122119759327531, -0.0028174939213690777, -0.0975309439715147, 
    -0.7257784945454897, -0.35434285002584825, -0.6372092679572258, 
    -0.6319290328885425, -0.37436223991967527, -0.37443160310530493, 
    0.7712831058493207, -0.349683030873482, 3.4741184775599305, 
    0.08921911486299862, 0.7438598290874257, 0.643189354538082, 
    -0.33321384013747096, -0.4478339589698615, -0.20026454391970344, 
    -0.6395319051152512, -0.6248707997445304, -0.38763462350750655, 
    -0.34508401858082666, 0.6528228780141483, -0.11024922321249885, 
    -0.12470615670462065, 2.310815790256898, -0.7362346401101137, 
    -0.00776224074056876, -0.004918644383724874, -0.08948642202040107, 
    -0.09507567152556495, -0.8092618187059747, -0.011663642603760032, 
    -0.036651869142258646, -0.024436507262009306, -0.01238515036740332, 
    -0.02618002418454278, -0.018609896340735923, -0.04122119759327531, 
    -0.0028174939213690777, -0.0975309439715147, 0.5317738841070986, 
    -0.3130407611127971, 1.602663889932865, 1.6051037177847889, 
    -0.37436223991967527, -0.37443160310530493, -1.435162311348284, 
    -0.016929597707948767, -0.37455970440553465, 0.7343425609306344, 
    -1.0266544017850858, -1.1387558712615415, -0.015620856278641512, 
    -0.4801968475158174, -0.28910340026287856, 1.6087590765792643, 
    1.6189552037455606, -0.38763462350750655, -0.3763870260680415, 
    -0.6563667617603728, -0.11024922321249885, -0.12470615670462065, 
    1.3320073011945015, -0.7362346401101137, -0.00776224074056876, 
    -0.004918644383724874, -0.08948642202040107, -0.09507567152556495, 
    -0.8092618187059747, -0.011663642603760032, -0.036651869142258646, 
    -0.024436507262009306, -0.01238515036740332, -0.02618002418454278, 
    -0.018609896340735923, -0.04122119759327531, -0.0028174939213690777, 
    -0.0975309439715147, 0.3221818209983339, -0.271738672199746, 
    1.602663889932865, 1.6051037177847889, -0.37436223991967527, 
    -0.37443160310530493, -1.3441748714638466, 0.03852930781964016, 
    -0.37455970440553465, 0.7343425609306344, -0.9724549865542945, 
    -1.0942072406165508, -0.06855302025511306, -0.4801968475158174, 
    -0.28910340026287856, 1.6087590765792643, 1.6189552037455606, 
    -0.38763462350750655, -0.3763870260680415, 0.6528228780141483, 
    -0.09642701884602016, 2.219311849644404, -1.1761894520278895, 
    0.7511112872365361, -0.007746568649453432, -0.004909691950666344, 
    -0.08948642202040107, -0.09507567152556495, -0.8092618187059747, 
    -0.011663642603760032, -0.036651869142258646, -0.024436507262009306, 
    -0.01238515036740332, -0.02618002418454278, -0.018609896340735923, 
    -0.04122119759327531, -0.0028174939213690777, -0.0975309439715147, 
    -0.6908464840273623, -0.29927339814178006, -0.6372092679572258, 
    -0.6319290328885425, -0.37436223991967527, -0.37443160310530493, 
    0.7712831058493207, -0.349683030873482, 0.8955040956430689, 
    -1.3320684772547615, -0.809856740861921, -0.002765789814281515, 
    -0.12148518423158469, 1.2026733568738908, -0.28910340026287856, 
    -0.6395319051152512, -0.6248707997445304, -0.38763462350750655, 
    -0.3763870260680415, 0.21642633142264128, -0.11024922321249885, 
    2.219311849644404, 0.7814275260969035, 0.7511112872365361, 
    -0.007737369813364, -0.004918644383724874, -0.08948642202040107, 
    -0.09507567152556495, -0.8092618187059747, -0.011663642603760032, 
    -0.036651869142258646, -0.024436507262009306, -0.01238515036740332, 
    -0.02618002418454278, -0.018609896340735923, -0.04122119759327531, 
    -0.0028174939213690777, -0.0975309439715147, -0.7257784945454897, 
    -0.3681102129968653, -0.6372092679572258, -0.6319290328885425, 
    -0.37436223991967527, -0.37443160310530493, 0.7712831058493207, 
    -0.349683030873482, -0.37455970440553465, 0.7343425609306344, 
    -1.0266544017850858, -1.1387558712615415, 2.36632652266258, 
    2.497188898712128, -0.28910340026287856, -0.6395319051152512, 
    -0.6248707997445304, -0.38763462350750655, -0.3763870260680415, 
    0.6528228780141483
};

const uint8_t y_test[10] = {1, 1, 0, 0, 1, 1, 0, 0, 1, 1}; // Actual labels for each sample

void setup() {
    Serial.begin(115200);
    while(!Serial);
    delay(1000);
    Serial.print("Start setup: ");

    static tflite::MicroErrorReporter micro_error_reporter;
    error_reporter = &micro_error_reporter;

    const tflite::Model* model = tflite::GetModel(network_model);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        Serial.println("Model version does not match schema version.");
        return;
    }
    Serial.println("Model loaded OK");

    static tflite::MicroMutableOpResolver<10> micro_op_resolver;
    micro_op_resolver.AddFullyConnected();
    micro_op_resolver.AddSoftmax();
    micro_op_resolver.AddQuantize();
    micro_op_resolver.AddDequantize();
    Serial.println("Ops registered");

    static tflite::MicroInterpreter static_interpreter(model, micro_op_resolver, tensor_arena, TENSOR_ARENA_SIZE, error_reporter);
    interpreter = &static_interpreter;
    Serial.println("Interpreter created");

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        Serial.println("Failed to allocate tensors!");
        return;
    }
    Serial.println("AllocateTensors() OK");

    Serial.print("Input type: ");  Serial.println(input->type);
    Serial.print("Output type: "); Serial.println(output->type);
    Serial.print("Input size: ");  Serial.println(input->bytes);

    input = interpreter->input(0);
    output = interpreter->output(0);
}

void loop() {
    for (uint8_t i = 0; i < 10; i++) {
        // Load the i-th test sample data into the input tensor
        for (int j = 0; j < NUMBER_OF_INPUTS; j++) {
            input->data.f[j] = X_test[i][j];
        }

        // Run the model on this input and check for error
        if (interpreter->Invoke() != kTfLiteOk) {
            Serial.println("Failed to invoke!");
            continue;
        }

        // Question 5: Deploying the Quantized Mode
        // (a) Implement code to obtain the prediction from the output tensor and determine the predicted class label.
        // Output has 2 neurons (index 0 = attack probability, index 1 = normal probability)
        float prob_attack = output->data.f[0];
        float prob_normal = output->data.f[1];
        // predicted class: 0 = attack, 1 = normal
        int predicted_class = (prob_normal > prob_attack) ? 1 : 0;

        // (b) Implement code to output Sample #, Predicted Class, and Actual Class for each sample to the serial monitor using Serial.print function.
        Serial.print("Sample #"); Serial.print(i + 1);
        Serial.print("  |  Predicted Class: ");
        Serial.print(predicted_class == 1 ? "normal" : "attack");
        Serial.print("  |  Actual Class: ");
        Serial.println(y_test[i] == 1 ? "normal" : "attack");

        // Delay between predictions
        delay(1000);
    }

    // Delay before repeating the tests
    delay(10000);
}
