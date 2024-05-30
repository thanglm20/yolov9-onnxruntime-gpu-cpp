// yolov9-ort-cpp-windows.cpp : This file contains the 'main' function. Program execution begins and ends there.
//

#include <iostream>
#include <algorithm>
#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <onnxruntime_cxx_api.h>
#define CLASS_NUM  2
#define PROBABILITY_THRESHOLD 0.4
#define NMS_THRESHOLD  0.4
int g_img_width = 0;
int g_img_height = 0;

struct Object {
    // The object class.
    int label{};
    // The detection's confidence probability.
    float probability{};
    // The object bounding box rectangle.
    cv::Rect_<float> rect;
    // Semantic segmentation mask
    cv::Mat boxMask;
    // Pose estimation key points
    std::vector<float> kps{};
};

// Function to load the ONNX model and create a session
Ort::Session loadModel(Ort::Env& env, const std::wstring& model_path, Ort::SessionOptions& session_options)
{
    return Ort::Session(env, model_path.c_str(), session_options);
}

// Function to load and preprocess the image
std::vector<float> preprocessImage(const std::string& image_path, const std::vector<int64_t>& input_shape)
{
    cv::Mat image = cv::imread(image_path);
    if (image.empty())
    {
        throw std::runtime_error("Could not read the image: " + image_path);
    }
    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(input_shape[2], input_shape[3]));

    resized_image.convertTo(resized_image, CV_32F, 1.0 / 255);

    std::vector<cv::Mat> channels(3);
    cv::split(resized_image, channels);

    std::vector<float> input_tensor_values;
    for (int c = 0; c < 3; ++c)
    {
        input_tensor_values.insert(input_tensor_values.end(), (float*)channels[c].data, (float*)channels[c].data + input_shape[2] * input_shape[3]);
    }

    return input_tensor_values;
}
std::vector<Object> postprocessDetect(std::vector<float>& featureVector) {
    auto numChannels = 6;
    auto numAnchors = 33600;

    auto numClasses = CLASS_NUM;

    std::vector<cv::Rect> bboxes;
    std::vector<float> scores;
    std::vector<int> labels;
    std::vector<int> indices;

    cv::Mat output = cv::Mat(numChannels, numAnchors, CV_32F, featureVector.data());
    output = output.t();

    // Get all the YOLO proposals
    for (int i = 0; i < numAnchors; i++) {
        auto rowPtr = output.row(i).ptr<float>();
        auto bboxesPtr = rowPtr;
        auto scoresPtr = rowPtr + 4;
        auto maxSPtr = std::max_element(scoresPtr, scoresPtr + numClasses);
        float score = *maxSPtr;
        if (score > PROBABILITY_THRESHOLD) {
            float x = *bboxesPtr++;
            float y = *bboxesPtr++;
            float w = *bboxesPtr++;
            float h = *bboxesPtr;

            float x0 = (x - 0.5 *w) * g_img_width / 1280;
            float y0 = (y - 0.5*h) * g_img_height / 1280;
            float x1 = (x + 0.5 * w) * g_img_width / 1280;
            float y1 = (y + 0.5 * h) * g_img_height / 1280;

            int label = maxSPtr - scoresPtr;
            cv::Rect_<float> bbox;
            bbox.x = x0;
            bbox.y = y0;
            bbox.width = x1 - x0;
            bbox.height = y1 - y0;

            bboxes.push_back(bbox);
            labels.push_back(label);
            scores.push_back(score);
        }
    }

    // Run NMS
    cv::dnn::NMSBoxesBatched(bboxes, scores, labels, PROBABILITY_THRESHOLD, NMS_THRESHOLD, indices);

    std::vector<Object> objects;

    // Choose the top k detections
    int cnt = 0;
    for (auto& chosenIdx : indices) {
        Object obj{};
        obj.probability = scores[chosenIdx];
        obj.label = labels[chosenIdx];
        obj.rect = bboxes[chosenIdx];
        objects.push_back(obj);

        cnt += 1;
    }

    return objects;
}
// Function to get the input name of the model
std::string getInputName(Ort::Session& session, Ort::AllocatorWithDefaultOptions& allocator)
{
    Ort::AllocatedStringPtr name_allocator = session.GetInputNameAllocated(0, allocator);
    return std::string(name_allocator.get());
}

// Function to get the output name of the model
void getOutputName(Ort::Session& session, Ort::AllocatorWithDefaultOptions& allocator,
                    std::vector<std::string>& outputs)
{
    std::cout << "Number of output layers: " << session.GetOutputCount() << std::endl;
    for (int i = 0; i < session.GetOutputCount(); ++i) {
        Ort::AllocatedStringPtr name_allocator = session.GetOutputNameAllocated(i, allocator);
        outputs.push_back(std::string(name_allocator.get()));
    }
}

// Function to run inference on the model
std::vector<float> runInference(Ort::Session& session, const std::vector<float>& input_tensor_values, const std::vector<int64_t>& input_shape)
{
    Ort::AllocatorWithDefaultOptions allocator;

    std::string input_name = getInputName(session, allocator);
    std::vector<std::string> outputs;
    getOutputName(session, allocator, outputs);
    std::cout << "Model input name: " << input_name << std::endl;
    std::cout << "Model output name: " << std::endl;
    for (int i = 0; i < outputs.size(); ++i) {
        std::cout << "Output " << i << ": " << outputs[i] << std::endl;

    }
    const size_t output_size = outputs.size();
    std::vector<const char*> input_node_names = { input_name.c_str() };
    std::vector<const char*> output_node_names = { outputs[0].c_str(),outputs[1].c_str()};

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, const_cast<float*>(input_tensor_values.data()), input_tensor_values.size(), input_shape.data(), input_shape.size());
    Ort::RunOptions run_options;
    auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, 
                            input_node_names.data(), &input_tensor, 1,
                            output_node_names.data(), 2);
    std::cout << "Onnx run successfully\n";
    float* floatarr = output_tensors[0].GetTensorMutableData<float>();
    size_t output_tensor_size = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
    std::cout << "output size: " << output_tensor_size << std::endl;
    return std::vector<float>(floatarr, floatarr + output_tensor_size);
}
//
//int main()
//{
//    std::cout << "ONNX Runtime Yolov9\n";
//    std::string image_path = "D:\\projects\\yolov9\\data\\images\\1.png";
//    std::wstring model_path(L"D:\\projects\\yolov9\\outputs\\best.onnx");
//    cv::Mat img = cv::imread(image_path);
//    bool _UseCuda = false;
//    bool _Debug = true;
// 
//    g_img_width = img.cols;
//    g_img_height = img.rows;
//
//    // Initialize ONNX Runtime environment
//    Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "ONNXRuntime");
//
//    // Create session options
//    Ort::SessionOptions session_options;
//    session_options.SetIntraOpNumThreads(1);
//    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
//    if (_UseCuda) {
//        OrtCUDAProviderOptions cudaOption;
//        cudaOption.device_id = 0;
//        session_options.AppendExecutionProvider_CUDA(cudaOption);
//    }
//    try
//    {
//        // Load model
//        Ort::Session session = loadModel(env, model_path, session_options);
//
//        // Define input shape (e.g., {1, 3, 1280, 1280})
//        std::vector<int64_t> input_shape = { 1, 3, 1280, 1280 };
//
//        // Preprocess image
//        std::vector<float> input_tensor_values = preprocessImage(image_path, input_shape);
//
//        // Run inference
//        std::vector<float> results = runInference(session, input_tensor_values, input_shape);
//
//        std::vector<Object> obj = postprocessDetect(results);
//        std::cout << "Number of objects:  " << obj.size() << std::endl;
//        for (auto o : obj) {
//            cv::rectangle(img,o.rect, cv::Scalar(0, 255, 0));
//        }
//        while (true) {
//            cv::imshow("Result", img);
//            cv::waitKey(1);
//        }
//    }
//    catch (const std::exception& e)
//    {
//        std::cerr << "Error: " << e.what() << std::endl;
//        return 1;
//    }
//
//
//    
//    std::cout << "Done\n";
//}

// Run program: Ctrl + F5 or Debug > Start Without Debugging menu
// Debug program: F5 or Debug > Start Debugging menu

// Tips for Getting Started: 
//   1. Use the Solution Explorer window to add/manage files
//   2. Use the Team Explorer window to connect to source control
//   3. Use the Output window to see build output and other messages
//   4. Use the Error List window to view errors
//   5. Go to Project > Add New Item to create new code files, or Project > Add Existing Item to add existing code files to the project
//   6. In the future, to open this project again, go to File > Open > Project and select the .sln file
