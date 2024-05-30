#include "Yolov9Detector.h"



Yolov9Detector::Yolov9Detector(const std::wstring& model_path, int class_num,
    float threshold, float nms_threshold) {
    loadModel(model_path);
    m_class_num = class_num;
    m_prob = threshold;
    m_nms_threshold = nms_threshold;
}

// Function to get the input name of the model
std::string Yolov9Detector::getInputName(Ort::Session& session, Ort::AllocatorWithDefaultOptions& allocator)
{
    Ort::AllocatedStringPtr name_allocator = session.GetInputNameAllocated(0, allocator);
    return std::string(name_allocator.get());
}

// Function to get the output name of the model
void Yolov9Detector::getOutputName(Ort::Session& session, Ort::AllocatorWithDefaultOptions& allocator,
    std::vector<std::string>& outputs)
{
    for (int i = 0; i < session.GetOutputCount(); ++i) {
        Ort::AllocatedStringPtr name_allocator = session.GetOutputNameAllocated(i, allocator);
        outputs.push_back(std::string(name_allocator.get()));
    }
}

void Yolov9Detector::loadModel(const std::wstring& model_path) {
	// Initialize ONNX Runtime environment
	m_env = Ort::Env(ORT_LOGGING_LEVEL_WARNING, "Yolov9Detector");
    Ort::SessionOptions session_options;
    session_options.SetIntraOpNumThreads(1);
    session_options.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
    if (m_UseCuda) {
        OrtCUDAProviderOptions cudaOption;
        cudaOption.device_id = 0;
        session_options.AppendExecutionProvider_CUDA(cudaOption);
    }
	m_session =  Ort::Session(m_env, model_path.c_str(), session_options);
    Ort::AllocatorWithDefaultOptions allocator;
    m_input_name = getInputName(m_session, allocator);
    getOutputName(m_session, allocator, m_output_names);
    
    // read input layer
    std::cout << "- Input model: name = " << m_input_name;
    auto type_info = m_session.GetInputTypeInfo(0);
    auto tensor_info = type_info.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType type = tensor_info.GetElementType();
    std::cout << ", type = " << type << ", shape[ ";
    m_input_node_dims = tensor_info.GetShape();
    for (int i : m_input_node_dims) {
        std::cout << i << " ";
    }
    std::cout << "]\n";

    // read output
    auto output_type_info = m_session.GetOutputTypeInfo(0);
    auto output_tensor_info = output_type_info.GetTensorTypeAndShapeInfo();
    ONNXTensorElementDataType output_type = output_tensor_info.GetElementType();
    m_output_node_dims = output_tensor_info.GetShape();
    std::cout << "- Ouput model: name = " << m_output_names[0];
    std::cout << ", type = " << output_type << ", shape[ ";
    for (int i : m_output_node_dims) {
        std::cout << i << " ";
    }
    std::cout << "]\n";

}

// Function to load and preprocess the image
void Yolov9Detector::preprocessImage(const cv::Mat& image,
                            std::vector<float>& input_tensor)
{
    if (image.empty())
    {
        return;
    }
    m_image_width = image.cols;
    m_image_height = image.rows;

    cv::cvtColor(image, image, cv::COLOR_BGR2RGB);

    cv::Mat resized_image;
    cv::resize(image, resized_image, cv::Size(m_input_node_dims[2], m_input_node_dims[3]));

    resized_image.convertTo(resized_image, CV_32F, 1.0 / 255);

    std::vector<cv::Mat> channels(3);
    cv::split(resized_image, channels);

    for (int c = 0; c < 3; ++c)
    {
        input_tensor.insert(input_tensor.end(), (float*)channels[c].data, (float*)channels[c].data + m_input_node_dims[2] * m_input_node_dims[3]);
    }
}

// Function to run inference on the model
 void Yolov9Detector::runInference(const std::vector<float>& input_tensor_values, std::vector<float>& featureVector)
{
    Ort::AllocatorWithDefaultOptions allocator;

    std::vector<const char*> input_node_names = { m_input_name.c_str() };
    std::vector<const char*> output_node_names = { m_output_names[0].c_str(),m_output_names[1].c_str() };

    Ort::MemoryInfo memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
    Ort::Value input_tensor = Ort::Value::CreateTensor<float>(memory_info, 
                            const_cast<float*>(input_tensor_values.data()), 
                            input_tensor_values.size(), m_input_node_dims.data(), m_input_node_dims.size());
    Ort::RunOptions run_options;
    auto output_tensors = m_session.Run(Ort::RunOptions{ nullptr },
        input_node_names.data(), &input_tensor, 1,
        output_node_names.data(), 2);
    float* floatarr = output_tensors[0].GetTensorMutableData<float>();
    size_t output_tensor_size = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();
    featureVector.assign(floatarr, floatarr + output_tensor_size);
}

void  Yolov9Detector::postprocessDetect(std::vector<float>& featureVector, std::vector<Object>& objects) {
    auto numChannels = m_output_node_dims[1];
    auto numAnchors = m_output_node_dims[2];;

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
        auto maxSPtr = std::max_element(scoresPtr, scoresPtr + m_class_num);
        float score = *maxSPtr;
        if (score > m_prob) {
            float x = *bboxesPtr++;
            float y = *bboxesPtr++;
            float w = *bboxesPtr++;
            float h = *bboxesPtr;

            float x0 = (x - 0.5 * w) * m_image_width / m_input_node_dims[3];
            float y0 = (y - 0.5 * h) * m_image_height / m_input_node_dims[2];
            float x1 = (x + 0.5 * w) * m_image_width / m_input_node_dims[3];
            float y1 = (y + 0.5 * h) * m_image_height / m_input_node_dims[2];

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
    cv::dnn::NMSBoxesBatched(bboxes, scores, labels, m_prob, m_nms_threshold, indices);

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
}

void Yolov9Detector::detect(const cv::Mat& image, std::vector<Object>& objects){
    std::vector<float> input_tensor;
    preprocessImage(image, input_tensor);
    std::vector<float> feature_vector;
    auto begin = std::chrono::steady_clock::now();

    runInference(input_tensor, feature_vector);
    auto end = std::chrono::steady_clock::now();
    std::cout << "Inference spent = " <<
        std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;

    postprocessDetect(feature_vector, objects);
}