#pragma once
#include <algorithm>
#include "opencv2/opencv.hpp"
#include "opencv2/core.hpp"
#include "opencv2/videoio.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <onnxruntime_cxx_api.h>

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

class Yolov9Detector
{
public:
	Yolov9Detector() = default;
    Yolov9Detector(const std::wstring& model_path, int class_num,
           float threshold, float nms_threshold);
	~Yolov9Detector(){}
    void detect(const cv::Mat& image, std::vector<Object>& objects);
private:
	void loadModel(const std::wstring& model_path);
    std::string getInputName(Ort::Session& session, Ort::AllocatorWithDefaultOptions& allocator);
    void getOutputName(Ort::Session& session, 
            Ort::AllocatorWithDefaultOptions& allocator,
            std::vector<std::string>& outputs);

	void preprocessImage(const cv::Mat& image,
		                std::vector<float>& input_tensor);
    void runInference(const std::vector<float>& input_tensor_values,
                    std::vector<float>& featureVector);

    void  postprocessDetect(std::vector<float>& featureVector, 
                            std::vector<Object>& objects);
private:
    bool m_UseCuda = true;
    Ort::Env m_env{ nullptr };
    Ort::Session m_session{ nullptr };
    int m_class_num = 2;
    float m_prob = 0.4;
    float m_nms_threshold = 0.4;
    int m_image_width = 0;
    int m_image_height = 0;
    std::vector<int64_t> m_input_node_dims;
    std::vector<int64_t> m_output_node_dims;
    std::string m_input_name{ "" };
    std::vector<std::string> m_output_names;
};

