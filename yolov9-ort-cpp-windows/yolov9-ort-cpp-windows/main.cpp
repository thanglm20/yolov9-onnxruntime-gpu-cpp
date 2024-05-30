

#include "Yolov9Detector.h"


int main() {
	std::cout << "ONNX Runtime Yolov9\n";
	std::string image_path = "D:\\projects\\yolov9\\data\\images\\1.png";
	std::wstring model_path(L"D:\\projects\\yolov9\\outputs\\best.onnx");

	Yolov9Detector yolo(model_path, 2, 0.4, 0.4);
	

	while (true) {
		cv::Mat img = cv::imread(image_path);
		std::vector<Object> objs;
		auto begin = std::chrono::steady_clock::now();
		yolo.detect(img, objs);
		auto end = std::chrono::steady_clock::now();
		std::cout << "Time spent = " << 
		std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count() << "[ms]" << std::endl;

		 std::cout << "Number of objects:  " << objs.size() << std::endl;
        for (auto o : objs) {
            cv::rectangle(img,o.rect, cv::Scalar(0, 255, 0));
        }
		cv::imshow("Result", img);
		if (cv::waitKey(1) == 'q') break;
        
	}
	std::cout << "Done\n";
}