

#include "Yolov9Detector.h"


int main() {
	std::cout << "ONNX Runtime Yolov9\n";
	std::string image_path = "D:\\projects\\yolov9\\data\\images\\1.png";
	std::wstring model_path(L"D:\\projects\\yolov9\\outputs\\best.onnx");

	Yolov9Detector yolo(model_path, 2, 0.4, 0.4);
	cv::VideoCapture cap("D:\\projects\\yolov9\\data\\videos\\1.mp4");
	//if (!cap.isOpened()) {
	//	std::cout << "Error opening video stream or file" << std::endl;
	//	return -1;
	//}

	while (cap.isOpened()) {
		//cv::Mat img = cv::imread(image_path);
		cv::Mat img;
		cap >> img;

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
		cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
		cv::resize(img, img, cv::Size(1280, 720));

		cv::imshow("Result", img);
		if (cv::waitKey(1) == 'q') break;
	}
	// When everything done, release the video capture object
	cap.release();
	std::cout << "Done\n";
}