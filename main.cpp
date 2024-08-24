#include <opencv2/opencv.hpp>
#include <iostream>
using namespace std;
using namespace cv;
int main() {
    string cascadePath = "/opt/homebrew/Cellar/opencv/4.9.0_12/share/opencv4/haarcascades/haarcascade_frontalface_default.xml";
    CascadeClassifier faceCascade;
    if (!faceCascade.load(cascadePath)) {
        cerr << "Error loading face detection model: " << cascadePath << endl;
        return -1;
    }
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Error opening camera." << endl;
        return -1;
    }
    Mat initialFrame;
    cout << "Press 'c' to capture your picture." << endl;
    while (true) {
        cap >> initialFrame;
        imshow("Capture Your Picture", initialFrame);
        if (waitKey(1) == 'c') {
            break;
        }
    }
    destroyWindow("Capture Your Picture");
    Mat grayInitial;
    cvtColor(initialFrame, grayInitial, COLOR_BGR2GRAY);
    vector<Rect> faces;
    faceCascade.detectMultiScale(grayInitial, faces);
    if (faces.empty()) {
        cerr << "No face detected in the captured image." << endl;
        return -1;
    }
    Mat capturedFace = grayInitial(faces[0]);
    string userName;
    cout << "Enter your name: ";
    cin >> userName;
    Mat referenceImage;
    resize(capturedFace, referenceImage, capturedFace.size());
    namedWindow("Face Detection", WINDOW_NORMAL);
    while (true) {
        Mat frame;
        cap >> frame;
        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);
        vector<Rect> faces;
        faceCascade.detectMultiScale(gray, faces);
        for (const auto& face : faces) {
            rectangle(frame, face, Scalar(0, 0, 255), 2);
            Mat faceROI = gray(face);
            Mat resizedReference;
            resize(referenceImage, resizedReference, faceROI.size());
            Mat result;
            matchTemplate(faceROI, resizedReference, result, TM_CCOEFF_NORMED);
            double minVal, maxVal;
            Point minLoc, maxLoc;
            minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc);
            double threshold = 0.7;
            if (maxVal >= threshold) {
                putText(frame, "He is " + userName, Point(face.x, face.y - 10),
                        FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
            } else {
                putText(frame, "Not Identified", Point(face.x, face.y - 10),
                        FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255), 2);
            }
        }
        imshow("Face Detection", frame);
        if (waitKey(1) == 'q') {
            break;
        }
    }
    cap.release();
    destroyAllWindows();

    return 0;
}
