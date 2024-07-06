#include <opencv2/opencv.hpp>
#include <iostream>

using namespace std;
using namespace cv;

int main() {
    // Paths to the Haar Cascade face detection model and images
    string cascadePath = "/Users/aryan/Library/Mobile Documents/com~apple~CloudDocs/Desktop/App Dev/OpencvCourse/4.9.0_3/share/opencv4/haarcascades/haarcascade_frontalface_default.xml";
    string referenceImagePath1 = "/Users/aryan/Desktop/OpenCVTest/aryan.jpg";
    string referenceImagePath2 = "/Users/aryan/Desktop/OpenCVTest/aryan2.jpg";

    // Load the Haar Cascade face detection model
    CascadeClassifier faceCascade;
    if (!faceCascade.load(cascadePath)) {
        cerr << "Error loading face detection model: " << cascadePath << endl;
        return -1;
    }

    // Load the reference images
    Mat referenceImage1 = imread(referenceImagePath1, IMREAD_GRAYSCALE);
    Mat referenceImage2 = imread(referenceImagePath2, IMREAD_GRAYSCALE);
    if (referenceImage1.empty() || referenceImage2.empty()) {
        cerr << "Error loading reference images." << endl;
        return -1;
    }

    // Open the default camera (typically laptop webcam)
    VideoCapture cap(0);
    if (!cap.isOpened()) {
        cerr << "Error opening camera." << endl;
        return -1;
    }

    namedWindow("Face Detection", WINDOW_NORMAL);

    // Perform face detection and identification
    while (true) {
        Mat frame;
        cap >> frame;

        // Convert frame to grayscale for face detection
        Mat gray;
        cvtColor(frame, gray, COLOR_BGR2GRAY);

        // Detect faces
        vector<Rect> faces;
        faceCascade.detectMultiScale(gray, faces);

        // Process each detected face
        for (const auto& face : faces) {
            // Draw rectangle around face
            rectangle(frame, face, Scalar(0, 0, 255), 2);

            // Extract face region of interest (ROI)
            Mat faceROI = gray(face);

            // Resize reference images to match face ROI size
            Mat resizedReference1, resizedReference2;
            resize(referenceImage1, resizedReference1, faceROI.size());
            resize(referenceImage2, resizedReference2, faceROI.size());

            // Match templates (compare face ROI with reference images)
            Mat result1, result2;
            matchTemplate(faceROI, resizedReference1, result1, TM_CCOEFF_NORMED);
            matchTemplate(faceROI, resizedReference2, result2, TM_CCOEFF_NORMED);

            // Find best matches
            double minVal1, maxVal1, minVal2, maxVal2;
            Point minLoc1, maxLoc1, minLoc2, maxLoc2;
            minMaxLoc(result1, &minVal1, &maxVal1, &minLoc1, &maxLoc1);
            minMaxLoc(result2, &minVal2, &maxVal2, &minLoc2, &maxLoc2);

            // Set thresholds for identification
            double threshold1 = 0.7; // Adjust as needed
            double threshold2 = 0.7; // Adjust as needed

            // Check matches and draw text accordingly
            if (maxVal1 >= threshold1) {
                putText(frame, "Identified as Aryan", Point(face.x, face.y - 10),
                        FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
            } else if (maxVal2 >= threshold2) {
                putText(frame, "Identified as Aryan", Point(face.x, face.y - 10),
                        FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 255, 0), 2);
            } else {
                putText(frame, "Not Identified", Point(face.x, face.y - 10),
                        FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255), 2);
            }
        }

        // Display frame
        imshow("Face Detection", frame);

        // Exit loop on 'q' key press
        if (waitKey(1) == 'q') {
            break;
        }
    }

    // Cleanup
    cap.release();
    destroyAllWindows();

    return 0;
}
