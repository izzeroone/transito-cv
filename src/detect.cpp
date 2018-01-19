/* HOG DETECTOR
 *
 */

#include <dlib/svm_threaded.h>
#include <dlib/gui_widgets.h>
#include <dlib/image_processing.h>
#include <dlib/data_io.h>
#include <dlib/image_transforms.h>
#include <dlib/cmd_line_parser.h>
#include <dlib/opencv.h>
#include<opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>        // Basic OpenCV structures (cv::Mat)
#include <opencv2/highgui/highgui.hpp>
#include <iostream>
#include <fstream>
#include <cstdlib>

using namespace std;
using namespace dlib;
using namespace cv;

struct TrafficSign {
  string name;
  string svm_path;
  rgb_pixel color;
  TrafficSign(string name, string svm_path, rgb_pixel color) :
    name(name), svm_path(svm_path), color(color) {};
};

static string outputFile = "result.txt";
int main(int argc, char** argv) {
  try {
    command_line_parser parser;

    parser.add_option("h","Display this help message.");
    parser.add_option("u", "Upsample each input image <arg> times. Each \
                      upsampling quadruples the number of pixels in the image \
                      (default: 0).", 1);
    parser.add_option("wait","Wait user input to show next image.");
      parser.add_option("v", "open video file", 1);

    parser.parse(argc, argv);
    parser.check_option_arg_range("u", 0, 8);

    const char* one_time_opts[] = {"h","u","v","wait"};
    parser.check_one_time_options(one_time_opts);

    // Display help message
    if (parser.option("h")) {
      cout << "Usage: " << argv[0] << " [options] <list of images>" << endl;
      parser.print_options();

      return EXIT_SUCCESS;
    }


    const unsigned long upsample_amount = get_option(parser, "u", 0);



      VideoCapture capture;
      if(parser.option("v")) {
          capture.open(get_option(parser, "v", ""));
          if(!capture.isOpened()){
              cout << "Unable to open video file";
              cin.get();
              return EXIT_FAILURE;
          }
      } else{
          cout << "You must pass the video path." << endl;
          cout << "\nTry the -h option for more information." << endl;
          return EXIT_FAILURE;
      }
      //open file
      ofstream outFile(outputFile);
      //export to video
      Size S = Size((int) capture.get(CV_CAP_PROP_FRAME_WIDTH),    // Acquire input size
                    (int) capture.get(CV_CAP_PROP_FRAME_HEIGHT));
      VideoWriter outVideo;
      outVideo.open("out.avi", CV_FOURCC('M', 'J', 'P', 'G'), capture.get(CV_CAP_PROP_FPS), S, true);

      typedef scan_fhog_pyramid<pyramid_down<6> > image_scanner_type;

      // Load SVM detectors
      std::vector<TrafficSign> signs;
      signs.push_back(TrafficSign("CAM_RE_TRAI", "../svm_detectors/CamReTrai_detector.svm",
                                  rgb_pixel(255,0,0)));
      signs.push_back(TrafficSign("CAM_RE_PHAI", "../svm_detectors/CamRePhai_detector.svm",
                                  rgb_pixel(255,0,0)));
      signs.push_back(TrafficSign("MOT_CHIEU", "../svm_detectors/MotChieu_detector.svm",
                                  rgb_pixel(255,0,0)));
      signs.push_back(TrafficSign("CAM_NGUOC_CHIEU", "../svm_detectors/CamNguocChieu_detector.svm",
                                  rgb_pixel(255,0,0)));
      signs.push_back(TrafficSign("PARE", "../svm_detectors/pare_detector.svm",
                                  rgb_pixel(255,0,0)));
      signs.push_back(TrafficSign("LOMBADA", "../svm_detectors/lombada_detector.svm",
                                  rgb_pixel(255,122,0)));
      signs.push_back(TrafficSign("PEDESTRE", "../svm_detectors/pedestre_detector.svm",
                                  rgb_pixel(255,255,0)));


      std::vector<object_detector<image_scanner_type> > detectors;

      for (int i = 0; i < signs.size(); i++) {
          object_detector<image_scanner_type> detector;
          deserialize(signs[i].svm_path) >> detector;
          detectors.push_back(detector);
      }

      image_window win;
      std::vector<rect_detection> rects;
      array2d<unsigned char> dlibImage;
      cv::Mat matImage;
      cv::Mat mat2Image;
      //namedWindow("img1",CV_WINDOW_FULLSCREEN);

      int numFrame = capture.get(CV_CAP_PROP_FRAME_COUNT);

      capture.set(CV_CAP_PROP_POS_FRAMES,100);
      for (unsigned long i = 100; i < numFrame; i++) {
          cout << "Load frame : " << i << endl;
          //capture.set(CV_CAP_PROP_POS_FRAMES,i);
          capture.read(matImage);
          if(i % 5 != 0)
              continue;
          mat2Image = matImage;
          cv_image<bgr_pixel> cimg(matImage);
          dlib::assign_image(dlibImage, cimg);
          //Check if frame number to detect
//          for(int j = 0; j < parser.number_of_arguments(); j++){
//              if(stoi(parser[j]) == i){
//                  cout << "Detect frame : " << i << endl;
//                  pyramid_up(dlibImage);
//                  evaluate_detectors(detectors, dlibImage, rects);
//                  for (unsigned long h = 0; h < rects.size(); ++h) {
//                      if(outFile.is_open())
//                  outFile << i << " " << rects[h].rect.left() << " " << rects[h].rect.bottom() << " " << rects[h].rect.right() << " " << rects[h].rect.top() << " " << signs[rects[j].weight_index].name << endl;
//                    cv::rectangle(mat2Image, cv::Point( rects[h].rect.left(),  rects[h].rect.bottom()),cv::Point( rects[h].rect.right(),  rects[h].rect.top()), Scalar(0, 255, 0 ));
//                    }
//                  break;
//              }
//          }
          pyramid_up(dlibImage);
          evaluate_detectors(detectors, dlibImage, rects);
          win.clear_overlay();
          win.set_image(dlibImage);
          cout << "Rect size : " << rects.size() << endl ;
          for (unsigned long h = 0; h < rects.size(); ++h) {
              if(outFile.is_open())
                  outFile << i << " " << rects[h].rect.left() << " " << rects[h].rect.bottom() << " " << rects[h].rect.right() << " " << rects[h].rect.top() << " " << signs[rects[h].weight_index].name << endl;
              win.add_overlay(rects[h].rect, signs[rects[h].weight_index].color,
                              signs[rects[h].weight_index].name);
              //cv::rectangle(mat2Image, cv::Point( rects[h].rect.left(),  rects[h].rect.bottom()),cv::Point( rects[h].rect.right(),  rects[h].rect.top()), Scalar(0, 255, 0 ));
          }


          //imshow("img1", mat2Image);
          //cv::waitKey(20);


          outVideo << mat2Image;


          if (parser.option("wait")) {
              cout << "Press any key to continue...";
              cin.get();
          }

      }

      outFile.close();
  }
  catch (exception& e) {
    cout << "\nexception thrown!" << endl;
    cout << e.what() << endl;
  }
}
