#include <opencv2/opencv.hpp>
#include <iostream>


#define _show(img) cv::imshow(#img,img);
#define _wait(ms)  cv::waitKey(ms);
#define _IMG(k)   cv::imread("img (" #k ").bmp")
#define _WRI(img) cv::imwrite("#img.bmp",img)


using namespace std;
using namespace cv;

typedef vector<Point> Points;
typedef Points Contour;
typedef vector<Points> Pointss;
typedef Pointss Contours;
typedef vector<Point2f> Point2fs;
typedef vector<Point2fs> Point2fss;

typedef struct {
  Point2f C;
  float R;
  float fitness;
  int match_cnt;
} Circle;
typedef vector<Circle> Circles;

struct {
  bool operator()(Circle& a, Circle& b) {
    return a.fitness > b.fitness;
  }
} CircleCompare;

struct
{
	bool operator()(Circle& a, Circle&b) {
		return a.R > b.R;
	}
} RadiusCompare;


vector<Mat> imgs;
void load_images() {
  imgs.push_back(_IMG(1));
  imgs.push_back(_IMG(2));
  imgs.push_back(_IMG(3));
  imgs.push_back(_IMG(4));
  imgs.push_back(_IMG(5));
  imgs.push_back(_IMG(6));
  imgs.push_back(_IMG(7));
  imgs.push_back(_IMG(8));
  imgs.push_back(_IMG(9));
  imgs.push_back(_IMG(10));
 /* imgs.push_back(_IMG(11));
  imgs.push_back(_IMG(12));
  imgs.push_back(_IMG(13));*/
  //for (auto&img : imgs)cvtColor(img, img, COLOR_RGBA2GRAY);
}

int hist_size = 256;        //histogram bin count
cv::Mat calc_histogram(Mat& img) {
  //paramaters
  float range[] = { 0,256 };  //
  const float* hist_range = { range };
  Mat hist;
  calcHist(&img, 1, 0, Mat(), hist, 1, &hist_size, &hist_range, true, false);
  return hist;
}

void draw_hist(cv::Mat& img, cv::Mat& hist, const Scalar& color = { 0,255,0 }) {
  //visualize histogram
  int bin_w = img.cols / 256;
  for (int i = 1; i < hist_size; i++) {
    line(img, { bin_w*(i - 1),img.rows - 1 - (int)hist.at<float>(i - 1) },
    { bin_w*i,img.rows - 1 - (int)hist.at<float>(i) }, color);
  }
}
//Detect histogram peaks using naive approach
//resutls: x = peak bin, y = peak val
vector<Point2f> find_peaks(cv::Mat& hist, bool maxima = true) {
  vector<Point2f> peaks;
  // lowpass filtering
  //blur(hist, hist, cv::Size(15,15));

  //calculate derivative
  Mat diff_hist = hist.clone();
  for (int i = 0; i < hist_size - 1; i++) {
    diff_hist.at<float>(i) = hist.at<float>(i + 1) - hist.at<float>(i);
  }

  Mat diff2_hist = diff_hist.clone();
  for (int i = 0; i < hist_size - 1; i++) {
    diff2_hist.at<float>(i) = diff_hist.at<float>(i + 1) - diff_hist.at<float>(i);
  }

  float radius = 1;
  if (maxima) {
    for (int i = hist_size - radius - 2; i > radius; i--) {
      if (diff_hist.at<float>(i + radius)*diff_hist.at<float>(i - radius) < 0 &&
        diff2_hist.at<float>(i) < 0) {
        peaks.push_back({ float(i),hist.at<float>(i) });
        i -= 2 * radius;
      }
    }
  }
  else {
    for (int i = hist_size - radius - 2; i > radius; i--) {
      if (diff_hist.at<float>(i + radius)*diff_hist.at<float>(i - radius) < 0 &&
        diff2_hist.at<float>(i) > 0) {
        peaks.push_back({ float(i),hist.at<float>(i) });
        i -= 2 * radius;
      }
    }
  }
  return peaks;
}


Contours find_contours(cv::Mat& edge_img) {
  Contours contours;
  findContours(edge_img, contours, RETR_LIST, CHAIN_APPROX_NONE);
  return contours;
}

Circles find_circles(Point2fs& pnts,
  const float min_distance = 50,
  const int min_rad = 200, 
  const int max_rad = 600,
  const float epsilon_rad = 5,
  const float match_threshold = 15,
  const float min_fitness = 0.7,
  const int sample_count = 1000, Mat& img=cv::Mat()) {
  Circles circles;
  for (int i = 0; i < sample_count; i++)
  {
    //random 3 points
    Point2f p0 = pnts[rand() % pnts.size()];
    Point2f p1 = p0;
    Point2f p2 = p1;
    while (norm(p1 - p0) < min_distance)p1 = pnts[rand() % pnts.size()];
    while (norm(p1 - p2) < min_distance || norm(p2 - p0) < min_distance)p2 = pnts[rand() % pnts.size()];
  
    if (!img.empty()) {
      circle(img, p0, 3, { 0,255,255 });
      circle(img, p1, 3, { 0,255,255 });
      circle(img, p2, 3, { 0,255,255 });
      _show(img);
      _wait();
    }

    Vec2f m1 = (p0 + p1) / 2.0;
    Vec2f m2 = (p1 + p2) / 2.0;
    Vec2f n1 = p1 - p0;
    Vec2f n2 = p2 - p1;
    Matx22f A(n1(0), n1(1), n2(0), n2(1));
    Matx21f b(m1.dot(n1), m2.dot(n2));
    if (determinant(A) != 0)
    {
      Matx21f X = A.inv()*b;
      Point2f center{ X(0), X(1) };
      float R = norm(center - p0);
      //check validity
      if (R < min_rad || R> max_rad)continue;

      float perimeter = CV_2PI*R;
      //scan for match
      int cnt = 0;
      for (auto&p : pnts) {
        if (abs(norm(center - p) - R) < epsilon_rad)cnt++;
      }

      float fitness = cnt / perimeter;
      //cout << "Fitness=" << fitness << endl;



      if (fitness > min_fitness) // this is a candidate circle
      {
        if (!img.empty()) {
          circle(img, center, R, { 0,255,0});
        _show(img);
          _wait();
        }
        bool exist = false;
        for (int k = 0; k < circles.size(); k++) {
          Circle& cir = circles[k];
          float match_value = norm(center - cir.C) + abs(R - cir.R);
          if (match_value < match_threshold) {
            exist = true;
            if (fitness > cir.fitness) //replace
            {
              cir = { center,R,fitness, cir.match_cnt };
            }
            cir.match_cnt++;
            break;
          }
        }
        if (!exist) {
          circles.push_back({ center,R,fitness ,0 });
          //cout << "Detected circle: " << center << ", R=" << R << endl;
        }
      }
    }
  }
  return circles;
}

cv::Mat adaptive_threshold(cv::Mat& gray_img,
  const int win_size = 10,
  const float margin=2
  ) {
  const int W = gray_img.cols;
  const int H = gray_img.rows;
  Mat gray_img_th = Mat::zeros(gray_img.size(),gray_img.type());

  for (int r = win_size; r < H - win_size; r++) {
    uchar* ptr = gray_img.ptr<uchar>(r);
    uchar* tptr = gray_img_th.ptr<uchar>(r);
    for (int c = win_size; c < W - win_size; c++) {
      if (ptr[c] == 0)continue;
      int sum = 0;
      int cnt = 0;
      uchar gval = 0;
      for (int k = -win_size; k <= win_size; k++) {
        uchar* wptr = ptr + k*W;
        for (int l = -win_size; l <= win_size; l++) {
          if ((gval=wptr[c + l]) > 0) {
            sum += gval;
            cnt++;
          }
        }
      }
      if (cnt == 0)continue;
      float avg = sum / float(cnt);
      if(ptr[c] < avg + margin) {
        tptr[c] = 255;
      }
    }
  }
  return gray_img_th;
}

std::vector<cv::RotatedRect> FittingEllipseForCircle2(Circles &cirles,cv::Mat &input_img, const float epsilon_rad = 5) {
	std::vector<std::vector<cv::Point2f>> points;
	std::vector<cv::RotatedRect> output_ellipse;

	points.resize(4);

	for (unsigned int i = 0; i != input_img.rows; i++) {
		uchar* Mi = input_img.ptr<uchar>(i);
		for (unsigned int j = 0; j != input_img.cols; j++) {
			cv::Point2f pixel_point = cv::Point2f(j, i);
			double temp_distance1 = cv::norm(cirles[0].C - pixel_point);
			double temp_distance2 = cv::norm(cirles[1].C - pixel_point);
			double temp_distance3 = cv::norm(cirles[2].C - pixel_point);
			double temp_distance4 = cv::norm(cirles[3].C - pixel_point);

			if (abs(temp_distance1 - cirles[0].R) <= epsilon_rad)
			{
				points[0].push_back(pixel_point);
			}
			else if (abs(temp_distance2 - cirles[1].R) <= epsilon_rad) points[1].push_back(pixel_point);
			else if (abs(temp_distance3 - cirles[2].R) <= epsilon_rad) points[2].push_back(pixel_point);
			else if (abs(temp_distance4 - cirles[3].R) <= epsilon_rad) points[3].push_back(pixel_point);
		}
	}
	// fitting ellispe
	for (int i = 0; i !=4; i++) {
		if (points[i].size() <= 5);
	    cv:RotatedRect tempt = cv::fitEllipse(points[i]);
		output_ellipse.push_back(tempt);
	}
	return output_ellipse;
}

std::vector<cv::RotatedRect> FittingEllipseForCirclePoints(Circles &cirles, Point2fs &pnts, const float epsilon_rad = 5) {
	std::vector<std::vector<cv::Point2f>> points;
	std::vector<cv::RotatedRect> output_ellipse;

	points.resize(4);

	for (unsigned int i = 0; i != pnts.size();i++) {
			double temp_distance1 = cv::norm(cirles[0].C - pnts[i]);
			double temp_distance2 = cv::norm(cirles[1].C - pnts[i]);
			double temp_distance3 = cv::norm(cirles[2].C - pnts[i]);
			double temp_distance4 = cv::norm(cirles[3].C - pnts[i]);

			if (abs(temp_distance1 - cirles[0].R) <= epsilon_rad)
			{
				points[0].push_back(pnts[i]);
			}
			else if (abs(temp_distance2 - cirles[1].R) <= epsilon_rad) points[1].push_back(pnts[i]);
			else if (abs(temp_distance3 - cirles[2].R) <= epsilon_rad) points[2].push_back(pnts[i]);
			else if (abs(temp_distance4 - cirles[3].R) <= epsilon_rad) points[3].push_back(pnts[i]);
		}

	// fitting ellispe
	for (int i = 0; i != 4; i++) {
		if (points[i].size() <= 5);
	cv:RotatedRect tempt = cv::fitEllipse(points[i]);
		output_ellipse.push_back(tempt);
	}
	return output_ellipse;
}


Contours filter_(Contours& contours,const Point2f& center,const int min_area= 50) 
{
  Contours out_contours;
  for (Contour& contour : contours) {
    if (contourArea(contour) < min_area)continue;
    out_contours.push_back(contour);
  }
  return out_contours;
}


const int max_defect_size = 1000000;
const int min_bound_area = 180;

void main() {
  //load images
  load_images();

  for (int i = 0;; i++) {
    cout << endl << "Processing img " << i << endl;
	string i_ptr = "img"+to_string(i)+".bmp";
    Mat img = cv::imread(i_ptr,1);
    Mat gray_img;
    //convert color
	//_show(img);
    cv::cvtColor(img, gray_img, COLOR_RGB2GRAY);

    Mat g1, g2;
    //blur 
  /*  tmr.tic();*/
    GaussianBlur(gray_img, g1, { 7,7 }, 2);
    GaussianBlur(gray_img, g2, { 15,15 }, 7);

    GaussianBlur(gray_img, gray_img, { 13,13}, 2);
   /* tmr.toc("Gaussian blur time: ");*/
   // _show(g1);
   // _show(g2);

    //_show((g2-g1)*20);

    Mat bw;
    double min, max;
    bw = (g2 - g1);
    minMaxLoc(bw, &min, &max);
    bw *= 1.5*255.0 / max;
    //_show(g1);
    //_show(g2);
    //_show(bw);



   /* tmr.tic();*/
    //histogram analysis
    Mat hist = calc_histogram(g2);
    //lowpass filter
    GaussianBlur(hist, hist, Size(15, 15), 7);
    //normalized
    normalize(hist, hist, 0, 500, NORM_MINMAX, -1, Mat());
    //draw
    draw_hist(img, hist);
    //detect peak
    vector<Point2f> peaks = find_peaks(hist);
    int bin_w = img.cols / 256;
    for (auto& peak : peaks) {
      line(img, { int(peak.x*bin_w),img.rows }, { int(peak.x*bin_w),img.rows - int(hist.at<float>(int(peak.x))) }, { 255,255,0 });
    }
   /* tmr.toc("Histogram calculation time: ");*/



    if (peaks.size() >= 2) {
     /* tmr.tic();*/
      int th = (peaks[0].x + peaks[1].x) / 2;
      //th = std::max(120, th);
      Mat edge_img;
      Canny(g1, edge_img, th, peaks[0].x);
      _show(edge_img);

      Contours contours = find_contours(edge_img.clone());
      /*tmr.toc("Find contour time: ");*/

      Point2fs pnts;
      //collect all points
      for (auto& contour : contours) for (auto&p : contour)pnts.push_back(Point2f(p));
      if (pnts.size() < 3)continue;
      //Detect circles
     /* tmr.tic();*/
      Circles circles = find_circles(pnts, 50, 100, 1000, 8, 20, .8, 1000);// , img);
      std::sort(circles.begin(), circles.end(), CircleCompare);
     /* tmr.toc("Detect circles time: ");*/

    /*  tmr.tic();*/
      //draw circles
      Mat msk1, msk2;
      msk1 = Mat::zeros(gray_img.size(), gray_img.type());
      msk2 = Mat::zeros(gray_img.size(), gray_img.type());
      Point2f center;
      if (circles.size() >= 4) {
		//[1]
		  Circles object1;
		  object1.push_back(circles[0]);
		  object1.push_back(circles[1]);
		  object1.push_back(circles[2]);
		  object1.push_back(circles[3]);
		  std::sort(object1.begin(), object1.end(), RadiusCompare);
        //draw circles
		  std::vector<cv::RotatedRect> ellipses = FittingEllipseForCircle2(object1,gray_img, 24);
		  for (int i = 0; i != ellipses.size(); i++) {
			  ellipse(img, ellipses[i], { 0,255,0 }, 2);
		  }

		  double distance_object1 = cv::norm(object1[0].C - object1[2].C);
		  double distance_object2 = cv::norm(object1[0].C - object1[3].C);
		  if (distance_object1 > distance_object2) {
			 /* circle(img, object1[0].C, object1[0].R, { 0,255,0 }, 2);
			  circle(img, object1[3].C, object1[3].R, { 0,255,0 }, 2);*/
			  // draw msk obejct 1
			 /* circle(msk1, object1[0].C, object1[0].R, { 255 }, -1);
			  circle(msk2, object1[3].C, object1[3].R, { 255 }, -1);*/
			  ellipse(msk1, ellipses[0], { 255 }, -1);
			  ellipse(msk2, ellipses[3], { 255 }, -1);
			 /* circle(img, object1[1].C, object1[1].R, { 0,0,255 }, 2);
			  circle(img, object1[2].C, object1[2].R, { 0,0,255 }, 2);*/
			  // draw msk object 2
			 /* circle(msk1, object1[1].C, object1[1].R, { 255 }, -1);
			  circle(msk2, object1[2].C, object1[2].R, { 255 }, -1);*/
			  ellipse(msk1, ellipses[1], { 255 }, -1);
			  ellipse(msk2, ellipses[2], { 255 }, -1);
		  }
		  else {
			 /* circle(img, object1[0].C, object1[0].R, { 0,255,0 }, 2);
			  circle(img, object1[2].C, object1[2].R, { 0,255,0 }, 2);*/
			  // draw msk obejct 1
			  /*circle(msk1, object1[0].C, object1[0].R, { 255 }, -1);
			  circle(msk2, object1[2].C, object1[2].R, { 255 }, -1);*/
			  ellipse(msk1, ellipses[0], { 255 }, -1);
			  ellipse(msk2, ellipses[2], { 255 }, -1);
			  center = (object1[0].C + object1[2].C) *0.5f;

			 /* circle(img, object1[1].C, object1[1].R, { 0,0,255 }, 2);
			  circle(img, object1[3].C, object1[3].R, { 0,0,255 }, 2);*/
			  // draw msk obejct 1
			 /* circle(msk1, object1[1].C, object1[1].R, { 255 }, -1);
			  circle(msk2, object1[3].C, object1[3].R, { 255 }, -1);*/
			  ellipse(msk1, ellipses[1], { 255 }, -1);
			  ellipse(msk2, ellipses[3], { 255 }, -1);
		  }
		
        //draw mask
       /* circle(msk1, object1[0].C, object1[0].R, { 255 }, -1);
        circle(msk2, object1[1].C, object1[1].R, { 255 }, -1);
        center = (object1[0].C + object1[1].C) *0.5f;*/
      }

      //calculate mask
      Mat msk = msk1^msk2;
      gray_img &= msk;
     

   
      erode(msk, msk, getStructuringElement(MORPH_ELLIPSE, { 53,53}));
      bw &= msk;
      Mat th_img = adaptive_threshold(gray_img, 20,-2);
      dilate(th_img, th_img, getStructuringElement(MORPH_CROSS, { 9,9 }));
    
	 
     _show(th_img);

     /* tmr.tic();*/
      //find contours
      contours = find_contours(th_img);
      const float min_ratio=0.3;

      // FILTERING
      for (int i = 0; i < contours.size(); i++) {
        //remove small artifacts
        if (contourArea(contours[i]) < min_bound_area)continue;
        auto rect = minAreaRect(contours[i]);
        //remove too big artifacts
        if (rect.size.area() > max_defect_size)continue;

        //TODO: MORE FILTER HERE
        //remove cocentric stuff
        //float w = max(rect.size.width, rect.size.height);
        //float h = min(rect.size.width, rect.size.height);
        //Vec2f v = rect.center - center;
        //float angle = abs(atan2(v(0), v(1)))*180/CV_PI;

        drawContours(img, contours, i, { 0,0,255 });
        Point2f rect_points[4]; rect.points(rect_points);
        for (int j = 0; j < 4; j++) line(img, rect_points[j], rect_points[(j + 1) % 4], { 255,0,255 }, 1, 8);
      }
     /* tmr.toc("Filter result time: ");*/
      //_show(edge_img);
      //_show(msk);
    }


    _show(img);
	cv::imwrite("detect"+i_ptr, img);
    //_show(gray_img);
    //_show(abs_img);
    //_show(blur_img);
    //_show(hist_img);
    char key = _wait();
    if (key == 'q')break;
  }
}
