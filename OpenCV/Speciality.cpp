//IP_Project Specialty Recognition
//Created by Armen and Amalia 


// ************************************************
#include <opencv2/objdetect/objdetect.hpp>
#include <opencv2/opencv.hpp>
#include <algorithm> 
#include <iostream>
#include <utility>
#include <cstdlib>
#include <vector>

//We assume there are 2 basic white color ranges for recognizing doctors' uniforms. Making compliance histogram
std::vector<int> ColorStatistic(const cv::Mat& img) {
	long q = img.rows * img.cols;
	std::vector<int> cv(9, 0);
	std::vector<long long> av(3, 0);
	std::vector<int> min(3, 0);
	std::vector<int> max(3, 0);

	for (int p = 0; p < 3; ++p) {
		for (int i = 0; i < img.rows; ++i) {
			for (int j = 0; j < img.cols; ++j) {
				av[p] += img.at<cv::Vec3b>(i, j)[p];
				if (img.at<cv::Vec3b>(i, j)[p] < min[p]) min[p] = img.at<cv::Vec3b>(i, j)[0];
				if (img.at<cv::Vec3b>(i, j)[p] > max[p]) max[p] = img.at<cv::Vec3b>(i, j)[0];
			}
		}
		cv[p] = av[p] / q;
	}

	cv[3] = min[0];
	cv[4] = min[1];
	cv[5] = min[2];
	cv[6] = max[0];
	cv[7] = max[1];
	cv[8] = max[2];

	std::cout << "\n============COLOR STATISTIC==============="
		<< "\n average B = " << cv[0]
		<< "\n average G = " << cv[1]
		<< "\n average R = " << cv[2]
		<< "\n min B = " << cv[3]
		<< "\n min G = " << cv[4]
		<< "\n min R = " << cv[5]
		<< "\n max B = " << cv[6]
		<< "\n max G = " << cv[7]
		<< "\n max R = " << cv[8]
		<< "\n============COLOR STATISTIC===============\n";

	return cv;
}

cv::Mat PredictionImage(char c, bool b, int i) {
	srand(time(0));
	std::vector<cv::Mat> rate;
	for (int i = 0; i < 21; ++i) {
		std::string name = cv::format("%d.jpg", i);
		cv::Mat img = cv::imread(name);
		if (img.empty())
		{
			std::cerr << "whaa " << name << " can't be loaded!" << std::endl;
			continue;
		}
		rate.push_back(img);
	}

	std::vector<cv::Mat> text;
	for (int i = 0; i < 6; ++i) {
		std::string name = cv::format("%d.jpg", i + 21);
		cv::Mat img = cv::imread(name);
		if (img.empty())
		{
			std::cerr << "whaa " << name << " can't be loaded!" << std::endl;
			continue;
		}
		text.push_back(img);
	}

	if (!b) {
		return text[3 + rand() % 3];
	}

	cv::Mat result;
	switch (c)
	{
	case 'n': {vconcat(text[0], rate[i / 5], result); return result; }
	case 'd': {vconcat(text[1], rate[i / 5], result); return result; }
	case 's': {vconcat(text[2], rate[i / 5], result); return result; }
	}
}


//We assume face and body have a similar color ranges for recognizing naked person. Making compliance histogram
std::vector<std::vector<int>> NakedColorsRange(const cv::Mat& img) {
	std::vector<std::vector<int>> ColorHist(3, std::vector<int>(13, 0));

	for (int i = 0; i < img.rows; ++i) {
		for (int j = 0; j < img.cols; ++j) {
			for (int p = 0; p < 3; ++p) {
				++ColorHist[p][img.at<cv::Vec3b>(i, j)[p] / 20];
			}
		}
	}

	for (int p = 0; p < 3; ++p) {
		std::cout << "\nColorHist [" << p << "] >>>";
		for (int i = 0; i < 13; ++i) {
			std::cout << ColorHist[p][i] << " - ";
		}
	}

	std::vector<std::vector<int>> max3(3, std::vector<int>(3, 0));

	//making max3 array of 3 highest values' indexes
	for (int i = 0; i < 3; ++i) {
		for (int p = 0; p < 3; ++p) {
			max3[i][p] = p;
			for (int j = p + 1; j < 13; ++j) {
				if (ColorHist[i][j] > ColorHist[i][max3[i][p]]) {
					max3[i][p] = j;
				}
			}
			std::swap(ColorHist[i][max3[i][p]], ColorHist[i][p]);
		}
	}

	for (int i = 0; i < 3; ++i) {
		std::cout << "\n";
		for (int j = 0; j < 3; ++j) {
			std::cout << max3[i][j] << " - ";
		}
		std::cout << "\n";
	}

	return max3;
}

//Verification of compliance. Is the man naked?
std::pair<bool, int> NakedChecking(const cv::Mat& img, const cv::Rect& body, const cv::Rect& face) {

	cv::Mat bodyRoi = img(body);
	std::vector<std::vector<int>> bodyMaxCol = NakedColorsRange(bodyRoi);

	cv::Mat faceRoi = img(face);
	std::vector<std::vector<int>> faceMaxCol = NakedColorsRange(faceRoi);

	std::vector<int> r(3, 0);
	for (int i = 0; i < 3; ++i) {
		for (int j = 0; j < 3; ++j) {
			for (int p = 0; p < 3; ++p) {
				if (bodyMaxCol[i][j] == faceMaxCol[i][p]) {
					++r[i];
					std::cout << r[i] << "\n";
				}
			}
		}
	}

	int rSum = r[0] + r[1] + r[2];
	int rNZM = (r[0] != 0) + (r[1] != 0) + (r[2] != 0); // rNZM = r None Zero Members

	std::cout << "\n\n\n\n" << rSum << "=rSum   -   rNZM=" << rNZM << "\n";

	if (rNZM < 3) {
		return std::make_pair(false, 0);
	}
	else {
		return std::make_pair(true, rSum * 10);
	}


}


//We assume there are 2 basic white color ranges for recognizing doctors' uniforms. Making compliance histogram
std::vector<int> DoctorPrimaryColorsHistogram(const cv::Mat& img) {
	std::vector<int> ColorHist(3, 0);

	for (int i = 0; i < img.rows; ++i) {
		for (int j = 0; j < img.cols; ++j) {
			if (img.at<cv::Vec3b>(i, j)[0] > 225 &&
				img.at<cv::Vec3b>(i, j)[1] > 220 &&
				img.at<cv::Vec3b>(i, j)[2] > 220) {
				++ColorHist[0];
				continue;
			}
			if (img.at<cv::Vec3b>(i, j)[0] >= 170 && img.at<cv::Vec3b>(i, j)[0] <= 225 &&
				img.at<cv::Vec3b>(i, j)[1] >= 160 && img.at<cv::Vec3b>(i, j)[1] <= 220 &&
				img.at<cv::Vec3b>(i, j)[2] >= 160 && img.at<cv::Vec3b>(i, j)[2] <= 220) {
				++ColorHist[1];
				continue;
			}
			++ColorHist[2];
		}
	}

	return ColorHist;
}

//Verification of compliance. Can a person be a doctor?
std::pair<bool, int> DoctorChecking(const cv::Mat& img, const cv::Rect& body, const cv::Rect& hat) {

	int rate = 0;

	cv::Mat bodyRoi = img(body);
	std::vector<int> bodyHist = DoctorPrimaryColorsHistogram(bodyRoi);


	cv::Rect bodyMidChestRect(body.x + body.width / 3, body.y, body.width / 3, body.height * 2 / 3);
	cv::Mat bodyMidChestRoi = img(bodyMidChestRect); //Region of interest
	std::vector<int> bodyMidChestHist = DoctorPrimaryColorsHistogram(bodyMidChestRoi);

	for (int i = 0; i < bodyHist.size(); ++i) {
		bodyHist[i] -= bodyMidChestHist[i];
	}

	cv::Mat hatRoi = img(hat);
	std::vector<int> hatHist = DoctorPrimaryColorsHistogram(hatRoi);

	int bodyS = bodyHist[0] + bodyHist[1] + bodyHist[2];
	int bodyColorProp_0 = bodyHist[0] * 100 / bodyS;
	int bodyColorProp_1 = bodyHist[1] * 100 / bodyS;
	int bodyColorProp_2 = bodyHist[2] * 100 / bodyS;
	int bodyColorProp_primary = (bodyS - bodyHist[2]) * 100 / bodyS;

	//statistical datas
	std::cout << "White - " << bodyHist[0] << ",  " << double(bodyHist[0]) / bodyS * 100 << "%\n";
	std::cout << "Dwhite - " << bodyHist[1] << ",  " << double(bodyHist[1]) / bodyS * 100 << "%\n";
	std::cout << "Others - " << bodyHist[2] << ",  " << double(bodyHist[2]) / bodyS * 100 << "%\n";

	int hatS = hatHist[0] + hatHist[1] + hatHist[2];
	int hatColorProp_0 = hatHist[0] * 100 / hatS;
	int hatColorProp_1 = hatHist[1] * 100 / hatS;
	int hatColorProp_2 = hatHist[2] * 100 / hatS;
	int hatColorProp_primary = (hatS - hatHist[2]) * 100 / hatS;

	std::cout << "Doctor hatColor prop = " << hatColorProp_0
		<< " - " << hatColorProp_1
		<< " - " << 100 - hatColorProp_primary << "\n";

	// feature_55 is true if the primary colors is distributed over 60% or more and main color is more than 45%
	bool bodyFeature_55 = (bodyColorProp_primary >= 55 && bodyColorProp_0 >= 45);
	std::cout << "\n=====Doctor bodyFeature_55 = " << bodyFeature_55 << "\n";

	// feature_75 is true if the primary colors is distributed over 75% or more and main color is more than 60%
	bool bodyFeature_75 = (bodyColorProp_primary >= 75 && bodyColorProp_0 >= 60);
	std::cout << "\n=====Doctor bodyFeature_75 = " << bodyFeature_75 << "\n";

	// feature_90 is true if the primary colors is distributed over 85% or more and main color is more than 75%
	bool bodyFeature_90 = (bodyColorProp_primary >= 90 && bodyColorProp_0 >= 75);
	std::cout << "\n=====Doctor bodyFeature_90 = " << bodyFeature_90 << "\n";

	//main compliance checking
	if (bodyColorProp_0 < 45) {
		return std::make_pair(false, 0);
	}

	if (bodyFeature_90) {
		return std::make_pair(true, 80);
	}
	else if (bodyFeature_75) {
		return std::make_pair(true, 65);
	}
	else if (bodyFeature_55) {
		return std::make_pair(true, 50);
	}
	else {
		return std::make_pair(false, 0);
	}
}

//We assume there are 4 basic color ranges for recognizing soldiers' uniforms. Making compliance histogram
std::vector<int> SoldierPrimaryColorsHistogram(const cv::Mat& img) {
	std::vector<int> ColorHist(5, 0);

	for (int i = 0; i < img.rows; ++i) {
		for (int j = 0; j < img.cols; ++j) {
			if (img.at<cv::Vec3b>(i, j)[0] >= 50 && img.at<cv::Vec3b>(i, j)[0] <= 90 &&
				img.at<cv::Vec3b>(i, j)[1] >= 85 && img.at<cv::Vec3b>(i, j)[1] <= 120 &&
				img.at<cv::Vec3b>(i, j)[2] >= 50 && img.at<cv::Vec3b>(i, j)[2] <= 105) {
				++ColorHist[0];
				continue;
			}
			if (img.at<cv::Vec3b>(i, j)[0] >= 100 && img.at<cv::Vec3b>(i, j)[0] <= 155 &&
				img.at<cv::Vec3b>(i, j)[1] >= 140 && img.at<cv::Vec3b>(i, j)[1] <= 200 &&
				img.at<cv::Vec3b>(i, j)[2] >= 150 && img.at<cv::Vec3b>(i, j)[2] <= 210) {
				++ColorHist[1];
				continue;
			}
			if (img.at<cv::Vec3b>(i, j)[0] >= 30 && img.at<cv::Vec3b>(i, j)[0] <= 85 &&
				img.at<cv::Vec3b>(i, j)[1] >= 55 && img.at<cv::Vec3b>(i, j)[1] <= 110 &&
				img.at<cv::Vec3b>(i, j)[2] >= 60 && img.at<cv::Vec3b>(i, j)[2] <= 130) {
				++ColorHist[2];
				continue;
			}
			if (img.at<cv::Vec3b>(i, j)[0] >= 10 && img.at<cv::Vec3b>(i, j)[0] <= 65 &&
				img.at<cv::Vec3b>(i, j)[1] >= 5 && img.at<cv::Vec3b>(i, j)[1] <= 65 &&
				img.at<cv::Vec3b>(i, j)[2] >= 10 && img.at<cv::Vec3b>(i, j)[2] <= 70) {
				++ColorHist[3];
				continue;
			}
			++ColorHist[4];
		}
	}

	return ColorHist;
}

//Verification of compliance. Can a person bn2e a soldier?
std::pair<bool, int> SoldierChecking(const cv::Mat& img, const cv::Rect& body, const cv::Rect& hat) {

	int rate = 0;

	cv::Mat bodyRoi = img(body);
	std::vector<int> bodyHist = SoldierPrimaryColorsHistogram(bodyRoi);

	cv::Rect bodyNeckRect(body.x + body.width / 3, body.y, body.width / 3, body.height / 3);
	cv::Mat bodyNeckRoi = img(bodyNeckRect);
	std::vector<int> bodyNeckHist = SoldierPrimaryColorsHistogram(bodyNeckRoi);

	for (int i = 0; i < bodyHist.size(); ++i) {
		bodyHist[i] -= bodyNeckHist[i];
	}

	cv::Mat hatRoi = img(hat);
	std::vector<int> hatHist = SoldierPrimaryColorsHistogram(hatRoi);

	int bodyS = bodyHist[0] + bodyHist[1] + bodyHist[2] + bodyHist[3] + bodyHist[4];
	int bodyColorProp_0 = bodyHist[0] * 100 / bodyS;
	int bodyColorProp_1 = bodyHist[1] * 100 / bodyS;
	int bodyColorProp_2 = bodyHist[2] * 100 / bodyS;
	int bodyColorProp_3 = bodyHist[3] * 100 / bodyS;
	int bodyColorProp_primary = (bodyS - bodyHist[4]) * 100 / bodyS;

	//statistical datas
	std::cout << "Green - " << bodyHist[0] << ",  " << double(bodyHist[0]) / bodyS * 100 << "%\n";
	std::cout << "Cream - " << bodyHist[1] << ",  " << double(bodyHist[1]) / bodyS * 100 << "%\n";
	std::cout << "Brown - " << bodyHist[2] << ",  " << double(bodyHist[2]) / bodyS * 100 << "%\n";
	std::cout << "Dark - " << bodyHist[3] << ",  " << double(bodyHist[3]) / bodyS * 100 << "%\n";
	std::cout << "Others - " << bodyHist[4] << ",  " << double(bodyHist[4]) / bodyS * 100 << "%\n";

	int hatS = hatHist[0] + hatHist[1] + hatHist[2] + hatHist[3] + hatHist[4];
	int hatColorProp_0 = hatHist[0] * 100 / hatS;
	int hatColorProp_1 = hatHist[1] * 100 / hatS;
	int hatColorProp_2 = hatHist[2] * 100 / hatS;
	int hatColorProp_3 = hatHist[3] * 100 / hatS;
	int hatColorProp_primary = (hatS - hatHist[4]) * 100 / hatS;

	std::cout << "hatColor prop = " << hatColorProp_0
		<< " - " << hatColorProp_1
		<< " - " << hatColorProp_2
		<< " - " << hatColorProp_3
		<< " - " << 100 - hatColorProp_primary << "\n";

	// feature_70 is true if none of the primary colors is distributed over 70% or more
	bool bodyFeature_70 = 1;
	if (bodyColorProp_primary != 0) {
		bool bodyFeature_70 = ((bodyColorProp_0 * 100 / bodyColorProp_primary < 70) &&
			(bodyColorProp_1 * 100 / bodyColorProp_primary < 70) &&
			(bodyColorProp_2 * 100 / bodyColorProp_primary < 70) &&
			(bodyColorProp_3 * 100 / bodyColorProp_primary < 70));
	}
	std::cout << "\n=====bodyFeature_70 = " << bodyFeature_70 << "\n";

	//feature_adc is true if at least two primary colors have a more or equal average distribution
	int bodyAverDist = bodyColorProp_primary / 4;
	int bodyFeature_adc = (bodyColorProp_0 >= bodyAverDist) +
		(bodyColorProp_1 >= bodyAverDist) +
		(bodyColorProp_2 >= bodyAverDist) +
		(bodyColorProp_3 >= bodyAverDist);
	std::cout << "\n=====bodyFeature_adc = " << bodyFeature_adc << "\n";

	//feature_hat is true if hypothetical hat is exist and has 35% or more primary colors proportion
	bool hatFeature_35 = hatColorProp_primary >= 35;
	std::cout << "\n=====hatFeature_35 = " << hatFeature_35 << "\n";

	//main compliance checking
	if (bodyColorProp_primary <= 40) {
		return std::make_pair(false, 0);
	}

	if (!bodyFeature_70) {
		if (hatFeature_35) {
			return std::make_pair(false, 0); //return std::make_pair(true, 0); 
		}
		else {
			return std::make_pair(false, 0);
		}
	}

	int h = 0;
	if (hatFeature_35) h = 5;

	if (bodyColorProp_primary < 50) {
		if (bodyFeature_adc >= 3) return std::make_pair(true, 40 + h);
		else if (bodyFeature_adc == 2) return std::make_pair(true, 35 + h);
		else return std::make_pair(true, 15 + h);
	}
	else if (bodyColorProp_primary < 60) {
		if (bodyFeature_adc >= 3) return std::make_pair(true, 50 + 2 * h);
		else if (bodyFeature_adc == 2) return std::make_pair(true, 45 + 2 * h);
		else return std::make_pair(true, 25 + 2 * h);
	}
	else if (bodyColorProp_primary < 70) {
		if (bodyFeature_adc >= 3) return std::make_pair(true, 65 + 3 * h);
		else if (bodyFeature_adc == 2) return std::make_pair(true, 60 + 3 * h);
		else return std::make_pair(true, 40 + 3 * h);
	}
	else if (bodyColorProp_primary < 80) {
		if (bodyFeature_adc >= 3) return std::make_pair(true, 70 + 3 * h);
		else if (bodyFeature_adc == 2) return std::make_pair(true, 65 + 3 * h);
		else return std::make_pair(true, 45 + 3 * h);
	}
	else {
		if (bodyFeature_adc >= 3) return std::make_pair(true, 80 + 4 * h);
		else if (bodyFeature_adc == 2) return std::make_pair(true, 75 + 4 * h);
		else return std::make_pair(true, 60 + 4 * h);
	}
}

int main() {
	cv::Mat img = cv::imread("A.png");

	cv::namedWindow("original_img", cv::WINDOW_AUTOSIZE);
	cv::imshow("original_img", img);

	cv::Mat grayImg;
	cv::cvtColor(img, grayImg, cv::COLOR_BGR2GRAY);
	//cv::equalizeHist(grayImg, grayImg);

	cv::CascadeClassifier face_cascade;
	std::vector<cv::Rect> face;

	if (!face_cascade.load("C:/opencv/sources/data/haarcascades/haarcascade_frontalface_alt_tree.xml")) {
		std::cout << "===LOADING ERROR(*_1.xml)===\n";
		return -1;
	};

	//=====face detection 1=====
	face_cascade.detectMultiScale(grayImg, face, 1.1, 3, 0, cv::Size(30, 30));

	if (!face.size()) {
		if (!face_cascade.load("C:/opencv/sources/data/haarcascades/haarcascade_frontalface_alt.xml")) {
			std::cout << "===LOADING ERROR(*_2.xml)===\n";
			return -1;                                       
		};
		//=====face detection 2=====
		face_cascade.detectMultiScale(grayImg, face, 1.1, 3, 0, cv::Size(30, 30)); 

		if (!face.size()) {
			std::cout << "=====SORRY, THE OBJECT WAS NOT FOUND=====\n";
			cv::Mat info = cv::imread("nwf.jpg");                            
			cv::namedWindow("info", cv::WINDOW_AUTOSIZE);
			cv::imshow("info", info);
			cv::waitKey(0);
			return 0;
		}
	}

	cv::Rect faceRect = face[0];
	cv::Mat faceRoi = img(faceRect);

	std::cout << "\n face color statistic \n";
	//ColorStatistic(faceRoi);
	NakedColorsRange(faceRoi);

	//=====face marking=====
	cv::Mat markedImg = img.clone();
	cv::rectangle(markedImg, faceRect, cv::Scalar(0, 255, 0), 2);
	//cv::namedWindow("marked_img", cv::WINDOW_AUTOSIZE);
	//cv::imwrite("marked.jpg", markedImg);
	//cv::imshow("marked_img", markedImg);

	//=====upperbody detection=====
	int upperBodyUpperLeft_x = std::max(0, int(faceRect.x - faceRect.width / 2));
	int upperBodyUpperLeft_y = std::min(img.rows - 1, int(faceRect.y + faceRect.height * 1.3));
	int upperBody_width = std::min(img.cols - upperBodyUpperLeft_x, faceRect.width * 2);
	int upperBody_height = std::min(img.rows - upperBodyUpperLeft_y, int(faceRect.height * 2));

	cv::Rect upperBodyRect(upperBodyUpperLeft_x, upperBodyUpperLeft_y, upperBody_width, upperBody_height);
	cv::Mat upperBodyRoi = img(upperBodyRect);

	std::cout << "\n body color statistic \n";
	//ColorStatistic(upperBodyRoi);
	NakedColorsRange(upperBodyRoi);

	//=====upperbody marking=====
	cv::rectangle(markedImg, upperBodyRect, cv::Scalar(0, 0, 255), 2);


	//=====hat detection=====
	int hatUpperLeft_x = faceRect.x + faceRect.width * 0.05;
	int hatUpperLeft_y = std::max(0, int(faceRect.y - (double)faceRect.height * 0.15));
	int hat_width = faceRect.width * 0.9;
	int hat_height = faceRect.y - hatUpperLeft_y + (double)faceRect.height * 0.1;

	cv::Rect hatRect(hatUpperLeft_x, hatUpperLeft_y, hat_width, hat_height);
	cv::Mat hatRoi = img(hatRect);

	//=====hat marking=====
	cv::rectangle(markedImg, hatRect, cv::Scalar(255, 0, 0), 2);

	cv::namedWindow("marked_img", cv::WINDOW_NORMAL);
	cv::imwrite("...marked.jpg", markedImg);
	cv::imshow("marked_img", markedImg);

	std::pair<bool, int> SoldierResult = SoldierChecking(img, upperBodyRect, hatRect);
	std::cout << "_______________________SoldierResult=(" << SoldierResult.first << ", " << SoldierResult.second << "%)\n";

	std::pair<bool, int> DoctorResult = DoctorChecking(img, upperBodyRect, hatRect);
	std::cout << "_______________________DoctorResult=(" << DoctorResult.first << ", " << DoctorResult.second << "%)\n";

	std::pair<bool, int> NakedResult = NakedChecking(img, upperBodyRect, faceRect);
	std::cout << "_______________________NakedResult=(" << NakedResult.first << ", " << NakedResult.second << "%)\n";

	cv::Mat resultImg;

	if (NakedResult.second >= DoctorResult.second && NakedResult.second >= SoldierResult.second) {
		resultImg = PredictionImage('n', NakedResult.first, NakedResult.second);
	}
	else if (DoctorResult.second >= SoldierResult.second) {
		resultImg = PredictionImage('d', DoctorResult.first, DoctorResult.second);
	}
	else {
		resultImg = PredictionImage('s', SoldierResult.first, SoldierResult.second);
	}

	cv::namedWindow("marked_img", cv::WINDOW_AUTOSIZE);
	cv::imwrite("result.jpg", resultImg);
	cv::imshow("marked_img", resultImg);

	cv::waitKey(0);

	return 0;
}

//*******************************************************