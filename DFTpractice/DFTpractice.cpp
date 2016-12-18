// DFTpractice.cpp : Defines the entry point for the console application.
//

#include "stdafx.h"
#include "opencv\cxcore.hpp"
#include "opencv2\imgproc\imgproc.hpp"
#include "opencv2\highgui\highgui.hpp"

using namespace cv;

void drawLines(Mat src, Mat dst)
{
	int r_offset = src.rows / 4;
	int c_offset = src.cols / 4;
	for (int i = 1; i <= 4; ++i)
	{
		line(src, Point(c_offset * i, 0), Point(0, r_offset * i), Scalar(255, 255, 255), 8);
	}
	for (int i = 1; i <= 3; ++i)
	{
		line(src, Point(src.cols, r_offset * i), Point(c_offset * i, src.rows), Scalar(255, 255, 255), 8);
	}
}

void MyrotImg(Mat src, Mat dst, int Rotdegree)
{
	Point center(src.cols / 2, src.rows / 2);
	Mat rotMat = getRotationMatrix2D(center, Rotdegree, 1.0);
	warpAffine(src, dst, rotMat, src.size(), 1, 0);
}

void Myshift(int x_offset, int y_offset, Mat src, Mat dst)
{
	Point2f s[3] = { Point2f(0, 0), Point2f(1, 0), Point2f(1, 1) };
	Point2f d[3] = { Point2f(x_offset, y_offset), Point2f(1 + x_offset, y_offset), Point2f(1 + x_offset, 1 + y_offset) };
	Mat M(2, 3, CV_32FC1);
	M = getAffineTransform(s, d);
	warpAffine(src, dst, M, src.size(), 1, 0);
}
int main(int argc, char **argv[])
{
	Mat src = imread("rect.png", CV_LOAD_IMAGE_GRAYSCALE);
	Mat transformedSrc = Mat::zeros(src.size(), src.type());
	Myshift(100, 100, src, transformedSrc);
	
	Mat padded;
	// Expand the image to an optimal size
	int r = getOptimalDFTSize(transformedSrc.rows);
	int c = cvGetOptimalDFTSize(transformedSrc.cols);
	copyMakeBorder(transformedSrc, padded, r - transformedSrc.rows, 0, c - transformedSrc.cols, 0, BORDER_CONSTANT);
	// Make place for both the complex and the real values
	Mat planes[] = { Mat_<float>(padded), Mat::zeros(padded.size(), CV_32F) };
	Mat complexSrc;
	merge(planes, 2, complexSrc);
	// Make the Discrete Fourier Transform
	dft(complexSrc, complexSrc);
	// Transform the real and complex values to magnitude
	split(complexSrc, planes);
	Mat mag(transformedSrc.size(), CV_32F);
	magnitude(planes[0], planes[1], mag);
	// Switch to a logarithmic scale
	mag += Scalar::all(1);
	log(mag, mag);
	int cr = mag.rows / 2; int cc = mag.cols / 2;
	Mat tl(mag, Rect(0, 0, cc, cr));
	Mat tr(mag, Rect(cc, 0, cc, cr));
	Mat bl(mag, Rect(0, cr, cc, cr));
	Mat br(mag, Rect(cc, cr, cc, cr));

	Mat temp;
	tl.copyTo(temp);
	br.copyTo(tl);
	temp.copyTo(br);

	tr.copyTo(temp);
	bl.copyTo(tr);
	temp.copyTo(bl);

	// Normalize
	normalize(mag, mag, 0, 1, CV_MINMAX);
	imshow("original", src);
	imshow("transformed", transformedSrc);
	imshow("DFT", mag);
	// output result
	imwrite("result\\transformed.png", transformedSrc);
	Mat test;
	mag.copyTo(test);
	mag.convertTo(test, CV_16UC1);
	imwrite("result\\tranDFT.jpg", test);
	waitKey(0);
	return 0;
}