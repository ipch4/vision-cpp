#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "iostream"
#include <QFileDialog>
#include <QDebug>
#include <QPixmap>
#include <QMessageBox>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>




using namespace cv;
using namespace std;

MainWindow::MainWindow(QWidget *parent) :
    QMainWindow(parent),
    ui(new Ui::MainWindow)
{

    ui->setupUi(this);
    int x = this->ui->fond->width();
    int y = this->ui->fond->height();
    QPixmap pix ("C:/Users/Dell/Documents/Test_OpenCV/fond.jpg");
    this->ui->fond->setPixmap(pix);

    // read an image
    //cv::Mat image = cv::imread("C:/Users/Dell/Documents/Test_OpenCV/banane.jpg", 1);
    // create image window named "My Image"
   //cv::namedWindow("My Image");
    // show the image on window
    //cv::imshow("My Image", image);
   // AfficherContours(image);


}

void MainWindow::AfficherContours(cv::Mat image)
{


    //cv::Mat image2 = cv::imread("C:/Users/Dell/Documents/Test_OpenCV/banane.jpg", 1);
    //getSimilarity(image, image2);

    //Prepare the image for findContours
    cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);
    cout<<"Cout apres cvtColor"<<endl;
    cv::threshold(image, image, 200, 255, cv::THRESH_BINARY);
    cout<<"Cout apres threshold"<<endl;
    //cv::imshow("My Image", image);




    //Find the contours. Use the contourOutput Mat so the original image doesn't get overwritten
    std::vector<std::vector<cv::Point> > contours;
    cv::Mat contourOutput = image.clone();
    cv::findContours( contourOutput, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE );

    //Draw the contours
    cv::Mat contourImage(image.size(), CV_8UC3, cv::Scalar(0,0,0));
    cv::Scalar colors[3];
    colors[0] = cv::Scalar(255, 0, 0);
    colors[1] = cv::Scalar(0, 255, 0);
    colors[2] = cv::Scalar(0, 0, 255);
    for (size_t idx = 0; idx < contours.size(); idx++) {
    cv::drawContours(contourImage, contours, idx, colors[idx % 3]);


    cout<<"idx = "<<idx<<endl;
    }

      // cv::imshow("Input Image", image);
      //cv::moveWindow("Input Image", 0, 0);
      //cv::imshow("Contours", contourImage);
      // cv::moveWindow("Contours", 200, 0);
      // cv::waitKey(0);
    cout<<"Ligne avant imwrite(contours.jpg)"<<endl;
       imwrite("Contours.jpg", contourImage);

      int x = this->ui->Image_Traitee->width();
      int y = this->ui->Image_Traitee->height();
      QPixmap *pix = new QPixmap("contours.jpg");
      this->ui->Image_Traitee->setPixmap(pix->scaled(x,y));





}

// Compare two images by getting the L2 error (square-root of sum of squared error).
double MainWindow::getSimilarity(cv::Mat image1, cv::Mat image2)
{
    if(image1.rows > 0 && image1.rows == image2.rows && image1.cols > 0 && image1.cols == image2.cols)
    {
        // Calculate the L2 relative error between images.
        double errorL2 = norm(image1, image2, NORM_L2 );
        // Convert to a reasonable scale, since L2 error is summed across all pixels of the image.
        double similiraty = errorL2 /(double)(image1.rows*image1.cols);
        cout<<"Return Similarity"<<similiraty<<endl;
        return similiraty;

    }
    else{
           cout<<"ERROR"<<endl;
        //Images have a different size
        return 100000000.0; //Return a bad value
    }
}

MainWindow::~MainWindow()
{
    delete ui;

}

void MainWindow::on_PB_Parcourir_clicked()
{
    QString fichier;
    QWidget image;
    fichier = QFileDialog :: getOpenFileName( this , tr( "Ouvrir un fichier" ) , "/ home" , tr( "Images (* .png * .xpm * .jpg)" ));
    setCheminImage(fichier);
    this->ui->LE_Fichier->setText(fichier);
}

void MainWindow::on_PB_Afficher_clicked()
{

    QString fichierOrigine = this -> ui -> LE_Fichier -> text();
    //fichier = fichierOrigine.toStdString().c_str();
    //cv::Mat image = cv::imread(fichier, 1);



            int x = ui->Image_Origine->width();
            int y = ui->Image_Origine->height();
            QPixmap *pix = new QPixmap(fichierOrigine);

            this->ui->Image_Origine->setPixmap(pix->scaled(x,y));




}

void MainWindow::setCheminImage(QString fichier)
{
    cheminImage = fichier;
}

void MainWindow::on_PB_Traitement_clicked()
{

    QString fichierOrigine = this -> ui -> LE_Fichier -> text();
    //const char* fichier = fichierOrigine.tos

        Mat image = imread(fichierOrigine.toStdString(), IMREAD_COLOR );
        Mat gray_image;
        Mat src_image;
        cvtColor( image, gray_image, COLOR_BGR2GRAY );
        cv :: threshold(gray_image, src_image, 200, 255, cv::THRESH_BINARY);

        //Find the contours. Use the contourOutput Mat so the original image doesn't get overwritten
           std::vector<std::vector<cv::Point> > contours;
           cv::Mat contourOutput = src_image.clone();
           cv::findContours( contourOutput, contours, cv::RETR_LIST, cv::CHAIN_APPROX_NONE );

           //Draw the contours
           cv::Mat contourImage(src_image.size(), CV_8UC3, cv::Scalar(0,0,0));
           cv::Scalar colors[3];
           colors[0] = cv::Scalar(255, 0, 0);
           colors[1] = cv::Scalar(0, 255, 0);
           colors[2] = cv::Scalar(0, 0, 255);
           for (size_t idx = 0; idx < contours.size(); idx++)
           {
            cv::drawContours(contourImage, contours, idx, colors[idx % 3]);
           }

        imwrite( "Gray_Image.jpg", gray_image );
        imwrite( "src_Image.jpg", src_image );
        imwrite( "Coutours.jpg", contourImage );

        convertHst();
        int x = this->ui->Image_Traitee->width();
        int y = this->ui->Image_Traitee->height();
        QPixmap *pixmap_img = new QPixmap("Coutours.jpg");
        this->ui->Image_Traitee->setPixmap(pixmap_img->scaled(x,y));








}
void MainWindow::convertHst()
{
    cv::Mat image0 = cv::imread("C:/Users/Dell/Documents/Test_OpenCV/banane0.jpg", 1);
    cv::Mat image1 = cv::imread("C:/Users/Dell/Documents/Test_OpenCV/fraise.jpg", 1);
    cv::Mat image2 = cv::imread("C:/Users/Dell/Documents/Test_OpenCV/fraise-banane.jpg", 1);
    //cv::Mat image3 = cv::imread("C:/Users/Dell/Documents/Test_OpenCV/banane3.jpg", 1);
    //cv::Mat image4 = cv::imread("C:/Users/Dell/Documents/Test_OpenCV/banane4.jpg", 1);

     Mat hsv_image0;
     Mat hsv_image1;
     Mat hsv_image;
      Mat hsv_image2;
     Mat hsv_half_down;
    QString fichierOrigine = this -> ui -> LE_Fichier -> text();
    //const char* fichier = fichierOrigine.tos

        Mat image = imread(fichierOrigine.toStdString(), IMREAD_COLOR );
        Mat gray_image;
        Mat src_image;

/// Convert to HSV
    cvtColor( image0, hsv_image0, COLOR_BGR2HSV );
    cvtColor( image1, hsv_image1, COLOR_BGR2HSV );
    cvtColor( image2, hsv_image2, COLOR_BGR2HSV );
    //cvtColor( image3, hsv_image3, COLOR_BGR2HSV );
    //cvtColor( image4, hsv_image4, COLOR_BGR2HSV );

    cvtColor( image, hsv_image, COLOR_BGR2HSV );


    hsv_half_down = hsv_image0( Range( hsv_image0.rows/2, hsv_image0.rows - 1 ), Range( 0, hsv_image0.cols - 1 ) );

    /// Using 50 bins for hue and 60 for saturation
    int h_bins = 50; int s_bins = 60;
    int histSize[] = { h_bins, s_bins };


    // hue varies from 0 to 179, saturation from 0 to 255
    float h_ranges[] = { 0, 180 };
    float s_ranges[] = { 0, 256 };

    const float* ranges[] = { h_ranges, s_ranges };


    // Use the o-th and 1-st channels
    int channels[] = { 0, 1 };


    /// Histograms
        MatND hist_image0;
        MatND hist_half_down;
        MatND hist_image;
        MatND hist_image1;
        MatND hist_image2;



    /// Calculate the histograms for the HSV images


    calcHist( &hsv_image0, 1, channels, Mat(), hist_image0, 2, histSize, ranges, true, false );
    normalize( hist_image0, hist_image0, 0, 1, NORM_MINMAX, -1, Mat() );

    calcHist( &hsv_half_down, 1, channels, Mat(), hist_half_down, 2, histSize, ranges, true, false );
    normalize( hist_half_down, hist_half_down, 0, 1, NORM_MINMAX, -1, Mat() );

    calcHist( &hsv_image, 1, channels, Mat(), hist_image, 2, histSize, ranges, true, false );
    normalize( hist_image, hist_image, 0, 1, NORM_MINMAX, -1, Mat() );

    calcHist( &hsv_image1, 1, channels, Mat(), hist_image1, 2, histSize, ranges, true, false );
    normalize( hist_image1, hist_image1, 0, 1, NORM_MINMAX, -1, Mat() );

    calcHist( &hsv_image2, 1, channels, Mat(), hist_image2, 2, histSize, ranges, true, false );
    normalize( hist_image2, hist_image2, 0, 1, NORM_MINMAX, -1, Mat() );


    /// Apply the histogram comparison methods
       for( int i = 0; i < 4; i++ )
       {
           int compare_method = i;
           double base_image0 = compareHist( hist_image0, hist_image0, compare_method );
           double base_half = compareHist( hist_image0, hist_half_down, compare_method );
           double base_image = compareHist( hist_image0, hist_image, compare_method );
           double base_image1 = compareHist( hist_image, hist_image1, compare_method );
           double base_image2 = compareHist( hist_image2, hist_image2, compare_method );


          // printf( " Method [%d] Perfect, Base-Half, Base-image0, Base-Test(2) : %f, %f, %f \n", i, base_image0, base_half , base_image);

           //printf( " Base-Half: %f  \n" ,base_half);
           //printf( " Base-image0: %f  \n" ,base_image0);
           printf( " Base-image: %f  \n" ,base_image);
            printf( " Base-fraisebana: %f  \n" ,base_image2);
           if (base_image<0.62)
           {
               this->ui->Result->setText("C'est une banane");
           }
           else{
               if (base_image1<0.63)
               {
                   this->ui->Result->setText("C'est une fraise");
               }else{


                this->ui->Result->setText("Ce n'est pas le bon fruit");
           }}

       }

       printf( " Terminé");


}




void MainWindow::on_pushButton_clicked()
{
     //read an image
    cv::Mat help = cv::imread("C:/Users/Dell/Documents/Test_OpenCV/help.jpg", 1);
    // create image window named "My Image"
   cv::namedWindow("help");
    // show the image on window
   cv::imshow("help", help);


}

void MainWindow::on_pushButton_2_clicked()
{
    //read an image
   cv::Mat aproposde = cv::imread("C:/Users/Dell/Documents/Test_OpenCV/aproposde.jpg", 1);
   // create image window named "My Image"
  cv::namedWindow("A propos de");
   // show the image on window
  cv::imshow("A propos de", aproposde);
}


