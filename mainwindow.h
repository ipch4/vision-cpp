#ifndef MAINWINDOW_H
#define MAINWINDOW_H

#include <QMainWindow>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace Ui {
class MainWindow;
}

class MainWindow : public QMainWindow
{
    Q_OBJECT

public:
    explicit MainWindow(QWidget *parent = 0);
    void AfficherContours(cv::Mat image);
    double getSimilarity(cv::Mat, cv::Mat);
    void convertHst();
    void camScreen();
    ~MainWindow();

private slots:
    void on_PB_Parcourir_clicked();
    void setCheminImage(QString fichier);
    void on_PB_Afficher_clicked();

    void on_PB_Traitement_clicked();

    void on_ValResultat_linkActivated(const QString &link);
    void on_Result_linkActivated(const QString &link);

    void on_Image_Traitee_linkActivated(const QString &link);

    void on_pushButton_clicked();

    void on_pushButton_2_clicked();

    void on_fond_linkActivated(const QString &link);

private:
    Ui::MainWindow *ui;
    QString cheminImage;
};

#endif // MAINWINDOW_H
