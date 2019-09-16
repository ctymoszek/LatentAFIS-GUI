import sys
import os
from shutil import rmtree
from PyQt4 import QtCore, QtGui, uic
from matching.Matching import RunMatcher
from GUI.feature_images import SaveFeatureImages, SaveCorrespondenceImage
 
MainWindowUI = "GUI/LatentMatcher.ui"
CorrespondenceUI = "GUI/correspondence.ui"
LargeLatentUI = "GUI/latentlarge.ui"
MAX_PAGE_NUM = 3
 
Ui_MainWindow, QtBaseClass = uic.loadUiType(MainWindowUI)
Ui_CorrespondenceWindow, QtBaseClass = uic.loadUiType(CorrespondenceUI)
Ui_LatentWindow, QtBaseClass = uic.loadUiType(LargeLatentUI)
 
class MainWindow(QtGui.QMainWindow, Ui_MainWindow):
    latent_data_path = ""
    latent_img_path = ""
    rank_list = []
    corr_list = []
    page_num = 0
    rank_labels = []
    score_labels = []
    result_imgs = []
    result_fnames = []
    latent_root = ""
    current_latent_file = ""
    dir = ""
    corr_imgs_path = ""
    def __init__(self):
        QtGui.QMainWindow.__init__(self)
        Ui_MainWindow.__init__(self)
        self.setupUi(self)
        self.rank_labels = [self.lbl_rank1, self.lbl_rank2, self.lbl_rank3,
                            self.lbl_rank4, self.lbl_rank5, self.lbl_rank6]
        self.score_labels = [self.lbl_score1, self.lbl_score2, self.lbl_score3,
                             self.lbl_score4, self.lbl_score5, self.lbl_score6]
        self.result_imgs = [self.lbl_image1, self.lbl_image2, self.lbl_image3,
                            self.lbl_image4, self.lbl_image5, self.lbl_image6]
        self.result_fnames = [self.lbl_fname1, self.lbl_fname2, self.lbl_fname3,
                              self.lbl_fname4, self.lbl_fname5, self.lbl_fname6]
        self.correspond_btns = [self.btn_Correspond1, self.btn_Correspond2,
                                self.btn_Correspond3, self.btn_Correspond4,
                                self.btn_Correspond5, self.btn_Correspond6]
        self.btn_InputLatent.clicked.connect(self.LoadLatent)
        self.btn_Search.clicked.connect(self.RunSearch)
        self.btn_Up.setEnabled(False)
        self.btn_Down.setEnabled(False)
        self.btn_Up.clicked.connect(self.PageUp)
        self.btn_Down.clicked.connect(self.PageDown)
        self.btn_Original.clicked.connect(self.ShowOriginal)
        self.btn_OF.clicked.connect(self.ShowOF)
        self.btn_Minutiae1.clicked.connect(self.ShowMinutiae1)
        self.btn_Minutiae2.clicked.connect(self.ShowMinutiae2)
        self.btn_ROI.clicked.connect(self.ShowROI)
        self.btn_ViewLarger.clicked.connect(self.ShowLatentNewWindow)
        self.btn_Correspond1.setEnabled(False)
        self.btn_Correspond2.setEnabled(False)
        self.btn_Correspond3.setEnabled(False)
        self.btn_Correspond4.setEnabled(False)
        self.btn_Correspond5.setEnabled(False)
        self.btn_Correspond6.setEnabled(False)
        self.btn_Correspond1.clicked.connect(self.ShowCorrespondenceWindow1)
        self.btn_Correspond2.clicked.connect(self.ShowCorrespondenceWindow2)
        self.btn_Correspond3.clicked.connect(self.ShowCorrespondenceWindow3)
        self.btn_Correspond4.clicked.connect(self.ShowCorrespondenceWindow4)
        self.btn_Correspond5.clicked.connect(self.ShowCorrespondenceWindow5)
        self.btn_Correspond6.clicked.connect(self.ShowCorrespondenceWindow6)
        
        self.dir = os.path.dirname(os.path.dirname(__file__))
        self.corr_imgs_path = os.path.join(self.dir, "Data/current_correspondences/")
 
    def LoadLatent(self):
        self.latent_img_path = QtGui.QFileDialog.getOpenFileName(self, 'Open file','c:\\')
        head, tail = os.path.split(unicode(self.latent_img_path))
        self.lbl_LatentFname.setText(tail)
        self.latent_root, ext = os.path.splitext(tail)
        dir = os.path.dirname(os.path.dirname(__file__))
        self.latent_data_path = os.path.join(dir, "Data/NSITSD27_2_minutiae/" + self.latent_root + ".dat")
        
        #get feature images
        feature_imgs_path = os.path.join(dir, "Data/current_latent_data/")
        if os.path.exists(feature_imgs_path):
            rmtree(feature_imgs_path)
        os.makedirs(feature_imgs_path)
        SaveFeatureImages(self.latent_img_path, self.latent_data_path, feature_imgs_path)
        
        self.current_latent_file = self.latent_img_path
        input_pixmap = QtGui.QPixmap(self.latent_img_path)
        input_pixmap = input_pixmap.scaledToWidth(self.input_img.frameGeometry().width())
        self.input_img.setPixmap(input_pixmap)
        self.input_img.show()
        #print(self.latent_img_path)
        #print(self.latent_data_path)
        return
    
    def RunSearch(self):
        print(self.latent_data_path)
        if len(self.latent_data_path) > 0:
            self.rank_list, self.corr_list = RunMatcher(self.latent_data_path)
            if len(self.rank_list) > 0:
                self.HandleResults()
        return
    
    def HandleResults(self):
        self.page_num = 0
        self.btn_Down.setEnabled(True)
        dir = os.path.dirname(os.path.dirname(__file__))
        #save correspondence images to disk
        if os.path.exists(self.corr_imgs_path):
            rmtree(self.corr_imgs_path)
        os.makedirs(self.corr_imgs_path)
            
        for i in range(6):
            self.rank_labels[i].setText("Rank: " + str(i+1))
            self.score_labels[i].setText("Score: " + str('%.2f'%self.rank_list[i][3]))
            
            root, ext = os.path.splitext(self.rank_list[i][0])
            rolled_img_path = os.path.join(dir, "Data/Rolled/" + root + ".bmp")
            rolled_data_path = os.path.join(dir, "Data/Rolled/" + root + ".dat")
            output_pixmap = QtGui.QPixmap(rolled_img_path)
            output_pixmap = output_pixmap.scaledToWidth(self.result_imgs[i].frameGeometry().width())
            self.result_imgs[i].setPixmap(output_pixmap)
            self.result_imgs[i].show()
            
            corr_img_file = os.path.join(self.corr_imgs_path, "corr" + root + ".jpg")
            SaveCorrespondenceImage(self.latent_data_path, self.latent_img_path,
                                    rolled_img_path, rolled_data_path,
                                    corr_img_file, self.corr_list[i])
            self.correspond_btns[i].setEnabled(True)
            
            self.result_fnames[i].setText("File: " + root + ".bmp")
        return
    
    def PageUp(self):
        self.page_num -= 1
        self.btn_Down.setEnabled(True)
        if(self.page_num == 0):
            self.btn_Up.setEnabled(False)
        for i in range(6):
            if(self.page_num == 0):
                self.correspond_btns[i].setEnabled(True)
            self.rank_labels[i].setText("Rank: " + str(i+1 + self.page_num*6))
            self.score_labels[i].setText("Score: " + str('%.2f'%self.rank_list[i + self.page_num*6][3]))
            
            root, ext = os.path.splitext(self.rank_list[i + self.page_num*6][0])
            dir = os.path.dirname(os.path.dirname(__file__))
            rolled_path = os.path.join(dir, "Data/Rolled/" + root + ".bmp")
            output_pixmap = QtGui.QPixmap(rolled_path)
            if output_pixmap.width() > output_pixmap.height():
                output_pixmap = output_pixmap.scaledToWidth(self.result_imgs[i].frameGeometry().width())
            else:
                output_pixmap = output_pixmap.scaledToHeight(self.result_imgs[i].frameGeometry().height())
            self.result_imgs[i].setPixmap(output_pixmap)
            self.result_imgs[i].show()
            
            self.result_fnames[i].setText("File: " + root + ".bmp")
        return
    
    def PageDown(self):
        self.page_num += 1
        self.btn_Up.setEnabled(True)
        if(self.page_num == MAX_PAGE_NUM):
            self.btn_Down.setEnabled(False)
        for i in range(6):
            self.correspond_btns[i].setEnabled(False)
            self.rank_labels[i].setText("Rank: " + str(i+1 + self.page_num*6))
            self.score_labels[i].setText("Score: " + str('%.2f'%self.rank_list[i + self.page_num*6][3]))
            
            root, ext = os.path.splitext(self.rank_list[i + self.page_num*6][0])
            dir = os.path.dirname(os.path.dirname(__file__))
            rolled_path = os.path.join(dir, "Data/Rolled/" + root + ".bmp")
            output_pixmap = QtGui.QPixmap(rolled_path)
            if output_pixmap.width() > output_pixmap.height():
                output_pixmap = output_pixmap.scaledToWidth(self.result_imgs[i].frameGeometry().width())
            else:
                output_pixmap = output_pixmap.scaledToHeight(self.result_imgs[i].frameGeometry().height())
            self.result_imgs[i].setPixmap(output_pixmap)
            self.result_imgs[i].show()
            
            self.result_fnames[i].setText("File: " + root + ".bmp")
        return
    
    def ShowOriginal(self):
        self.current_latent_file = self.latent_img_path
        input_pixmap = QtGui.QPixmap(self.latent_img_path)
        input_pixmap = input_pixmap.scaledToWidth(self.input_img.frameGeometry().width())
        self.input_img.setPixmap(input_pixmap)
        self.input_img.show()
        return
    
    def ShowROI(self):
        dir = os.path.dirname(os.path.dirname(__file__))
        self.current_latent_file = os.path.join(dir, "Data/current_latent_data/" + self.latent_root + "_ROI.jpg")
        
        input_pixmap = QtGui.QPixmap(self.current_latent_file)
        input_pixmap = input_pixmap.scaledToWidth(self.input_img.frameGeometry().width())
        self.input_img.setPixmap(input_pixmap)
        self.input_img.show()
        return

    def ShowMinutiae1(self):
		dir = os.path.dirname(os.path.dirname(__file__))
		self.current_latent_file = os.path.join(dir, "Data/current_latent_data/" + self.latent_root + "_minu1.jpg")
		
		input_pixmap = QtGui.QPixmap(self.current_latent_file)
		input_pixmap = input_pixmap.scaledToWidth(self.input_img.frameGeometry().width())
		self.input_img.setPixmap(input_pixmap)
		self.input_img.show()
		
		return
    
    def ShowMinutiae2(self):
		dir = os.path.dirname(os.path.dirname(__file__))
		self.current_latent_file = os.path.join(dir, "Data/current_latent_data/" + self.latent_root + "_minu2.jpg")
		
		input_pixmap = QtGui.QPixmap(self.current_latent_file)
		input_pixmap = input_pixmap.scaledToWidth(self.input_img.frameGeometry().width())
		self.input_img.setPixmap(input_pixmap)
		self.input_img.show()
		return
        
    def ShowOF(self):
        dir = os.path.dirname(os.path.dirname(__file__))
        self.current_latent_file = os.path.join(dir, "Data/current_latent_data/" + self.latent_root + "_OF.jpg")
        
        input_pixmap = QtGui.QPixmap(self.current_latent_file)
        input_pixmap = input_pixmap.scaledToWidth(self.input_img.frameGeometry().width())
        self.input_img.setPixmap(input_pixmap)
        self.input_img.show()
        return
    
    def ShowCorrespondenceWindow1(self):
        root, ext = os.path.splitext(self.rank_list[0][0])
        window = CorrespondenceWindow(self, os.path.join(self.corr_imgs_path, "corr" + root + ".jpg"))
        window.show()
        return
    
    def ShowCorrespondenceWindow2(self):
        root, ext = os.path.splitext(self.rank_list[1][0])
        window = CorrespondenceWindow(self, os.path.join(self.corr_imgs_path, "corr" + root + ".jpg"))
        window.show()
        return
    
    def ShowCorrespondenceWindow3(self):
        root, ext = os.path.splitext(self.rank_list[2][0])
        window = CorrespondenceWindow(self, os.path.join(self.corr_imgs_path, "corr" + root + ".jpg"))
        window.show()
        return
    
    def ShowCorrespondenceWindow4(self):
        root, ext = os.path.splitext(self.rank_list[3][0])
        window = CorrespondenceWindow(self, os.path.join(self.corr_imgs_path, "corr" + root + ".jpg"))
        window.show()
        return
    
    def ShowCorrespondenceWindow5(self):
        root, ext = os.path.splitext(self.rank_list[4][0])
        window = CorrespondenceWindow(self, os.path.join(self.corr_imgs_path, "corr" + root + ".jpg"))
        window.show()
        return
    
    def ShowCorrespondenceWindow6(self):
        root, ext = os.path.splitext(self.rank_list[5][0])
        window = CorrespondenceWindow(self, os.path.join(self.corr_imgs_path, "corr" + root + ".jpg"))
        window.show()
        return
    
    def ShowLatentNewWindow(self, event):
        window = LatentWindow(self, self.current_latent_file)
        window.show()
        return
    
class CorrespondenceWindow(QtGui.QMainWindow, Ui_CorrespondenceWindow):
    def __init__(self, parent, img_file):
        super(CorrespondenceWindow, self).__init__(parent)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setupUi(self)
        pixmap = QtGui.QPixmap(img_file).scaledToWidth(self.lbl_CorrImg.frameGeometry().width())
        self.lbl_CorrImg.setPixmap(pixmap)
        self.lbl_CorrImg.show()
        
class LatentWindow(QtGui.QMainWindow, Ui_LatentWindow):
    def __init__(self, parent, img_file):
        super(LatentWindow, self).__init__(parent)
        self.setAttribute(QtCore.Qt.WA_DeleteOnClose)
        self.setupUi(self)
        pixmap = QtGui.QPixmap(img_file).scaledToWidth(self.lbl_LatentImage.frameGeometry().width())
        self.lbl_LatentImage.setPixmap(pixmap)
        self.lbl_LatentImage.show()
    
if __name__ == "__main__":
    app = QtGui.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
