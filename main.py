from PyQt5 import QtWidgets
from ui.mainWindow import Ui_MainWindow
import sys


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # self.ui.showImageButton.clicked.connect(lambda: self.click_show_image_button())


if __name__ == '__main__':
    app = QtWidgets.QApplication([])
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
