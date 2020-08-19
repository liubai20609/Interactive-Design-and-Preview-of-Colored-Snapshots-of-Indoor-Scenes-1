# coding=utf-8
import numpy as np
import cv2
from PyQt4.QtCore import *
from PyQt4.QtGui import *

try:
    from PyQt4.QtCore import QString
except ImportError:
    QString = str
from .ui_control import UIControl

from data import lab_gamut
from skimage import color
import os
import datetime
import glob
import sys
from semantic_figure import semantic_figure
from findCloseColor import findclosestColor_label,findNeedColor


class GUIDraw(QWidget):
    def __init__(self, model, dist_model=None, load_size=256, win_size=512):
        QWidget.__init__(self)
        self.model = None
        self.image_file = None
        self.pos = None
        self.model = model
        self.dist_model = dist_model  # distribution predictor, could be empty
        self.win_size = win_size
        self.load_size = load_size
        self.setFixedSize(win_size, win_size)
        self.uiControl = UIControl(win_size=win_size, load_size=load_size)
        self.move(win_size, win_size)
        self.movie = True
        self.init_color()  # initialize color
        self.im_gray3 = None
        self.eraseMode = False
        self.ui_mode = 'none'  # stroke or point
        self.image_loaded = False
        self.use_gray = True
        self.total_images = 0
        self.image_id = 0
        self.method = 'with_dist'

        self.i = 0
        self.image_dir = ''
        self.semantic_fig=np.zeros((450, 800, 3), np.uint8)
        self.semantic_Max=0
        self.mask_split256=np.zeros((256, 256, 3), np.uint8)
        self.virtualMask=np.zeros((256, 256, 3), np.uint8)
        self.zhuangshi=np.zeros((450, 800, 3), np.uint8)

        self.loadImage=False #点击load按钮变True

        self.im_ab0_Procrssing = None
        self.maskmask = None
        self.newIM = np.zeros((256, 256, 3), np.uint8)

    def clock_count(self):
        self.count_secs -= 1
        self.update()

    def init_result(self, image_file):  # 最开始读入图片的方法
        # yanhai_begin
        print("==================================")
        print("This is gui_draw-->>init_result-->firstLoadImage!!!!")
        print("==================================")
        # yanhai_end
        self.read_image(image_file.encode('utf-8'))  # read an image
        # yanhai_begin
        pathh = self.findDir(image_file)
        # pppath=os.getcwd()
        # print("pppath===="+pppath)
        # print("os.getcwd(image_file)"+pathh[0]) #/home/mitc/yanhai/interactive-deep-colorization/test_imgs/huidutu
        # print("os.getcwd(image_file)" + pathh[1])#huidutu.png
        # self.get_batches(pathh[0])
        # print("init_result===="+self.read_image)
        # yamhai_end
        self.reset()

    # yanhai_begin
    def findDir(self, image_file):
        # image_dir=os.path.split(os.path.realpath(image_file))
        # yanhai_begin
        print("==================================")
        print("This is gui_draw-->>finDir!!!!")
        print("==================================")
        # yanhai_end
        print("findDir--image_file" + image_file)       #具体的图片
        self.image_dir = os.path.dirname(image_file)    # tests_imgs/huidutu
        print("os.path.dirname(image_file)====" + self.image_dir)
        print("I find the image_file")
        self.img_list = sorted(glob.glob(os.path.join(self.image_dir, '*.png')))
        print("self.img_list====")
        print(self.img_list)
        listdir = os.listdir(self.image_dir)
        print("os.listdir====")
        print(listdir)

        # show self.mask_split256 存储288*512 max 256*256
        # self.semantic_fig,self.semantic_Max,self.mask_split256,self.loadImage=semantic_figure(self.image_dir,self.loadImage)
        # return image_dir
        return self.img_list




    # yanhai_end
    def get_batches(self, img_dir):
        self.img_list = sorted(glob.glob(os.path.join(img_dir, '*.png')))
        self.total_images = len(self.img_list)
        img_first = self.img_list[0]
        # yanhai_begin
        # import matplotlib.pyplot as plt
        # plt.figure('hudutu2')
        # plt.imshow(img_first)
        # plt.show()
        print("self.img_list===")
        # print(self.img_list)
        print("self.total_images====")
        # print(self.total_images)
        # yanhai_end
        self.init_result(img_first)

    def nextImage(self):
        self.save_result()
        self.image_id += 1
        if self.image_id == self.total_images:
            print('you have finished all the results')
            sys.exit()
        img_current = self.img_list[self.image_id]
        # self.reset()
        self.init_result(img_current)
        self.reset_timer()

    def read_image(self, image_file):
        # self.result = None
        # yanhai_begin
        print("==================================")
        print("This is gui_draw-->>read_image!!!!")
        print("==================================")
        # yanhai_end
        # yanhai_begin
        # image_file = image_file[:-4] + "1" + image_file[-4:]
        yy = self.findDir(image_file)[0]
        print("get_batches======" + yy)
        # self.get_batches(yy)
        # import matplotlib.pyplot as plt
        # plt.figure('hudutu')
        # im1 = cv2.cvtColor(cv2.imread(image_file, 1), cv2.COLOR_BGR2RGB)
        # plt.imshow(im1)
        # plt.show()
        print("image_file====" + image_file)

        # yanhai_end

        self.image_loaded = True
        self.image_file = image_file
        print(image_file)
        im_bgr = cv2.imread(image_file)
        self.im_full = im_bgr.copy()
        # get image for display
        h, w, c = self.im_full.shape
        max_width = max(h, w)
        r = self.win_size / float(max_width)
        self.scale = float(self.win_size) / self.load_size
        print('scale = %f' % self.scale)
        rw = int(round(r * w / 4.0) * 4)
        rh = int(round(r * h / 4.0) * 4)

        self.im_win = cv2.resize(self.im_full, (rw, rh), interpolation=cv2.INTER_CUBIC)

        self.dw = int((self.win_size - rw) // 2)
        self.dh = int((self.win_size - rh) // 2)
        self.win_w = rw
        self.win_h = rh
        self.uiControl.setImageSize((rw, rh))
        im_gray = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2GRAY)
        # 20200107显示彩色白膜（注释下面,改为下一行
        # self.im_gray3 = cv2.cvtColor(im_gray, cv2.COLOR_GRAY2BGR)
        self.im_gray3 = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)

        self.gray_win = cv2.resize(self.im_gray3, (rw, rh), interpolation=cv2.INTER_CUBIC)
        im_bgr = cv2.resize(im_bgr, (self.load_size, self.load_size), interpolation=cv2.INTER_CUBIC)
        self.im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
        lab_win = color.rgb2lab(self.im_win[:, :, ::-1])

        self.im_lab = color.rgb2lab(im_bgr[:, :, ::-1])
        self.im_l = self.im_lab[:, :, 0]
        self.l_win = lab_win[:, :, 0]
        self.im_ab = self.im_lab[:, :, 1:]
        self.im_size = self.im_rgb.shape[0:2]

        self.im_ab0 = np.zeros((2, self.load_size, self.load_size))
        self.im_mask0 = np.zeros((1, self.load_size, self.load_size))
        self.brushWidth = 2 * self.scale

        self.model.load_image(image_file)

        if (self.dist_model is not None):
            self.dist_model.set_image(self.im_rgb)
            self.predict_color()
        return self.im_rgb

    def update_im(self):
        # yanhai_begin
        print("==================================")
        print("This is gui_draw-->>update_im!!!!")
        print("==================================")
        # yanhai_end
        self.update()
        QApplication.processEvents()

    def update_ui(self, move_point=True):
        # yanhai_begin
        print("==================================")
        print("This is gui_draw-->>update_ui!!!!")
        print("==================================")
        # yanhai_end
        if self.ui_mode == 'none':
            return False
        is_predict = False
        snap_qcolor = self.calibrate_color(self.user_color, self.pos)
        self.color = snap_qcolor
        self.emit(SIGNAL('update_color'), QString('background-color: %s' % self.color.name()))

        if self.ui_mode == 'point':
            print("this is pointing!!!!!")
            if move_point:
                self.uiControl.movePoint(self.pos, snap_qcolor, self.user_color, self.brushWidth)
            else:
                self.user_color, self.brushWidth, isNew = self.uiControl.addPoint(self.pos, snap_qcolor,
                                                                                  self.user_color, self.brushWidth)
                if isNew:
                    is_predict = True
                    # self.predict_color()

        if self.ui_mode == 'stroke':
            print("this is stroking!!!!!!!!!!")
            self.uiControl.addStroke(self.prev_pos, self.pos, snap_qcolor, self.user_color, self.brushWidth)
        if self.ui_mode == 'erase':
            print("this is erasing!!!!!!!!!!!")
            isRemoved = self.uiControl.erasePoint(self.pos)
            if isRemoved:
                is_predict = True
                # self.predict_color()
        return is_predict

    def reset(self):
        self.ui_mode = 'none'
        self.pos = None
        self.result = None
        self.user_color = None
        self.color = None
        self.uiControl.reset()
        self.init_color()
        self.compute_result()
        self.predict_color()
        self.update()

    def scale_point(self, pnt):
        print("this is scale_point!!!!!!!!")
        x = int((pnt.x() - self.dw) / float(self.win_w) * self.load_size)
        y = int((pnt.y() - self.dh) / float(self.win_h) * self.load_size)
        return x, y

    def valid_point(self, pnt):
        if pnt is None:
            print('WARNING: no point\n')
            return None
        else:
            if pnt.x() >= self.dw and pnt.y() >= self.dh and pnt.x() < self.win_size - self.dw and pnt.y() < self.win_size - self.dh:
                x = int(np.round(pnt.x()))
                y = int(np.round(pnt.y()))
                return QPoint(x, y)
            else:
                print('WARNING: invalid point (%d, %d)\n' % (pnt.x(), pnt.y()))
                return None

    def init_color(self):
        self.user_color = QColor(128, 128, 128)  # default color red
        self.color = self.user_color

    def change_color(self, pos=None):
        # yanhai_begin
        print("==================================")
        print("This is gui_draw-->>change_color!!!!")
        print("==================================")
        # yanhai_end
        if pos is not None:
            x, y = self.scale_point(pos)
            L = self.im_lab[y, x, 0]
            self.emit(SIGNAL('update_gamut'), L)
            rgb_colors = self.suggest_color(h=y, w=x, K=9)
            #yanhai_bagin
            print("this is gui_draw.py->suggest_color->rgb_colors::-->")
            print(rgb_colors)
            # rgb_colors[-1, :] = 0.5 #20190530改

            self.emit(SIGNAL('suggest_colors'), rgb_colors)
            used_colors = self.uiControl.used_colors()
            self.emit(SIGNAL('used_colors'), used_colors)
            snap_color = self.calibrate_color(self.user_color, pos)
            c = np.array((snap_color.red(), snap_color.green(), snap_color.blue()), np.uint8)

            self.emit(SIGNAL('update_ab'), c)

    def calibrate_color(self, c, pos):
        x, y = self.scale_point(pos)

        # snap color based on L color
        color_array = np.array((c.red(), c.green(), c.blue())).astype(
            'uint8')
        mean_L = self.im_l[y, x]
        snap_color = lab_gamut.snap_ab(mean_L, color_array)
        snap_color = color_array       #20190603改-->保证色板与C颜色一致
        snap_qcolor = QColor(snap_color[0], snap_color[1], snap_color[2])
        return snap_qcolor

    def set_color(self, c_rgb):
        # yanhai_begin
        print("==================================")
        print("This is gui_draw-->>set_color!!!!")
        print("==================================")
        # yanhai_end
        c = QColor(c_rgb[0], c_rgb[1], c_rgb[2])
        self.user_color = c
        snap_qcolor = self.calibrate_color(c, self.pos)

        self.color = snap_qcolor
        self.emit(SIGNAL('update_color'), QString('background-color: %s' % self.color.name()))
        self.uiControl.update_color(snap_qcolor, self.user_color)
        self.compute_result()

    def erase(self):
        self.eraseMode = not self.eraseMode

    def load_image(self):  # UI load时调用
        # yanhai_begin


        print("==================================")
        print("This is gui_draw-->>load_image!!!!")
        print("==================================")
        # yanhai_end
        img_path = unicode(QFileDialog.getOpenFileName(self, 'load an input image'))
        # yanhai_bengin
        self.image_dir = os.path.dirname(img_path)
        self.semantic_fig, self.semantic_Max, self.mask_split256,self.virtualMask,self.zhuangshi,self.white,self.black = semantic_figure(self.image_dir)
        print("QFileDialog.getOpenFileName====" + img_path)
        # yanhai_end
        self.init_result(img_path)


    def save_result(self):
        # yanhai_begin
        print("==================================")
        print("This is gui_draw-->>save_result!!!!")
        print("==================================")
        # yanhai_end
        path = os.path.abspath(self.image_file)
        path, ext = os.path.splitext(path)

        suffix = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
        save_path = "_".join([path, self.method, suffix])

        print('saving result to <%s>\n' % save_path)
        if not os.path.exists(save_path):
            os.mkdir(save_path)

        # np.save(os.path.join(save_path, 'im_l.npy'), self.model.img_l)
        # np.save(os.path.join(save_path, 'im_ab.npy'), self.im_ab0)
        # np.save(os.path.join(save_path, 'im_mask.npy'), self.im_mask0)

        result_bgr = cv2.cvtColor(self.result, cv2.COLOR_RGB2BGR)
        mask = self.im_mask0.transpose((1, 2, 0)).astype(np.uint8) * 255
        cv2.imwrite(os.path.join(save_path, 'input_mask.png'), mask)
        cv2.imwrite(os.path.join(save_path, 'ours.png'), result_bgr)
        cv2.imwrite(os.path.join(save_path, 'ours_fullres.png'), self.model.get_img_fullres()[:, :, ::-1])
        cv2.imwrite(os.path.join(save_path, 'input_fullres.png'), self.model.get_input_img_fullres()[:, :, ::-1])
        cv2.imwrite(os.path.join(save_path, 'input.png'), self.model.get_input_img()[:, :, ::-1])
        cv2.imwrite(os.path.join(save_path, 'input_ab.png'), self.model.get_sup_img()[:, :, ::-1])

    def enable_gray(self):
        self.use_gray = not self.use_gray
        self.update()

    def predict_color(self):
        # yanhai_begin
        print("==================================")
        print("This is gui_draw-->>predict_color!!!!")
        print("==================================")
        # yanhai_end
        if self.dist_model is not None and self.image_loaded:
            im, mask = self.uiControl.get_input()
            im_mask0 = mask > 0.0
            self.im_mask0 = im_mask0.transpose((2, 0, 1))
            im_lab = color.rgb2lab(im).transpose((2, 0, 1))
            self.im_ab0 = im_lab[1:3, :, :]
            self.dist_model.net_forward(self.im_ab0, self.im_mask0)

    def  suggest_color(self, h, w, K=5):
        # yanhai_begin
        print("==================================")
        print("This is gui_draw-->>suggest_color!!!!")
        print("==================================")
        # yanhai_end

        if self.dist_model is not None and self.image_loaded:
            ab, conf = self.dist_model.get_ab_reccs(h=h, w=w, K=K, N=25000, return_conf=True)
            L = np.tile(self.im_lab[h, w, 0], (K, 1))
            colors_lab = np.concatenate((L, ab), axis=1)
            colors_lab3 = colors_lab[:, np.newaxis, :]
            colors_rgb = np.clip(np.squeeze(color.lab2rgb(colors_lab3)), 0, 1)
            # colors_rgb_withcurr = np.concatenate((self.model.get_img_forward()[h, w, np.newaxis, :] / 255., colors_rgb),
            #                                   axis=0)
            # colors_rgb_withcurr = [94, 235, 0, 233, 122, 0, 183, 48, 0,
            #                        49, 117, 227, 98, 242, 227, 224, 118, 227,
            #                        204, 229, 138, 236, 244, 45, 117, 75, 45, 234, 243, 205]
            # colors_rgb_withcurr = [94, 235, 0]

            colors_rgb_withcurr = self.ExpertsSuggest(h,w)
            # colors_rgb_withcurr = [[94, 235, 0], [233, 122, 0], [183, 48, 0],
            #                        [49, 117, 227], [98, 242, 227], [224, 118, 227],
            #                        [204, 229, 138], [236, 244, 45], [117, 75, 45], [234, 243, 205]]
            return colors_rgb_withcurr
        else:
            return None

    def compute_result(self):
        print("==================================")
        print("This is gui_draw-->>compute_result====")
        print("==================================")
        # yanhai myCommentAllThis
        # im, mask = self.uiControl.get_input()
        # im_mask0 = mask > 0.0
        # self.im_mask0 = im_mask0.transpose((2, 0, 1))
        # im_lab = color.rgb2lab(im).transpose((2, 0, 1))
        # self.im_ab0 = im_lab[1:3, :, :]
        #
        # self.model.net_forward(self.im_ab0, self.im_mask0)
        # ab = self.model.output_ab.transpose((1, 2, 0))
        # ab_win = cv2.resize(ab, (self.win_w, self.win_h), interpolation=cv2.INTER_CUBIC)
        # pred_lab = np.concatenate((self.l_win[..., np.newaxis], ab_win), axis=2)
        # pred_rgb = (np.clip(color.lab2rgb(pred_lab), 0, 1) * 255).astype('uint8')
        # self.result = pred_rgb
        # self.emit(SIGNAL('update_result'), self.result)
        # self.update()

    def paintEvent(self, event):
        print("==================================")
        print("This is gui_draw-->>paintEvent====")
        print("==================================")
        painter = QPainter()
        painter.begin(self)
        painter.fillRect(event.rect(), QColor(49, 54, 49))  # tianchong
        painter.setRenderHint(QPainter.Antialiasing)
        if self.use_gray or self.result is None:
            im = self.gray_win
        else:
            im = self.result

        if im is not None:
            qImg = QImage(im.tostring(), im.shape[1], im.shape[0], QImage.Format_RGB888)
            painter.drawImage(self.dw, self.dh, qImg)

        self.uiControl.update_painter(painter)
        painter.end()

    def wheelEvent(self, event):
        print("==================================")
        print("This is gui_draw-->>wheelEvent!!!!")
        print("==================================")
        d = event.delta() / 120
        self.brushWidth = min(4.05 * self.scale, max(0, self.brushWidth + d * self.scale))
        print('update brushWidth = %f' % self.brushWidth)
        self.update_ui(move_point=True)
        self.update()

    def is_same_point(self, pos1, pos2):
        if pos1 is None or pos2 is None:
            return False
        dx = pos1.x() - pos2.x()
        dy = pos1.y() - pos2.y()
        d = dx * dx + dy * dy
        # print('distance between points = %f' % d)
        return d < 25

    def mousePressEvent(self, event):
        print("==================================")
        print("This is gui_draw-->>mousePressEvent!!!!")
        print("==================================")
        print('mouse press', event.pos())
        pos = self.valid_point(event.pos())

        if pos is not None:
            if event.button() == Qt.LeftButton:
                self.pos = pos
                self.ui_mode = 'point'
                self.change_color(pos)
                self.update_ui(move_point=False)
                # self.compute_result()

            if event.button() == Qt.RightButton:
                # draw the stroke
                self.pos = pos
                self.ui_mode = 'erase'
                self.update_ui(move_point=False)
                # self.compute_result()

    def mouseMoveEvent(self, event):
        print("==================================")
        print("This is gui_draw-->>mouseMoveEvent!!!!")
        print("==================================")
        self.pos = self.valid_point(event.pos())
        if self.pos is not None:
            if self.ui_mode == 'point':
                self.update_ui(move_point=True)
                self.compute_result()

    def mouseReleaseEvent(self, event):
        pass

    def sizeHint(self):
        return QSize(self.win_size, self.win_size)  # 28 * 8

    # yanhai_begin
    def okPress(self):
        print("==================================")
        print("This is gui_draw-->>compute_result====")
        print("==================================")
        import time
        start = time.clock()
        # yanhaiok_begin

        # self.l_norm = 1.
        # self.ab_norm = 1.
        # self.l_mean = 50.
        # self.ab_mean = 0.
        # self.mask_mult = 1.

        # im_bgr = cv2.imread(image_file)
        # im_bgr = cv2.resize(im_bgr, (self.load_size, self.load_size), interpolation=cv2.INTER_CUBIC)
        # self.im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
        # self.img_lab = color.rgb2lab(self.im_rgb).transpose((2, 0, 1))
        # self.img_lab_mc = self.img_lab / np.array((self.l_norm, self.ab_norm, self.ab_norm))[:, np.newaxis,
        #                                  np.newaxis] - np.array(
        #     (self.l_mean / self.l_norm, self.ab_mean / self.ab_norm, self.ab_mean / self.ab_norm))[:, np.newaxis,
        #                                                np.newaxis]
        # self.img_l_mc = self.img_lab_mc[[0], :, :]
        # yanhaiok_end

        # yanhai_okpress_begin
        n = len(self.img_list)
        import datetime
        now = datetime.datetime.now()
        yymmdd = now.strftime("%Y--%m--%d %H:%M:%S")
        image_dir = self.image_dir + '/saveResult' + yymmdd
        if not os.path.exists(image_dir):
            os.mkdir(image_dir)

        im, mask_orginal = self.uiControl.get_input()  #获取最原始的BGR用户点和mask
        cv2.imwrite(os.path.join(image_dir,'yuanshi_IM_RGB.png'),im[:,:,::-1])

        im1,haveLabel_color=self.colorVirtualPoint(self.virtualMask,self.mask_split256,im)
        cv2.imwrite(os.path.join(image_dir,'yuanshi_xuni_IM.png'),im1)        #BGR
        #获取条件概率的label:(R,G,B)
        needColor_label = self.hadLabel_label(haveLabel_color)
        #生成新的IM
        newIM = np.zeros((256,256,3),np.uint8)
        if needColor_label:
            newIM = self.AddcolorVirtualPoint(needColor_label, self.virtualMask, self.mask_split256,im1[:,:,::-1])
            if newIM is not None:
                cv2.imwrite(os.path.join(image_dir,'tiaojiangailv_IM_RGB.png'), newIM[:, :, ::-1])
                # cv2.imwrite(os.path.join(image_dir,'tiaojiangailv_IM_BGR.png'), newIM)

        im = im[:, :, ::-1] + newIM[:,:,::-1]*(~mask_orginal/255)
        # self.newIM = im1
        # im = im[:,:,::-1]+im1

        im = im[:,:,::-1] #BGR格式

        mask =mask_orginal + self.virtualMask  #256*256*1 +256*256*3
        cv2.imwrite(os.path.join(image_dir,'Xin_Mask.png'),mask)

        # yanhai_begin
        h = self.load_size
        w = self.load_size
        # yanhai_end
        im_mask0 = mask > 0.0
        self.im_mask0 = im_mask0.transpose((2, 0, 1))
        im_lab = color.rgb2lab(im).transpose((2, 0, 1))
        # imRGB=im.copy()[:,:,::-1]   #将获取的BGRim转为RGBim
        imRGB=im.copy()   #将获取的BGRim转为RGBim
        cv2.imwrite(os.path.join(image_dir,'xin_IMRGB.png'), imRGB[:,:,::-1]) #imRGB是RGB格式,取反cv2写入刚好为RGB
        input_im = imRGB.transpose((2, 0, 1))    #获取的彩色RGB用户点
        # input_im = im.copy()
        self.im_ab0 = im_lab[1:3, :, :]
        net_im_ab0 = self.im_ab0.copy()

        # yanhai_begin
        # self.im_mask0=np.zeros((1,h, w), np.uint8)
        # self.im_mask0[0,2:40,3:50]=0
        # self.im_ab0=self.im_ab0*self.im_mask0
        imgAllGray = np.zeros((n, h, w, 3), np.uint8) #所有灰度图[256,256,3]
        imgAllGrayT = np.zeros((n, 3, h, w), np.uint8)#所有灰度图转[3,256,256]
        maskk = np.zeros((1, h, w), np.uint8)

        input_maskAllGray = np.zeros((n, 1, h, w), np.uint8) #所有灰度图对应的mask

        for i in range(n):
            image_file = self.img_list[i]
            #20190702晚改
            im_bgr = cv2.imread(image_file)
            im_bgr = cv2.resize(im_bgr, (self.load_size, self.load_size), interpolation=cv2.INTER_CUBIC)
            im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
            imgAllGray[i] = im_rgb


            # imgAllGray[i] = self.read_image(image_file)


            cv2.imwrite(os.path.join(image_dir, 'imgAllGray' + str(i) + '.jpg'), imgAllGray[i])

            imgAllGrayT[i] = imgAllGray[i].transpose(2, 0, 1)
            # cv2.imwrite(os.path.join(image_dir, 'imggLab[i]' + str(i) + '.png'), imgAllGrayT[i])

        CopyMaskOrignal = self.im_mask0
        ExpendMask = np.zeros((1,256,256),np.uint8)  #存储mask用户点对应的灰度图index
        hwPosition=[]
        seg_index = []  # 记录语义标签label--灰度图index
        mask_split256T=self.mask_split256.transpose(2, 0, 1)
        input_maskAllGrayII = np.zeros((n, 1, h, w), np.uint8)


        for k in range(self.semantic_Max):
            segDotMask = (mask_split256T[0, :, :] == k) * CopyMaskOrignal
            for x in range(h):
                for y in range(w):
                    if segDotMask[0][x][y] == 1:
                        hwPosition.append((x, y))
                        a = self.findEdge(x, y, segDotMask)
                        segDotMask[0, x:x + a, y:y + a] = 0
                        sandbox = []
                        for u in range(a):
                            for v in range(a):
                                ## 单独计算灰度
                                r = int(input_im[0][x + u][y + v])
                                g = int(input_im[1][x + u][y + v])
                                b = int(input_im[2][x + u][y + v])
                                m = r * 0.299 + g * 0.587 + b * 0.144


                                # m = self.im_lab[y+v,x+u,0]  #gui_draw.py->change_color()方法有调用
                                for i in range(n):
                                    # sandbox.append(imgAllGrayT[i, 0, x + u, y + v] * 0.299 + imgAllGrayT[
                                    #     i, 1, x + u, y + v] * 0.587 + imgAllGrayT[i, 2, x + u, y + v] * 0.144 - m)
                                    sandbox.append(abs(imgAllGrayT[i, 0, x + u, y + v]-m)) # 灰度图三维是相同的
                                # sandbox = imgAllGrayT[:, 0, x + u, y + v] * 0.299 + imgAllGrayT[:, 1, x + u,
                                #                                                     y + v] * 0.587 + imgAllGrayT[:, 2,
                                #                                                                      x + u,
                                #                                                                      y + v] * 0.144 - m
                                sandtarget = np.argsort(sandbox)  # 排序灰度值差，得到从小到大排序后的index
                                sandbox = []
                                wanted_index = sandtarget[0]  # 找到灰度值最接近的index
                                input_maskAllGrayII[wanted_index][0][x + u][y + v] = 1
                                if mask_orginal[x+u,y+v,0]!=0:   #20191101增加用户点分配灰度图的比重
                                    input_maskAllGrayII[wanted_index][0][x + u][y + v] = 5
                        ExpendMask[0, x:x + a, y:y + a] = wanted_index  # 赋值所有的index

            graysum = [0] * n

            for indexSum in range(n):  # 找到最多用户点所属的index
                graysum[indexSum] = np.sum(input_maskAllGrayII[indexSum][0, :, :])
            SumMaxIndex = np.argsort(graysum)[-1]
            for index in range(n):
                input_maskAllGrayII[index][0, :, :] = 0
            input_maskAllGray[SumMaxIndex][:, :, :] = ((~(mask_split256T[0, :, :] == k)) * input_maskAllGray[
                                                                                              SumMaxIndex][0, :, :] + (
                                                                  mask_split256T[0, :, :] == k) * CopyMaskOrignal)[0,:,:]
            seg_index.append((k,SumMaxIndex))

        # if len(hwPosition)!=0:
        #     print("hwPosition:::::::::::::::::::")
        #     print(hwPosition)
        #     for i in range(len(hwPosition)):
        #         hh = hwPosition[i][0]
        #         ww = hwPosition[i][1]
        #
        #         print("hhhhhhhhhh"+str(hh)+"wwwwwwwwww"+str(ww))
        #         seg_index.append((self.mask_split256[hh, ww, 0], ExpendMask[0, hh, ww]))
        #
        # seg_index = list(set(seg_index))
        # seg_index.sort()

        # ExpendMask = cv2.resize(ExpendMask[0,:,:], (512, 288), interpolation=cv2.INTER_NEAREST)
        # return blackXY
        # cv2.imwrite(os.path.join(image_dir, 'input_maskk0.png'), input_maskAllGray[0])
        # cv2.imwrite(os.path.join(image_dir, 'input_maskk1.png'), input_maskAllGray[1])
        # cv2.imwrite(os.path.join(image_dir, 'input_maskk2.png'), input_maskAllGray[2])
        allResult=np.zeros((n, 450, 800, 3), np.uint8) #20190510晚改

        for i in range(n):
            #20190704晚注释下面两行
            image_file = self.img_list[i]
            imgAllGray[i] = self.read_image(image_file)


            # im_bgrr = cv2.imread(image_file)
            # im_bgr = cv2.resize(im_bgrr, (self.load_size, self.load_size), interpolation=cv2.INTER_CUBIC)
            # im_rgb = cv2.cvtColor(im_bgr, cv2.COLOR_BGR2RGB)
            # imgAllGray[i] = im_rgb

            # im, mask = self.uiControl.get_input()  # 获取最原始的BGR用户点和mask
            # im1,haveLabel_color = self.colorVirtualPoint(self.virtualMask, self.mask_split256, im)
            #
            # # 获取条件概率的label:(R,G,B)
            # newIM = np.zeros((256, 256, 3), np.uint8)
            # needColor_label = self.hadLabel_label(haveLabel_color)
            # # 生成新的IM
            # if needColor_label:
            #     newIM = self.AddcolorVirtualPoint(needColor_label, self.virtualMask, self.mask_split256, im1[:,:,::-1])
            #     if newIM is not None:
            #         cv2.imwrite(os.path.join(image_dir,'tiaojiangailv_IM_RGB.png'), newIM[:, :, ::-1])
            #
            # im = im[:, :, ::-1] + newIM[:,:,::-1]
            #
            # # im = im[:, :, ::-1] + im1
            # im = im[:, :, ::-1]
            # mask = mask + self.virtualMask
            # im_mask0 = mask > 0.0
            # self.im_mask0 = im_mask0.transpose((2, 0, 1))
            # im_lab = color.rgb2lab(im).transpose((2, 0, 1))
            # # imRGB=im.copy()[:,:,::-1]   #将获取的BGRim转为RGBim
            # imRGB = im.copy()  # 将获取的BGRim转为RGBim
            # cv2.imwrite(os.path.join(image_dir,'xin_IM1_RGB.png'), imRGB[:,:,::-1])
            # input_im = imRGB.transpose((2, 0, 1))  # 获取的彩色RGB用户点
            # # input_im = im.copy()
            # self.im_ab0 = im_lab[1:3, :, :]

            self.im_ab0 = net_im_ab0    #获取最后使用的mask_ab




            # cv2.imwrite(os.path.join(image_dir, 'imgg' + str(i) + '.jpg'), imgAllGray[i])

            blackXY = []

            # bianliImage

            # yanhai_end
            self.im_mask0 = input_maskAllGray[i]
            self.im_ab0_Procrssing = self.im_ab0.copy()
            self.im_ab0_Procrssing[0, :, :] = self.im_ab0_Procrssing[0, :, :] * self.im_mask0
            self.im_ab0_Procrssing[1, :, :] = self.im_ab0_Procrssing[1, :, :] * self.im_mask0

            # dilate = cv2.dilate(mask, None, iterations=1)
            # cv2.imshow('dilate', dilate)
            # cv2.waitKey(0)
            # erode = cv2.erode(dilate, None, iterations=1)
            # cv2.bitwise_and(self.im_mask0, self.im_mask0, mask=erode)

            #20190703
            self.maskmask = self.im_mask0
            # self.dist_model.set_image(im_rgb)
            self.dist_model.net_forward(self.im_ab0_Procrssing, self.maskmask)

            self.model.net_forward(self.im_ab0_Procrssing, self.im_mask0)
            ab = self.model.output_ab.transpose((1, 2, 0))

            ab_win = cv2.resize(ab, (self.win_w, self.win_h), interpolation=cv2.INTER_CUBIC)
            pred_lab = np.concatenate((self.l_win[..., np.newaxis], ab_win), axis=2)
            pred_rgb = (np.clip(color.lab2rgb(pred_lab), 0, 1) * 255).astype('uint8')


            self.result = pred_rgb
            # 存储不同灰度图着色结果


            self.emit(SIGNAL('update_result'), self.result)
            self.update()

            # image_file = self.img_list[self.i]  # 20190119night
            # if image_file is not None:
            #     self.read_image(image_file)
            #     self.i += 1

            result_bgr = cv2.cvtColor(self.result, cv2.COLOR_RGB2BGR)
            mask = self.im_mask0.transpose((1, 2, 0)).astype(np.uint8) * 255




            #20190707分别将灰度图与ab合成
            # im_bgrr = cv2.resize(im_bgrr, (512, 288), interpolation=cv2.INTER_CUBIC)
            # im_bgrr = cv2.cvtColor(im_bgrr, cv2.COLOR_BGR2RGB)
            # im_bgrr = cv2.cvtColor(im_bgrr, cv2.COLOR_RGB2LAB)
            #
            # result_bgr = cv2.cvtColor(result_bgr, cv2.COLOR_BGR2RGB)
            # result_bgr = cv2.cvtColor(result_bgr, cv2.COLOR_RGB2LAB)
            # result_bgr[:,:,0]= im_bgrr[:,:,0]
            # result_bgr = cv2.cvtColor(result_bgr, cv2.COLOR_LAB2BGR)

            # allResult[i, :, :, :] = result_bgr  # 存储所有着色结果
            allResult[i,:,:,:]=self.model.get_img_fullres()[:, :, ::-1]

            cv2.imwrite(os.path.join(image_dir, 'input_mask' + str(i) + '.jpg'), mask)
            cv2.imwrite(os.path.join(image_dir, 'ours' + str(i) + '.jpg'), result_bgr)
            cv2.imwrite(os.path.join(image_dir, 'input_ab' + str(i) + '.jpg'), self.model.get_sup_img()[:, :, ::-1])
            cv2.imwrite(os.path.join(image_dir, 'input_fullers' + str(i) + '.jpg'), self.model.get_input_img_fullres()[:, :, ::-1])
            maskkkk = maskk.transpose((1, 2, 0)).astype(np.uint8) * 255
            cv2.imwrite(os.path.join(image_dir, 'maskk.jpg'), maskkkk)

            #Fullre_image
            cv2.imwrite(os.path.join(image_dir, 'ours_fullres'+ str(i) + '.jpg'), self.model.get_img_fullres()[:, :, ::-1])

        finalResult=np.zeros((450, 800, 3), np.uint8)  #20190510晚改

        print("seg_index-----------------------------")
        print(seg_index)
        print("seg_index-----------------------------")
        if seg_index!=[]:
            for i in range(self.semantic_Max):
                for j in range(i, self.semantic_Max):
                    segTag = seg_index[j][0]
                    grayResultIndex = seg_index[j][1]
                    if segTag == i:
                        print("segTag+++++++++++++++++++++++++"+str(i)+"ij"+str(j)+str(segTag))
                        print("grayResultIndex+++++++++++++++++++++++++" +str(i)+"ij"+str(j)+ str(grayResultIndex))
                        finalResult = (~(self.semantic_fig[:, :, :] == j)) *finalResult + (self.semantic_fig[:, :, :] == j) * allResult[grayResultIndex]
                        break

        #贴黑白图
        white_index = []
        black_index = []
        numPoint = len(hwPosition)
        for p in range(numPoint):
            rr = imRGB[hwPosition[p][0],hwPosition[p][1],0]
            gg = imRGB[hwPosition[p][0],hwPosition[p][1],1]
            bb = imRGB[hwPosition[p][0],hwPosition[p][1],2]
            if (abs(rr-gg)+abs(rr-bb)+abs(gg-bb))<5 :
                if rr<50:
                    black_index.append(mask_split256T[0,hwPosition[p][0],hwPosition[p][1]])
                elif rr>220:
                    white_index.append(mask_split256T[0,hwPosition[p][0],hwPosition[p][1]])
        white_index = list(set(white_index))
        black_index = list(set(black_index))
        if len(white_index)>0:
            for w in range(len(white_index)):
                finalResult = (~(self.semantic_fig[:, :, :] == white_index[w])) * finalResult + (
                        self.semantic_fig[:, :, :] == white_index[w]) * self.white
        if len(black_index)>0:
            for b in range(len(black_index)):
                finalResult = (~(self.semantic_fig[:, :, :] == black_index[b])) * finalResult + (
                            self.semantic_fig[:, :, :] == black_index[b]) * self.black


        #增加装饰家具
        finalResult = (~(self.semantic_fig[:, :, :] == 18)) * finalResult + (self.semantic_fig[:, :, :] == 18) * self.zhuangshi
        # 处理家具间白边
        finalResult = self.processingBorder(finalResult,self.semantic_fig)


        finalResult = cv2.cvtColor(finalResult, cv2.COLOR_RGB2BGR)
        finalResult_win = cv2.resize(finalResult, (self.win_w, self.win_h), interpolation=cv2.INTER_CUBIC)
        self.emit(SIGNAL('update_result'), finalResult_win)
        self.update()

        # import datetime
        end = time.clock()
        print("Time Used=========", end - start)
        file = open("time.txt", "a")
        file.write(str(datetime.datetime.now()) + '\n')
        file.write(self.image_dir + '====>>' + "runtime==" + str(end - start) + "s" + '\n')
        file.close()


        # finalResult = finalResult.transpose((1, 2, 0)).astype(np.uint8) * 255
        finalResult = cv2.cvtColor(finalResult, cv2.COLOR_BGR2RGB)
        cv2.imwrite(os.path.join(image_dir, 'finalResult.jpg'), finalResult)

        # print("white_index======="+str(white_index))
        # print("black_index======="+str(black_index))

        # max(self.semantic_fig[:,:,0]) should be rewrite
        # for i in range(self.semantic_Max):
        #     sections=(self.semantic_fig[:,:,0]==i)*ExpendMask
        #     maskIndex=np.reshape(sections, [1, -1]).tolist()[0]
        #     a = np.bincount(maskIndex)
        #     print("^^^^^^^^^^^^^^^^^^^^^^" +str(i)+"i++"+ str(a))
        #     final=self.get_secondmost(a)
        #     print("^^^^^^^^^^^^^^^^^^^^^^"+str(final))

        #显示白模
        baomo = self.image_dir + '/baimo.jpg'
        self.read_image(baomo)

        return self.result
        # yanhai_onpress_end

    # yanhai_end
    # def get_secondmost(self,array1):
    #     max = 0
    #     sec = 0
    #     len = array1.size
    #     for i in range(len):
    #         if (array1[max] < array1[i]):
    #             max = i
    #     array1[max] = 0
    #     for i in range(len):
    #         if ((array1[sec] < array1[i])):
    #             sec = i
    #     return sec


    # yanhai_begin
    def okImageE(self, result):
        self.result = result
        result_bgr = cv2.cvtColor(self.result, cv2.COLOR_RGB2BGR)
        mask = self.im_mask0.transpose((1, 2, 0)).astype(np.uint8) * 255
        image_dir = 'test_imgs/huidutu/okImageE'
        cv2.imwrite(os.path.join(image_dir, 'input_mask.png'), mask)
        cv2.imwrite(os.path.join(image_dir, 'ours.png'), result_bgr)
        cv2.imwrite(os.path.join(image_dir, 'input_ab.png'), self.model.get_sup_img()[:, :, ::-1])
        pass

    def findEdge(self, x, y, mask):
        if x==255 or y==255:
            return  1

        rtn = 1

        while mask[0][x][y + rtn] == 1 and mask[0][x + rtn][y] == 1 and (x+rtn)<=255 and (y+rtn)<=255:
            rtn += 1
            if x + rtn > 255 or y + rtn > 255:
                return rtn
                # a,b=findEdge(x + 1, y + 1,mask)
        return rtn
    #根据label建议家具颜色
    def ExpertsSuggest(self, h, w):
        #h,w是256x256的坐标,左上角为(0,0)
        label = self.mask_split256[h, w, 0]
        #colors_rgb_withcurr应该按照RGB顺序赋值
        colors_rgb_withcurr = [[94, 235, 0], [233, 122, 0], [183, 48, 0],
                               [49, 117, 227], [98, 242, 227], [224, 118, 227],
                               [204, 229, 138], [236, 244, 45], [117, 75, 45], [234, 243, 0]]
        if label == 1:
            colors_rgb_withcurr = [[213, 197, 171], [222, 205, 179], [220, 216, 207], [173, 173, 171], [255, 255, 255],
                                   [177, 145, 120], [194, 200, 198], [237, 217, 184], [163, 189, 216], [97, 48, 18]]
            return colors_rgb_withcurr
        elif label == 2:
            colors_rgb_withcurr = [[114, 71, 36], [40, 40, 40], [71, 71, 59], [255, 255, 255], [213, 197, 171],
                                   [173, 173, 171], [102, 82, 57], [114, 71, 36], [194, 200, 198], [74, 54, 29]]
            return colors_rgb_withcurr
        elif label == 3:
            colors_rgb_withcurr = [[114, 71, 36], [213, 197, 171], [74, 54, 29], [194, 200, 198], [145, 118, 89],
                                   [102, 82, 57], [220, 216, 207], [173, 173, 171], [159, 137, 114], [142, 144, 159]]
            return colors_rgb_withcurr
        elif label == 4:
            colors_rgb_withcurr = [[114, 71, 36], [71, 71, 59], [184, 184, 156], [194, 200, 198], [213, 197, 171],
                                   [173, 173, 171], [40, 40, 40], [178, 209, 177], [120, 115, 121], [140, 134, 110]]
            return colors_rgb_withcurr
        elif label == 5:
            colors_rgb_withcurr = [[114, 71, 36], [40, 40, 40], [255, 255, 255], [71, 71, 59], [220, 216, 207],
                                   [173, 173, 171], [102, 82, 57], [114, 71, 36], [145, 118, 89], [152, 123, 83]]
            return colors_rgb_withcurr
        elif label == 6:
            colors_rgb_withcurr = [[213, 197, 171], [255, 255, 255], [222, 205, 179], [237, 217, 184], [220, 216, 207],
                                   [173, 173, 171], [194, 200, 198], [114, 71, 36], [184, 184, 156], [255, 255, 235]]
            return colors_rgb_withcurr
        elif label == 7:
            colors_rgb_withcurr = [[114, 71, 36], [145, 118, 89], [220, 216, 207], [140, 134, 110], [213, 197, 171],
                                   [194, 177, 125], [194, 200, 198], [71, 71, 59], [255, 255, 255], [131, 169, 154]]
            return colors_rgb_withcurr
        elif label == 8:
            colors_rgb_withcurr = [[114, 71, 36], [173, 173, 171], [194, 200, 198], [255, 255, 255], [183, 209, 122],
                                   [145, 118, 89], [194, 177, 125], [151, 62, 44], [220, 216, 207], [102, 82, 57]]
            return colors_rgb_withcurr
        elif label == 9:
            colors_rgb_withcurr = [[114, 71, 36], [213, 197, 171], [255, 255, 255], [152, 123, 83], [173, 173, 171],
                                   [40, 40, 40], [145, 118, 89], [220, 216, 207], [159, 137, 114], [102, 82, 57]]
            return colors_rgb_withcurr
        elif label == 10:
            colors_rgb_withcurr = [[114, 71, 36], [255, 255, 255], [152, 123, 83], [40, 40, 40], [173, 173, 171],
                                   [120, 115, 121], [102, 82, 57], [213, 197, 171], [145, 118, 89], [147, 125, 49]]
            return colors_rgb_withcurr
        elif label == 11:
            colors_rgb_withcurr = [[114, 71, 36], [40, 40, 40], [152, 123, 83], [102, 82, 57], [255, 255, 255],
                                   [159, 137, 114], [74, 54, 29], [120, 115, 121], [173, 173, 171], [68, 18, 27]]
            return colors_rgb_withcurr
        elif label == 12:
            colors_rgb_withcurr = [[114, 71, 36], [213, 197, 171], [255, 255, 255], [194, 177, 125], [40, 40, 40],
                                   [145, 118, 89], [152, 123, 83], [222, 205, 179], [159, 137, 114], [177, 145, 120]]
            return colors_rgb_withcurr
        elif label == 13:
            colors_rgb_withcurr = [[114, 71, 36], [74, 54, 29], [102, 82, 57], [71, 71, 59], [40, 40, 40],
                                   [194, 200, 198], [186, 135, 88], [115, 26, 28], [253, 204, 137], [152, 123, 83]]
            return colors_rgb_withcurr
        elif label == 14:
            colors_rgb_withcurr = [[114, 71, 36], [173, 173, 171], [213, 197, 171], [184, 184, 156], [140, 134, 110],
                                   [119, 137, 115], [222, 205, 179], [252, 204, 184], [219, 162, 95], [151, 62, 44]]
            return colors_rgb_withcurr
        elif label == 15:
            colors_rgb_withcurr = [[114, 71, 36], [255, 255, 255], [255, 255, 235], [184, 184, 156], [220, 216, 207],
                                   [202, 164, 185], [194, 200, 198], [255, 255, 205], [173, 173, 171], [194, 177, 125]]
            return colors_rgb_withcurr
        elif label == 16:
            colors_rgb_withcurr = [[213, 197, 171], [114, 71, 36], [255, 255, 255], [222, 205, 179], [237, 217, 184],
                                   [173, 173, 171], [220, 216, 207], [255, 255, 235], [254, 218, 220], [167, 202, 222]]
            return colors_rgb_withcurr
        elif label == 17:
            colors_rgb_withcurr = [[114, 71, 36], [152, 123, 83], [173, 173, 171], [255, 255, 255], [102, 82, 57],
                                   [145, 118, 89], [71, 71, 59], [114, 71, 36], [220, 216, 207], [177, 145, 120]]
            return colors_rgb_withcurr
        else:
            colors_rgb_withcurr = [[0, 0, 0], [0, 0, 0], [0, 0, 0],
                                   [0, 0, 0], [0, 0, 0], [0, 0, 0],
                                   [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
            return
    #对虚拟点赋予颜色
    def colorVirtualPoint(self,Allmask,mask_split256,rgbMask):#虚拟mask.语义mask,UI-im
        newmask = np.zeros([256, 256, 3], np.uint8)
        ab = np.zeros([256, 256, 3], np.uint8)
        haveLabel_color = {}
        # for i in range(1, 18):
        #     haveLabel_color[i] = 0
        for label in range(1, 18):
            ab[:, :, 0] = (mask_split256[:, :, 0] == label) * rgbMask[:, :, 0]
            ab[:, :, 1] = (mask_split256[:, :, 0] == label) * rgbMask[:, :, 1]
            ab[:, :, 2] = (mask_split256[:, :, 0] == label) * rgbMask[:, :, 2]
            k1 = 0
            k2 = 0
            k3 = 0
            for i in range(256):
                t = False
                for j in range(256):
                    if ab[i][j][0] != 0 or ab[i][j][1] != 0 or ab[i][j][2] != 0:
                        k1 = ab[i][j][0]
                        k2 = ab[i][j][1]
                        k3 = ab[i][j][2]
                        haveLabel_color[label] = (k1,k2,k3)
                        t = True
                        break
                if t:
                    break
            newmask[:, :, 0] = ((mask_split256[:, :, 0] == label) * Allmask[:, :, 0]) / 255 * k3 + newmask[:, :, 0]
            newmask[:, :, 1] = ((mask_split256[:, :, 0] == label) * Allmask[:, :, 0]) / 255 * k2 + newmask[:, :, 1]
            newmask[:, :, 2] = ((mask_split256[:, :, 0] == label) * Allmask[:, :, 0]) / 255 * k1 + newmask[:, :, 2]
        return newmask,haveLabel_color

    def hadLabel_label(self,rgb):
        print("===========hadLabel_label====rgb0=========")
        print(rgb)
        if not rgb:
            return
        print("===========hadLabel_label====rgb1=========")
        print(rgb)
        for key,value in rgb.items():
            rgb[key] = findclosestColor_label(value)  #jiajulabel:color_label
        print("===========hadLabel_label====rgb2=========")
        print(rgb)
        needColor_label = findNeedColor(rgb)
        print("===========hadLabel_label====needColor_label=========")
        print(needColor_label)
        if not needColor_label:
            return
        file = open('needColor_label.txt','w')
        file.write(str(needColor_label))
        file.close()
        return needColor_label      #{2: (21, 0.08163265306122448)}

    def AddcolorVirtualPoint(self,needcolor_label,Allmask,mask_split256,rgbMask):
        newIM=rgbMask
        label = np.unique(mask_split256[:,:,0])
        if not needcolor_label:
            return None
        for key,value in needcolor_label.items():
            if key in label:
                newIM[:, :, 0] = ((mask_split256[:, :, 0] == key) * Allmask[:, :, 0]) / 255 * int(value[0]) + newIM[:, :, 0]
                newIM[:, :, 1] = ((mask_split256[:, :, 0] == key) * Allmask[:, :, 0]) / 255 * int(value[1]) + newIM[:, :, 1]
                newIM[:, :, 2] = ((mask_split256[:, :, 0] == key) * Allmask[:, :, 0]) / 255 * int(value[2]) + newIM[:, :, 2]
        return newIM            #RGB格式

    def processingBorder(self,result,mask):
        # n,m,l = result.shape()
        for i in range(1,287):
            for j in range(1,511):
                if mask[i,j,0]==0:
                    result[i][j][0] = (np.sum(result[i-1:i+2,j-1:j+2,0])-result[i,j,0])/8
                    result[i][j][1] = (np.sum(result[i-1:i+2,j-1:j+2,1])-result[i,j,1])/8
                    result[i][j][2] = (np.sum(result[i-1:i+2,j-1:j+2,2])-result[i,j,2])/8
        # result1 = result.copy()
        # finalResult = (~(self.semantic_fig[:, :, :] == 0)) * result + (self.semantic_fig[:, :, :] == 0) * result1

        return result



