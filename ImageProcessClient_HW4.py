# -*- coding: utf-8 -*-
from PyQt4.QtGui import *
from PyQt4.QtCore import *
import numpy as np
#from random import *
import sys
import math


QTextCodec.setCodecForTr(QTextCodec.codecForName("utf8"))

class HisDialog(QDialog):
	def __init__(self, hist_arr):
		super(HisDialog, self).__init__()
		self.setFixedSize(600,500)
		self.setWindowTitle("Image Histogram")
		self.drawArr = hist_arr
	
	def paintEvent(self, e):
		qp = QPainter()
		qp.begin(self)
		self.drawLine(qp)
		qp.end()
		
	def drawLine(self, qp):
		color = QColor(0,0,0)
		qp.setPen(color)
		qp.drawLine(10, 10, 10, 410)
		qp.drawLine(10, 410, 260 * 2, 410)
		maxn = 0
		for i in range(1, 256):
			if maxn < self.drawArr[i - 1]:
				maxn = self.drawArr[i - 1]
		scaleD = 400 / float(maxn)
		for i in range(1, 256):
			qp.drawLine(10 + i * 2 ,410, 10 + i * 2,  410 - math.floor(scaleD * self.drawArr[i - 1]+0.5))
		
		
class MainWidget(QMainWindow):
	def __init__(self, parent = None, ):
		super(MainWidget, self).__init__(parent)
		
		self.setWindowTitle("Image Process")
		self.imgFile = ""
		self.imgTemp = QImage()
		self.level = 256
		self.width = 384
		self.height = 256
		
		window = QWidget()
		self.setCentralWidget(window)
		
		self.layout = QGridLayout()
		
		#scalePushButton = QPushButton(self.tr("缩放图片"))
		#levelPushButton = QPushButton(self.tr("降低灰度"))

		self.imgLabel = QLabel()
		#self.histLabel = HisLabel()
		
		self.layout.addWidget(self.imgLabel, 0, 0)
		#layout.addWidget(self.histLabel, 0, 1)
		#layout.addWidget(scalePushButton, 1, 0)
		#layout.addWidget(levelPushButton, 2, 0)
		
		window.setLayout(self.layout)
		self.createMenu()
		#self.connect(scalePushButton, SIGNAL("clicked()"), self.slotCustom)
		#self.connect(levelPushButton, SIGNAL("clicked()"), self.slotLevel)

	def createMenu(self):
		layoutMenu = self.menuBar().addMenu(self.tr("文件"))
		arrange = QAction(self.tr("读取图片"), self)
		self.connect(arrange, SIGNAL("triggered()"), self.openFile)
		layoutMenu.addAction(arrange)

		tile = QAction(self.tr("保存图片"), self)
		self.connect(tile, SIGNAL("triggered()"), self.saveFile)
		layoutMenu.addAction(tile)
		
		layoutMenu2 = self.menuBar().addMenu(self.tr("图片处理"))
		scalePic = QAction(self.tr("图片缩放"), self)
		self.connect(scalePic, SIGNAL("triggered()"), self.slotCustom)
		layoutMenu2.addAction(scalePic)
		
		quantizePic = QAction(self.tr("灰度变换"), self)
		self.connect(quantizePic, SIGNAL("triggered()"), self.slotLevel)
		layoutMenu2.addAction(quantizePic)
		layoutMenu2.addSeparator()
		
		addNoisePic1 = QAction(self.tr("添加高斯噪声"), self)
		self.connect(addNoisePic1, SIGNAL("triggered()"), self.slotAddGaussian)
		layoutMenu2.addAction(addNoisePic1)
		
		addNoisePic2 = QAction(self.tr("添加椒盐噪声"), self)
		self.connect(addNoisePic2, SIGNAL("triggered()"), self.slotAddImpluse)
		layoutMenu2.addAction(addNoisePic2)
		layoutMenu2.addSeparator()
		
		drawHisAct = QAction(self.tr("绘制直方图"), self)
		self.connect(drawHisAct, SIGNAL("triggered()"), self.plot_hist)
		layoutMenu2.addAction(drawHisAct)
		
		HE_trans = QAction(self.tr("直方图均衡化"), self)
		self.connect(HE_trans, SIGNAL("triggered()"), self.equanlize_hist)
		layoutMenu2.addAction(HE_trans)
		
		patchAct = QAction(self.tr("提取Patch"), self)
		self.connect(patchAct, SIGNAL("triggered()"), self.slotPatch)
		layoutMenu2.addAction(patchAct)

		maxFilterAct = QAction(self.tr("MAX滤波"), self)
		self.connect(maxFilterAct, SIGNAL("triggered()"), self.slotMaxFilter)
		layoutMenu2.addAction(maxFilterAct)
		
		minFilterAct = QAction(self.tr("MIN滤波"), self)
		self.connect(minFilterAct, SIGNAL("triggered()"), self.slotMinFilter)
		layoutMenu2.addAction(minFilterAct)	

		medianFilterAct = QAction(self.tr("MEDIAN滤波"), self)
		self.connect(medianFilterAct, SIGNAL("triggered()"), self.slotMedianFilter)
		layoutMenu2.addAction(medianFilterAct)
		
		GmeanFilterAct = QAction(self.tr("几何平均滤波"), self)
		self.connect(GmeanFilterAct, SIGNAL("triggered()"), self.slotGMean)
		layoutMenu2.addAction(GmeanFilterAct)
		
		aveFilterAct = QAction(self.tr("算术平均滤波"), self)
		self.connect(aveFilterAct, SIGNAL("triggered()"), self.slotAveFilter)
		layoutMenu2.addAction(aveFilterAct)

		harFilterAct = QAction(self.tr("调和平均滤波"), self)
		self.connect(harFilterAct, SIGNAL("triggered()"), self.slotHarmFilter)
		layoutMenu2.addAction(harFilterAct)

		conHarFilterAct = QAction(self.tr("反调和平均滤波"), self)
		self.connect(conHarFilterAct, SIGNAL("triggered()"), self.slotConHarmFilter)
		layoutMenu2.addAction(conHarFilterAct)

		lapFilterAct = QAction(self.tr("Laplacian滤波"), self)
		self.connect(lapFilterAct, SIGNAL("triggered()"), self.slotLapFilter)
		layoutMenu2.addAction(lapFilterAct)

		sobFilterAct = QAction(self.tr("Sobel滤波"), self)
		self.connect(sobFilterAct, SIGNAL("triggered()"), self.slotSobFilter)
		layoutMenu2.addAction(sobFilterAct)
		
		sob2FilterAct = QAction(self.tr("Sobel滤波2"), self)
		self.connect(sob2FilterAct, SIGNAL("triggered()"), self.slotSob2Filter)
		layoutMenu2.addAction(sob2FilterAct)
		layoutMenu2.addSeparator()
		
		RGBEqualAct1 = QAction(self.tr("RGB直方图均衡TEST1"), self)
		self.connect(RGBEqualAct1, SIGNAL("triggered()"), self.slotRGBEqual1)
		layoutMenu2.addAction(RGBEqualAct1)
		
		RGBEqualAct2 = QAction(self.tr("RGB直方图均衡TEST2"), self)
		self.connect(RGBEqualAct2, SIGNAL("triggered()"), self.slotRGBEqual2)
		layoutMenu2.addAction(RGBEqualAct2)
		
		
	# scale函数采用双线性插值的方法来对图像进行放缩
	def scale(self):
		# 加载图像
		img = QImage()
		img.load(self.imgFile)
		
		# 获取宽度和高度
		i_width = img.width()
		i_height = img.height()
		
		# 获取放缩比例
		scale_x = i_width / float(self.width)
		scale_y = i_height / float(self.height)
		# s变量用于存放image的data
		s = ''
		
		# 为新的image的每个pixel做插值, 因此使用新图的宽和高作为循环的range
		for i in range(self.height):
			fx = float((i + 0.5) * scale_y - 0.5)
			sx = math.floor(fx)
			fx -= sx
			# 对sx和fx做一个范围的限制,主要防止下标的越界
			sx = max(min(i_height - 2, sx),0)
			fx = max(0, fx)
			# 对浮点数做放大,防止精度损失
			factor_x0 = int((1.0 - fx) * 2048)
			factor_x1 = 2048 - factor_x0
			for j in range(self.width):
				fy = float((j + 0.5)*scale_x - 0.5)
				sy = math.floor(fy)
				fy -= sy
				sy = max(min(i_width - 2, sy), 0)
				fy = max(0, fy)
				factor_y0 = int((1.0 - fy) * 2048)
				factor_y1 = 2048 - factor_y0
				# 获取2X2各个点的灰度值,并且做一个双线性插值 
				s += chr((factor_x0 * factor_y0 * QColor(img.pixel(sy, sx)).red() +
					factor_x0 * factor_y1 * QColor(img.pixel(sy + 1,sx)).red() +
					factor_x1 * factor_y0 * QColor(img.pixel(sy, sx + 1)).red() +
					factor_x1 * factor_y1 * QColor(img.pixel(sy + 1, sx + 1)).red()) >> 22)
		# 显示图片
		img_solve = QImage(s, self.width, self.height, 3)
		self.imgTemp = img_solve.copy()
		self.imgLabel.setPixmap(QPixmap.fromImage(img_solve))
		
	# 对输入图像进行降灰度处理
	def quantize(self):
		# 加载图像
		img = QImage()
		img.load(self.imgFile)
		# 获取宽度和高度
		i_width = img.width()
		i_height = img.height()

		# 获取相应灰度级别的阈值,并存放到node数组里面
		num = 254 % (self.level - 1)
		total = 254 / (self.level - 1)
		node = []
		tail = 0
		for i in range(self.level - 1):
			node.append(tail)
			tail += total
			if num > 0:
				tail += 1
				num -= 1
		node.append(255)
		
		# 遍历一遍每个像素, 对每个像素点做一个量化
		s = ''
		for i in range(i_height):
			for j in range(i_width):
				for k in range(self.level):
					if QColor(img.pixel(j, i)).red() < node[k]:
						if QColor(img.pixel(j, i)).red() >= (node[k] + node[k - 1])/2:
							k += 1
							break
						else:
							break
					elif QColor(img.pixel(j, i)).red() == node[k]:
						k += 1
						break
				s += chr(node[k - 1])
		# 显示图片
		img_solve = QImage(s, i_width, i_height, 3)
		self.imgTemp = img_solve.copy()
		self.imgLabel.setPixmap(QPixmap.fromImage(img_solve))

	def plot_hist(self):
		# 加载图像
		#img = QImage()
		#img.load(self.imgFile)
		# 获取宽度和高度
		i_width = self.imgTemp.width()
		i_height = self.imgTemp.height()

		hist_arr = [0 for i in range(256)]
		s = self.imgTemp.bits().asstring(self.imgTemp.byteCount())
		x = 0
		for i in range(i_height):
			for j in range(i_width):
				hist_arr[ord(s[x])] += 1
				x += 1
		
		dialog = HisDialog(hist_arr)
		dialog.exec_()
	
	def plot_hist(self, img):
		i_width = img.width()
		i_height = img.height()
		hist_arr = np.zeros(256)
		imgArr = np.array([ord(ch) for ch in img.bits().asstring(img.byteCount())])
		# 获取每个像素值的频数
		for i in range(i_height):
			for j in range(i_width):
				hist_arr[imgArr[i * i_width + j]] += 1
		return hist_arr
		
	def equanlize_hist(self):
		img = QImage()
		img.load(self.imgFile)
		
		i_width = img.width()
		i_height = img.height()
		
		totalPixels = i_width * i_height
		
		hist_arr = [0 for i in range(256)]
		for i in range(i_height):
			for j in range(i_width):
				for k in range(QColor(img.pixel(j, i)).red(), 256):
					hist_arr[k] += 1
		
		s = ""
		for i in range(i_height):
			for j in range(i_width):
				newPixel = math.floor(255 * (hist_arr[QColor(img.pixel(j, i)).red()]/float(totalPixels)) + 0.5)
				s += chr(int(newPixel))
		
		img_solve = QImage(s, i_width, i_height, 3)
		self.imgTemp = img_solve.copy()
		self.imgLabel.setPixmap(QPixmap.fromImage(img_solve))
	
	def equanlize_hist(self, img):		
		i_width = img.width()
		i_height = img.height()
		
		totalPixels = i_width * i_height
		# get the pure pic array
		imgArr = np.array([ord(ch) for ch in img.bits().asstring(img.byteCount())])
		hist_arr = [0 for i in range(256)]
		for i in range(i_height):
			for j in range(i_width):
				value = int(imgArr[j + i * i_width])
				for k in range(value, 256):
					hist_arr[k] += 1
		#s = ''
		s = np.array([])
		for i in range(i_height):
			for j in range(i_width):
				value = int(imgArr[j + i * i_width])
				newPixel = math.floor(255 * (hist_arr[value]/float(totalPixels)) + 0.5)
				s = np.append(s, int(newPixel))
				#s += chr(int(newPixel))
		return s
		#img_solve = QImage(s, i_width, i_height, 3)
		#self.imgTemp = img_solve.copy()
		#self.imgLabel.setPixmap(QPixmap.fromImage(img_solve))
		
	def match_hist(self, img, histArr):
		i_height = img.height()
		i_width = img.width()
		destArr = np.zeros(i_height * i_width)
		imgArr = np.array([ord(ch) for ch in img.bits().asstring(img.byteCount())])
		# 获得频率
		pArr = self.plot_hist(img) / (i_width * i_height)
		# 累计
		for i in range(len(pArr) - 1):
			pArr[i + 1] += pArr[i]
			
		for i in range(i_height):
			for j in range(i_width):
				# 获取该点像素值
				value = int(imgArr[j + i * i_width])
				# 单通道该值频率imgArr[value] 而平均histArr[value]
				for idx in range(len(histArr)):
					if histArr[idx] == pArr[value]:
						destArr[j + i * i_width] = idx
						break
					elif histArr[idx] > pArr[value]:
						if idx == 0:
							destArr[j + i * i_width] = 0
						elif (pArr[value] - histArr[idx - 1]) >= (histArr[idx] - pArr[value]):
							destArr[j + i * i_width] = idx
						else: destArr[j + i * i_width] = idx - 1
						break
		return destArr

	def view_as_window(self, img, patch_size):
		(patchWidth, patchHeight) = patch_size
		
		i_width = img.width()
		i_height = img.height()
		
		imgArr = np.array([ord(ch) for ch in img.bits().asstring(img.byteCount())]).reshape(i_height, img.bytesPerLine())

		patch_arr = []
		for i in range(i_height - patchHeight + 1):
			patch_tmp = []
			for j in range(i_width - patchWidth + 1):
				patch_tmp.append(imgArr[i:i+patchWidth,j:j+patchHeight])
			patch_arr.append(patch_tmp)
		#for i in range(8):
		#	ran_height = int(math.floor(random() * (i_height - patchHeight) + 0.5))
		#	ran_width = int(math.floor(random() * (i_width - patchWidth) + 0.5))
		#	print str(i) + ":"
		#	print patch_arr[ran_height][ran_width]
		
		return patch_arr

		
	def filter2d(self, input_img, filter, size):
		# fill the edge with 0
		sizeBi = size / 2
		#print sizeBi
		width = input_img.width() + sizeBi * 2 # --> 386 0-->385 [0 385] 1 - 384
		height = input_img.height() + sizeBi * 2
		newImg = QImage(width, height, QImage.Format_Indexed8)
		inputList = [ord(ch) for ch in input_img.bits().asstring(input_img.byteCount())]  # sip.voidptr

		new_Ptr = newImg.bits()    # sip.voidptr
		new_Ptr.setsize((width + 2) * height)

		for i in range(height):
			for j in range(width):
				if (j < sizeBi) or (j >= width - sizeBi):
					new_Ptr.__setitem__(newImg.bytesPerLine() * i + j, chr(0))
				elif (i < sizeBi) or (i >= height - sizeBi):
					new_Ptr.__setitem__(newImg.bytesPerLine() * i + j, chr(0))
				else:
					#print input_img.width() * (i - sizeBi) + j - sizeBi
					new_Ptr.__setitem__(newImg.bytesPerLine() * i + j, chr(inputList[input_img.width() * (i - sizeBi) + j - sizeBi]))
		# return newImg
		# dot product
		destImg = QImage(input_img.width(), input_img.height(), QImage.Format_Indexed8)
		destPtr = destImg.bits()
		destPtr.setsize((input_img.width() + 2) * input_img.height())
		patch_arr = self.view_as_window(newImg, (size, size))

		index = 0
		count = size * size
		for i in range(destImg.height()):
			for j in range(destImg.width()):
				arr = patch_arr[i][j]
				sum = 0
				for k in range(size):
					for l in range(size):
						sum += arr[k][l] * int(filter[k][l])
				if sum / count > 255:
					destPtr.__setitem__(destImg.bytesPerLine() * i + j, chr(255))
				elif sum / count < 0:
					destPtr.__setitem__(destImg.bytesPerLine() * i + j, chr(0))
				else:
					destPtr.__setitem__(destImg.bytesPerLine() * i + j, chr(int(sum / count)))
					
		return destImg
	# NEW HW4
	def conHarmonic(self,  input_img, Q, size):
		# fill the edge with 0
		sizeBi = size / 2
		#print sizeBi
		width = input_img.width() + sizeBi * 2 # --> 386 0-->385 [0 385] 1 - 384
		height = input_img.height() + sizeBi * 2
		newImg = QImage(width, height, QImage.Format_Indexed8)
		print input_img.width(), input_img.height(), input_img.bytesPerLine()
		inputList = [ord(ch) for ch in input_img.bits().asstring(input_img.byteCount())]  # sip.voidptr

		new_Ptr = newImg.bits()    # sip.voidptr
		new_Ptr.setsize(newImg.bytesPerLine() * height)
		print width,height
		for i in range(height):
			for j in range(width):
				if (j < sizeBi) or (j >= width - sizeBi):
					new_Ptr.__setitem__(newImg.bytesPerLine() * i + j, chr(0))
				elif (i < sizeBi) or (i >= height - sizeBi):
					new_Ptr.__setitem__(newImg.bytesPerLine() * i + j, chr(0))
				else:
					#print input_img.width() * (i - sizeBi) + j - sizeBi
					new_Ptr.__setitem__(newImg.bytesPerLine() * i + j, chr(inputList[input_img.width() * (i - sizeBi) + j - sizeBi]))
		# dot product
		destImg = QImage(input_img.width(), input_img.height(), QImage.Format_Indexed8)
		destPtr = destImg.bits()
		destPtr.setsize(destImg.bytesPerLine() * destImg.height())
		patch_arr = self.view_as_window(newImg, (size, size))

		for i in range(destImg.height()):
			for j in range(destImg.width()):
				arr = patch_arr[i][j]
				#destPtr.__setitem__(destImg.bytesPerLine() * i + j, chr(arr[1][1]))
				sum1 = 0
				sum2 = 0
				flag = False
				for k in range(size):
					for l in range(size):
						if Q < 0 and arr[k][l] == 0:
							flag = True
							break
						else:
							sum1 += pow(arr[k][l], Q+1)
							sum2 += pow(arr[k][l], Q)
				if (flag == True) or (sum2 != 0 and sum1 / sum2 < 0):
					destPtr.__setitem__(destImg.bytesPerLine() * i + j, chr(0))
				elif (sum2 == 0) or (sum1 / sum2 > 255):
					destPtr.__setitem__(destImg.bytesPerLine() * i + j, chr(255))
				else:
					destPtr.__setitem__(destImg.bytesPerLine() * i + j, chr(int(sum1 / float(sum2))))
		return destImg

	def staFilter2d(self, input_img, size, func):
		# fill the edge with 0
		sizeBi = size / 2
		#print sizeBi
		width = input_img.width() + sizeBi * 2 # --> 386 0-->385 [0 385] 1 - 384
		height = input_img.height() + sizeBi * 2
		newImg = QImage(width, height, QImage.Format_Indexed8)
		inputList = [ord(ch) for ch in input_img.bits().asstring(input_img.byteCount())] # sip.voidptr
		
		new_Ptr = newImg.bits()    # sip.voidptr
		new_Ptr.setsize(newImg.bytesPerLine() * height)

		for i in range(height):
			for j in range(width):
				if (j < sizeBi) or (j >= width - sizeBi):
					new_Ptr.__setitem__(newImg.bytesPerLine() * i + j, chr(0))
				elif (i < sizeBi) or (i >= height - sizeBi):
					new_Ptr.__setitem__(newImg.bytesPerLine() * i + j, chr(0))
				else:
					new_Ptr.__setitem__(newImg.bytesPerLine() * i + j, chr(inputList[input_img.width() * (i - sizeBi) + j - sizeBi]))
		# dot product
		destImg = QImage(input_img.width(), input_img.height(), QImage.Format_Indexed8)
		destPtr = destImg.bits()
		destPtr.setsize(destImg.bytesPerLine() * destImg.height())
		patch_arr = self.view_as_window(newImg, (size, size))
		for i in range(destImg.height()):
			for j in range(destImg.width()):
				arr = patch_arr[i][j]
				destPtr.__setitem__(destImg.bytesPerLine() * i + j, chr(int(func(arr.ravel()))))
		return destImg

	def keepIn(self, value):
		if value < 0: return 0
		elif value > 255: return 255
		else: return int(value)
		
	def slotLapFilter(self):
		img = QImage()
		img.load(self.imgFile)
		self.imgTemp = self.filter2d(img, [[-1,-1,-1],[-1,9,-1],[-1,-1,-1]], 3)
		self.imgLabel.setPixmap(QPixmap.fromImage(self.imgTemp))
	
	def slotAddGaussian(self):
		img = QImage()
		img.load(self.imgFile)
		meanString, ok = QInputDialog.getText(self, self.tr("PARAMETER设置"),
			                            self.tr("输入均值: "), QLineEdit.Normal, "")
		if ok and (not meanString.isEmpty()):
			mean = int(meanString)
		scaleString, ok = QInputDialog.getText(self, self.tr("PARAMETER设置"),
			                            self.tr("输入标准差: "), QLineEdit.Normal, "")
		if ok and (not scaleString.isEmpty()):
			scale = int(scaleString)
		img = self.rgba2g(img)
		print img.width(), img.bytesPerLine()
		# get the pure pic array
		imgArr = np.array([ord(ch) for ch in img.bits().asstring(img.byteCount())])
		#print imgArr[0],imgArr[1],imgArr[2], imgArr[3],imgArr[4],imgArr[5], imgArr[6], imgArr[7]
		#imgNoise = zeros(img.height() * img.bytesPerLine())
		# get a guassian noise
		imgNoise = np.random.normal(mean, scale, (img.height() * img.bytesPerLine()))
		#imgNoise[:,:img.bytesPerLine()] = gauArr
		# G = P + N
		imgArr = np.add(imgArr, imgNoise)
		destImg = QImage(img.width(), img.height(), QImage.Format_Indexed8)
		destPtr = destImg.bits()
		destPtr.setsize(img.width() * img.height())
		print "dest"
		print destImg.width(), destImg.height()
		for i in range(destImg.height()):
			for j in range(destImg.width()):
				destPtr.__setitem__(destImg.bytesPerLine() * i + j, chr(self.keepIn(imgArr[j + i * img.bytesPerLine()])))
		self.imgTemp = destImg
		print self.imgTemp.width(), self.imgTemp.bytesPerLine()
		self.imgLabel.setPixmap(QPixmap.fromImage(self.imgTemp))
		
	def slotAddImpluse(self):
		img = QImage()
		img.load(self.imgFile)
		saltString, ok = QInputDialog.getText(self, self.tr("PARAMETER设置"),
			                            self.tr("输入盐噪声概率: "), QLineEdit.Normal, "")
		if ok and (not saltString.isEmpty()):
			salt = float(saltString)
		pupperString, ok = QInputDialog.getText(self, self.tr("PARAMETER设置"),
			                            self.tr("输入椒噪声概率: "), QLineEdit.Normal, "")
		if ok and (not pupperString.isEmpty()):
			pupper = float(pupperString)
		imgT = self.rgba2g(img)
		print "imgT"
		print imgT.width(),imgT.bytesPerLine()
		# get the pure pic array
		imgArr = np.array([ord(ch) for ch in imgT.bits().asstring(imgT.byteCount())])
		destImg = QImage(imgT.width(), imgT.height(), QImage.Format_Indexed8)
		destPtr = destImg.bits()
		destPtr.setsize(imgT.width() * imgT.height())
		for i in range(destImg.height()):
			for j in range(destImg.width()):
				pro = np.random.random()
				if salt > pro > 0:
					value = 255
				elif pupper > pro >= salt:
					value = 0
				else:  value = imgArr[j + i * imgT.bytesPerLine()]
				destPtr.__setitem__(destImg.bytesPerLine() * i + j, chr(value))
		self.imgTemp = destImg
		self.imgLabel.setPixmap(QPixmap.fromImage(self.imgTemp))
	
	# convert rgb to gray
	def rgba2g(self, img):
		if img.bytesPerLine() / float(img.width()) < 2: 
			#debug
			print "1"
			return img
		#debug
		if img.bytesPerLine() / float(img.width()) == 4: 
			print "4"
		# get the pure pic array
		imgArr = np.array([ord(ch) for ch in img.bits().asstring(img.byteCount())])
		print img.bytesPerLine() * img.height()
		print len(imgArr)
		destImg = QImage(img.width(), img.height(), QImage.Format_Indexed8)
		destPtr = destImg.bits()
		destPtr.setsize(img.width() * img.height())
		for i in range(destImg.height()):
			for j in range(destImg.width()):
				value = imgArr[j * 4 + i * img.bytesPerLine()]
				destPtr.__setitem__(destImg.bytesPerLine() * i + j, chr(value))
		return destImg
	
	def rgba2one(self, img, offset):
		if img.bytesPerLine() / float(img.width()) < 2: 
			#debug
			print "1"
			return img
		#debug
		if img.bytesPerLine() / float(img.width()) == 4: 
			print "4"
		# get the pure pic array
		imgArr = np.array([ord(ch) for ch in img.bits().asstring(img.byteCount())])
		print img.bytesPerLine() * img.height()
		print len(imgArr)
		destImg = QImage(img.width(), img.height(), QImage.Format_Indexed8)
		destPtr = destImg.bits()
		destPtr.setsize(img.width() * img.height())
		for i in range(destImg.height()):
			for j in range(destImg.width()):
				value = imgArr[j * 4 + offset + i * img.bytesPerLine()]
				destPtr.__setitem__(destImg.bytesPerLine() * i + j, chr(value))
		return destImg		
	
	def combineHist(self, RPArr, GPArr, BPArr):
		sumA = np.sum(RPArr) + np.sum(GPArr) + np.sum(BPArr)
		# 特定RGB值频率
		proArr = (RPArr + GPArr + BPArr)/float(sumA)
		for i in range(len(proArr) - 1):
			proArr[i + 1] += proArr[i]
		return proArr
	
	def rebuildRGB(self, RP, GP, BP, width, height):
		destImg = QImage(width, height, QImage.Format_RGB32)
		destPtr = destImg.bits()
		destPtr.setsize(destImg.bytesPerLine() * destImg.height())
		index = 0
		for i in range(destImg.height()):
			for j in range(0,destImg.bytesPerLine(),4):
				destPtr.__setitem__(destImg.bytesPerLine() * i + j, chr(int(RP[index])))#R
				destPtr.__setitem__(destImg.bytesPerLine() * i + j + 1, chr(int(GP[index])))#G
				destPtr.__setitem__(destImg.bytesPerLine() * i + j + 2, chr(int(BP[index])))#B
				destPtr.__setitem__(destImg.bytesPerLine() * i + j + 3, chr(255))
				index += 1
		return destImg
	
	def geometricMean(self, ls):
		product = 1
		for num in ls:
			product = product * pow(num, 1/float(len(ls)))
		return product

	def slotGMean(self):
		img = QImage()
		img.load(self.imgFile)
		imgT = self.rgba2g(img)
		self.imgTemp = self.staFilter2d(imgT, 5, self.geometricMean)
		self.imgLabel.setPixmap(QPixmap.fromImage(self.imgTemp))
		
	def slotMaxFilter(self):
		img = QImage()
		img.load(self.imgFile)
		# 将4通转为单通道
		imgT = self.rgba2g(img)
		self.imgTemp = self.staFilter2d(imgT, 5, np.max)
		self.imgLabel.setPixmap(QPixmap.fromImage(self.imgTemp))

	def slotMinFilter(self):
		img = QImage()
		img.load(self.imgFile)
		imgT = self.rgba2g(img)
		self.imgTemp = self.staFilter2d(imgT, 5, np.min)
		self.imgLabel.setPixmap(QPixmap.fromImage(self.imgTemp))
	
	def slotMedianFilter(self):
		img = QImage()
		img.load(self.imgFile)
		imgT = self.rgba2g(img)
		self.imgTemp = self.staFilter2d(imgT, 5, np.median)
		self.imgLabel.setPixmap(QPixmap.fromImage(self.imgTemp))
	
	def slotHarmFilter(self):
		img = QImage()
		img.load(self.imgFile)
		sizeString, ok = QInputDialog.getText(self, self.tr("PATCH设置"),
			                            self.tr("输入Size: "), QLineEdit.Normal, "")
		if ok and (not sizeString.isEmpty()):
			size = int(sizeString)
		imgT = self.rgba2g(img)
		print imgT.width(), imgT.bytesPerLine()
		self.imgTemp = self.conHarmonic(imgT, -1, size)
		self.imgLabel.setPixmap(QPixmap.fromImage(self.imgTemp))
	
	def slotConHarmFilter(self):
		img = QImage()
		img.load(self.imgFile)
		sizeString, ok = QInputDialog.getText(self, self.tr("设置"),
			                            self.tr("输入Size: "), QLineEdit.Normal, "")
		if ok and (not sizeString.isEmpty()):
			size = int(sizeString)
		QValue, ok = QInputDialog.getText(self, self.tr("设置"),
			                            self.tr("输入Q: "), QLineEdit.Normal, "")
		if ok and (not QValue.isEmpty()):
			Q = float(QValue)
		imgT = self.rgba2g(img)
		self.imgTemp = self.conHarmonic(imgT, Q, size)
		self.imgLabel.setPixmap(QPixmap.fromImage(self.imgTemp))
		
	def slotAveFilter(self):
		img = QImage()
		img.load(self.imgFile)
		sizeString, ok = QInputDialog.getText(self, self.tr("PATCH设置"),
										 self.tr("输入Size: "),
										 QLineEdit.Normal, "")
		if ok and (not sizeString.isEmpty()):
			size = int(sizeString)
		imgT = self.rgba2g(img)
		filter = np.ones(size * size).reshape(size, size)
		self.imgTemp = self.filter2d(imgT, filter, size)
		self.imgLabel.setPixmap(QPixmap.fromImage(self.imgTemp))
		
	def slotSobFilter(self):
		img = QImage()
		img.load(self.imgFile)
		self.imgTemp = self.filter2d(img, [[-1,-2,-1],[0,0,0],[1,2,1]], 3)
		self.imgLabel.setPixmap(QPixmap.fromImage(self.imgTemp))		

	def slotSob2Filter(self):
		img = QImage()
		img.load(self.imgFile)
		self.imgTemp = self.filter2d(img, [[-1,0,1],[-2,0,2],[-1,0,1]], 3)
		self.imgLabel.setPixmap(QPixmap.fromImage(self.imgTemp))	
		
	def slotPatch(self):
		widthString, ok = QInputDialog.getText(self, self.tr("PATCH设置"),
										 self.tr("输入宽度: "),
										 QLineEdit.Normal, "")
		if ok and (not widthString.isEmpty()):
			patchWidth = int(widthString)
		heightString, ok = QInputDialog.getText(self, self.tr("PATCH设置"),
										 self.tr("输入高度: "),
										 QLineEdit.Normal, "")
		if ok and (not heightString.isEmpty()):
			patchHeight = int(heightString)
		self.view_as_window(self.imgTemp, (patchWidth, patchHeight))
		
	def slotCustom(self):
		widthString, ok = QInputDialog.getText(self, self.tr("设置"),
										 self.tr("输入宽度: "),
										 QLineEdit.Normal, "")
		if ok and (not widthString.isEmpty()):
			self.width = int(widthString)
		heightString, ok = QInputDialog.getText(self, self.tr("设置"),
										 self.tr("输入高度: "),
										 QLineEdit.Normal, "")
		if ok and (not heightString.isEmpty()):
			self.height = int(heightString)
		self.scale()	
	
	def slotLevel(self):
		levelString, ok = QInputDialog.getText(self, self.tr("设置"),
										 self.tr("输入灰度级别(<256): "),
										 QLineEdit.Normal, "")		
		if ok and (not levelString.isEmpty()):
			self.level = int(levelString)
			
		self.quantize()
	
	# FOR HW4  RGBEqualize
	def slotRGBEqual1(self):
		img = QImage()
		img.load(self.imgFile)
		RP = self.rgba2one(img, 0)
		GP = self.rgba2one(img, 1)
		BP = self.rgba2one(img, 2)
		RPArr = self.equanlize_hist(RP)
		GPArr = self.equanlize_hist(GP)
		BPArr = self.equanlize_hist(BP)
		self.imgTemp = self.rebuildRGB(RPArr, GPArr, BPArr,img.width(), img.height())
		self.imgLabel.setPixmap(QPixmap.fromImage(self.imgTemp))
		
	def slotRGBEqual2(self):
		img = QImage()
		img.load(self.imgFile)
		RP = self.rgba2one(img, 0)
		GP = self.rgba2one(img, 1)
		BP = self.rgba2one(img, 2)
		# 获得频数矩阵
		RPArr = self.plot_hist(RP)
		GPArr = self.plot_hist(GP)
		BPArr = self.plot_hist(BP)
		# 结合计算平均直方图
		GArr = self.combineHist(RPArr, GPArr, BPArr)
		RPArr = self.match_hist(RP, GArr)
		GPArr = self.match_hist(GP, GArr)
		BPArr = self.match_hist(BP, GArr)
		self.imgTemp = self.rebuildRGB(RPArr, GPArr, BPArr, img.width(), img.height())
		self.imgLabel.setPixmap(QPixmap.fromImage(self.imgTemp))
	
	def openFile(self):
		self.imgFile = QFileDialog.getOpenFileName(self, "open file Dialog", "C:\Users\acer\Desktop\DIP\src\hw4_input", "Image file(*.png)")
		if not self.imgFile.isEmpty():
			img = QImage()
			img.load(self.imgFile)
			print img.width(), img.bytesPerLine()
			self.imgTemp = img.copy()
			self.imgLabel.setPixmap(QPixmap.fromImage(img))
			self.resize(img.width(), img.height())
	
	def saveFile(self):
		fileName = QFileDialog.getSaveFileName(self, "save file Dialog", ".", "Image Files (*.png *.jpg *.bmp)")
		if fileName.isEmpty():
			return
		QPixmap.fromImage(self.imgTemp).save(fileName)

app = QApplication(sys.argv)
main = MainWidget()
main.show()
app.exec_()