import numpy as np
import cv2

class Colors:
    """
    Ultralytics default color palette https://ultralytics.com/.

    This class provides methods to work with the Ultralytics color palette, including converting hex color codes to
    RGB values.

    Attributes:
        palette (list of tuple): List of RGB color values.
        n (int): The number of colors in the palette.
        pose_palette (np.array): A specific color palette array with dtype np.uint8.
    """

    def __init__(self):
        """Initialize colors as hex = matplotlib.colors.TABLEAU_COLORS.values()."""
        hexs = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
                '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        self.palette = [self.hex2rgb(f'#{c}') for c in hexs]
        self.n = len(self.palette)
        self.pose_palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0], [255, 153, 255],
                                      [153, 204, 255], [255, 102, 255], [255, 51, 255], [102, 178, 255], [51, 153, 255],
                                      [255, 153, 153], [255, 102, 102], [255, 51, 51], [153, 255, 153], [102, 255, 102],
                                      [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]],
                                     dtype=np.uint8)

    def __call__(self, i, bgr=False):
        """Converts hex color codes to RGB values."""
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        """Converts hex color codes to RGB values (i.e. default PIL order)."""
        return tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4))

class GetTextSizeForLabel:
    def __init__(self, label, im_shape, line_width = None):
        self.label = label
        self.lw = line_width or max(round(sum(im_shape) / 2 * 0.003), 2)
        self.tf = max(self.lw - 1, 1)
        self.sf = self.lw / 3
    def getText(self):
        return (cv2.getTextSize(self.label, 0, fontScale = self.sf, thickness = self.tf)[0], self.sf, self.tf)
    
class Annotator:
    # Initialize class
    def __init__(self, img, xyxys, classnames, confidences, colors, lw = 2):
        self.img = img
        self.xyxys = xyxys
        self.classnames = classnames
        self.confidences = confidences
        self.colors = colors
        self.lw = lw

    # Draw label only
    def drawClass(self):
        for i in range(len(self.classnames)):
            # Draw bounding box
            p1, p2 = (int(self.xyxys[i][0]), int(self.xyxys[i][1])), (int(self.xyxys[i][2]), int(self.xyxys[i][3]))
            cv2.rectangle(self.img, p1, p2, self.colors[i], int(round(self.lw)))
            # Draw text
            labelText = f'{self.classnames[i]}'
            (w, h), sf, tf = GetTextSizeForLabel(label = labelText, im_shape = self.img.shape, line_width = int(round(self.lw))).getText()
            outside = p1[1] - h >= 3
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(self.img, p1, p2, self.colors[i], -1, cv2.LINE_AA)  # filled
            cv2.putText(self.img,
                            labelText, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                            0,
                            sf,
                            (255, 255, 255),
                            thickness=tf,
                            lineType=cv2.LINE_AA)
        return self.img

    # Draw quantity only
    def drawQuantity(self, classnames_xyxys_count_object):
        for i in range(len(classnames_xyxys_count_object)):
            # Draw bounding box
            p1, p2 = (int(self.xyxys[i][0]), int(self.xyxys[i][1])), (int(self.xyxys[i][2]), int(self.xyxys[i][3]))
            cv2.rectangle(self.img, p1, p2, self.colors[i], int(round(self.lw)))
            # Draw text
            labelText = str(classnames_xyxys_count_object[i]['qty'])
            (w, h), sf, tf = GetTextSizeForLabel(label = labelText, im_shape = self.img.shape, line_width = int(round(self.lw))).getText()
            outside = p1[1] - h >= 3
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(self.img, p1, p2, self.colors[i], -1, cv2.LINE_AA)
            cv2.putText(self.img,
                            labelText, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                            0,
                            sf,
                            (255, 255, 255),
                            thickness=tf,
                            lineType=cv2.LINE_AA)
        return self.img
    
    # Draw label and confidence
    def drawClassAndConfidence(self):
        for i in range(len(self.classnames)):
            # Draw bounding box
            p1, p2 = (int(self.xyxys[i][0]), int(self.xyxys[i][1])), (int(self.xyxys[i][2]), int(self.xyxys[i][3]))
            cv2.rectangle(self.img, p1, p2, self.colors[i], int(round(self.lw)))
            # Draw text
            labelText = f'{self.classnames[i]}: {int(self.confidences[i] * 100)} %'
            (w, h), sf, tf = GetTextSizeForLabel(label = labelText, im_shape = self.img.shape, line_width = int(round(self.lw))).getText()
            outside = p1[1] - h >= 3
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(self.img, p1, p2, self.colors[i], -1, cv2.LINE_AA)  # filled
            cv2.putText(self.img,
                            labelText, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2),
                            0,
                            sf,
                            (255, 255, 255),
                            thickness=tf,
                            lineType=cv2.LINE_AA)
        return self.img

# Create object of Colors class
colors = Colors()