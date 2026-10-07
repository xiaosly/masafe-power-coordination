"""Optional image display for vectorized environments."""


class ImageViewer:
    def __init__(self):
        self.figure = None
        self.artist = None

    @property
    def isopen(self):
        import matplotlib.pyplot as plt
        return self.figure is not None and plt.fignum_exists(self.figure.number)

    def imshow(self, image):
        import matplotlib.pyplot as plt
        if not self.isopen:
            self.figure, axis = plt.subplots()
            axis.set_axis_off()
            self.artist = axis.imshow(image)
        else:
            self.artist.set_data(image)
        self.figure.canvas.draw_idle()
        plt.show(block=False)
        self.figure.canvas.flush_events()

    def close(self):
        import matplotlib.pyplot as plt
        if self.figure is not None:
            plt.close(self.figure)
            self.figure = None
