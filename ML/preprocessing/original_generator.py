

class trainGenerator(object):
    def __init__(self):
        self.img_generator=[]
        self.anno_generator=[]

    def reset(self):
        self.img_generator=[]
        self.anno_generator=[]

    def flow(self, imgs, annos, batch=4):
        while True:
            for anno, img in zip(annos, imgs):
                self.img_generator.append(img)
                self.anno_generator.append(anno)
                if len(self.anno_generator)==batch:
                    input_img = np.array(self.img_generator)
                    input_anno = np.array(self.anno_generator)
                    self.reset()
                    yield input_img, input_anno

if __name__=='__main__':
    batch_size=4
    gene = trainGenerator().flow(X_train, y_train, batch=batch_size)
