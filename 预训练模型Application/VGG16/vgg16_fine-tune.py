#encoding:utf-8

'''
https://blog.csdn.net/sinat_26917383/article/details/72861152
keras系列︱图像多分类训练与利用bottleneck features进行微调(三)
'''

from keras.applications import VGG16
from keras.layers import Flatten, Reshape, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers

model = VGG16(weights='imagenet', include_top=False, input_shape=(150, 150, 3))

x = model.output

print model.input
print model.output

x = Reshape((4,4, 512))(x)
x = Flatten(name='flatten')(x)
x = Dense(256, activation='relu', name='fc1')(x)
x = Dropout(0.5)(x)
predictions = Dense(5, activation='softmax')(x)

from keras.models import Model

vgg_model = Model(inputs=model.input, outputs=predictions)

from keras.utils import plot_model
plot_model(vgg_model, to_file='vgg_model.png')

for layer in vgg_model.layers[:15]:
    layer.trainable = False

print len(vgg_model.layers)

# compile the model with a SGD/momentum optimizer
vgg_model.compile(loss='binary_crossentropy', optimizer=optimizers.SGD(lr=1e-4, momentum=0.9), metrics=['accuracy'])

print vgg_model.summary()

# 准备数据
train_data_dir = './re/train'
validation_data_dir = './re/test'
img_width, img_height = 150, 150
nb_train_samples = 500
nb_validation_samples = 100
epochs = 2# 50
batch_size = 32 # 16

# 导入并显示图片
# img_path = './re/train/320.jpg'
# img = image.load_img(img_path)

# 图片预处理生成器
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

# 图片generator
train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_height, img_width),
        batch_size=32,
        class_mode='categorical')
validation_generator = test_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_height, img_width),
        batch_size=32,
        class_mode='categorical')

# 训练
vgg_model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size)
