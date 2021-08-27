import os

from tensorflow.keras.applications import VGG16
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator


def add_top_model(base_model, num_classes): 
    top_model = base_model.output
    top_model = Flatten(name = "flatten")(top_model)
    top_model = Dense(256, activation = "relu")(top_model)
    top_model = Dropout(0.3)(top_model)
    top_model = Dense(num_classes, activation = "softmax")(top_model)
    
    return top_model

img_rows, img_cols = 224, 224

vgg16 = VGG16(
    input_shape=(img_rows, img_cols, 3), 
    include_top=False, 
    weights='imagenet')

for layers in vgg16.layers:
    layers.trainable = False

num_classes = 17
complete_model = add_top_model(vgg16, num_classes)

model = Model(inputs=vgg16.input, outputs=complete_model)
print(model.summary())

my_path = os.path.abspath(os.path.dirname(__file__))

train_data_dir = os.path.join(my_path, "./17_flowers/train")
validation_data_dir = os.path.join(my_path, "./17_flowers/validation")

train_data_gen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=45,
    width_shift_range=0.3,
    height_shift_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest')

validation_data_gen = ImageDataGenerator(
    rescale=1./255)

batch_size = 32
epochs = 25

train_generator = train_data_gen.flow_from_directory(
    train_data_dir,
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = validation_data_gen.flow_from_directory(
    validation_data_dir,
    target_size=(img_rows, img_cols),
    batch_size=batch_size,
    class_mode='categorical')


checkpoint = ModelCheckpoint("flower_classifier_vgg16.h5",
                             monitor="val_loss",
                             mode="min",
                             save_best_only = True,
                             verbose=1)

earlystop = EarlyStopping(monitor = 'val_loss', 
                          min_delta = 0, 
                          patience = 3,
                          verbose = 1,
                          restore_best_weights = True)

callbacks = [checkpoint, earlystop]

model.compile(loss = 'categorical_crossentropy',
              optimizer = RMSprop(lr = 0.001),
              metrics = ['accuracy'])


nb_train_samples = 1190
nb_validation_samples = 170

history = model.fit_generator(
    train_generator,
    steps_per_epoch = nb_train_samples // batch_size,
    epochs = epochs,
    callbacks = callbacks,
    validation_data = validation_generator,
    validation_steps = nb_validation_samples // batch_size)
