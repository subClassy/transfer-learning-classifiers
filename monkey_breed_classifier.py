import os

from tensorflow.keras.applications import MobileNet
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.optimizer_v2.rmsprop import RMSProp


def add_top_model(base_model, num_classes):
    top_model = base_model.output
    top_model = GlobalAveragePooling2D()(top_model)
    top_model = Dense(1024, activation='relu')(top_model)
    top_model = Dense(1024, activation='relu')(top_model)
    top_model = Dense(512, activation='relu')(top_model)
    top_model = Dense(num_classes, activation='softmax')(top_model)
    
    return top_model

img_rows, img_cols = 224, 224

mobile_net = MobileNet(
    input_shape=(img_rows, img_cols, 3), 
    include_top=False, 
    weights='imagenet')

for layers in mobile_net.layers:
    layers.trainable = False

num_classes = 10
complete_model = add_top_model(mobile_net, num_classes)

model = Model(inputs=mobile_net.input, outputs=complete_model)
print(model.summary())

my_path = os.path.abspath(os.path.dirname(__file__))

train_data_dir = os.path.join(my_path, "./monkey_breed/train")
validation_data_dir = os.path.join(my_path, "./monkey_breed/validation")

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
epochs = 5

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


checkpoint = ModelCheckpoint("monkey_breed_mobileNet.h5",
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


nb_train_samples = 1098
nb_validation_samples = 272

history = model.fit_generator(
    train_generator,
    steps_per_epoch = nb_train_samples // batch_size,
    epochs = epochs,
    callbacks = callbacks,
    validation_data = validation_generator,
    validation_steps = nb_validation_samples // batch_size)