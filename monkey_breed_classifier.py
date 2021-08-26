from tensorflow.keras.applications import MobileNet
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model


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
