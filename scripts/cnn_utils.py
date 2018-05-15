


class CNNDataSet:
    def __init__(name=None):
        name = name
        
    def get_input_shape(self, channel_count=3):
        from keras import backend as K
        K.set_image_dim_ordering('tf')
        if K.image_data_format() == 'channels_first':
            x_train = x_train.reshape(x_train.shape[0], channel_count, img_rows, img_cols)
            x_test = x_test.reshape(x_test.shape[0], channel_count, img_rows, img_cols)
            input_shape = (channel_count, img_rows, img_cols)
        else:
            x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, channel_count)
            x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, channel_count)
            input_shape = (img_rows, img_cols, channel_count)
    
        return input_shape