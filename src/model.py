import tensorflow as tf
from keras.layers import Layer, InputSpec, Reshape, Activation, Conv2D, Conv2DTranspose, SeparableConv2D, Dropout
from keras.layers import Input, Add, Concatenate, Lambda,LeakyReLU,AveragePooling2D, BatchNormalizationm, LayerNormalization,Add, MultiHeadAttention, Flatten
from keras.optimizers import Adam
from keras.models import Model
from keras.initializers import RandomNormal
from losses import *


class ReflectionPadding2D(Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        if type(padding) == int:
            padding = (padding, padding)
        self.padding = padding
        self.input_spec = [InputSpec(ndim=4)]
        super(ReflectionPadding2D, self).__init__(**kwargs)
        
    def get_config(self):
      cfg = super().get_config()
      return cfg   

    def compute_output_shape(self, s):
        """ If you are using "channels_last" configuration"""
        return (s[0], s[1] + 2 * self.padding[0], s[2] + 2 * self.padding[1], s[3])

    def call(self, x, mask=None):
        w_pad,h_pad = self.padding
        return tf.pad(x, [[0,0], [h_pad,h_pad], [w_pad,w_pad], [0,0] ], 'REFLECT')
      
      
def novel_residual_block(X_input, filters,base):

    name_base = base + '/branch'
    X = X_input
    X = ReflectionPadding2D((1,1),name=name_base + '1/rf')(X)
    X = SeparableConv2D(filters, kernel_size=(3,3), strides=(1,1),dilation_rate=1, padding='valid',name=name_base + '1/sepconv')(X)
    X = BatchNormalization(axis=3, center=True, scale=True, name=name_base + '1/BNorm')(X)
    X = LeakyReLU(alpha=0.2,name=name_base + '1/LeakyRelu')(X)

    ## Branch 1 ext1
    X_branch_1 = ReflectionPadding2D((1,1),name=name_base + '1_1/rf')(X)
    X_branch_1 = SeparableConv2D(filters, kernel_size=(3,3), strides=(1,1), padding='valid',name=name_base + '1_1/sepconv')(X_branch_1)
    X_branch_1 = BatchNormalization(axis=3, center=True, scale=True, name=name_base + '1_1/BNorm')(X_branch_1)
    X_branch_1 = LeakyReLU(alpha=0.2,name=name_base + '1_1/LeakyRelu')(X_branch_1)

    ## Branch 2
    X_branch_2 = ReflectionPadding2D((2,2),name=name_base + '2/rf')(X)
    X_branch_2 = SeparableConv2D(filters, kernel_size=(3,3), strides=(1,1), dilation_rate=2, padding='valid',name=name_base + '2/sepconv')(X_branch_2)
    X_branch_2 = BatchNormalization(axis=3, center=True, scale=True, name=name_base + '2/BNorm')(X_branch_2)
    X_branch_2 = LeakyReLU(alpha=0.2,name=name_base + '2/LeakyRelu')(X_branch_2)
    X_add_branch_1_2 = Add(name=name_base + '1/add_branch1_2')([X_branch_2,X_branch_1])

    X = Add(name=name_base + '1/add_skip')([X_input, X_add_branch_1_2])
    return X

def Attention(X,filters,i):
    X_input = X
    X = Conv2D(filters, kernel_size=(3,3), strides=(1,1), padding='same',kernel_initializer=RandomNormal(stddev=0.02),name="Attention_"+str(i+1)+"/conv1")(X)
    X = BatchNormalization(name="Attention_"+str(i+1)+"/BNorm1")(X)
    X = LeakyReLU(alpha=0.2,name="Attention_"+str(i+1)+"/leakyReLU1")(X)
    X = Add(name="Attention_"+str(i+1)+"/add1")([X_input,X])

    X = Conv2D(filters, kernel_size=(3,3), strides=(1,1), padding='same',kernel_initializer=RandomNormal(stddev=0.02),name="Attention_"+str(i+1)+"/conv2")(X)
    X = BatchNormalization(name="Attention_"+str(i+1)+"/BNorm2")(X)
    X = LeakyReLU(alpha=0.2,name="Attention_"+str(i+1)+"/leakyReLU2")(X)

    X = Add(name="Attention_"+str(i+1)+"/add2")([X_input,X])
    return X
def encoder_block(X,down_filters,i):
    X = Conv2D(down_filters, kernel_size=(3,3), strides=(2,2), padding='same',kernel_initializer=RandomNormal(stddev=0.02),name="down_conv_"+str(i+1))(X)
    X = BatchNormalization(name="down_bn_"+str(i+1))(X)
    X = LeakyReLU(alpha=0.2,name="down_leakyRelu_"+str(i+1))(X)
    return X

def decoder_block(X,up_filters,i):
    X = Conv2DTranspose(filters=up_filters, kernel_size=(3,3), strides=(2,2), padding='same',kernel_initializer=RandomNormal(stddev=0.02),name="up_convtranpose_"+str(i+1) )(X)
    X = BatchNormalization(name="up_bn_"+str(i+1))(X)
    X = LeakyReLU(alpha=0.2,name="up_leakyRelu_"+str(i+1))(X)
    return X

def coarse_generator(img_shape=(64, 64, 1),ncf=64, n_downsampling=2, n_blocks=9, n_channels=1):
    X_input = Input(img_shape)
    X = ReflectionPadding2D((3,3))(X_input)
    X = Conv2D(ncf, kernel_size=(7,7), strides=(1,1), padding='valid',kernel_initializer=RandomNormal(stddev=0.02),name="conv1")(X)
    X = BatchNormalization(name="bn_1")(X)
    X_pre_down = LeakyReLU(alpha=0.2,name="leakyRelu_1")(X)

    # Downsampling layers
    down_filters = ncf * pow(2,0) * 2
    X_down1 = encoder_block(X,down_filters,0)
    down_filters = ncf * pow(2,1) * 2
    X_down2 = encoder_block(X_down1,down_filters,1)
    X = X_down2


    # Novel Residual Blocks
    res_filters = pow(2,n_downsampling)
    for i in range(n_blocks):
        X = novel_residual_block(X, ncf*res_filters,base="block_"+str(i+1))


    # Upsampling layers
    up_filters  =int(ncf * pow(2,(n_downsampling - 0)) / 2) 
    X_up1 = decoder_block(X,up_filters,0)
    X_up1_att = Attention(X_down1,128,0)
    X_up1_add = Add(name="skip_1")([X_up1_att,X_up1])
    up_filters  =int(ncf * pow(2,(n_downsampling - 1)) / 2) 
    X_up2 = decoder_block(X_up1_add,up_filters,1)
    X_up2_att = Attention(X_pre_down,64,1)
    X_up2_add = Add(name="skip_2")([X_up2_att,X_up2])
    feature_out = X_up2_add
    print("X_feature",feature_out.shape)
    X = ReflectionPadding2D((3,3),name="final/rf")(X_up2_add)
    X = Conv2D(n_channels, kernel_size=(7,7), strides=(1,1), padding='valid',kernel_initializer=RandomNormal(stddev=0.02),name="final/conv")(X)
    X = Activation('tanh',name="tanh")(X)


    model = Model(X_input, [X,feature_out],name='G_Coarse')
    model.compile(loss=['mse',None], optimizer=Adam(lr=0.0002, beta_1=0.5, beta_2=0.999))

    model.summary()
    return model

def fine_generator(x_coarse_shape=(32,32,64),input_shape=(64, 64, 1), nff=64, n_blocks=3, n_coarse_gen=1,n_channels = 1):

    
    X_input = Input(shape=input_shape,name="input")
    X_coarse = Input(shape=x_coarse_shape,name="x_input")
    print("X_coarse",X_coarse.shape)
    for i in range(1, n_coarse_gen+1):
        
        
        # Downsampling layers
        down_filters = nff * (2**(n_coarse_gen-i))
        X = ReflectionPadding2D((3,3),name="rf_"+str(i))(X_input)
        X = Conv2D(down_filters, kernel_size=(7,7), strides=(1,1), padding='valid',kernel_initializer=RandomNormal(stddev=0.02),name="conv_"+str(i))(X)
        X = BatchNormalization(name="in_"+str(i))(X)
        X_pre_down = LeakyReLU(alpha=0.2,name="leakyRelu_"+str(i))(X)


        X_down1 = encoder_block(X,down_filters,i-1)
        # Connection from Coarse Generator
        X = Add(name="add_X_coarse")([X_coarse,X_down1])

        X = SeparableConv2D(down_filters*2, kernel_size=(3,3), strides=(1,1), padding='same',kernel_initializer=RandomNormal(stddev=0.02),name="sepconv_"+str(i))(X)
        X = BatchNormalization(name="sep_in_"+str(i))(X)
        X = LeakyReLU(alpha=0.2,name="sep_leakyRelu_"+str(i))(X)
        for j in range(n_blocks-1):
            res_filters = nff * (2**(n_coarse_gen-i)) * 2
            X = novel_residual_block(X, res_filters,base="block_"+str(j+1))

        # Upsampling layers
        up_filters = nff * (2**(n_coarse_gen-i))
        X_up1 = decoder_block(X,up_filters,i-1)
        X_up1_att = Attention(X_pre_down,up_filters,i-1)
        X_up1_add = Add(name="skip_"+str(i))([X_up1_att,X_up1])

    X = ReflectionPadding2D((3,3),name="final/rf")(X_up1_add)
    X = Conv2D(n_channels, kernel_size=(7,7), strides=(1,1), padding='valid',name="final/conv")(X)
    X = Activation('tanh',name="tanh")(X)

    model = Model([X_input,X_coarse], X, name='G_Fine')
    model.compile(loss='mse', optimizer=Adam(lr=0.0002, beta_1=0.5, beta_2=0.999))

    model.summary()
    return model

class Patches(Layer):
    def __init__(self, patch_size):
        super(Patches, self).__init__()
        self.patch_size = patch_size
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'patch_size': self.patch_size,
        })
        return config
    def call(self, images):
        batch_size = tf.shape(images)[0]
        patches = tf.image.extract_patches(
            images=images,
            sizes=[1, self.patch_size, self.patch_size, 1],
            strides=[1, self.patch_size, self.patch_size, 1],
            rates=[1, 1, 1, 1],
            padding="VALID",
        )
        patch_dims = patches.shape[-1]
        patches = tf.reshape(patches, [batch_size, -1, patch_dims])
        return patches
      
class PatchEncoder(Layer):
    def __init__(self, num_patches, projection_dim):
        super(PatchEncoder, self).__init__()
        self.num_patches = num_patches
        self.projection = layers.Dense(units=projection_dim)
        self.position_embedding = layers.Embedding(
            input_dim=num_patches, output_dim=projection_dim
        )
    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'num_patches': self.num_patches,
            'projection': self.projection,
            'position_embedding': self.position_embedding,
        })
        return config
    def call(self, patch):
        positions = tf.range(start=0, limit=self.num_patches, delta=1)
        encoded = self.projection(patch) + self.position_embedding(positions)
        return encoded
def mlp(x, hidden_units, dropout_rate):
    for units in hidden_units:
        x = layers.Dense(units, activation=tf.nn.gelu)(x)
        x = layers.Dropout(dropout_rate)(x)
    return x
   
def vit_discriminator(input_shape_fundus=(512, 512, 3),input_shape_angio= (512, 512, 1),image_size = 512,projection_dim=4096, patch_size=64,transformer_layers = 8, num_heads = 4,
                          mlp_head_units = [128,64], num_classes=2,activation='tanh', name='VTGAN'):
    num_patches = (image_size // patch_size) ** 2
    transformer_units = [projection_dim * 2,projection_dim] 
    #inputs = Input(shape=input_shape)
    X_input_fundus = Input(shape=input_shape_fundus,name="input_fundus")
    X_input_angio = Input(shape=input_shape_angio,name="input_angio")
    X = Concatenate(axis=-1,name="concat")([X_input_fundus, X_input_angio])
    feat =[]
    # Create patches.
    patches = Patches(patch_size)(X)
    # Encode patches.
    encoded_patches = PatchEncoder(num_patches, projection_dim)(patches)
    # Create multiple layers of the Transformer block.
    for _ in range(transformer_layers):
        # Layer normalization 1.
        x1 = LayerNormalization(epsilon=1e-6)(encoded_patches)
        # Create a multi-head attention layer.
        attention_output = MultiHeadAttention(
            num_heads=num_heads, key_dim=projection_dim, dropout=0.1
        )(x1, x1)
        # Skip connection 1.
        x2 = Add()([attention_output, encoded_patches])
        # Layer normalization 2.
        x3 = LayerNormalization(epsilon=1e-6)(x2)
        # MLP.
        x3 = mlp(x3, hidden_units=transformer_units, dropout_rate=0.1)
        # Skip connection 2.
        encoded_patches = Add()([x3, x2])
        feat.append(encoded_patches)
        #print(feat[0].shape)
    feat = Concatenate(axis=-1)([feat[0], feat[1],feat[2],feat[3]])
    # Create a [batch_size, projection_dim] tensor.
    representation = LayerNormalization(epsilon=1e-6)(encoded_patches)
    # Patchgan output
    X_reshape = Reshape((projection_dim, projection_dim,1), name='reshape')(representation)
    X = Conv2D(1, kernel_size=(4,4), strides=(1,1), padding='same',kernel_initializer=RandomNormal(stddev=0.02))(X_reshape)
    out_hinge = Activation(activation,name='patchgan')(X)
    # Class output
    representation = Conv2D(64, kernel_size=(4,4), strides=(1,1), padding='same',kernel_initializer=RandomNormal(stddev=0.02))(X_reshape)
    # Add MLP.
    features = GlobalAveragePooling2D()(representation)
    # Classify outputs.
    classses = Dense(num_classes)(features)
    out_class = Activation('softmax',name='class')(classses)
    # Create the Keras model.
    model = Model(inputs=[X_input_fundus,X_input_angio], outputs=[out_hinge,out_class,feat],name=name)
    model.compile(loss=['mse','categorical_crossentropy',ef_loss], optimizer=Adam(lr=0.0002, beta_1=0.5, beta_2=0.999))
    model.summary()
    return model

def vtgan(g_model_fine,g_model_coarse, d_model1, d_model2,image_shape_fine,image_shape_coarse,image_shape_x_coarse,label_shape_fine,label_shape_coarse): #d_model3, d_model4
    # Discriminator NOT trainable
    d_model1.trainable = False
    d_model2.trainable = False

    in_fine= Input(shape=image_shape_fine)
    in_coarse = Input(shape=image_shape_coarse)
    in_x_coarse = Input(shape=image_shape_x_coarse)

    # Generators
    gen_out_coarse, _ = g_model_coarse(in_coarse)
    gen_out_fine = g_model_fine([in_fine,in_x_coarse])

    # Discriminators Fine
    dis_out_1_fake = d_model1([in_fine,gen_out_fine]) 

    # Discriminators Coarse
    dis_out_2_fake = d_model2([in_coarse, gen_out_coarse]) 


    model = Model([in_fine,in_coarse,in_x_coarse],[dis_out_1_fake[0],
                                                    dis_out_2_fake[0],
                                                    dis_out_1_fake[1],
                                                    dis_out_2_fake[1],
                                                    dis_out_1_fake[2],
                                                    dis_out_2_fake[2],
                                                    gen_out_coarse,
                                                    gen_out_fine,
                                                    gen_out_coarse,
                                                    gen_out_fine,
                                                    gen_out_coarse,
                                                    gen_out_fine
                                                    ])

    opt = Adam(lr=0.0002, beta_1=0.5)
    model.compile(loss=['hinge', 
                    'hinge',
                    'categorical_crossentropy',
                    'categorical_crossentropy',
                    ef_loss,
                    ef_loss,
                    'hinge',
                    'hinge',
                    'mse',
                    'mse',
                    perceptual_loss_coarse,
                    perceptual_loss_fine
                    ], 
              optimizer=opt,loss_weights=[1,1,10,10,
                                          1,1,
                                          10,10,10,10,10,10
                                          ])
    model.summary()
    return model
