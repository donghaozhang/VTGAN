from src.model import coarse_generator,fine_generator,vtgan,vit_discriminator
from src.visualization import summarize_performance, summarize_performance_global, plot_history, to_csv
from src.data_loader import resize, generate_fake_data_coarse, generate_fake_data_fine, generate_real_data, load_real_data
import argparse
import time
from numpy import load
import gc
import keras.backend as K




def train(d_model1, d_model2,g_global_model, g_local_model, gan_model, dataset, n_epochs=20, n_batch=1, savedir='AAGAN',n_patch=[64,32]): #
    
    if not os.path.exists(savedir):
      os.makedirs(savedir)
    # unpack dataset
    trainA, trainB, trainC = dataset
    # calculate the number of batches per training epoch
    bat_per_epo = int(len(trainA) / n_batch)
    # calculate the number of training iterations
    n_steps = bat_per_epo * n_epochs
    
    # lists for storing loss, for plotting later
    d1_hist, d2_hist, d3_hist, d4_hist, d1_cls_hist, d2_cls_hist =  list(),list(), list(), list(), list(), list() #d5_hist, d6_hist, d7_hist, d8_hist = list(),list(), list(), list()
    ef1_hist,ef2_hist =list(),list() 
    g_global_hist, g_local_hist, gan_hist =  list(), list(), list()
    g_global_percp_hist, g_local_percp_hist, g_global_recon_hist, g_local_recon_hist =list(),list(), list(), list()
    evf = np.ones((n_batch,64,256))
    # manually enumerate epochs
    b = 92
    start_time = time.time()
    for k in range(n_epochs):
        for i in range(bat_per_epo):
          d_model1.trainable = True
          d_model2.trainable = True
          gan_model.trainable = False
          g_global_model.trainable = False
          g_local_model.trainable = False
          for j in range(2):
              # select a batch of real samples 
              [X_realA, X_realB], [y1,y2] = generate_real_data(dataset, n_batch,n_patch)#,y3 = 

              
              # generate a batch of fake samples for Coarse Generator
              out_shape = (int(X_realA.shape[1]/2),int(X_realA.shape[2]/2))
              [X_realA_half,X_realB_half] = resize(X_realA,X_realB,out_shape)
              [X_fakeB_half, x_global], y1_coarse = generate_fake_data_coarse(g_global_model, X_realA_half, n_patch) #[y1_coarse,y2_coarse] =


              # generate a batch of fake samples for Fine Generator
              X_fakeB, y1_fine = generate_fake_data_fine(g_local_model, X_realA, x_global, n_patch)  # [y1_fine,y2_fine] = 


              ## FINE DISCRIMINATOR  
              # update discriminator for real samples
              d_feat1_real = d_model1.predict([X_realA,X_realB])
              d_loss1= d_model1.train_on_batch([X_realA, X_realB], [y1,y2,d_feat1_real[2]])[0] # [,X_realB]
              # update discriminator for generated samples
              d_feat1_fake = d_model1.predict([X_realA,X_fakeB])
              d_loss2 = d_model1.train_on_batch([X_realA, X_fakeB] , [y1_fine,y2,d_feat1_fake[2]])[0] #[,X_fakeB]

              ## COARSE DISCRIMINATOR  
              # update discriminator for real samples

              d_feat2_real = d_model2.predict([X_realA_half,X_realB_half])
              d_loss3 = d_model2.train_on_batch([X_realA_half,X_realB_half],[y1,y2,d_feat2_real[2]])[0] #[, X_realB_half]
              # update discriminator for generated samples
              d_feat2_fake = d_model2.predict([X_realA_half,X_fakeB_half])
              d_loss4 = d_model2.train_on_batch([X_realA_half,X_fakeB_half],[y1_coarse,y2,d_feat2_fake[2]])[0] # [,X_fakeB_half]

          


          # turn Global G1 trainable
          d_model1.trainable = False
          d_model2.trainable = False
          gan_model.trainable = False
          g_global_model.trainable = True
          g_local_model.trainable = False
          
          

          # select a batch of real samples for Fine generator
          [X_realA, X_realB], _ = generate_real_data(dataset, n_batch, n_patch)

          # Global Generator image fake and real
          out_shape = (int(X_realA.shape[1]/2),int(X_realA.shape[2]/2))
          [X_realA_half,X_realB_half] = resize(X_realA,X_realB,out_shape)
          [X_fakeB_half, x_global], _ = generate_fake_data_coarse(g_global_model, X_realA_half, n_patch)
          

          # update the Coarse generator
          g_global_loss,_ = g_global_model.train_on_batch(X_realA_half, [X_realB_half,_])

          
          d_model1.trainable = False
          d_model2.trainable = False
          #d_model3.trainable = False
          #d_model4.trainable = False
          gan_model.trainable = False
          g_global_model.trainable = False
          g_local_model.trainable = True
          
          # update the Fine generator 
          g_local_loss = g_local_model.train_on_batch([X_realA,x_global], X_realB)
          

          # turn G1, G2 and GAN trainable, not D1,D2 
          d_model1.trainable = False
          d_model2.trainable = False
          gan_model.trainable = True
          g_global_model.trainable = True
          g_local_model.trainable = True

          d_feat1 = d_model1.predict([X_realA,X_realB])
          d_feat2 = d_model2.predict([X_realA_half,X_realB_half])
          gan_loss,_,_,d_class1,d_class2,ef1_loss,ef2_loss,_,_,g_global_recon_loss, g_local_recon_loss, g_global_percp_loss, g_local_percp_loss = gan_model.train_on_batch([X_realA,X_realA_half,x_global],
                                                                                                                                                      [y1, y1, y2, y2, d_feat1[2], d_feat2[2],
                                                                                                                                                      X_realB_half,X_realB,
                                                                                                                                                      X_realB_half,X_realB,
                                                                                                                                                      X_realB_half,X_realB
                                                                                                                                          ])

          # print losses  
          print('>%d, d1[%.3f] d2[%.3f] d3[%.3f] d4[%.3f] ef1[%.3f] ef2[%.3f] g_g[%.3f] g_l[%.3f] g_g_r[%.3f] g_l_r[%.3f] g_g_p[%.3f] g_l_p[%.3f] gan[%.3f]' % 
                (i+1, d_loss1, d_loss2, d_loss3, d_loss4,ef1_loss, ef2_loss, g_global_loss, g_local_loss, 
                g_global_recon_loss, g_local_recon_loss, g_global_percp_loss, g_local_percp_loss, gan_loss))
                                                                                                                            
          d1_hist.append(d_loss1)
          d2_hist.append(d_loss2)
          d3_hist.append(d_loss3)
          d4_hist.append(d_loss4)
          d1_cls_hist.append(d_class1)
          d2_cls_hist.append(d_class2)
          ef1_hist.append(ef1_loss)
          ef2_hist.append(ef2_loss)
          g_global_hist.append(g_global_loss)
          g_local_hist.append(g_local_loss)
          g_global_recon_hist.append(g_global_recon_loss)
          g_local_recon_hist.append(g_local_recon_loss)
          g_global_percp_hist.append(g_global_percp_loss)
          g_global_percp_hist.append(g_local_percp_loss)
          gan_hist.append(gan_loss)
        
        # summarize model performance
   
        summarize_performance_global(b, g_global_model, d_model1, dataset, n_samples=3,savedir=savedir)
        summarize_performance(b, g_global_model,g_local_model, d_model2, dataset, n_samples=3,savedir=savedir)
        b = b+1
        per_epoch_time = time.time()
        total_per_epoch_time = (per_epoch_time - start_time)/3600.0
        print(total_per_epoch_time)
    plot_history(d1_hist, d2_hist, d3_hist, d4_hist, d1_cls_hist, d2_cls_hist, ef1_hist, ef2_hist,g_global_hist,g_local_hist, 
                 g_global_percp_hist, g_local_percp_hist, g_global_recon_hist, g_local_recon_hist, gan_hist,savedir=savedir)
    to_csv(d1_hist, d2_hist, d3_hist, d4_hist, d1_cls_hist,d2_cls_hist,ef1_hist1, ef2_hist, g_global_hist,g_local_hist, 
                 g_global_percp_hist, g_local_percp_hist, g_global_recon_hist, g_local_recon_hist, gan_hist,savedir=savedir)
    
    

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--npz_file', type=str, default='vtgan', help='path/to/npz/file')
    parser.add_argument('--input_dim', type=int, default=512)
    parser.add_argument('--n_patch', type=int, default=64)
    parser.add_argument('--savedir', type=str, required=False, help='path/to/save_directory',default='VTGAN')
    parser.add_argument('--resume_training', type=str, required=False,  default='no', choices=['yes','no'])
    args = parser.parse_args()
    
    K.clear_session()
    gc.collect()
    start_time = time.time()
    dataset = load_real_data(args.npz_file+'.npz')
    print('Loaded', dataset[0].shape, dataset[1].shape)
    
    
    # define input shape based on the loaded dataset
    in_size = args.input_dim
    image_shape_coarse = (in_size//2,in_size//2,3)
    label_shape_coarse = (in_size//2,in_size//2,1)


    image_shape_fine = (in_size,in_size,3)
    label_shape_fine = (in_size,in_size,1)
    
    image_shape_x_coarse = (in_size//2,in_size//2,64)
    
    ncf=64
    nff=64
    patch_size_fine= args.n_patch
    patch_size_coarse= patch_size_fine//2
    # define discriminator models
    d_model1 = vit_discriminator(image_shape_fine,label_shape_fine, image_size = 512,projection_dim=64, patch_size_fine,
                                 transformer_layers = 4, num_heads = 4, mlp_head_units = [128,64], activation='tanh', name='VT_fine')

    d_model2 = vit_discriminator(image_shape_coarse,label_shape_coarse, image_size = 256, projection_dim=64, patch_size_coarse,
                                 transformer_layers = 4, num_heads = 4, mlp_head_units = [128,64], activation='tanh', name='VT_coarse') 
    
    # define generator models
    g_model_coarse = coarse_generator(image_shape_coarse,n_downsampling=2, n_blocks=9, ncf=ncf,n_channels=1)
    g_model_fine = fine_generator(x_coarse_shape=image_shape_x_coarse,input_shape=image_shape_fine,nff=nff,n_blocks=3)
    
    if args.resume_training =='yes':
        #weight_name_global = "global_model_000067.h5"
        g_model_coarse.load_weights(weight_name_global)

        #weight_name_local = "local_model_000067.h5"
        g_model_fine.load_weights(weight_name_local)
        
    # define the composite model

    vt_model = vtgan(g_model_fine,g_model_coarse, d_model1, d_model2, image_shape_fine,image_shape_coarse,image_shape_x_coarse,label_shape_fine,label_shape_coarse)
    
    train(d_model1, d_model2,  g_model_coarse, g_model_fine, vt_model, dataset, n_epochs=args.epochs, 
          n_batch=args.batch_size, savedir=args.savedir, n_patch=[args.n_patch,args.n_patch])
    
    
    end_time = time.time()
    time_taken = (end_time-start_time)/3600.0
    print(time_taken)
