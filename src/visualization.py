import matplotlib.pyplot as plt
import pandas as pd 
import os
from src.dataloader import *

def summarize_performance(step,g_global_model,g_local_model, d_model, dataset, n_samples=3,savedir='VTGAN'):
    # select a sample of input images
    n_patch = [1,1,1]
    [X_realA, X_realB], _ = generate_real_data(dataset, n_samples, n_patch)
    #######################################
    # Resize to half
    out_shape = (int(X_realA.shape[1]/2),int(X_realA.shape[2]/2))
    [X_realA_half,X_realB_half] = resize(X_realA,X_realB,out_shape)
    # generate a batch of fake samples
    [X_fakeB_half, x_global], _ = generate_fake_data_coarse(g_global_model, X_realA_half, n_patch)
    ##########################################
    # generate a batch of fake samples
    X_fakeB, _ = generate_fake_data_fine(g_local_model, X_realA, x_global, n_patch)
    # scale all pixels from [-1,1] to [0,1]
    X_realA = (X_realA + 1) / 2.0
    X_realB = (X_realB + 1) / 2.0
    X_fakeB = (X_fakeB + 1) / 2.0
    # plot real source images
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(X_realA[i])
    # plot generated target image
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + n_samples + i)
        pyplot.axis('off')
        twoD_img = X_fakeB[:,:,:,0]
        pyplot.imshow(twoD_img[i],cmap="gray")
    # plot real target image
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + n_samples*2 + i)
        pyplot.axis('off')
        twoD_img = X_realB[:,:,:,0]
        pyplot.imshow(twoD_img[i],cmap="gray")
    # save plot to file
    filename1 = savedir+'/local_plot_%06d.png' % (step+1)
    pyplot.savefig(filename1)
    pyplot.close()
    # save the generator model
    filename2 = savedir+'/local_gmodel_%06d.h5' % (step+1)
    filename3 = savedir+'/local_dmodel_%06d.h5' % (step+1)
    g_local_model.save(filename2)
    d_model.save(filename3)
    print('>Saved: %s and %s' % (filename1, filename2))

def summarize_performance_global(step, g_model,d_model, dataset, n_samples=3,savedir='VTGAN'):
    # select a sample of input images
    n_patch = [1,1,1]
    [X_realA, X_realB], _ = generate_real_data(dataset, n_samples, n_patch)
    # Resize to half
    out_shape = (int(X_realA.shape[1]/2),int(X_realA.shape[2]/2))
    [X_realA_half,X_realB_half] = resize(X_realA,X_realB,out_shape)
    # generate a batch of fake samples
    [X_fakeB_half, x_global], _ = generate_fake_data_coarse(g_model, X_realA_half, n_patch)
    # scale all pixels from [-1,1] to [0,1]
    X_realA_half = (X_realA_half + 1) / 2.0
    X_realB_half = (X_realB_half + 1) / 2.0
    X_fakeB_half = (X_fakeB_half + 1) / 2.0
    # plot real source images
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + i)
        pyplot.axis('off')
        pyplot.imshow(X_realA_half[i])
    # plot generated target image
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + n_samples + i)
        pyplot.axis('off')
        twoD_img = X_fakeB_half[:,:,:,0]
        pyplot.imshow(twoD_img[i],cmap="gray")
    # plot real target image
    for i in range(n_samples):
        pyplot.subplot(3, n_samples, 1 + n_samples*2 + i)
        pyplot.axis('off')
        twoD_img = X_realB_half[:,:,:,0]
        pyplot.imshow(twoD_img[i],cmap="gray")
    # save plot to file
    filename1 = savedir+'/global_plot_%06d.png' % (step+1)
    pyplot.savefig(filename1)
    pyplot.close()
    # save the generator model
    filename2 = savedir+'/global_gmodel_%06d.h5' % (step+1)
    filename3 = savedir+'/global_dmodel_%06d.h5' % (step+1)
    g_model.save(filename2)
    d_model.save(filename3)
    print('>Saved: %s and %s' % (filename1, filename2))
    #return x_global
def plot_history(d1_hist, d2_hist, d3_hist, d4_hist, d1_cls_hist,d2_cls_hist,fm1_hist, fm2_hist,  #d5_hist, d6_hist, d7_hist, d8_hist, fm1_hist, fm2_hist, fm3_hist, fm4_hist,
                 g_global_hist,g_local_hist, 
                 g_global_percp_hist, 
                 g_local_percp_hist, g_global_recon_hist, g_local_recon_hist, gan_hist,savedir='VTGAN'):
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    pyplot.plot(d1_hist, label='dloss1')
    pyplot.plot(d2_hist, label='dloss2')
    pyplot.plot(d3_hist, label='dloss3')
    pyplot.plot(d4_hist, label='dloss4')
    pyplot.plot(d1_cls_hist, label='dclass1')
    pyplot.plot(d2_cls_hist, label='dclass2')
    pyplot.plot(fm1_hist, label='fm1')
    pyplot.plot(fm2_hist, label='fm2')
    pyplot.plot(g_global_hist, label='g_g_loss')
    pyplot.plot(g_local_hist, label='g_l_loss')
    pyplot.plot(g_global_percp_hist, label='g_g_per')
    pyplot.plot(g_local_percp_hist, label='g_l_per')
    pyplot.plot(g_global_recon_hist, label='g_g_rec')
    pyplot.plot(g_local_recon_hist, label='g_l_rec')
    pyplot.plot(gan_hist, label='gan_loss')
    pyplot.legend()
    filename = savedir+'/plot_line_plot_loss.png'
    pyplot.savefig(filename)
    pyplot.close()
    print('Saved %s' % (filename))
def to_csv(d1_hist, d2_hist, d3_hist, d4_hist, d1_cls_hist, d2_cls_hist,  fm1_hist, fm2_hist,
           #d5_hist, d6_hist, d7_hist, d8_hist, fm1_hist, fm2_hist, fm3_hist, fm4_hist, 
           g_global_hist,g_local_hist, g_global_percp_hist, g_local_percp_hist, g_global_recon_hist, g_local_recon_hist, gan_hist
          ,savedir='VTGAN'):
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    d1 = np.array(d1_hist)
    d2 = np.array(d2_hist)
    d3 = np.array(d3_hist)
    d4 = np.array(d4_hist)
    d5 = np.array(d1_cls_hist)
    d6 = np.array(d2_cls_hist)
    fm1 = np.array(fm1_hist)
    fm2 = np.array(fm2_hist)
    g_global = np.array(g_global_hist)
    g_local = np.array(g_local_hist)
    g_g_per = np.array(g_global_percp_hist)
    g_l_per = np.array(g_local_percp_hist)
    g_g_rec = np.array(g_global_recon_hist)
    g_l_rec = np.array(g_local_recon_hist)
    gan = np.array(gan_hist)
    df = pd.DataFrame(data=(d1,d2,d3,d4,d5,d6,fm1,fm2,g_global,g_local,g_g_per,g_l_per,g_g_rec,g_l_rec,gan)).T
    df.columns=["d1","d2","d3","d4","d5","d6","fm1","fm2","g_global","g_local","g_g_per","g_l_per","g_g_rec","g_l_rec","gan"]
    filename = save_dir+"/vtgan-loss.csv"
    df.to_csv(filename)
