# -*- coding:utf-8 -*-
# author: Xinge
# @file: train_cylinder_asym.py
import os
import time
import argparse
import sys
import numpy as np
import torch
import torch.optim as optim
from tqdm import tqdm

from utils.metric_util import per_class_iu, fast_hist_crop
from dataloader.pc_dataset import get_SemKITTI_label_name, get_cityscape_label_name
from builder import data_builder, model_builder, loss_builder
from config.config import load_config_data

from utils.load_save_util import load_checkpoint

import warnings

warnings.filterwarnings("ignore")


def main(args):
    pytorch_device = torch.device('cuda:0')

    config_path = args.config_path
    print("Loading config : " , config_path)
    configs = load_config_data(config_path)

    dataset_config = configs['dataset_params']
    train_dataloader_config = configs['train_data_loader']
    val_dataloader_config = configs['val_data_loader']

    val_batch_size = val_dataloader_config['batch_size']
    train_batch_size = train_dataloader_config['batch_size']

    model_config = configs['model_params']
    train_hypers = configs['train_params']

    grid_size = model_config['output_shape']
    num_class = model_config['num_class']
    ignore_label = dataset_config['ignore_label']
    expname = train_hypers['exp_path']
    model_load_path = os.path.join(expname,train_hypers['model_load_path'])
    model_save_path = os.path.join(expname,train_hypers['model_save_path'])
    os.makedirs(expname, exist_ok=True)
    frozen_lr = configs['f_lr']['frozen_lr']
    print(frozen_lr)

    if args.nus32 == False:
        cityscape_label_name = get_cityscape_label_name(dataset_config["label_mapping"])
        unique_label = np.asarray(sorted(list(cityscape_label_name.keys())))
        unique_label_str = [cityscape_label_name[x] for x in unique_label ]
        print('cityscape_label_name', cityscape_label_name)
        print('unique_label', unique_label)
        print('unique_label_str', unique_label_str)
    else :
        nuscenes32_label_name = get_SemKITTI_label_name(dataset_config["label_mapping"])
        unique_label = np.asarray(sorted(list(nuscenes32_label_name.keys())))
        unique_label_str = [nuscenes32_label_name[x] for x in unique_label ]
        print('\nnuscenes_label_name', nuscenes32_label_name)
        print('unique_label', unique_label)
        print('unique_label_str', unique_label_str)
    
    # exit()
    my_model = model_builder.build(model_config)
    if (args.resume or args.eval) and len([f for f in os.listdir(expname) if f.endswith('.pth')])>0:
        print('load model')
        try:
            print("loading " , model_load_path)
            # model_load_path = os.path.join('/SSD_DISK/users/kuangshaochen/Cylinder3D/exp/nerflidar_city_v6/000015.pth')
            my_model = load_checkpoint(model_load_path, my_model)
        except:
            ckpt = sorted(os.listdir(expname))[-1]
            print("without version_pth, loading" , os.path.join(expname,ckpt))
            my_model = load_checkpoint(os.path.join(expname,ckpt), my_model)

    my_model.to(pytorch_device)
    optimizer = optim.Adam(my_model.parameters(), lr=train_hypers["learning_rate"])

    loss_func, lovasz_softmax = loss_builder.build(wce=True, lovasz=True,
                                                   num_class=num_class, ignore_label=ignore_label)
    train_dataset_loader, val_dataset_loader = data_builder.build(dataset_config,
                                                                  train_dataloader_config,
                                                                  val_dataloader_config,
                                                                  grid_size=grid_size)
    # quit()
    # training
    epoch = 0
    best_val_miou = 0
    my_model.train()
    global_iter = 0
    check_iter = train_hypers['eval_every_n_steps']
    
    def eval_val(my_model):
        print('begin eval')
        
        my_model.eval()
        hist_list = []
        val_loss_list = []
        with torch.no_grad():
            for i_iter_val, (_, val_vox_label, val_grid, val_pt_labs, val_pt_fea) in enumerate(
                    val_dataset_loader):
                val_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in
                                    val_pt_fea]
                val_grid_ten = [torch.from_numpy(i).to(pytorch_device) for i in val_grid]
                val_label_tensor = val_vox_label.type(torch.LongTensor).to(pytorch_device)
                val_batch_size = val_vox_label.shape[0]
                predict_labels = my_model(val_pt_fea_ten, val_grid_ten, val_batch_size)

                # aux_loss = loss_fun(aux_outputs, point_label_tensor)
                loss = lovasz_softmax(torch.nn.functional.softmax(predict_labels).detach(), val_label_tensor,
                                        ignore=ignore_label) + loss_func(predict_labels.detach(), val_label_tensor)
                predict_labels = torch.argmax(predict_labels, dim=1)
                predict_labels = predict_labels.cpu().detach().numpy()
                ################visiualize
                # temp_label = val_vox_label.detach().numpy()
                # np.save('/SSD_DISK/users/kuangshaochen/Cylinder3D/visiual_real/predicts/{:04d}.npy'.format(i_iter_val),predict_labels)
                # print("val num" , predict_labels)
            
                ##################################
                for count, i_val_grid in enumerate(val_grid):
                    # print("val num" , i_iter_val)
                    # np.save('/SSD_DISK/users/kuangshaochen/Cylinder3D/visiual_real/labels/{:04d}.npy'.format(i_iter_val), 
                    # predict_labels[count, val_grid[count][:, 0], val_grid[count][:, 1],
                                                        # val_grid[count][:, 2]].flatten())

                    hist_list.append(fast_hist_crop(predict_labels[
                                                        count, val_grid[count][:, 0], val_grid[count][:, 1],
                                                        val_grid[count][:, 2]], val_pt_labs[count],#unique_label
                                                    unique_label[:-1]))
                val_loss_list.append(loss.detach().cpu().numpy())
                if i_iter_val == 9:
                    quit()
                # import pdb;pdb.set_trace()
        iou = per_class_iu(sum(hist_list))
        # import pdb;pdb.set_trace()
        # import pdb;pdb.set_trace()
        # print('Validation per class iou: ')
        for class_name, class_iou in zip(unique_label_str, iou):
            print('%s : %.2f%%' % (class_name, class_iou * 100))
        val_miou = np.nanmean(iou) * 100
        del val_vox_label, val_grid, val_pt_fea, val_grid_ten
        return val_miou,val_loss_list
    if args.eval:
        my_model.eval()
        eval_miou,_ = eval_val(my_model)
        print('The miou is  %.3f' % (eval_miou))
        exit()
    while epoch < train_hypers['max_num_epochs']:
        loss_list = []
        pbar = tqdm(total=len(train_dataset_loader))
        time.sleep(10)
        if epoch>1 and epoch %5==0:
            torch.save(my_model.state_dict(), os.path.join(expname,'{:06d}.pth'.format(epoch)))
        # lr_scheduler.step(epoch)
        for i_iter, (_, train_vox_label, train_grid, _, train_pt_fea) in enumerate(train_dataset_loader):
            if global_iter % check_iter == 0  and epoch>1:
            # if global_iter % check_iter == 0 or epoch % 10==0:
                # At this time eval the model on the validataion set
                my_model.eval()
                val_miou,val_loss_list = eval_val(my_model)    
                my_model.train()
                # save model if performance is improved
                if best_val_miou < val_miou:
                    best_val_miou = val_miou
                    torch.save(my_model.state_dict(), model_save_path)

                print('Current val miou is %.3f while the best val miou is %.3f' %
                      (val_miou, best_val_miou))
                with open(os.path.join(expname,'results.txt'),'a+') as f:
                    f.write('\n')
                    f.write('Current iter is %6d ,val miou is %.3f while the best val miou is %.3f' % (global_iter,val_miou, best_val_miou))
                print('Current val loss is %.3f' %
                      (np.mean(val_loss_list)))

            train_pt_fea_ten = [torch.from_numpy(i).type(torch.FloatTensor).to(pytorch_device) for i in train_pt_fea]
            # train_grid_ten = [torch.from_numpy(i[:,:2]).to(pytorch_device) for i in train_grid]
            train_vox_ten = [torch.from_numpy(i).to(pytorch_device) for i in train_grid]
            point_label_tensor = train_vox_label.type(torch.LongTensor).to(pytorch_device)
            train_batch_size = train_vox_label.shape[0]
            # forward + backward + optimize
            outputs = my_model(train_pt_fea_ten, train_vox_ten, train_batch_size)
            loss = lovasz_softmax(torch.nn.functional.softmax(outputs), point_label_tensor, ignore=ignore_label) + loss_func(
                outputs, point_label_tensor)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            # if i_iter == 50 :
            #     print("against virus" , "save in " , model_save_path)
            #     torch.save(my_model.state_dict(), model_save_path)
            if global_iter % 1000 == 0:
                if len(loss_list) > 0:
                    print('epoch %d iter %5d, loss: %.3f\n' %
                          (epoch, i_iter, np.mean(loss_list)))
                else:
                    print('loss error')

            optimizer.zero_grad()
            if frozen_lr :
                # print("frozen lr")
                for params in optimizer.param_groups: 
                    params['lr'] = 0.001
            pbar.update(1)
            global_iter += 1
            if global_iter % check_iter == 0:
                if len(loss_list) > 0:
                    print('epoch %d iter %5d, loss: %.3f\n' %
                          (epoch, i_iter, np.mean(loss_list)))
                else:
                    print('loss error')
        torch.save(my_model.state_dict(), os.path.join(expname,'iter{:06d}.pth'.format(global_iter)))
        pbar.close()
        epoch += 1


if __name__ == '__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('-y', '--config_path', default='config/NeRFlidar_city.yaml')
    parser.add_argument('--eval',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--nus32',default = False)
    parser.add_argument('--frozen_lr',action='store_true')
    args = parser.parse_args()

    print(' '.join(sys.argv))
    print(args)
    main(args)
