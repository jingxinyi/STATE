import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import data_loader.data_loaders as module_data
# import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
import torch.nn as nn
import os
import imageio
import numpy as np

import skimage.io
from torchvision import datasets, transforms


def main(config):
    logger = config.get_logger('test')

    output_dir = config['object_test_dataset']['args']['output_dir']
    output_target_dir = config['object_test_dataset']['args'][
        'output_target_dir']
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    if not os.path.exists(output_target_dir):
        os.makedirs(output_target_dir)

    # setup data_loader instances
    dataset = getattr(module_data, config['object_test_dataset']['type'])(
        config['object_test_dataset']['args']['data_dir'],
        config['object_test_dataset']['args']['num_input'],
        config['object_test_dataset']['args']['object_type'],
        config['object_test_dataset']['args']['image_size'],
    )
    data_loader = DataLoader(
        dataset=dataset,
        batch_size=config['object_test_dataset']['args']['batch_size'],
        shuffle=False,
        num_workers=2,
        drop_last=True)
    # build model architecture
    model = config.init_obj('arch', module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    metric_fns = [getattr(module_metric, met) for met in config['metrics']]

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        model = torch.nn.DataParallel(model)
    model.load_state_dict(state_dict)

    # prepare model for testing
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    model.eval()

    total_metrics = torch.zeros(len(metric_fns)).cuda()

    s = 0

    with torch.no_grad():
        for i, (source_imgs, source_poses, target_img, target_pose,
                target_img_raw) in enumerate(tqdm(data_loader)):
            for i, source_img in enumerate(source_imgs):
                source_imgs[i] = source_img.to(device)

            for i, source_pose in enumerate(source_poses):
                source_poses[i] = source_pose.to(device)

            target_img = target_img.to(device)
            target_pose = target_pose.to(device)
            target_img_raw = target_img_raw.to(device)

            # output_flow_2_fusion = model.FlowModule2(source_imgs, source_poses, target_pose)
            output_flow_3d_fusion, visual_img, _, weight = model.FlowModule3D(
                source_imgs, source_poses, target_pose)
            # print(weight.shape)
            # output_gated = model.GatedModule(
            # torch.cat([output_flow_2_fusion.detach(), output_flow_3d_fusion.detach()], 1))

            batch_size = target_img.shape[0]
            output_flow_3d_fusion_list = torch.split(output_flow_3d_fusion,
                                                     1,
                                                     dim=0)
            # weight_list = torch.split(weight, 1, dim=0)
            # output_target_list = torch.split(target_img, 1, dim=0)
            # -------------------add
            # visual_img_list = torch.split(visual_img, 1, dim=0)
            # for i, img in enumerate(visual_img_list):
            #     img_name = "%05d" % (i + batch_size * s, )
            #     print(img.shape)
            #     img = torch.nn.functional.interpolate(img,
            #                                           size=(40, 40),
            #                                           mode="bilinear",
            #                                           align_corners=False)
            #     img = ((img) * 255).clamp(min=0, max=255)
            #     img = torch.squeeze(img, 0).transpose(1, 0).transpose(2, 1)
            #     img = np.around(img.cpu().numpy()).astype(
            #         np.uint8)  # around floor
            #     imageio.imwrite(os.path.join(output_dir, img_name + '.png'),
            #                     img)
            # ----------------------over

            # for i, img in enumerate(output_flow_3d_fusion_list):
            #     img_name = "%05d" % (i + batch_size * s, )
            #     # print(img.shape)
            #     img = torch.nn.functional.interpolate(img,
            #                                           size=(256, 256),
            #                                           mode="bilinear",
            #                                           align_corners=False)
            #     img = ((img + 1) * 255 / 2).clamp(min=0, max=255)
            #     img = torch.squeeze(img, 0).transpose(1, 0).transpose(2, 1)
            #     img = np.around(img.cpu().numpy()).astype(
            #         np.uint8)  # around floor
            # imageio.imwrite(os.path.join(output_dir, img_name+'.png'), img)

            for i, img in enumerate(weight):
                # img_name = "%05d" % (i + batch_size * s, )
                img = torch.nn.functional.interpolate(img,
                                                      size=(40, 40),
                                                      mode="bilinear",
                                                      align_corners=False)
                # img = ((img) * 255).clamp(min=0, max=255)
                # img = torch.squeeze(img, 0)
                img_list = torch.split(img, 1, dim=0)
                for j, img1 in enumerate(img_list):
                    img_name = "%05d" % (j + batch_size * s, )
                    img1 = torch.squeeze(img1, 0)
                    # img1 = torch.squeeze(img1, 0)
                    # resize = transforms.Resize([256, 256])
                    # img1 = resize(img1)
                    img1 = torch.squeeze(img1, 0)
                    # img1 = torch.unsqueeze(img1, 0)
                    # img1 = torch.squeeze(img1, 0).transpose(1,
                    #                                         0).transpose(2, 1)
                    # img2 = torch.squeeze(output_flow_3d_fusion_list[j], 0)
                    # img1 = torch.mul(img1, img2)
                    # img1 = img1 * img2
                    # img1 = img1.transpose(1, 0).transpose(2, 1)
                    img1 = ((img1) * 255).clamp(min=0, max=255)
                    img1 = np.around(img1.cpu().numpy()).astype(
                        np.uint8)  # around floor
                    # skimage.io.imsave(
                    #     os.path.join(output_dir,
                    #                  img_name + '_' + str(i) + '.png'), img1)
                # imageio.imwrite(os.path.join(output_dir, img_name + '.png'),
                #                 img)
            '''
            for i, img in enumerate(output_target_list):
                img_name = "%05d" % (i + batch_size * s,)
                img = torch.squeeze(img, 0).transpose(1, 0).transpose(2, 1)
            
            # #     # img = torch.squeeze(img, 0)
            #     # img = nn.UpsamplingNearest2d(size=256)(img)
            #     # print(img.shape)
                imageio.imwrite(os.path.join(output_target_dir, img_name+'.png'), img.cpu())
            
            '''
            s += 1

            # for i, metric in enumerate(metric_fns):
            #     total_metrics[i] += metric(nn.UpsamplingNearest2d(size=256)(output_gated), target_img_raw) * batch_size
            for i, metric in enumerate(metric_fns):
                total_metrics[i] += metric(output_flow_3d_fusion,
                                           target_img) * batch_size

    n_samples = len(data_loader.sampler)

    log = {}
    log.update({
        met.__name__: total_metrics[i].item() / n_samples
        for i, met in enumerate(metric_fns)
    })
    logger.info(log)


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='PyTorch Template')
    args.add_argument('-c',
                      '--config',
                      default=None,
                      type=str,
                      help='config file path (default: None)')
    args.add_argument('-r',
                      '--resume',
                      default=None,
                      type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d',
                      '--device',
                      default=None,
                      type=str,
                      help='indices of GPUs to enable (default: all)')

    config = ConfigParser.from_args(args)
    main(config)
