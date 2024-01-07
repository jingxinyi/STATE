import torch
from base import BaseTrainer
from utils import inf_loop, MetricTracker
import gc
import torch.nn as nn


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader, gan_start_step,
                 valid_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader

        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.gan_criterion = nn.BCEWithLogitsLoss()
        print('Length of data loader: {}'.format(len(self.data_loader)))

        self.do_validation = self.valid_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = 100
        self.max_bs_to_show = 2
        self.gan_start_step = gan_start_step
        self.gan_flag = False
        self.gan_gen_k = 0
        self.gan_disc_k = 0

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _forward(self):
        # Flow 3d module
        self.output_flow_3d_fusion, self.outputs_flow_x, self.outputs_flow_y = self.model.FlowModule3D(self.source_imgs, self.source_poses, self.target_pose)

    def _backward_g(self):
        total_loss = 0

        loss_flow_3d_fusion, self.loss_flow_3d_fusion_list = self.criterion(self.output_flow_3d_fusion, self.target_img)
        total_loss += loss_flow_3d_fusion

        self.outputs_img = [self.output_flow_3d_fusion]

        loss_gen = 0
        for output_img in self.outputs_img:
            gen = self.model.DiscriminatorModule(output_img)
            # loss_gen += torch.mean((gen - torch.ones_like(gen)) ** 2) / len(self.outputs_img) * self.gan_gen_k
            loss_gen += self.gan_criterion(gen, torch.ones_like(gen)) / len(self.outputs_img) * self.gan_gen_k
        total_loss += loss_gen

        self.loss_gen = loss_gen.detach().item()

        total_loss.backward()
        self.total_loss = total_loss.detach().item()

    def _backward_d(self):
        loss_disc = 0
        real = self.model.DiscriminatorModule(self.target_img)
        loss_disc += torch.mean((real - torch.ones_like(real)) ** 2) * self.gan_disc_k
        for output_img in self.outputs_img:
            fake = self.model.DiscriminatorModule(output_img.detach())
            # loss_disc += torch.mean(fake - torch.zeros_like(fake)) ** 2 / len(self.outputs_img) * self.gan_disc_k
            loss_disc += self.gan_criterion(fake, torch.zeros_like(fake)) / len(self.outputs_img) * self.gan_disc_k

        loss_disc.backward()
        self.loss_disc = loss_disc.detach().item()

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, data in enumerate(self.data_loader):
            gc.collect()

            self.source_imgs, self.source_poses, self.target_img, self.target_pose = self.parse_input_data(*data)
            bs, _, _, _ = self.source_imgs[0].shape
            total_step = (epoch - 1) * self.len_epoch + batch_idx

            if not self.gan_flag:
                if total_step >= self.gan_start_step:
                    print('GAN started.')
                    self.gan_flag = True
                    self.gan_gen_k = 0.05
                    self.gan_disc_k = 1

            self._forward()
            self.optimizer['flow_3d'].zero_grad()
            self._backward_g()
            self.optimizer['flow_3d'].step()

            self.optimizer['disc'].zero_grad()
            self._backward_d()
            self.optimizer['disc'].step()

            # Logging
            self.writer.set_step(total_step)

            self.writer.add_scalar('loss_flow_3d_fusion/loss_l1_flow_3d', self.loss_flow_3d_fusion_list[0])
            self.writer.add_scalar('loss_flow_3d_fusion/loss_ssim_flow_3d', self.loss_flow_3d_fusion_list[1])
            self.writer.add_scalar('loss_flow_3d_fusion/loss_vgg_flow_3d', self.loss_flow_3d_fusion_list[2])
            # self.writer.add_scalar('loss_flow_3d_fusion/loss_vgg_style_flow_3d', self.loss_flow_3d_fusion_list[3])

            self.writer.add_scalar('loss_gan/loss_disc', self.loss_disc)
            self.writer.add_scalar('loss_gan/loss_gen', self.loss_gen)

            self.train_metrics.update('loss', self.total_loss)

            if batch_idx % self.log_step == 0:
                for met in self.metric_ftns:
                    self.train_metrics.update(met.__name__, met(self.output_flow_3d_fusion, self.target_img).item())

                self.logger.debug('Train Epoch: {} {} '
                                  'loss: {:.3f}'.format(epoch, self._progress(batch_idx), self.total_loss))
                #self.logger.debug('L1 loss:{}, ssim loss:{}, vgg loss:{}'.format(self.loss_flow_3d_fusion_list[0], self.loss_flow_3d_fusion_list[1], self.loss_flow_3d_fusion_list[2]))

                final_vis = self.get_visualization(self.source_imgs, self.target_img, self.outputs_flow_x, self.outputs_flow_y, self.output_flow_3d_fusion)
                for i in range(min(bs, self.max_bs_to_show)):
                    self.writer.add_image('visualization/{}'.format(i), final_vis[i, ...])

            if batch_idx == self.len_epoch:
                break

        log = self.train_metrics.result()

        for _ in self.lr_scheduler.values():
            _.step()

        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})

        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, data in enumerate(self.valid_data_loader):
                self.source_imgs, self.source_poses, self.target_img, self.target_pose = self.parse_input_data(*data)

                bs, _, _, _ = self.source_imgs[0].shape
                total_step = (epoch - 1) * self.len_epoch + batch_idx

                self._forward()
                self.writer.set_step(total_step, mode='val')

                if batch_idx % self.log_step == 0:
                    final_vis = self.get_visualization(self.source_imgs, self.target_img, self.outputs_flow_x,
                                                       self.outputs_flow_y, self.output_flow_3d_fusion)

                for i in range(min(bs, self.max_bs_to_show)):
                    self.writer.add_image('visualization/{}'.format(i), final_vis[i, ...])

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)

    @staticmethod
    def get_visualization(source_imgs, target_img, outputs_flow_x, outputs_flow_y, output_flow_3d_fusion):

        source_imgs_vis = (torch.cat(source_imgs, -1) + 1) / 2
        output_flow_3d_fusion_vis = torch.cat(
            [(output_flow_3d_fusion + 1) / 2] + [torch.ones_like(target_img)] * (len(source_imgs) - 1), -1)

        Upsample = nn.Upsample(size=(target_img.shape[-2], target_img.shape[-1]))
        outputs_flow_x = [Upsample(v) for v in outputs_flow_x]
        outputs_flow_y = [Upsample(v) for v in outputs_flow_y]
        outputs_flow_x_vis = (torch.cat(outputs_flow_x, -1).repeat(1, 3, 1, 1) + 1) / 2
        outputs_flow_y_vis = (torch.cat(outputs_flow_y, -1).repeat(1, 3, 1, 1) + 1) / 2

        left_vis = torch.cat([torch.ones_like(target_img),
                              torch.ones_like(target_img),
                              torch.ones_like(target_img),
                              (target_img + 1) / 2], -2)
        right_vis = output_flow_3d_fusion_vis
        '''
        torch.cat([torch.ones_like(target_img),
                               torch.ones_like(target_img),
                               torch.ones_like(target_img),
                               output_flow_3d_fusion_vis], -2)
        '''
        return right_vis#torch.cat([left_vis, right_vis], -1)

    def parse_input_data(self, source_imgs, source_poses, target_img, target_pose):
        for i, source_img in enumerate(source_imgs):
            source_imgs[i] = source_img.to(self.device)

        for i, source_pose in enumerate(source_poses):
            source_poses[i] = source_pose.to(self.device)

        target_img = target_img.to(self.device)
        target_pose = target_pose.to(self.device)

        return source_imgs, source_poses, target_img, target_pose
