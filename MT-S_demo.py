# ========================================================================
# Perceptual Quality Assessment of Smartphone Photography
# PyTorch Version 1.0 by Hanwei Zhu
# Copyright(c) 2020 Yuming Fang, Hanwei Zhu, Yan Zeng, Kede Ma, and Zhou Wang
# All Rights Reserved.
#
# ----------------------------------------------------------------------
# Permission to use, copy, or modify this software and its documentation
# for educational and research purposes only and without fee is hereby
# granted, provided that this copyright notice and the original authors'
# names appear on all copies and supporting documentation. This program
# shall not be used, rewritten, or adapted as the basis of a commercial
# software or hardware product without first obtaining permission of the
# authors. The authors make no representations about the suitability of
# this software for any purpose. It is provided "as is" without express
# or implied warranty.
# ----------------------------------------------------------------------
# This is an implementation of Multi-Task learning from scene Semantics (MT-S)
# for blind image quality assessment.
# Please refer to the following paper:
#
# Y. Fang et al., "Perceptual Quality Assessment of Smartphone Photography" 
# in IEEE Conference on Computer Vision and Pattern Recognition, 2020
#
# Kindly report any suggestions or corrections to hanwei.zhu@outlook.com
# ========================================================================

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
import torchvision
from Prepare_image import Image_load
from PIL import Image
import argparse
import  os

class MTS(nn.Module):
	def __init__(self, config):
		super(MTS, self).__init__()
		self.config = config
		self.backbone_semantic = torchvision.models.resnet50(pretrained=False)
		self.backbone_quality = torchvision.models.resnet50(pretrained=False)
		fc_feature = self.backbone_quality.fc.in_features
		self.backbone_quality.fc = nn.Linear(fc_feature, 1, bias=True)
		self.backbone_semantic.fc = nn.Linear(fc_feature, self.config.output_channels, bias=True)

	def forward(self, x):
		batch_size = x.size()[0]

		#Shared layers
		x = self.backbone_quality.conv1(x)
		x = self.backbone_quality.bn1(x)
		x = self.backbone_quality.relu(x)
		x = self.backbone_quality.maxpool(x)
		x = self.backbone_quality.layer1(x)
		x = self.backbone_quality.layer2(x)
		x = self.backbone_quality.layer3(x)


		#Image quality task
		x1 = self.backbone_quality.layer4(x)
		x2 = self.backbone_quality.avgpool(x1)
		x2 = x2.squeeze(2).squeeze(2)

		quality_result = self.backbone_quality.fc(x2)
		quality_result = quality_result.view(batch_size, -1)

		#Scen semantic task
		xa = self.backbone_semantic.layer4(x)
		xb = self.backbone_semantic.avgpool(xa)
		xb = xb.squeeze(2).squeeze(2)

		semantic_result = self.backbone_semantic.fc(xb)
		semantic_result = semantic_result.view(batch_size, -1)


		return quality_result, semantic_result

class Demo(object):
	def __init__(self, config, load_weights=True, checkpoint_dir='./weights/MT-S_release.pt' ):
		self.config = config
		self.load_weights = load_weights
		self.checkpoint_dir = checkpoint_dir

		self.prepare_image = Image_load(size=512, stride=224)

		self.model = MTS(self.config)
		self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
		self.model.to(self.device)
		self.model_name = type(self.model).__name__

		if self.load_weights:
			self.initialize()

	def predit_quality(self):
		image_1 = self.prepare_image(Image.open(self.config.image_1).convert("RGB"))
		image_2 = self.prepare_image(Image.open(self.config.image_2).convert("RGB"))

		image_1 = image_1.to(self.device)
		self.model.eval()
		score_1, _ = self.model(image_1)
		print(score_1.mean().item())
		image_2 = image_2.to(self.device)
		score_2, _ = self.model(image_2)
		print(score_2.mean().item())

	def initialize(self):
		ckpt_path = self.checkpoint_dir
		could_load = self._load_checkpoint(ckpt_path)
		if could_load:
			print('Checkpoint load successfully!')
		else:
			raise IOError('Fail to load the pretrained model')

	def _load_checkpoint(self, ckpt):
		if os.path.isfile(ckpt):
			print("[*] loading checkpoint '{}'".format(ckpt))
			checkpoint = torch.load(ckpt)
			self.model.load_state_dict(checkpoint['state_dict'])
			return True
		else:
			return False

def parse_config():
	parser = argparse.ArgumentParser()
	parser.add_argument('--image_1', type=str, default='./images/05293.png')
	parser.add_argument('--image_2', type=str, default='./images/00914.png')
	parser.add_argument('--output_channels', type=int, default=9)
	return parser.parse_args()

def main():
	cfg = parse_config()
	t = Demo(config=cfg)
	t.predit_quality()

if __name__ == '__main__':
	main()
		









