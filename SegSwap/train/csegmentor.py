import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
sys.path.append('../')
from model.dinov2vit import dinov2_vit_base, dinov2_vit_large, dinov2_vit_giant
from model.dinov3vit import dinov3_vit_base, dinov3_vit_large
from model.dinov3convnext import get_convnext_arch
    
class ClsBranch(nn.Module):
    def __init__(self, token_dim):
        super(ClsBranch, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(token_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
    def forward(self, cls_token):
        if len(cls_token.shape) > 2: # [batch_size, 1, token_dim]
            cls_token = cls_token.squeeze(1)
        return self.classifier(cls_token)
    
class ConditionalSegmentationModel(nn.Module):
    def __init__(self, 
                 feat_extractor='cn2_tiny',  # Options: 'convnextv2', 'vit'
                 extractor_depth=2,  # Feature map downsampling level (e.g., 16, 32)
                 backbone_size='base',  # Options: 'base', 'large'
                 image_size=480,
                 upsampler='bilinear',  # Options: 'bilinear', 'convex'
                 backbone_type='mae',  # Options: 'mae', 'dinov2', 'dinov3', 'deit3'
                 n_aux_layers=1,
                 num_register_tokens=0):
        super().__init__()
        
        self.extractor_depth = extractor_depth
        self.backbone_type = backbone_type
        if self.backbone_type == 'dinov2':
            self.patch_size = 14
        else:
            self.patch_size = 16

        encoders = {
            'dinov3_cn_tiny': (get_convnext_arch('convnext_tiny'), {1: 192, 2: 384, 3: 768}),
            'dinov3_cn_small': (get_convnext_arch('convnext_small'), {1: 192, 2: 384, 3: 768}),
            'dinov3_cn_base': (get_convnext_arch('convnext_base'), {1: 256, 2: 512, 3: 1024}),
            'dinov3_cn_large': (get_convnext_arch('convnext_large'), {1: 384, 2: 768, 3: 1536}),
        }

        if feat_extractor in encoders:
            encoder_fn, feat_dims = encoders[feat_extractor]
            self.encoder = encoder_fn()
            self.feat_dim = feat_dims.get(self.extractor_depth)

        self.backbone, embed_dim = self._init_backbone(backbone_size, image_size, num_register_tokens)
            
        self.sigmoid = nn.Sigmoid()
        self.projector = nn.Conv2d(self.feat_dim, embed_dim, kernel_size=1)
        
        # Upsampler for final mask generation
        if upsampler == 'bilinear' : 
            self.upsample_func = self.bilinear_upsampler
        elif upsampler == 'convex' : 
            self.upsample_func = self.convex_upsampler
            self.upsampler = nn.Sequential(
                nn.Conv2d(embed_dim, embed_dim //8, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(embed_dim //8, 9 * self.patch_size * self.patch_size, kernel_size=1)
            )
        
        self.cls_branch = ClsBranch(embed_dim)
        self.final = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim // 2, 1, kernel_size=1, padding = 0)
        )
        self.n_aux_layers = n_aux_layers
        self.num_register_tokens = num_register_tokens
        
    def _init_backbone(self, backbone_size, img_size, num_register_tokens):
        
        if self.backbone_type == 'dinov2':
            if backbone_size == 'base':
                return dinov2_vit_base(img_size=img_size, patch_size=self.patch_size, num_register_tokens=num_register_tokens), 768
            elif backbone_size == 'large':
                return dinov2_vit_large(img_size=img_size, patch_size=self.patch_size, num_register_tokens=num_register_tokens), 1024
            elif backbone_size == 'giant':
                return dinov2_vit_giant(img_size=img_size, patch_size=self.patch_size, num_register_tokens=num_register_tokens), 1536
            else:
                raise ValueError("Unsupported backbone size")
        
        if self.backbone_type == 'dinov3':
            if backbone_size == 'base':
                return dinov3_vit_base(img_size=img_size, patch_size=self.patch_size, n_storage_tokens=num_register_tokens), 768
            elif backbone_size == 'large':
                return dinov3_vit_large(img_size=img_size, patch_size=self.patch_size, n_storage_tokens=num_register_tokens), 1024
            else:
                raise ValueError("Unsupported backbone size")
            
        raise ValueError("Unsupported backbone type")
          
    def compute_conditional_feature(self, source_features, source_mask):

        source_features = source_features[self.extractor_depth]
        B, C, H, W = source_features.shape

        source_mask_resized = F.interpolate(source_mask, size=(H, W), mode='bilinear', align_corners=False)
        weighted_features = source_features * source_mask_resized
        conditional_feature = torch.sum(weighted_features, dim=(2, 3), keepdim=True) / (torch.sum(source_mask_resized, dim=(2, 3), keepdim=True) + 1e-6)
        
        return conditional_feature
      
    def upsample_mask(self, mask, weight, kernel_size=3):
        N, C, H, W = mask.shape
        weight = weight.view(N, 1, kernel_size * kernel_size, self.patch_size, self.patch_size, H, W).contiguous() # should be the patch size in vit, not the extractor depth
        weight = torch.softmax(weight, dim=2)
        
        up_mask = F.unfold(mask, kernel_size, padding=kernel_size // 2)
        up_mask = up_mask.view(N, C, kernel_size * kernel_size, 1, 1, H, W)
        up_mask = torch.sum(weight * up_mask, dim=2)
        up_mask = up_mask.permute(0, 1, 4, 2, 5, 3).contiguous()   
        return up_mask.reshape(N, C, self.patch_size*H, self.patch_size*W).contiguous()
    
    def convex_upsampler(self, encoded_features) : 
        weight = self.upsampler(encoded_features)
        upsample_feat = self.upsample_mask(encoded_features, weight)
        return upsample_feat
    
    def bilinear_upsampler(self, encoded_features) : 
        return F.interpolate(encoded_features, scale_factor=self.patch_size, mode='bilinear')

    def forward(self, source_img, source_mask, target_img):
        source_features = self.encoder(source_img) 
        # source_features[self.extractor_depth]: [1, 512, 32, 32], source_img: [1, 3, 518, 518]
        weighted_source_features = self.compute_conditional_feature(source_features, source_mask) 
        # [1, 512, 1, 1]
        weighted_source_features = self.projector(weighted_source_features) 
        # [1, 1024, 1, 1]
        
        encoded_features_list, cls_token_list, cond_feat_list = self.backbone(weighted_source_features, target_img, self.n_aux_layers)
        mask_list = [self.final(encoded_features_i) for encoded_features_i in encoded_features_list]
        cls_score_list = [self.cls_branch(cls_token_i).squeeze(1) for cls_token_i in cls_token_list]
        final_mask_list = [self.upsample_func(mask_i) for mask_i in mask_list]
        
        return final_mask_list, cls_score_list

if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test DINOv3 integration
    print("Testing DINOv3 Base integration...")
    model_dinov3_base = ConditionalSegmentationModel(
        image_size=518, 
        backbone_type='dinov3', 
        backbone_size='base'
    ).to(device)
    
    print("Testing DINOv3 Large integration...")
    model_dinov3_large = ConditionalSegmentationModel(
        image_size=518, 
        backbone_type='dinov3', 
        backbone_size='large'
    ).to(device)
    
    # DINOv3 uses patch_size=14, so the image size should be a multiple of 14
    source_img = torch.randn(2, 3, 518, 518).to(device)  
    source_mask = torch.randn(2, 1, 518, 518).to(device)
    target_img = torch.randn(2, 3, 518, 518).to(device)

    print("Testing DINOv3 Base forward pass...")
    output_mask_list_base, cls_score_list_base = model_dinov3_base(source_img, source_mask, target_img)
    print("DINOv3 Base - Output mask shape:", output_mask_list_base[-1].shape)
    print("DINOv3 Base - Cls score shape:", cls_score_list_base[-1].shape)
    
    print("Testing DINOv3 Large forward pass...")
    output_mask_list_large, cls_score_list_large = model_dinov3_large(source_img, source_mask, target_img)
    print("DINOv3 Large - Output mask shape:", output_mask_list_large[-1].shape)
    print("DINOv3 Large - Cls score shape:", cls_score_list_large[-1].shape)
    
    print("DINOv3 integration test completed successfully!") 
       